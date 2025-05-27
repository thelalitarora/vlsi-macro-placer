import json
import random
import math
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import numpy as np # For numerical operations, especially gradients

# --- Data Structures (Same as before) ---
class PlacedMacro:
    """Represents a macro block with its dimensions, current position, and orientation."""
    def __init__(self, name, width, height, x=0, y=0, orientation='N'):
        self.name = name
        self.width = width
        self.height = height
        self.x = x # Bottom-left X coordinate
        self.y = y # Bottom-left Y coordinate
        self.orientation = orientation # 'N', 'R90', 'R180', 'R270'

    def get_corners(self):
        """Returns the (x_min, y_min, x_max, y_max) of the macro based on its current orientation."""
        # Note: For NLO, we fix orientation after initial random assignment.
        # So width/height will be fixed for the optimization loop.
        if self.orientation in ['N', 'R180']:
            return self.x, self.y, self.x + self.width, self.y + self.height
        else: # R90, R270
            return self.x, self.y, self.x + self.height, self.y + self.width

    def get_center(self):
        """Returns the (center_x, center_y) of the macro based on its current orientation."""
        if self.orientation in ['N', 'R180']:
            return self.x + self.width / 2, self.y + self.height / 2
        else: # R90, R270
            return self.x + self.height / 2, self.y + self.width / 2

    def set_orientation_randomly(self):
        """Randomly sets initial orientation and adjusts dimensions."""
        orientations = ['N', 'R90', 'R180', 'R270']
        self.orientation = random.choice(orientations)
        if self.orientation in ['R90', 'R270']:
            self.width, self.height = self.height, self.width # Swap dimensions

    def __repr__(self):
        return f"Macro(name={self.name}, x={self.x:.2f}, y={self.y:.2f}, w={self.width:.2f}, h={self.height:.2f}, ori={self.orientation})"

class Port:
    """Represents a fixed die port with its location."""
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

    def get_center(self):
        """Returns the (x, y) coordinates of the port."""
        return self.x, self.y

    def __repr__(self):
        return f"Port(name={self.name}, x={self.x}, y={self.y})"

class Net:
    """Represents an interconnection between macros and/or ports."""
    def __init__(self, objects, weight=1.0):
        self.objects = objects # List of object names (macros or ports)
        self.weight = weight

    def __repr__(self):
        return f"Net(objects={self.objects}, weight={self.weight})"

class Blockage:
    """Represents a fixed forbidden area on the die."""
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.width = x_max - x_min
        self.height = y_max - y_min

    def get_corners(self):
        """Returns the (x_min, y_min, x_max, y_max) of the blockage."""
        return self.x_min, self.y_min, self.x_max, self.y_max

    def __repr__(self):
        return f"Blockage(x_min={self.x_min}, y_min={self.y_min}, w={self.width}, h={self.height})"

# --- Data Loading Function (Same as before) ---
def load_placement_data(json_file_path):
    """
    Loads placement data from a JSON file.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    die_x_min, die_y_min = data['die'][0][0], data['die'][0][1]
    die_x_max, die_y_max = data['die'][2][0], data['die'][2][1] # Assuming standard rectangular die from corners
    die_width = die_x_max - die_x_min
    die_height = die_y_max - die_y_min
    die_origin = (die_x_min, die_y_min)

    macros = [PlacedMacro(m['name'], m['width'], m['height']) for m in data['macros']]
    ports = [Port(p['name'], p['x'], p['y']) for p in data['ports']]
    nets = [Net(n['objects'], n.get('weight', 1.0)) for n in data['nets']]
    blockages = []
    for b_coords in data['blockages']:
        x_min = min(b_coords[0][0], b_coords[1][0])
        y_min = min(b_coords[0][1], b_coords[1][1])
        x_max = max(b_coords[0][0], b_coords[1][0])
        y_max = max(b_coords[0][1], b_coords[1][1])
        blockages.append(Blockage(x_min, y_min, x_max, y_max))

    return die_origin, die_width, die_height, macros, ports, nets, blockages

# --- Cost and Gradient Calculation Function (Same as before) ---
def calculate_cost_and_gradient(current_macros, all_ports, all_nets, die_width, die_height, die_origin_x, die_origin_y, blockages,
                                wirelength_weight=1.0, overlap_weight=1000.0, boundary_weight=500.0, blockage_weight=2000.0):
    """
    Calculates the total cost of a given macro placement and its gradient.
    The cost function is a weighted sum of squared Euclidean wirelength,
    quadratic penalties for macro overlaps, out-of-bounds placement,
    and overlaps with blockages.
    """
    total_cost = 0.0
    gradients = {m.name: [0.0, 0.0] for m in current_macros} # [grad_x, grad_y]
    macro_map = {m.name: m for m in current_macros}
    port_map = {p.name: p for p in all_ports}

    # --- 1. Wirelength Cost (Squared Euclidean Distance) ---
    for net in all_nets:
        net_obj_centers = []
        for obj_name in net.objects:
            if obj_name in macro_map:
                net_obj_centers.append((macro_map[obj_name], macro_map[obj_name].get_center()))
            elif obj_name in port_map:
                net_obj_centers.append((port_map[obj_name], port_map[obj_name].get_center()))

        if len(net_obj_centers) < 2:
            continue

        for i in range(len(net_obj_centers)):
            obj1, (cx1, cy1) = net_obj_centers[i]
            for j in range(i + 1, len(net_obj_centers)):
                obj2, (cx2, cy2) = net_obj_centers[j]

                dx = cx1 - cx2
                dy = cy1 - cy2
                
                cost_term = net.weight * (dx**2 + dy**2)
                total_cost += wirelength_weight * cost_term

                if isinstance(obj1, PlacedMacro):
                    gradients[obj1.name][0] += wirelength_weight * 2 * net.weight * dx
                    gradients[obj1.name][1] += wirelength_weight * 2 * net.weight * dy
                if isinstance(obj2, PlacedMacro):
                    gradients[obj2.name][0] += wirelength_weight * 2 * net.weight * (-dx)
                    gradients[obj2.name][1] += wirelength_weight * 2 * net.weight * (-dy)

    # --- 2. Macro-Macro Overlap Penalty (Quadratic) ---
    for i in range(len(current_macros)):
        m1 = current_macros[i]
        m1_x1, m1_y1, m1_x2, m1_y2 = m1.get_corners()
        
        for j in range(i + 1, len(current_macros)):
            m2 = current_macros[j]
            m2_x1, m2_y1, m2_x2, m2_y2 = m2.get_corners()

            overlap_x_len = max(0, min(m1_x2, m2_x2) - max(m1_x1, m2_x1))
            overlap_y_len = max(0, min(m1_y2, m2_y2) - max(m1_y1, m2_y1))

            if overlap_x_len > 0 and overlap_y_len > 0:
                overlap_area = overlap_x_len * overlap_y_len
                cost_term = overlap_area**2
                total_cost += overlap_weight * cost_term

                # Simplified gradients for overlap area
                grad_x_m1 = 0
                if m1_x1 < m2_x2 and m1_x2 > m2_x1:
                    if m1_x1 < m2_x1:
                        grad_x_m1 = -2 * overlap_area * overlap_y_len * (1 if m1_x2 < m2_x2 else 0)
                    else:
                        grad_x_m1 = 2 * overlap_area * overlap_y_len * (1 if m1_x1 > m2_x1 else 0)
                
                grad_y_m1 = 0
                if m1_y1 < m2_y2 and m1_y2 > m2_y1:
                    if m1_y1 < m2_y1:
                        grad_y_m1 = -2 * overlap_area * overlap_x_len * (1 if m1_y2 < m2_y2 else 0)
                    else:
                        grad_y_m1 = 2 * overlap_area * overlap_x_len * (1 if m1_y1 > m2_y1 else 0)

                gradients[m1.name][0] += overlap_weight * grad_x_m1
                gradients[m1.name][1] += overlap_weight * grad_y_m1
                gradients[m2.name][0] -= overlap_weight * grad_x_m1
                gradients[m2.name][1] -= overlap_weight * grad_y_m1


    # --- 3. Out-of-Bounds Penalty (Quadratic) ---
    for macro in current_macros:
        m_x1, m_y1, m_x2, m_y2 = macro.get_corners()

        if m_x1 < die_origin_x:
            dist = die_origin_x - m_x1
            cost_term = dist**2
            total_cost += boundary_weight * cost_term
            gradients[macro.name][0] += boundary_weight * 2 * dist * (-1)

        if m_x2 > die_origin_x + die_width:
            dist = m_x2 - (die_origin_x + die_width)
            cost_term = dist**2
            total_cost += boundary_weight * cost_term
            gradients[macro.name][0] += boundary_weight * 2 * dist * (1)

        if m_y1 < die_origin_y:
            dist = die_origin_y - m_y1
            cost_term = dist**2
            total_cost += boundary_weight * cost_term
            gradients[macro.name][1] += boundary_weight * 2 * dist * (-1)

        if m_y2 > die_origin_y + die_height:
            dist = m_y2 - (die_origin_y + die_height)
            cost_term = dist**2
            total_cost += boundary_weight * cost_term
            gradients[macro.name][1] += boundary_weight * 2 * dist * (1)

    # --- 4. Blockage Overlap Penalty (Quadratic) ---
    for macro in current_macros:
        m_x1, m_y1, m_x2, m_y2 = macro.get_corners()
        for blockage in blockages:
            b_x1, b_y1, b_x2, b_y2 = blockage.get_corners()

            ix1 = max(m_x1, b_x1)
            iy1 = max(m_y1, b_y1)
            ix2 = min(m_x2, b_x2)
            iy2 = min(m_y2, b_y2)

            overlap_x_len = max(0, ix2 - ix1)
            overlap_y_len = max(0, iy2 - iy1)

            if overlap_x_len > 0 and overlap_y_len > 0:
                overlap_area = overlap_x_len * overlap_y_len
                cost_term = overlap_area**2
                total_cost += blockage_weight * cost_term

                grad_x_macro = 0
                if m_x1 < b_x2 and m_x2 > b_x1:
                    if m_x1 < b_x1:
                        grad_x_macro = -2 * overlap_area * overlap_y_len * (1 if m_x2 < b_x2 else 0)
                    else:
                        grad_x_macro = 2 * overlap_area * overlap_y_len * (1 if m_x1 > b_x1 else 0)

                grad_y_macro = 0
                if m_y1 < b_y2 and m_y2 > b_y1:
                    if m_y1 < b_y1:
                        grad_y_macro = -2 * overlap_area * overlap_x_len * (1 if m_y2 < b_y2 else 0)
                    else:
                        grad_y_macro = 2 * overlap_area * overlap_x_len * (1 if m_y1 > b_y1 else 0)

                gradients[macro.name][0] += blockage_weight * grad_x_macro
                gradients[macro.name][1] += blockage_weight * grad_y_macro

    return total_cost, gradients

# --- Non-linear Optimization (Gradient Descent) Algorithm (Same as before) ---
def nlo_placement(macros, ports, nets, die_width, die_height, die_origin_x, die_origin_y, blockages,
                  learning_rate, num_iterations, random_seed=None):
    """
    Performs macro placement using a basic Non-linear Optimization (Gradient Descent) algorithm.
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Initial random placement for macros within the die boundaries
    # And fix random orientation for NLO phase
    for macro in macros:
        macro.x = random.randint(die_origin_x, die_origin_x + die_width - macro.width)
        macro.y = random.randint(die_origin_y, die_origin_y + die_height - macro.height)
        macro.set_orientation_randomly() # Set and fix initial random orientation

    current_placement = copy.deepcopy(macros)

    print("Starting Non-linear Optimization (Gradient Descent)...")
    for iteration in range(num_iterations):
        cost, gradients = calculate_cost_and_gradient(
            current_placement, ports, nets, die_width, die_height, die_origin_x, die_origin_y, blockages
        )

        for macro in current_placement:
            grad_x, grad_y = gradients[macro.name]
            macro.x -= learning_rate * grad_x
            macro.y -= learning_rate * grad_y

            # Clamp positions to stay within die bounds during optimization
            macro.x = max(die_origin_x, min(macro.x, die_origin_x + die_width - macro.width))
            macro.y = max(die_origin_y, min(macro.y, die_origin_y + die_height - macro.height))
        
        if iteration % 100 == 0:
            print(f"Iteration: {iteration}, Cost: {cost:.2f}")

    final_cost = calculate_cost_and_gradient(current_placement, ports, nets, die_width, die_height, die_origin_x, die_origin_y, blockages)[0]
    print(f"Optimization finished after {num_iterations} iterations. Final Cost: {final_cost:.2f}")
    return current_placement, final_cost

# --- NEW: Legalization Function to remove overlaps ---
def legalize_placement(macros, die_width, die_height, die_origin_x, die_origin_y, blockages, grid_unit=1.0, max_legalization_iters=100):
    """
    Legalizes the placement by iteratively resolving macro-macro and macro-blockage overlaps,
    and snapping macros to a grid.

    This is a simplified greedy legalization. More advanced legalizers exist.
    """
    print("\nStarting Legalization...")

    # Helper function to check if two rectangles overlap
    def check_overlap(rect1_x1, rect1_y1, rect1_x2, rect1_y2, rect2_x1, rect2_y1, rect2_x2, rect2_y2):
        return not (rect1_x2 <= rect2_x1 or rect1_x1 >= rect2_x2 or
                    rect1_y2 <= rect2_y1 or rect1_y1 >= rect2_y2)

    # Helper function to calculate overlap amount (positive if overlapping)
    def calculate_overlap_amount(rect1_x1, rect1_y1, rect1_x2, rect1_y2, rect2_x1, rect2_y1, rect2_x2, rect2_y2):
        overlap_x = max(0, min(rect1_x2, rect2_x2) - max(rect1_x1, rect2_x1))
        overlap_y = max(0, min(rect1_y2, rect2_y2) - max(rect1_y1, rect2_y1))
        return overlap_x, overlap_y

    for iteration in range(max_legalization_iters):
        overlaps_resolved_in_iter = 0

        # Create a copy for iteration to avoid issues with modifying list while iterating
        # Sort by x for consistent processing (e.g., left-to-right push)
        macros_to_process = sorted(macros, key=lambda m: m.x) 

        # 1. Resolve Macro-Macro Overlaps
        for i in range(len(macros_to_process)):
            m1 = macros_to_process[i]
            m1_x1, m1_y1, m1_x2, m1_y2 = m1.get_corners()

            for j in range(i + 1, len(macros_to_process)):
                m2 = macros_to_process[j]
                m2_x1, m2_y1, m2_x2, m2_y2 = m2.get_corners()

                if check_overlap(m1_x1, m1_y1, m1_x2, m1_y2, m2_x1, m2_y1, m2_x2, m2_y2):
                    overlap_x, overlap_y = calculate_overlap_amount(m1_x1, m1_y1, m1_x2, m1_y2, m2_x1, m2_y1, m2_x2, m2_y2)

                    # Simple greedy resolution: move the 'later' macro (m2) out
                    # Choose axis with smaller overlap amount to minimize disturbance
                    if overlap_x < overlap_y: # Shift along X-axis
                        if m1_x1 < m2_x1: # m1 is to the left of m2
                            # Move m2 to the right of m1
                            m2.x = m1_x2 
                        else: # m1 is to the right of m2
                            # Move m2 to the left of m1
                            m2.x = m1_x1 - (m2_x2 - m2_x1) 
                    else: # Shift along Y-axis
                        if m1_y1 < m2_y1: # m1 is below m2
                            # Move m2 above m1
                            m2.y = m1_y2
                        else: # m1 is above m2
                            # Move m2 below m1
                            m2.y = m1_y1 - (m2_y2 - m2_y1)

                    overlaps_resolved_in_iter += 1
                    # Re-get m2's corners after shift for subsequent checks in this pass
                    m2_x1, m2_y1, m2_x2, m2_y2 = m2.get_corners()


        # 2. Resolve Macro-Blockage Overlaps and ensure within die boundaries
        for macro in macros:
            m_x1, m_y1, m_x2, m_y2 = macro.get_corners()

            # Ensure within die boundaries first
            die_x_max_boundary = die_origin_x + die_width
            die_y_max_boundary = die_origin_y + die_height

            if m_x1 < die_origin_x:
                macro.x += (die_origin_x - m_x1) # Shift right to boundary
                overlaps_resolved_in_iter += 1
            if m_x2 > die_x_max_boundary:
                macro.x -= (m_x2 - die_x_max_boundary) # Shift left to boundary
                overlaps_resolved_in_iter += 1
            if m_y1 < die_origin_y:
                macro.y += (die_origin_y - m_y1) # Shift up to boundary
                overlaps_resolved_in_iter += 1
            if m_y2 > die_y_max_boundary:
                macro.y -= (m_y2 - die_y_max_boundary) # Shift down to boundary
                overlaps_resolved_in_iter += 1

            # Recalculate corners after boundary adjustments
            m_x1, m_y1, m_x2, m_y2 = macro.get_corners() 

            # Resolve overlaps with blockages
            for blockage in blockages:
                b_x1, b_y1, b_x2, b_y2 = blockage.get_corners()

                if check_overlap(m_x1, m_y1, m_x2, m_y2, b_x1, b_y1, b_x2, b_y2):
                    overlap_x, overlap_y = calculate_overlap_amount(m_x1, m_y1, m_x2, m_y2, b_x1, b_y1, b_x2, b_y2)

                    # Try to shift macro out of blockage
                    # Prioritize the smaller overlap axis
                    if overlap_x < overlap_y: # Shift along X-axis
                        if m_x1 < b_x1: # Macro is to the left of blockage
                            macro.x = b_x1 - (m_x2 - m_x1) # Move macro to the left of blockage
                        else: # Macro is to the right of blockage
                            macro.x = b_x2 # Move macro to the right of blockage
                    else: # Shift along Y-axis
                        if m_y1 < b_y1: # Macro is below blockage
                            macro.y = b_y1 - (m_y2 - m_y1) # Move macro below blockage
                        else: # Macro is above blockage
                            macro.y = b_y2 # Move macro above blockage
                    
                    # Re-clamp after shift to ensure it's still within die
                    current_macro_width = m_x2 - m_x1 # Use its current dimensions
                    current_macro_height = m_y2 - m_y1
                    macro.x = max(die_origin_x, min(macro.x, die_x_max_boundary - current_macro_width))
                    macro.y = max(die_origin_y, min(macro.y, die_y_max_boundary - current_macro_height))
                    
                    overlaps_resolved_in_iter += 1

        if overlaps_resolved_in_iter == 0:
            print(f"Legalization converged after {iteration+1} iterations.")
            break
        
        if iteration % 10 == 0:
            print(f"Legalization Iteration {iteration}: {overlaps_resolved_in_iter} overlaps resolved.")

    # 3. Snap to Grid
    # This should be the last step after all overlaps are resolved
    for macro in macros:
        macro.x = round(macro.x / grid_unit) * grid_unit
        macro.y = round(macro.y / grid_unit) * grid_unit

        # After snapping, ensure it's still within bounds (might slightly push out)
        current_macro_width = macro.get_corners()[2] - macro.get_corners()[0]
        current_macro_height = macro.get_corners()[3] - macro.get_corners()[1]
        macro.x = max(die_origin_x, min(macro.x, die_origin_x + die_width - current_macro_width))
        macro.y = max(die_origin_y, min(macro.y, die_origin_y + die_height - current_macro_height))

    print("Legalization complete.")
    return macros


# --- Visualization Function (Same as before) ---
def visualize_placement(macros, ports, nets, die_width, die_height, die_origin_x, die_origin_y, blockages, title="Placement"):
    """
    Visualizes the current placement of macros, ports, blockages, and nets on the die.
    """
    fig, ax = plt.subplots(1, figsize=(10, 8))

    # Draw Die
    die_rect = patches.Rectangle((die_origin_x, die_origin_y), die_width, die_height,
                                 linewidth=2, edgecolor='black', facecolor='lightgray', fill=True, label='Die')
    ax.add_patch(die_rect)

    # Draw Blockages
    for blockage in blockages:
        blockage_rect = patches.Rectangle((blockage.x_min, blockage.y_min), blockage.width, blockage.height,
                                          linewidth=1, edgecolor='red', facecolor='salmon', hatch='///', label='Blockage')
        ax.add_patch(blockage_rect)

    # Draw Macros
    macro_colors = plt.cm.tab20.colors # Use a colormap for different macro colors
    for i, macro in enumerate(macros):
        x1, y1, x2, y2 = macro.get_corners()
        
        macro_rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       linewidth=1, edgecolor='blue', facecolor=macro_colors[i % len(macro_colors)], alpha=0.8)
        ax.add_patch(macro_rect)
        ax.text(macro.x + (x2-x1) / 2, macro.y + (y2-y1) / 2, macro.name, # use current effective width/height
                ha='center', va='center', fontsize=8, color='black', weight='bold')


    # Draw Ports
    for port in ports:
        ax.plot(port.x, port.y, 'ro', markersize=8, alpha=0.7) # Red circles for ports
        ax.text(port.x, port.y + 10, port.name, ha='center', va='bottom', fontsize=8, color='red', weight='bold')

    # Draw Nets (represented as lines connecting centers of connected objects)
    macro_map = {m.name: m for m in macros}
    port_map = {p.name: p for p in ports}
    net_lines = []
    for net in nets:
        object_centers = []
        for obj_name in net.objects:
            if obj_name in macro_map:
                object_centers.append(macro_map[obj_name].get_center())
            elif obj_name in port_map:
                object_centers.append(port_map[obj_name].get_center())

        if len(object_centers) > 1:
            # Draw lines between all connected components (clique model for visualization)
            for i in range(len(object_centers)):
                for j in range(i + 1, len(object_centers)):
                    net_lines.append([object_centers[i], object_centers[j]])

    if net_lines:
        line_collection = LineCollection(net_lines, colors='gray', linestyle=':', linewidth=0.5, alpha=0.7)
        ax.add_collection(line_collection)

    ax.set_title(title)
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    
    # Set plot limits to cover the die and provide some padding
    ax.set_xlim(die_origin_x - 50, die_origin_x + die_width + 50)
    ax.set_ylim(die_origin_y - 50, die_origin_y + die_height + 50)
    
    ax.set_aspect('equal', adjustable='box') # Maintain aspect ratio
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# --- Main Execution Block ---
def main():
    json_data_path = 'placement_data.json'

    # --- Create the JSON input file with your provided data ---
    json_content = """
    {
      "die": [
        [0, 0], [0, 600], [800, 600], [800, 0], [0, 0]
      ],
      "blockages": [
        [[100, 100], [200, 200]],
        [[500, 300], [600, 400]]
      ],
      "macros": [
        { "name": "M1", "width": 80, "height": 80 },
        { "name": "M2", "width": 80, "height": 80 },
        { "name": "M3", "width": 80, "height": 80 },
        { "name": "M4", "width": 80, "height": 80 },
        { "name": "M5", "width": 80, "height": 80 },
        { "name": "M6", "width": 60, "height": 80 },
        { "name": "M7", "width": 60, "height": 80 },
        { "name": "M8", "width": 60, "height": 80 },
        { "name": "M9", "width": 60, "height": 80 },
        { "name": "M10", "width": 60, "height": 80 },
        { "name": "M11", "width": 50, "height": 50 },
        { "name": "M12", "width": 50, "height": 50 },
        { "name": "M13", "width": 50, "height": 50 },
        { "name": "M14", "width": 50, "height": 50 },
        { "name": "M15", "width": 50, "height": 50 },
        { "name": "M18", "width": 100, "height": 100 },
        { "name": "M19", "width": 100, "height": 100 }
      ],
      "ports": [
        { "name": "P1", "x": 300, "y": 0 },
        { "name": "P2", "x": 400, "y": 0 },
        { "name": "P3", "x": 500, "y": 0 },
        { "name": "P4", "x": 500, "y": 600 },
        { "name": "P5", "x": 600, "y": 600 },
        { "name": "P6", "x": 700, "y": 600 }
      ],
      "nets": [
        { "objects": ["M1", "M18"], "weight": 4.0 },
        { "objects": ["M2", "M18"], "weight": 4.0 },
        { "objects": ["M3", "M18"], "weight": 4.0 },
        { "objects": ["M4", "M18"], "weight": 4.0 },
        { "objects": ["M5", "M18"], "weight": 4.0 },
        { "objects": ["M6", "M18"], "weight": 4.0 },
        { "objects": ["M7", "M18"], "weight": 4.0 },
        { "objects": ["M8", "M18"], "weight": 4.0 },
        { "objects": ["M9", "M18"], "weight": 4.0 },
        { "objects": ["M10", "M18"], "weight": 4.0 },
        { "objects": ["M11", "M19"], "weight": 4.0 },
        { "objects": ["M12", "M19"], "weight": 4.0 },
        { "objects": ["M13", "M19"], "weight": 4.0 },
        { "objects": ["M14", "M19"], "weight": 4.0 },
        { "objects": ["M15", "M19"], "weight": 4.0 },
        { "objects": ["M19", "P1"], "weight": 6.0 },
        { "objects": ["M19", "P2"], "weight": 6.0 },
        { "objects": ["M19", "P3"], "weight": 6.0 },
        { "objects": ["M18", "P4"], "weight": 6.0 },
        { "objects": ["M18", "P5"], "weight": 6.0 },
        { "objects": ["M18", "P6"], "weight": 6.0 }
      ]
    }
    """
    with open(json_data_path, 'w') as f:
        f.write(json_content)
    print(f"Created '{json_data_path}' with provided data.")

    # Load data from the JSON file
    die_origin, die_width, die_height, macros, ports, nets, blockages = load_placement_data(json_data_path)

    print("\n--- Initial Placement (Random) ---")
    initial_macros_copy = copy.deepcopy(macros)
    # Ensure initial positions are set for the copy before cost calculation
    for macro in initial_macros_copy:
        macro.x = random.randint(die_origin[0], die_origin[0] + die_width - macro.width)
        macro.y = random.randint(die_origin[1], die_origin[1] + die_height - macro.height)
        macro.set_orientation_randomly()
    initial_cost, _ = calculate_cost_and_gradient(initial_macros_copy, ports, nets, die_width, die_height, die_origin[0], die_origin[1], blockages)
    visualize_placement(initial_macros_copy, ports, nets, die_width, die_height, die_origin[0], die_origin[1], blockages, "Initial Random Placement")
    print(f"Initial Cost: {initial_cost:.2f}")


    # --- Non-linear Optimization Parameters ---
    learning_rate = 0.5
    num_iterations = 5000
    random_seed = 42

    print("\n--- Starting Non-linear Optimization (Global Placement) ---")
    optimized_placement, nlo_final_cost = nlo_placement(
        macros, ports, nets, die_width, die_height, die_origin[0], die_origin[1], blockages,
        learning_rate, num_iterations, random_seed
    )

    print("\n--- Placement After NLO (May have overlaps) ---")
    # Recalculate cost with the actual optimized positions
    nlo_final_cost_check, _ = calculate_cost_and_gradient(optimized_placement, ports, nets, die_width, die_height, die_origin[0], die_origin[1], blockages)
    visualize_placement(optimized_placement, ports, nets, die_width, die_height, die_origin[0], die_origin[1], blockages, f"After NLO Global Placement (Cost: {nlo_final_cost_check:.2f})")
    print(f"NLO Final Cost: {nlo_final_cost_check:.2f}")

    # --- NEW: Legalization Step ---
    print("\n--- Starting Legalization (Detailed Placement) ---")
    legalized_placement = legalize_placement(
        optimized_placement, die_width, die_height, die_origin[0], die_origin[1], blockages,
        grid_unit=1.0, max_legalization_iters=200 # A higher max_iters might be needed for complex cases
    )

    print("\n--- Final Legalized Placement ---")
    # It's important to understand that legalization typically increases wirelength
    # because it resolves overlaps that the global placer preferred.
    final_legal_cost, _ = calculate_cost_and_gradient(legalized_placement, ports, nets, die_width, die_height, die_origin[0], die_origin[1], blockages)
    visualize_placement(legalized_placement, ports, nets, die_width, die_height, die_origin[0], die_origin[1], blockages, f"Final Legalized Placement (Cost: {final_legal_cost:.2f})")
    print(f"Final Legalized Cost: {final_legal_cost:.2f}")


    # --- Dump Final Macro Coordinates to JSON ---
    output_data = []
    for macro in legalized_placement: # Dump the legalized placement
        output_data.append({
            "name": macro.name,
            "x": macro.x,
            "y": macro.y,
            "width": macro.width,
            "height": macro.height,
            "orientation": macro.orientation
        })

    output_json_path = 'final_macro_placement_legalized.json'
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"\nFinal legalized macro coordinates dumped to '{output_json_path}'")

    print("\nLegalized Macro Positions:")
    for macro in legalized_placement:
        print(f"  {macro.name}: (x={macro.x:.2f}, y={macro.y:.2f}, w={macro.width:.2f}, h={macro.height:.2f}, ori='{macro.orientation}')")

if __name__ == "__main__":
    main()
 
