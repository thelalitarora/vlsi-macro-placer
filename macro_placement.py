import json
import random
import math
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

# --- Data Structures ---
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

    def rotate(self):
        """Rotates the macro 90 degrees clockwise and updates its dimensions."""
        if self.orientation == 'N':
            self.orientation = 'R90'
        elif self.orientation == 'R90':
            self.orientation = 'R180'
        elif self.orientation == 'R180':
            self.orientation = 'R270'
        else: # 'R270'
            self.orientation = 'N'
        self.width, self.height = self.height, self.width # Swap dimensions

    def __repr__(self):
        return f"Macro(name={self.name}, x={self.x}, y={self.y}, w={self.width}, h={self.height}, ori={self.orientation})"

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

# --- Data Loading Function ---
def load_placement_data(json_file_path):
    """
    Loads placement data from a JSON file.

    Args:
        json_file_path (str): Path to the JSON input file.

    Returns:
        tuple: (die_origin, die_width, die_height, macros, ports, nets, blockages)
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Die dimensions
    die_coords = data['die']
    # Assuming die is a rectangle defined by its corners, e.g., [[0,0],[0,Y],[X,Y],[X,0],[0,0]]
    die_x_coords = [p[0] for p in die_coords]
    die_y_coords = [p[1] for p in die_coords]
    die_x_min, die_y_min = min(die_x_coords), min(die_y_coords)
    die_x_max, die_y_max = max(die_x_coords), max(die_y_coords)
    die_width = die_x_max - die_x_min
    die_height = die_y_max - die_y_min
    die_origin = (die_x_min, die_y_min)

    # Macros
    macros = [PlacedMacro(m['name'], m['width'], m['height']) for m in data['macros']]

    # Ports
    ports = [Port(p['name'], p['x'], p['y']) for p in data['ports']]

    # Nets
    nets = [Net(n['objects'], n.get('weight', 1.0)) for n in data['nets']] # Default weight to 1.0

    # Blockages
    blockages = []
    for b_coords in data['blockages']:
        # Blockage format assumed: [[x1, y1], [x2, y2]] (bottom-left, top-right)
        x_min = min(b_coords[0][0], b_coords[1][0])
        y_min = min(b_coords[0][1], b_coords[1][1])
        x_max = max(b_coords[0][0], b_coords[1][0])
        y_max = max(b_coords[0][1], b_coords[1][1])
        blockages.append(Blockage(x_min, y_min, x_max, y_max))

    return die_origin, die_width, die_height, macros, ports, nets, blockages

# --- Cost Function ---
def calculate_cost(current_macros, all_ports, all_nets, die_width, die_height, die_origin_x, die_origin_y, blockages):
    """
    Calculates the total cost of a given macro placement.
    The cost function is a weighted sum of wirelength, macro overlaps, out-of-bounds placement,
    and overlaps with blockages.
    """
    cost = 0
    macro_map = {m.name: m for m in current_macros}
    port_map = {p.name: p for p in all_ports}

    # --- 1. Wirelength (HPWL - Half-Perimeter Wirelength) ---
    # Calculates HPWL for each net and sums them up, weighted by net weight.
    for net in all_nets:
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')

        valid_net_objects = 0
        for obj_name in net.objects:
            center_x, center_y = None, None
            if obj_name in macro_map:
                macro = macro_map[obj_name]
                center_x, center_y = macro.get_center()
            elif obj_name in port_map:
                port = port_map[obj_name]
                center_x, center_y = port.get_center()

            if center_x is not None:
                min_x = min(min_x, center_x)
                max_x = max(max_x, center_x)
                min_y = min(min_y, center_y)
                max_y = max(max_y, center_y)
                valid_net_objects += 1

        if valid_net_objects > 1: # Only calculate HPWL if there's more than one object
            hpwl = (max_x - min_x) + (max_y - min_y)
            cost += net.weight * hpwl

    # --- 2. Overlap Penalty (Macros overlapping with each other) ---
    # High penalty for overlaps to encourage non-overlapping solutions.
    overlap_penalty_weight = 100000 
    for i in range(len(current_macros)):
        for j in range(i + 1, len(current_macros)):
            m1 = current_macros[i]
            m2 = current_macros[j]

            # AABB (Axis-Aligned Bounding Box) intersection test
            m1_x1, m1_y1, m1_x2, m1_y2 = m1.get_corners()
            m2_x1, m2_y1, m2_x2, m2_y2 = m2.get_corners()

            if (m1_x1 < m2_x2 and m1_x2 > m2_x1 and
                m1_y1 < m2_y2 and m1_y2 > m2_y1):
                # Calculate overlap area and add to cost
                overlap_x = max(0, min(m1_x2, m2_x2) - max(m1_x1, m2_x1))
                overlap_y = max(0, min(m1_y2, m2_y2) - max(m1_y1, m2_y1))
                cost += overlap_penalty_weight * (overlap_x * overlap_y)

    # --- 3. Out-of-Bounds Penalty (Macros placed outside the die) ---
    # High penalty for macros spilling outside the defined die area.
    out_of_bounds_penalty_weight = 50000 
    for macro in current_macros:
        m_x1, m_y1, m_x2, m_y2 = macro.get_corners()

        # Check if any part of the macro is outside the die boundary
        if (m_x1 < die_origin_x or m_x2 > die_origin_x + die_width or
            m_y1 < die_origin_y or m_y2 > die_origin_y + die_height):
            
            # Calculate how much it's out of bounds and add to cost
            penalty_amount = (
                max(0, die_origin_x - m_x1) + # Left side spill
                max(0, m_x2 - (die_origin_x + die_width)) + # Right side spill
                max(0, die_origin_y - m_y1) + # Bottom side spill
                max(0, m_y2 - (die_origin_y + die_height)) # Top side spill
            )
            cost += out_of_bounds_penalty_weight * penalty_amount

    # --- 4. Blockage Penalty (Macros overlapping with fixed blockages) ---
    # Very high penalty for overlapping with blockages, as these are usually forbidden areas.
    blockage_penalty_weight = 200000 
    for macro in current_macros:
        m_x1, m_y1, m_x2, m_y2 = macro.get_corners()
        for blockage in blockages:
            b_x1, b_y1, b_x2, b_y2 = blockage.get_corners()

            if (m_x1 < b_x2 and m_x2 > b_x1 and
                m_y1 < b_y2 and m_y2 > b_y1):
                # Calculate overlap area with blockage and add to cost
                overlap_x = max(0, min(m_x2, b_x2) - max(m_x1, b_x1))
                overlap_y = max(0, min(m_y2, b_y2) - max(m_y1, b_y1))
                cost += blockage_penalty_weight * (overlap_x * overlap_y)

    return cost

# --- Simulated Annealing Algorithm ---
def simulated_annealing_placement(macros, ports, nets, die_width, die_height, die_origin_x, die_origin_y, blockages,
                                  initial_temp, cooling_rate, iterations_per_temp, random_seed=None):
    """
    Performs macro placement using the Simulated Annealing optimization algorithm.

    Args:
        macros (list): List of PlacedMacro objects.
        ports (list): List of Port objects.
        nets (list): List of Net objects.
        die_width (int): Width of the die.
        die_height (int): Height of the die.
        die_origin_x (int): X-coordinate of the die origin.
        die_origin_y (int): Y-coordinate of the die origin.
        blockages (list): List of Blockage objects.
        initial_temp (float): Starting temperature for annealing.
        cooling_rate (float): Rate at which temperature decreases (e.g., 0.99).
        iterations_per_temp (int): Number of moves attempted at each temperature step.
        random_seed (int, optional): Seed for random number generation for reproducibility.

    Returns:
        tuple: (best_placement_macros, best_cost)
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Initial random placement for macros within the die boundaries
    # This also initializes their x, y coordinates
    for macro in macros:
        # Place randomly within die bounds initially, accounting for macro dimensions
        macro.x = random.randint(die_origin_x, die_origin_x + die_width - macro.width)
        macro.y = random.randint(die_origin_y, die_origin_y + die_height - macro.height)
        # Optionally, randomize initial orientation
        if random.random() > 0.5:
            macro.rotate()

    current_placement = copy.deepcopy(macros) # Store current state of macros
    current_cost = calculate_cost(current_placement, ports, nets, die_width, die_height, die_origin_x, die_origin_y, blockages)
    
    best_placement = copy.deepcopy(current_placement) # Store the best placement found so far
    best_cost = current_cost

    temperature = initial_temp
    
    print("Starting Simulated Annealing...")
    while temperature > 0.1: # Annealing stops when temperature is very low
        print(f"Temp: {temperature:.2f} | Current Cost: {current_cost:.2f} | Best Cost: {best_cost:.2f}")

        for _ in range(iterations_per_temp):
            new_placement = copy.deepcopy(current_placement) # Create a copy to modify for the new state

            # --- Perform a random move on a randomly chosen macro ---
            # Randomly select a macro from the new_placement list
            macro_to_move = random.choice(new_placement)
            
            # Store its original state in case the move is rejected
            original_x, original_y, original_width, original_height, original_orientation = \
                macro_to_move.x, macro_to_move.y, macro_to_move.width, macro_to_move.height, macro_to_move.orientation

            # Choose a random type of move: 0=shift, 1=swap, 2=rotate
            move_type = random.randint(0, 2)

            if move_type == 0: # Shift a macro
                # Shift range decreases with temperature to allow finer adjustments later
                shift_limit = max(1, int(die_width * 0.05 * (temperature / initial_temp)))
                macro_to_move.x += random.randint(-shift_limit, shift_limit)
                macro_to_move.y += random.randint(-shift_limit, shift_limit)

                # Heuristic: Clamp coordinates to stay somewhat within the die.
                # This helps prevent macros from wandering too far and incurring massive penalties,
                # guiding the search towards feasible regions.
                macro_to_move.x = max(die_origin_x, min(macro_to_move.x, die_origin_x + die_width - macro_to_move.width))
                macro_to_move.y = max(die_origin_y, min(macro_to_move.y, die_origin_y + die_height - macro_to_move.height))

            elif move_type == 1 and len(new_placement) > 1: # Swap two macros
                # Ensure there's another macro to swap with
                other_macros = [m for m in new_placement if m.name != macro_to_move.name]
                if not other_macros: 
                    continue # Skip if no other macros are available
                macro2_to_swap = random.choice(other_macros)

                # Swap positions and orientations (and thus dimensions)
                # This is a powerful move for global exploration
                macro_to_move.x, macro2_to_swap.x = macro2_to_swap.x, macro_to_move.x
                macro_to_move.y, macro2_to_swap.y = macro2_to_swap.y, macro_to_move.y
                macro_to_move.orientation, macro2_to_swap.orientation = macro2_to_swap.orientation, macro_to_move.orientation
                macro_to_move.width, macro2_to_swap.width = macro2_to_swap.width, macro_to_move.width 
                macro_to_move.height, macro2_to_swap.height = macro2_to_swap.height, macro_to_move.height

            elif move_type == 2: # Rotate a macro
                macro_to_move.rotate()

            # Calculate cost of the new placement
            new_cost = calculate_cost(new_placement, ports, nets, die_width, die_height, die_origin_x, die_origin_y, blockages)

            # Decide whether to accept the new placement
            if new_cost < current_cost:
                # Always accept better solutions
                current_placement = new_placement
                current_cost = new_cost
                # Update overall best if current is better
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_placement = copy.deepcopy(current_placement)
            else:
                # Accept worse solutions with a probability based on temperature and cost difference
                delta_cost = new_cost - current_cost
                acceptance_probability = math.exp(-delta_cost / temperature)
                if random.random() < acceptance_probability:
                    current_placement = new_placement
                    current_cost = new_cost
                else:
                    # If the move is rejected, revert the macro to its original state
                    # This is important because macro_to_move is a reference to an object in new_placement.
                    # We need to explicitly undo the change if not accepted.
                    macro_to_move.x = original_x
                    macro_to_move.y = original_y
                    macro_to_move.width = original_width
                    macro_to_move.height = original_height
                    macro_to_move.orientation = original_orientation

        # Cool down the temperature for the next iteration
        temperature *= cooling_rate

    print("Simulated Annealing finished.")
    return best_placement, best_cost

# --- Visualization Function ---
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
        # Use get_corners() to get dimensions based on current orientation
        x1, y1, x2, y2 = macro.get_corners()
        
        macro_rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       linewidth=1, edgecolor='blue', facecolor=macro_colors[i % len(macro_colors)], alpha=0.8)
        ax.add_patch(macro_rect)
        ax.text(macro.x + macro.width / 2, macro.y + macro.height / 2, macro.name, # text uses stored w/h, which is fine for center
                ha='center', va='center', fontsize=8, color='black', weight='bold')
        # Optionally show orientation in text for debugging
        # ax.text(macro.x + macro.width / 2, macro.y + macro.height / 2 - 15, f"({macro.orientation})",
        #         ha='center', va='center', fontsize=6, color='black')


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
    json_data_path = 'placement_data.json' # The input JSON file

    # --- Create the JSON input file with your provided data ---
    json_content = """
    {
      "die": [
        [0, 0], [0, 0], [800, 600], [800, 0], [0, 600]
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
    # Visualize the initial random placement before optimization
    visualize_placement(macros, ports, nets, die_width, die_height, die_origin[0], die_origin[1], blockages, "Initial Random Placement")
    initial_cost = calculate_cost(macros, ports, nets, die_width, die_height, die_origin[0], die_origin[1], blockages)
    print(f"Initial Cost: {initial_cost:.2f}")

    # --- Simulated Annealing Parameters ---
    # These parameters are critical and often require tuning for different designs.
    initial_temp = 50000.0       # Starting temperature: high enough to accept many worse moves initially.
    cooling_rate = 0.995        # How quickly the temperature decreases (0.95 to 0.999 are common).
                                # Closer to 1 means slower cooling, more exploration, potentially better but longer.
    iterations_per_temp = 200   # Number of attempts/moves at each temperature step.
                                # More iterations allow better exploration at each temperature.
    random_seed = 42            # For reproducible results. Set to None for truly random runs.

    print("\n--- Starting Simulated Annealing Optimization ---")
    final_placement, final_cost = simulated_annealing_placement(
        macros, ports, nets, die_width, die_height, die_origin[0], die_origin[1], blockages,
        initial_temp, cooling_rate, iterations_per_temp, random_seed
    )

    print("\n--- Final Placement Results ---")
    # Visualize the final optimized placement
    visualize_placement(final_placement, ports, nets, die_width, die_height, die_origin[0], die_origin[1], blockages, "Final Optimized Placement")
    print(f"Final Cost: {final_cost:.2f}")

    # --- Dump Final Macro Coordinates to JSON ---
    output_data = []
    for macro in final_placement:
        output_data.append({
            "name": macro.name,
            "x": macro.x,
            "y": macro.y,
            "width": macro.width, # Note: width/height reflect final orientation
            "height": macro.height,
            "orientation": macro.orientation
        })

    output_json_path = 'final_macro_placement.json'
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"\nFinal macro coordinates dumped to '{output_json_path}'")

    print("\nOptimized Macro Positions:")
    for macro in final_placement:
        print(f"  {macro.name}: (x={macro.x:.2f}, y={macro.y:.2f}, w={macro.width:.2f}, h={macro.height:.2f}, ori='{macro.orientation}')")

if __name__ == "__main__":
    main()
