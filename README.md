# Fast and Visualizable VLSI Macro Placement Optimizer

This project implements a fast and scalable Python-based macro placement engine for VLSI physical design. It considers macro connectivity, criticality weights, die boundaries, I/O ports, and blockage regions. The algorithm aims to optimize total wirelength (HPWL) while avoiding overlaps and respecting design constraints.

## 🔍 Features

✅ Die area defined as rectilinear polygon
✅ Macro placement with fixed and movable options
✅ Weighted netlist (macro ↔ macro, macro ↔ port)
✅ Blockage-aware placement
✅ Force-directed heuristic with repulsion to avoid overlaps
✅ Visualization of final placement and net connections
✅ JSON-based input format for portability

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/vlsi-macro-placer.git
cd vlsi-macro-placer
```

## 📦 Input Format
<pre><code>```json 
  { 
    "die": [[0, 0], [0, 1000], [1000, 1000], [1000, 0]], 
    "blockages": [ [[100, 100], [100, 300], [300, 300], [300, 100]] ], 
    "macros": [ 
      { "name": "M1", "width": 50, "height": 60 }, 
      { "name": "M2", "width": 40, "height": 50, "x": 100, "y": 100, "fixed": true } 
    ], 
    "ports": [ 
      { "name": "P1", "x": 0, "y": 500 }, 
      { "name": "P2", "x": 1000, "y": 800 } 
    ], 
    "nets": [ 
      { "objects": ["M1", "P1"], "weight": 2.0 }, 
      { "objects": ["M1", "M2"], "weight": 1.0 } 
    ] 
  } 
  ```</code></pre>

## 📊 Output
<pre><code>```text 
  Final Macro Placements: 
  M1 @ (432.2, 718.6) 
  M2 @ (100.0, 100.0) ... 
  Total HPWL: 1347.92 
  ``` 
  
  A matplotlib-based plot will display: 
  - **Die boundary** in black 
  - **Macros** in blue (movable) and red (fixed) 
  - **Ports** as green dots with labels 
  - **Blockage areas** in gray 
  - **Net connections** as dashed cyan lines 

## 🧠 Algorithm Notes
  - Starts with legal random placement avoiding blockages and die violations 
  - Uses force-directed repulsion to reduce macro overlaps 
  - Penalizes collisions with blockages 
  - Evaluates HPWL cost function on each iteration 
  - Ports are treated as fixed connection points 
  - Uses R-tree spatial indexing for fast overlap queries

## 🧩 Future Work
  - Add macro clustering and hierarchy awareness 
  - Integrate timing-aware placement and slack analysis 
  - Add export to DEF/GDSII formats 
  - Congestion and density-aware placement cost models 
  - Parallelization of placement loops for large-scale designs 
  - GUI-based interactive placement tweaking

## 🤝 Contributions
Feel free to fork and contribute! Pull requests, improvements, and suggestions are always welcome.
