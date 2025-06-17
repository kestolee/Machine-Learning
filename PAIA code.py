#!/usr/bin/env python3
# The above "shebang" line indicates that this script should run with Python 3 interpreter.

import tkinter as tk
# tkinter is a standard GUI library in Python, providing tools to create graphical interfaces.
from tkinter import messagebox
# messagebox provides simple dialogs like error, info, warning pop-ups.
import heapq
# heapq implements a min-heap priority queue useful for graph searches like A*.
import time
# time is used to measure elapsed times and for delaying steps in simulation.
import math
# math module provides mathematical utilities such as infinity (math.inf).

#####################################
# Utility and Data Structures
#####################################
# Defines the four possible directions (up, down, left, right) for a 2D grid in terms of (dx, dy).
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def manhattan(a, b):
    # Computes the Manhattan distance between two grid positions a and b.
    # The Manhattan distance is |ax - bx| + |ay - by|.
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def compare_keys(key1, key2):
    # Compares two priority keys (tuples) and returns a negative number if key1 < key2,
    # positive if key1 > key2, zero if equal.
    # Each key is typically a tuple: (primary_priority, secondary_priority).
    if key1[0] != key2[0]:
        return key1[0] - key2[0]
    return key1[1] - key2[1]

#####################################
# Grid Class
#####################################
class Grid:
    def __init__(self, width, height):
        # Creates a grid with width * height cells, each cell having a default cost of 1.0.
        self.width = width
        self.height = height
        self.cost = {}
        # For every position in the grid, store a default traversal cost.
        for x in range(width):
            for y in range(height):
                self.cost[(x, y)] = 1.0

    def in_bounds(self, pos):
        # Checks if the position (x, y) is within the grid boundaries (0 <= x < width, etc.).
        (x, y) = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def get_cost(self, pos):
        # Returns the cost of a particular grid cell, or infinity if not found.
        return self.cost.get(pos, math.inf)

    def set_cost(self, pos, new_cost):
        # Sets a new cost for the cell 'pos' if it's in the grid bounds.
        if self.in_bounds(pos):
            self.cost[pos] = new_cost

    def get_neighbors(self, pos):
        # Returns the valid neighboring cells of 'pos' in the four cardinal directions.
        (x, y) = pos
        results = []
        for dx, dy in DIRS:
            nxt = (x + dx, y + dy)
            if self.in_bounds(nxt):
                results.append(nxt)
        return results

    def get_predecessors(self, pos):
        # In an undirected grid, predecessors and neighbors are effectively the same set of cells.
        return self.get_neighbors(pos)

#####################################
# Simple Prediction Model
#####################################
class SimplePredictionModel:
    def __init__(self):
        # Holds predictions in a dictionary with keys (time_step, x, y), values as cost (float or inf).
        self.predictions = {}
    
    def add_prediction(self, time_step, pos, cost):
        # Adds or updates a predicted cost at time_step for (x, y) = pos.
        self.predictions[(time_step, pos[0], pos[1])] = cost
    
    def clear(self):
        # Clears all prediction entries.
        self.predictions = {}
    
    def get_predicted_cost(self, pos, time_step):
        # Returns the cost of position 'pos' at 'time_step'. If not predicted, returns None.
        return self.predictions.get((time_step, pos[0], pos[1]), None)
    
    def get_future_impact(self, pos, eta, H=8, severity=10):
        # Calculates a penalty cost for future predictions within a horizon H steps into the future.
        # 'eta' is the estimated time of arrival at pos; severity is used to scale infinite predictions.
        penalty = 0.0
        for t in range(eta+1, eta+H+1):
            pred = self.get_predicted_cost(pos, t)
            if pred is not None:
                # If prediction is infinite, apply a severity penalty reduced by how far out it is.
                if pred == math.inf:
                    pen = severity / max(1, (t - eta))
                else:
                    # Otherwise divide the predicted cost by the "distance" in time from ETA.
                    pen = pred / max(1, (t - eta))
                penalty += pen
        return penalty

#####################################
# Basic A* Algorithm
#####################################
def basic_a_star(grid, start, goal):
    # Implements a standard version of A* on a static grid with a Manhattan heuristic.
    open_set = []
    # Push the start cell into the priority queue with priority = h(start, goal).
    heapq.heappush(open_set, (manhattan(start, goal), start))
    came_from = {}
    cost_so_far = {start: 0}
    nodes_expanded = 0
    st = time.time()
    while open_set:
        # Pop the cell with the lowest f-value (priority).
        _, current = heapq.heappop(open_set)
        nodes_expanded += 1
        # If we've reached the goal, stop searching.
        if current == goal:
            break
        # Iterate over the neighbors of the current cell.
        for nxt in grid.get_neighbors(current):
            new_cost = cost_so_far[current] + grid.get_cost(nxt)
            # If we haven't visited next or found a cheaper path, update it.
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + manhattan(nxt, goal)
                heapq.heappush(open_set, (priority, nxt))
                came_from[nxt] = current
    search_time = time.time() - st
    # If the goal isn't in came_from (unless start=goal), there's no path.
    if goal not in came_from and goal != start:
        return None, None, nodes_expanded, search_time
    # Reconstruct the path by backtracking from goal to start.
    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = came_from[cur]
    path.append(start)
    path.reverse()
    return path, cost_so_far.get(goal, math.inf), nodes_expanded, search_time

#####################################
# LPABasedSearch (LPA* and PAIA*)
#####################################
class LPABasedSearch:
    # LPA* uses incremental updates of g and rhs values. This class also acts as a base for PAIAStar.
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.g = {}
        self.rhs = {}
        self.parent = {}
        # U is the priority queue. U_map maps positions to their current valid keys for quick check.
        self.U = []
        self.U_map = {}
        self.nodes_expanded = 0
        # Initialize g and rhs of all nodes to infinity.
        for pos in grid.cost.keys():
            self.g[pos] = math.inf
            self.rhs[pos] = math.inf
        # For the start node, rhs is 0 (it requires no cost to reach itself).
        self.rhs[self.start] = 0
        key = self.calculate_key(self.start)
        # Push start into the queue with this key.
        heapq.heappush(self.U, (key, self.start))
        self.U_map[self.start] = key
        self.search_time = 0

    def heuristic(self, pos):
        # In basic LPA*, the heuristic is just the Manhattan distance to the goal.
        return manhattan(pos, self.goal)

    def calculate_key(self, pos):
        # The key is (min(g, rhs) + heuristic(pos), min(g, rhs)).
        ks = min(self.g[pos], self.rhs[pos])
        h_val = self.heuristic(pos)
        return (ks + h_val, ks)

    def update_vertex(self, pos):
        # Updates rhs for the vertex 'pos' by looking at all its predecessors.
        # Then, if g != rhs, we push or update its entry in the priority queue.
        if pos != self.start:
            min_rhs = math.inf
            best_pred = None
            # Evaluate cost from each predecessor to find the minimal one.
            for pred in self.grid.get_predecessors(pos):
                candidate = self.g[pred] + self.grid.get_cost(pos)
                if candidate < min_rhs:
                    min_rhs = candidate
                    best_pred = pred
            self.rhs[pos] = min_rhs
            if best_pred is not None:
                self.parent[pos] = best_pred
        # Remove the vertex from U if it was there before.
        if pos in self.U_map:
            del self.U_map[pos]
        # If g[pos] != rhs[pos], this node requires re-expansion, so push it with a new key.
        if self.g[pos] != self.rhs[pos]:
            key = self.calculate_key(pos)
            heapq.heappush(self.U, (key, pos))
            self.U_map[pos] = key

    def compute_shortest_path(self, time_limit=None):
        # The main LPA* loop. Continues until U is empty or goal is 'consistent' (g=rhs).
        st = time.time()
        self.nodes_expanded = 0
        while self.U:
            top_key, current = self.U[0]
            goal_key = self.calculate_key(self.goal)
            # If the top key is >= goal_key and the goal is consistent, we can stop.
            if compare_keys(top_key, goal_key) >= 0 and self.g[self.goal] == self.rhs[self.goal]:
                break
            # Pop the top node from the priority queue.
            key, u = heapq.heappop(self.U)
            # If the node's key is outdated (due to re-insertion), skip it.
            if u not in self.U_map or self.U_map[u] != key:
                continue
            del self.U_map[u]
            self.nodes_expanded += 1
            # If g[u] > rhs[u], set g[u] = rhs[u] and update its neighbors.
            if self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for neigh in self.grid.get_neighbors(u):
                    self.update_vertex(neigh)
            else:
                # Otherwise, overconsistent case: reset g[u] to inf, update itself and neighbors.
                self.g[u] = math.inf
                self.update_vertex(u)
                for neigh in self.grid.get_neighbors(u):
                    self.update_vertex(neigh)
            # If a time_limit is specified and we've exceeded it, break out of the loop.
            if time.time() - st > (time_limit if time_limit is not None else float('inf')):
                break
        self.search_time = time.time() - st
        # If g-value of goal is finite, we can reconstruct the path.
        if self.g[self.goal] < math.inf:
            return self.reconstruct_path(), self.g[self.goal]
        else:
            # No valid path if the goal's cost is still infinity.
            return None, math.inf

    def reconstruct_path(self):
        # Traces from the goal back to the start using the parent dictionary.
        path = []
        pos = self.goal
        while pos != self.start:
            path.append(pos)
            pos = self.parent.get(pos, self.start)
        path.append(self.start)
        path.reverse()
        return path

    def notify_environment_change(self, changed_positions):
        # If the environment changes at certain positions, we call update_vertex so LPA* can re-evaluate.
        for pos in changed_positions:
            self.update_vertex(pos)

#####################################
# PAIAStar Implementation (Predictive Anytime Incremental A*)
#####################################
class PAIAStar(LPABasedSearch):
    # PAIAStar extends LPA* with predictive heuristics and a weight factor for anytime search.
    def __init__(self, grid, start, goal, weight=1.5, use_predictive=True, predictor=None, current_time=0):
        self.weight = weight
        self.use_predictive = use_predictive
        self.predictor = predictor
        self.current_time = current_time
        # We call the predecessor class (LPA*) initializer with super().
        super().__init__(grid, start, goal)
    
    def heuristic(self, pos):
        # The base heuristic is still Manhattan distance to the goal.
        # If 'use_predictive' is True and we have a predictor, add a penalty for future predictions.
        base = manhattan(pos, self.goal)
        if not self.use_predictive or self.predictor is None:
            return base
        # Calculate cost_so_far by taking the minimum of g and rhs at position pos.
        cost_so_far = min(self.g.get(pos, math.inf), self.rhs.get(pos, math.inf))
        # Estimate arrival time at pos by adding cost_so_far (rounded) to current_time.
        eta = self.current_time + (round(cost_so_far) if cost_so_far < math.inf else 0)
        # Retrieve penalty from the prediction model.
        penalty = self.predictor.get_future_impact(pos, eta, H=8, severity=self.grid.width)
        return base + penalty

    def calculate_key(self, pos):
        # Weighted A* style key: (g/rhs + w * heuristic, g/rhs).
        ks = min(self.g[pos], self.rhs[pos])
        h_val = self.heuristic(pos)
        return (ks + self.weight * h_val, ks)

    def proactive_check(self, path, lookahead=3):
        """Examines the next 'lookahead' steps in the path for predicted infinite costs at future times.
        If any such steps are predicted as infinite => replan."""
        if path is None or len(path) == 0:
            return False
        # Check up to 'lookahead' cells in front of the agent's path.
        for i in range(1, min(lookahead, len(path))):
            pos = path[i]
            eta = self.current_time + i
            pred = self.predictor.get_predicted_cost(pos, eta)
            # If 'pred' is infinite, that means we'd be blocked in future => need replan.
            if pred == math.inf:
                return True
        return False

#####################################
# Simulator / UI Application (Two Columns Layout)
#####################################
class PAIASimulatorApp:
    # This class manages the Tkinter UI, drawing the grid and handling user interactions.
    # Define color constants for different cell states in the visualization.
    COLOR_EMPTY = "lightgrey"
    COLOR_OBSTACLE = "darkgrey"
    COLOR_START = "green"
    COLOR_GOAL = "red"
    COLOR_PATH  = "blue"
    COLOR_AGENT = "orange"
    COLOR_PREDICTION = "purple"

    def __init__(self, master):
        # Constructor sets up initial simulator state and UI components.
        self.master = master
        self.master.title("PAIA*/LPA*/A* Simulator")
        self.cell_size = 25
        self.grid_width = 20
        self.grid_height = 20
        # Create a grid and a simple prediction model object here.
        self.grid = Grid(self.grid_width, self.grid_height)
        self.predictor = SimplePredictionModel()
        self.start_pos = None
        self.goal_pos = None
        self.path = []
        self.agent_pos = None
        self.simulation_time = 0
        self.active_algorithm = None
        # Variables to store the selected algorithm type, predictive usage, and weight factor.
        self.algo_type = tk.StringVar(value="PAIA")
        self.use_predictive = tk.BooleanVar(value=True)
        self.weight_var = tk.DoubleVar(value=1.5)
        self.simulation_running = False

        self.create_widgets()
        self.draw_grid()

    def create_widgets(self):
        # Lays out the two columns (left: controls, right: drawing canvas).
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.controls_frame = tk.Frame(main_frame, bd=2, relief=tk.GROOVE)
        self.controls_frame.grid(row=0, column=0, sticky="nsw", padx=5, pady=5)
        self.canvas_frame = tk.Frame(main_frame, bd=2, relief=tk.SUNKEN)
        self.canvas_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        main_frame.columnconfigure(1, weight=1)

        # ---- Grid Setup Section ----
        gs_frame = tk.Frame(self.controls_frame)
        gs_frame.pack(padx=5, pady=5, fill=tk.X)
        tk.Label(gs_frame, text="Grid Setup", font=("Arial", 10, "bold")).pack()
        gs_inner = tk.Frame(gs_frame)
        gs_inner.pack(pady=2)
        tk.Label(gs_inner, text="Width:").grid(row=0, column=0, sticky="e")
        self.entry_width = tk.Entry(gs_inner, width=5)
        self.entry_width.insert(0, str(self.grid_width))
        self.entry_width.grid(row=0, column=1)
        tk.Label(gs_inner, text="Height:").grid(row=1, column=0, sticky="e")
        self.entry_height = tk.Entry(gs_inner, width=5)
        self.entry_height.insert(0, str(self.grid_height))
        self.entry_height.grid(row=1, column=1)
        self.btn_create_reset = tk.Button(gs_frame, text="Create/Reset Grid", command=self.create_reset_grid)
        self.btn_create_reset.pack(pady=2)

        # ---- Interaction Mode Section ----
        im_frame = tk.Frame(self.controls_frame)
        im_frame.pack(padx=5, pady=5, fill=tk.X)
        tk.Label(im_frame, text="Interaction Mode", font=("Arial", 10, "bold")).pack()
        self.mode = tk.StringVar(value="SetStart")
        # Provide radio buttons to select the user action to perform on each click (start, goal, obstacle, prediction).
        for text, val in [("Set Start", "SetStart"), ("Set Goal", "SetGoal"),
                          ("Add Obstacle", "Obstacle"), ("Add Prediction", "Prediction")]:
            tk.Radiobutton(im_frame, text=text, variable=self.mode, value=val).pack(anchor="w")
        
        # ---- Prediction Input Section ----
        pred_frame = tk.Frame(self.controls_frame)
        pred_frame.pack(padx=5, pady=5, fill=tk.X)
        tk.Label(pred_frame, text="Prediction Input", font=("Arial", 10, "bold")).pack()
        pi_inner = tk.Frame(pred_frame)
        pi_inner.pack(pady=2)
        tk.Label(pi_inner, text="Time:").grid(row=0, column=0, sticky="e")
        self.entry_pred_time = tk.Entry(pi_inner, width=5)
        self.entry_pred_time.grid(row=0, column=1)
        tk.Label(pi_inner, text="Cost ('inf'):").grid(row=1, column=0, sticky="e")
        self.entry_pred_cost = tk.Entry(pi_inner, width=5)
        self.entry_pred_cost.insert(0, "inf")
        self.entry_pred_cost.grid(row=1, column=1)
        self.btn_add_pred = tk.Button(pred_frame, text="Add Prediction", command=self.add_prediction)
        self.btn_add_pred.pack(pady=2)
        self.btn_clear_pred = tk.Button(pred_frame, text="Clear Predictions", command=self.clear_predictions)
        self.btn_clear_pred.pack(pady=2)

        # ---- Algorithm & Simulation Controls Section ----
        algo_frame = tk.Frame(self.controls_frame)
        algo_frame.pack(padx=5, pady=5, fill=tk.X)
        tk.Label(algo_frame, text="Algorithm Choice", font=("Arial", 10, "bold")).pack()
        for text, val in [("PAIA*", "PAIA"), ("LPA*", "LPA"), ("Basic A*", "Basic")]:
            tk.Radiobutton(algo_frame, text=text, variable=self.algo_type, value=val,
                           command=self.update_algo_options).pack(anchor="w")
        self.chk_predictive = tk.Checkbutton(algo_frame, text="Use Predictive", variable=self.use_predictive)
        self.chk_predictive.pack(anchor="w", pady=2)
        weight_frame = tk.Frame(algo_frame)
        weight_frame.pack(pady=2, fill=tk.X)
        tk.Label(weight_frame, text="Weight (w):").pack(side=tk.LEFT)
        self.entry_weight = tk.Entry(weight_frame, width=5, textvariable=self.weight_var)
        self.entry_weight.pack(side=tk.LEFT, padx=3)
        sim_frame = tk.Frame(algo_frame)
        sim_frame.pack(padx=5, pady=5, fill=tk.X)
        self.btn_find_path = tk.Button(sim_frame, text="Find Path", command=self.find_path)
        self.btn_find_path.pack(side=tk.LEFT, padx=2, pady=2)
        self.btn_run = tk.Button(sim_frame, text="▶ Run", command=self.run_simulation)
        self.btn_run.pack(side=tk.LEFT, padx=2, pady=2)
        self.btn_pause = tk.Button(sim_frame, text="❚❚ Pause", command=self.pause_simulation)
        self.btn_pause.pack(side=tk.LEFT, padx=2, pady=2)
        self.btn_step = tk.Button(sim_frame, text="Step →", command=self.step_simulation)
        self.btn_step.pack(side=tk.LEFT, padx=2, pady=2)
        self.btn_reset_sim = tk.Button(sim_frame, text="Reset Simulation", command=self.reset_simulation)
        self.btn_reset_sim.pack(side=tk.LEFT, padx=2, pady=2)

        # ---- Metrics Display Section ----
        metric_frame = tk.Frame(self.controls_frame)
        metric_frame.pack(padx=5, pady=5, fill=tk.X)
        tk.Label(metric_frame, text="Results & Metrics", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=2)
        self.label_init_time = tk.Label(metric_frame, text="Init Search Time: -")
        self.label_init_time.grid(row=1, column=0, sticky="w")
        self.label_nodes = tk.Label(metric_frame, text="Nodes Expanded: -")
        self.label_nodes.grid(row=2, column=0, sticky="w")
        self.label_path_cost = tk.Label(metric_frame, text="Path Cost: -")
        self.label_path_cost.grid(row=3, column=0, sticky="w")
        self.label_path_len = tk.Label(metric_frame, text="Path Length: -")
        self.label_path_len.grid(row=4, column=0, sticky="w")
        self.label_sim_time = tk.Label(metric_frame, text="Final Sim Time: -")
        self.label_sim_time.grid(row=5, column=0, sticky="w")

        # ---- Right Column: Canvas ----
        self.canvas = tk.Canvas(self.canvas_frame, width=self.grid_width*self.cell_size+2,
                                height=self.grid_height*self.cell_size+2, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # Bind left-clicks on the canvas to a callback that handles user editing modes.
        self.canvas.bind("<Button-1>", self.canvas_click)

    def update_algo_options(self):
        # Called when the user selects a different algorithm radio button.
        # If it's PAIA*, we enable predictive/weight options. Otherwise we disable them.
        if self.algo_type.get() == "PAIA":
            self.entry_weight.config(state="normal")
            self.chk_predictive.config(state="normal")
        else:
            self.entry_weight.config(state="disabled")
            self.chk_predictive.config(state="disabled")
            self.use_predictive.set(False)

    def create_reset_grid(self):
        # Resets the grid to new dimensions, clears predictions, and refreshes internal states.
        try:
            w = int(self.entry_width.get())
            h = int(self.entry_height.get())
        except:
            messagebox.showerror("Error", "Invalid width/height.")
            return
        self.grid_width = w
        self.grid_height = h
        self.grid = Grid(w, h)
        self.predictor.clear()
        self.start_pos = None
        self.goal_pos = None
        self.path = []
        self.agent_pos = None
        self.simulation_time = 0
        self.active_algorithm = None
        self.draw_grid()
        print(f"[Info] Grid created/reset: {w} x {h}")

    def canvas_click(self, event):
        # Handles a user left-click on the canvas. The action depends on the selected mode.
        x = event.x // self.cell_size
        y = event.y // self.cell_size
        pos = (x, y)
        if not self.grid.in_bounds(pos):
            return
        mode = self.mode.get()
        if mode == "SetStart":
            # Don't allow setting start on an obstacle.
            if self.grid.get_cost(pos) == math.inf:
                messagebox.showwarning("Warning", "Cannot set start on an obstacle.")
                return
            self.start_pos = pos
            self.agent_pos = pos
            print("[Info] Start set at:", pos)
        elif mode == "SetGoal":
            # Don't allow setting goal on an obstacle.
            if self.grid.get_cost(pos) == math.inf:
                messagebox.showwarning("Warning", "Cannot set goal on an obstacle.")
                return
            self.goal_pos = pos
            print("[Info] Goal set at:", pos)
        elif mode == "Obstacle":
            # Toggle obstacle: If it's inf, turn it into cost=1, otherwise set cost=inf.
            current = self.grid.get_cost(pos)
            if current == math.inf:
                self.grid.set_cost(pos, 1.0)
                print("[Info] Obstacle removed at:", pos)
            else:
                # Prevent placing obstacles on start/goal.
                if pos == self.start_pos or pos == self.goal_pos:
                    messagebox.showwarning("Warning", "Cannot place obstacle on start/goal.")
                    return
                self.grid.set_cost(pos, math.inf)
                print("[Info] Obstacle added at:", pos)
            # If an LPA*-based algorithm is active, notify it about the changed cell.
            if self.active_algorithm is not None:
                self.active_algorithm.notify_environment_change([pos])
            # If we have a start and goal, automatically replan to reflect the new obstacle state.
            if self.start_pos and self.goal_pos:
                self.find_path()
        elif mode == "Prediction":
            # Store pos for a future prediction. Also autofill the predicted time as current_time+5.
            self.current_pred_pos = pos
            self.entry_pred_time.delete(0, tk.END)
            self.entry_pred_time.insert(0, str(self.simulation_time+5))
            print("[Info] Prediction position set to:", pos)
        # Redraw the grid to reflect any changes in start, goal, obstacle.
        self.draw_grid()

    def draw_grid(self):
        # Clears and redraws the entire grid canvas.
        self.canvas.delete("all")
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                pos = (x, y)
                color = self.COLOR_EMPTY
                # If cost is inf => draw as an obstacle cell.
                if self.grid.get_cost(pos) == math.inf:
                    color = self.COLOR_OBSTACLE
                # Highlight the start/goal cells.
                if self.start_pos == pos:
                    color = self.COLOR_START
                elif self.goal_pos == pos:
                    color = self.COLOR_GOAL
                # Compute the pixel coordinates for the rectangle.
                x1 = x*self.cell_size
                y1 = y*self.cell_size
                x2 = x1+self.cell_size
                y2 = y1+self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
        # Draw the path cells in blue, if a path is available.
        if self.path:
            for pos in self.path:
                x, y = pos
                x1 = x*self.cell_size
                y1 = y*self.cell_size
                x2 = x1+self.cell_size
                y2 = y1+self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.COLOR_PATH, outline="black")
        # Draw the agent's current position as an orange circle if it exists.
        if self.agent_pos:
            x, y = self.agent_pos
            x1 = x*self.cell_size + self.cell_size//4
            y1 = y*self.cell_size + self.cell_size//4
            x2 = x1 + self.cell_size//2
            y2 = y1 + self.cell_size//2
            self.canvas.create_oval(x1, y1, x2, y2, fill=self.COLOR_AGENT)
        # Draw prediction markers as purple small circles for each predicted (time, x, y).
        for (t, px, py), pred_cost in self.predictor.predictions.items():
            x1 = px*self.cell_size + self.cell_size//3
            y1 = py*self.cell_size + self.cell_size//3
            x2 = x1 + self.cell_size//3
            y2 = y1 + self.cell_size//3
            self.canvas.create_oval(x1, y1, x2, y2, fill=self.COLOR_PREDICTION)

    def add_prediction(self):
        # Takes user inputs for time and cost, then stores it in the prediction model.
        if not hasattr(self, "current_pred_pos"):
            messagebox.showwarning("Warning", "Click on grid to choose prediction position.")
            return
        try:
            t = int(self.entry_pred_time.get())
        except:
            messagebox.showerror("Error", "Invalid prediction time.")
            return
        cost_val = self.entry_pred_cost.get().strip().lower()
        if cost_val == "inf":
            cost = math.inf
        else:
            try:
                cost = float(cost_val)
            except:
                messagebox.showerror("Error", "Invalid prediction cost.")
                return
        self.predictor.add_prediction(t, self.current_pred_pos, cost)
        print(f"[Info] Prediction added at time {t} for pos {self.current_pred_pos} with cost {cost}.")
        self.draw_grid()
        # If we're using the PAIA* algorithm, add_prediction triggers a replan to incorporate the new prediction.
        if self.algo_type.get() == "PAIA" and self.start_pos and self.goal_pos:
            self.find_path()

    def clear_predictions(self):
        # Clears all predictions from the prediction model, and if using PAIA*, replan.
        self.predictor.clear()
        print("[Info] Predictions cleared.")
        self.draw_grid()
        if self.algo_type.get() == "PAIA" and self.start_pos and self.goal_pos:
            self.find_path()

    def find_path(self):
        # Determines which algorithm to use and performs the initial path search.
        if self.start_pos is None or self.goal_pos is None:
            messagebox.showerror("Error", "Set both Start and Goal positions.")
            return
        print(f"[Info] Finding path from {self.start_pos} to {self.goal_pos}.")
        algo = self.algo_type.get()
        if algo == "Basic":
            # Basic A* usage: we call the basic_a_star function.
            path, cost, nodes, search_time = basic_a_star(self.grid, self.start_pos, self.goal_pos)
            self.path = path if path is not None else []
            print(f"[Basic A*] Search Time: {search_time:.4f}s | Nodes Expanded: {nodes} | Path Cost: {cost} | Path Length: {len(self.path) if self.path else 0}")
            self.label_init_time.config(text=f"Init Search Time: {search_time:.4f}s")
            self.label_nodes.config(text=f"Nodes Expanded: {nodes}")
            self.label_path_cost.config(text=f"Path Cost: {cost:.2f}")
            self.label_path_len.config(text=f"Path Length: {len(self.path) if self.path else '-'}")
            self.active_algorithm = None
        else:
            # For LPA* or PAIA*, instantiate the relevant class and compute the shortest path.
            if algo == "LPA":
                self.active_algorithm = LPABasedSearch(self.grid, self.start_pos, self.goal_pos)
            elif algo == "PAIA":
                w = self.weight_var.get()
                self.active_algorithm = PAIAStar(self.grid, self.start_pos, self.goal_pos,
                                                  weight=w, use_predictive=self.use_predictive.get(),
                                                  predictor=self.predictor, current_time=self.simulation_time)
            path, cost = self.active_algorithm.compute_shortest_path()
            self.path = path if path is not None else []
            print(f"[{algo}] Search Time: {self.active_algorithm.search_time:.4f}s | Nodes Expanded: {self.active_algorithm.nodes_expanded} | Path Cost: {cost} | Path Length: {len(self.path) if self.path else 0}")
            self.label_init_time.config(text=f"Init Search Time: {self.active_algorithm.search_time:.4f}s")
            self.label_nodes.config(text=f"Nodes Expanded: {self.active_algorithm.nodes_expanded}")
            self.label_path_cost.config(text=f"Path Cost: {cost:.2f}")
            self.label_path_len.config(text=f"Path Length: {len(self.path) if self.path else '-'}")
        self.draw_grid()

    def run_simulation(self):
        # Starts a loop where the agent moves step-by-step until it's paused or reaches the goal.
        if not self.path:
            messagebox.showwarning("Warning", "No path found!")
            return
        self.simulation_running = True
        self.btn_run.config(state="disabled")
        self.btn_pause.config(state="normal")
        self.simulation_loop()

    def simulation_loop(self):
        # A recurring loop which calls step_simulation and schedules itself 200ms later, until paused.
        if not self.simulation_running:
            return
        self.step_simulation()
        self.master.after(200, self.simulation_loop)

    def pause_simulation(self):
        # Pauses continuous simulation; user can resume or step manually.
        self.simulation_running = False
        self.btn_run.config(state="normal")
        self.btn_pause.config(state="disabled")
        print(f"[Info] Simulation paused at time {self.simulation_time}.")

    def step_simulation(self):
        # Moves the agent one cell along the path, checks predictions, triggers replan if needed.
        if self.agent_pos == self.goal_pos:
            # If agent already at goal, end simulation and record finishing time.
            self.simulation_running = False
            print(f"[Info] Goal reached at simulation time {self.simulation_time}.")
            self.label_sim_time.config(text=f"Final Sim Time: {self.simulation_time}")
            return
        # If no path or agent not on the path, attempt to replan.
        if not self.path or self.agent_pos not in self.path:
            print("[Info] No valid path. Triggering replan.")
            self.find_path()
        # If we're dealing with a PAIAStar, set current_time and detect obstacles predicted on next move.
        if isinstance(self.active_algorithm, PAIAStar):
            self.active_algorithm.current_time = self.simulation_time
            try:
                current_idx = self.path.index(self.agent_pos)
                next_cell = self.path[current_idx+1]
                predicted = self.predictor.get_predicted_cost(next_cell, self.simulation_time + 1)
                if predicted == math.inf:
                    print("[PAIA*] Prediction on next step detected! Replanning before moving.")
                    self.find_path()
            except (ValueError, IndexError):
                # Means agent_pos isn't in path or no 'next' cell => do nothing special.
                pass
        # Move one step along the path if possible.
        try:
            idx = self.path.index(self.agent_pos)
        except ValueError:
            idx = 0
        if idx < len(self.path)-1:
            self.agent_pos = self.path[idx+1]
            self.simulation_time += 1
            print(f"[Sim] Moved to {self.agent_pos} at time {self.simulation_time}.")
        else:
            # If idx is the last cell, we assume we've reached the goal or the path end.
            self.agent_pos = self.goal_pos
            self.simulation_running = False
            print(f"[Sim] Reached goal at time {self.simulation_time}.")
        self.label_sim_time.config(text=f"Final Sim Time: {self.simulation_time}")
        self.draw_grid()

    def reset_simulation(self):
        # Resets agent position, time, path, stops simulation.
        self.simulation_running = False
        self.simulation_time = 0
        self.agent_pos = self.start_pos
        self.path = []
        self.active_algorithm = None
        self.btn_run.config(state="normal")
        self.btn_pause.config(state="disabled")
        self.label_sim_time.config(text="Final Sim Time: 0")
        print("[Info] Simulation reset.")
        self.draw_grid()

#####################################
# Main Script Execution
#####################################
if __name__ == "__main__":
    # Main entry point. Create a Tk root, instantiate PAIASimulatorApp, then run mainloop.
    root = tk.Tk()
    app = PAIASimulatorApp(root)
    root.mainloop()