import sys
import math
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, safe for batch/CLI
import matplotlib.pyplot as plt
from search import (Problem, breadth_first_graph_search, depth_first_graph_search,
                    greedy_best_first_graph_search, astar_search,
                    recursive_best_first_search, iterative_deepening_search)


# ─────────────────────────────────────────────
# CLI ARGUMENTS
# ─────────────────────────────────────────────

if len(sys.argv) < 3:
    print("Usage: python program.py <filename> <method>")
    print("       python program.py <filename> <method> --graph")
    sys.exit(1)

filename = sys.argv[1]
method   = sys.argv[2].lower()  # convert to lowercase for case-insensitive matching


# ─────────────────────────────────────────────
# FILE PARSER
# Supports format:
#   Nodes:   1 4 1
#   Edges:   1 3 5
#   Start:   2
#   Goal:    5;4
# ─────────────────────────────────────────────

nodes = {}   # { node_id (str) : (x, y) }
edges = {}   # { node_id (str) : [(neighbour, weight), ...] }
graph = {}   # { node_id (str) : [neighbour, ...] }
start = None
goal  = None

with open(filename, 'r') as f:
    section = None
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # Detect section headers
        if line.startswith('Nodes'):
            section = 'nodes'
            continue
        elif line.startswith('Edges'):
            section = 'edges'
            continue
        elif line.startswith('Start'):
            start = line.split(':')[1].strip()
            continue
        elif line.startswith('Goal'):
            raw  = line.split(':')[1].strip()
            goal = [g.strip() for g in raw.split(';')]  # e.g. ['5', '4']
            continue

        if section == 'nodes':
            # Format: 1 4 1  (node_id x y)
            parts   = line.split()
            node_id = parts[0]
            x, y    = float(parts[1]), float(parts[2])
            nodes[node_id] = (x, y)
            graph[node_id] = []
            edges[node_id] = []

        elif section == 'edges':
            # Format: 1 3 5  (from to weight)
            parts      = line.split()
            frm, to, w = parts[0], parts[1], int(parts[2])
            edges[frm].append((to, w))
            graph[frm].append(to)


# ─────────────────────────────────────────────
# HEURISTIC — Euclidean distance to nearest goal
# Used by GBFS, A* (AS), and CUS2
# ─────────────────────────────────────────────

def euclidean_heuristic(goal_nodes):
    def h(node):
        if node.state not in nodes or not goal_nodes:
            return 0
        return min(
            math.sqrt((nodes[node.state][0] - nodes[g][0])**2 +
                      (nodes[node.state][1] - nodes[g][1])**2)
            for g in goal_nodes
        )
    return h


# ─────────────────────────────────────────────
# GRAPH VISUALIZER
# Only runs when --graph flag is passed
# Saves graph as graph.png in current directory
# ─────────────────────────────────────────────

def draw_graph(nodes, edges):
    plt.figure()
    for node in edges:
        for neighbor, weight in edges[node]:
            x1, y1 = nodes[node]
            x2, y2 = nodes[neighbor]
            plt.plot([x1, x2], [y1, y2], 'b-')
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            plt.text(mx, my, str(weight), color='red', fontsize=8)
    for node, (x, y) in nodes.items():
        plt.scatter(x, y, s=600, zorder=5)
        plt.text(x, y, node, ha='center', va='center',
                 color='white', fontweight='bold')
    plt.title("Weighted Graph")
    plt.axis('equal')
    plt.savefig("graph.png")
    plt.close()
    print("Graph saved to graph.png")

# Draw graph only if --graph flag is provided
if '--graph' in sys.argv:
    draw_graph(nodes, edges)


# ─────────────────────────────────────────────
# GRAPH PROBLEM DEFINITION
# Wraps the parsed graph into the AIMA Problem interface
# ─────────────────────────────────────────────

class GraphProblem(Problem):
    def __init__(self, initial, goal=None):
        super().__init__(initial, goal)
        # Normalize goal to always be a list for multi-destination support
        self.goals = goal if isinstance(goal, list) else [goal]

    def actions(self, state):
        """Return list of reachable neighbours from current state."""
        return graph.get(state, [])

    def result(self, state, action):
        """Moving to a neighbour returns that neighbour as the new state."""
        return action

    def goal_test(self, state):
        """Check if current state is any of the destination nodes."""
        return state in self.goals



# ─────────────────────────────────────────────
# SEARCH METHOD DISPATCHER
# ─────────────────────────────────────────────

problem  = GraphProblem(start, goal)
heuristic = euclidean_heuristic(goal if isinstance(goal, list) else [goal])

methods = {
    'dfs' : lambda p: depth_first_graph_search(p),
    'bfs' : lambda p: breadth_first_graph_search(p),
    'gbfs': lambda p: greedy_best_first_graph_search(p, heuristic),
    'as'  : lambda p: astar_search(p, heuristic),
    'cus1': lambda p: iterative_deepening_search(p),
    'cus2': lambda p: recursive_best_first_search(p, heuristic),
}

if method not in methods:
    print(f"Unknown method '{method}'. Available: {[m.upper() for m in methods.keys()]}")
    sys.exit(1)

# ─────────────────────────────────────────────
# DRAW GRAPHS OF SEARCH SOLUTIONS
# ─────────────────────────────────────────────

def draw_solution(nodes, edges, path, filename, method):
    plt.figure()
    
    # Draw all edges in default colour first
    for node in edges:
        for neighbor, weight in edges[node]:
            x1, y1 = nodes[node]
            x2, y2 = nodes[neighbor]
            plt.plot([x1, x2], [y1, y2], 'b-', linewidth=1, alpha=0.3)
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            plt.text(mx, my, str(weight), color='gray', fontsize=8)

    # Draw path edges in red on top
    for i in range(len(path) - 1):
        curr, next_ = path[i], path[i + 1]
        x1, y1 = nodes[curr]
        x2, y2 = nodes[next_]
        plt.plot([x1, x2], [y1, y2], 'r-', linewidth=3)

    # Draw all nodes
    for node, (x, y) in nodes.items():
        if node == path[0]:
            color = 'green'   # start node
        elif node == path[-1]:
            color = 'red'     # goal node
        elif node in path:
            color = 'orange'  # visited path nodes
        else:
            color = 'steelblue'  # unvisited nodes
        plt.scatter(x, y, s=600, color=color, zorder=5)
        plt.text(x, y, node, ha='center', va='center', color='white', fontweight='bold')

    # Add a legend
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',    markersize=10, label='Start'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',      markersize=10, label='Goal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',   markersize=10, label='On path'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',markersize=10, label='Unvisited'),
        Line2D([0], [0], color='red',      linewidth=3, label='Solution path'),
        Line2D([0], [0], color='blue',     linewidth=1, alpha=0.3, label='Other edges'),
    ]
    plt.legend(handles=legend, loc='upper left')

    plt.title(f"Solution path using {method.upper()}")
    plt.axis('equal')

    # Save with method name so each run produces a unique file
    output_name = f"solution_{method}.png"
    plt.savefig(output_name)
    plt.close()
    print(f"Solution graph saved to {output_name}")

# ─────────────────────────────────────────────
# RUN SEARCH & PRINT RESULT
# Output format per spec:
#   filename method
#   goal number_of_nodes
#   path
# ─────────────────────────────────────────────

solution = methods[method](problem)

if solution:
    path      = solution.solution()
    num_nodes = len(solution.path())
    reached   = path[-1]  # the goal node that was reached
    full_path = [start] + path
    
    print(f"{filename} {method.upper()}")
    print(f"{reached} {num_nodes}")
    print(' -> '.join([start] + path))
    
    draw_solution(nodes, edges, full_path, filename, method)
else:
    print(f"{filename} {method.upper()}")
    print("No path found")

