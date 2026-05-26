import sys
import math
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, safe for batch/CLI
import matplotlib.pyplot as plt
from utils import memoize
from search import (Problem, breadth_first_graph_search, depth_first_graph_search, greedy_best_first_graph_search, astar_search, recursive_best_first_search, iterative_deepening_search)


# CLI arguements
if len(sys.argv) != 3:
    print("Usage: python bfs.py <filename> <method>")
    sys.exit(1)

filename = sys.argv[1]
method   = sys.argv[2]

# file parser
nodes = {}
edges = {}
graph = {}
start = None
goal  = None

with open(filename, 'r') as f:
    section = None
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('Nodes'):
            section = 'nodes'; 
            continue
        elif line.startswith('Edges'):
            section = 'edges'; 
            continue
        elif line.startswith('Start'):
            start = line.split(':')[1].strip(); 
            continue
        elif line.startswith('Goal'):
            raw = line.split(':')[1].strip()
            goal = raw.split(';')   # e.g. ['5', '4']
            continue

        if section == 'nodes':
            # format: 1 4 1
            parts = line.split()
            node_id = parts[0]
            x, y = float(parts[1]), float(parts[2])
            nodes[node_id] = (x, y)
            graph[node_id] = []
            edges[node_id] = []

        elif section == 'edges':
            # format: 1 3 5   (from, to, weight)
            parts = line.split()
            frm, to, w = parts[0], parts[1], int(parts[2])
            edges[frm].append((to, w))
            graph[frm].append(to)

#heuristics for Greedy best first and A*, euclidian distance used as we have node coordinates
def euclidean_heuristic(goal_nodes):
    def h(node):
        # node.state is the current node ID e.g. '3'
        if node.state not in nodes or not goal_nodes:
            return 0
        # if multiple goals, take the minimum distance to any goal
        return min(
            math.sqrt((nodes[node.state][0] - nodes[g][0])**2 +
                      (nodes[node.state][1] - nodes[g][1])**2)
            for g in goal_nodes
        )
    return h

# graph visualizer
def draw_graph(nodes, graph):
    plt.figure()
    for node in graph:
        for neighbor, weight in graph[node]:
            x1, y1 = nodes[node]
            x2, y2 = nodes[neighbor]
            plt.plot([x1, x2], [y1, y2], 'b-')
            mx, my = (x1+x2)/2, (y1+y2)/2
            plt.text(mx, my, str(weight), color='red', fontsize=8)
    for node, (x, y) in nodes.items():
        plt.scatter(x, y, s=600, zorder=5)
        plt.text(x, y, node, ha='center', va='center', color='white', fontweight='bold')
    plt.title("Weighted Graph")
    plt.axis('equal')
    plt.savefig("graph.png")   # saves instead of plt.show() for CLI use
    plt.close()

draw_graph(nodes, edges)

# GraphProblem
class GraphProblem(Problem):
    def __init__(self, initial, goal=None):
        super().__init__(initial, goal)
        # normalize goal to always be a list
        self.goals = goal if isinstance(goal, list) else [goal]

    def actions(self, state):
        return graph[state]

    def result(self, state, action):
        return action
        
    # checks against all goals
    def goal_test(self, state):
        return state in self.goals   

# Run the selected search method
problem = GraphProblem(start, goal)

heuristic = euclidean_heuristic(goal if isinstance(goal, list) else [goal])

methods = {
    'bfs' : lambda p: breadth_first_graph_search(p),
    'dfs' : lambda p: depth_first_graph_search(p),
    'gbfs' : lambda p: greedy_best_first_graph_search(p, heuristic),
    'as' : lambda p: astar_search(p, heuristic),
    'cus1' : lambda p: iterative_deepening_search(p),
    'cus2' : lambda p: recursive_best_first_search(p, heuristic)
}

if method not in methods:
    print(f"Unknown method '{method}'. Available: {list(methods.keys())}")
    sys.exit(1)

solution = methods[method](problem)

# output in correct format
if solution:
    path       = solution.solution()
    num_nodes  = len(solution.path())
    # the actual goal node reached
    reached    = path[-1]            
    print(f"{filename} {method}")
    print(f"{reached} {num_nodes}")
    print(' -> '.join([start] + path))
else:
    print(f"{filename} {method}")
    print("No solution found.")