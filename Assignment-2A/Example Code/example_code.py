import sys
import heapq
import math
from collections import deque

# ─────────────────────────────────────────────
# FILE PARSER
# ─────────────────────────────────────────────

def parse_file(filename):
    """
    Reads a problem file and returns:
      nodes       : dict  { node_id (int) : (x, y) }
      edges       : dict  { from_node (int) : [(to_node, cost), ...] }
      origin      : int
      destinations: list of int
    """
    nodes = {}
    edges = {}
    origin = None
    destinations = []

    section = None  # tracks which block we're in

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Detect section headers
            if line == 'Nodes:':
                section = 'nodes'
            elif line == 'Edges:':
                section = 'edges'
            elif line == 'Origin:':
                section = 'origin'
            elif line == 'Destinations:':
                section = 'destinations'

            # Parse node:  1: (4,1)
            elif section == 'nodes':
                parts = line.split(':')
                node_id = int(parts[0].strip())
                coords = parts[1].strip().strip('()')
                x, y = map(int, coords.split(','))
                nodes[node_id] = (x, y)
                edges[node_id] = []  # initialise adjacency list

            # Parse edge:  (2,1): 4
            elif section == 'edges':
                parts = line.split(':')
                endpoints = parts[0].strip().strip('()').split(',')
                from_node = int(endpoints[0].strip())
                to_node   = int(endpoints[1].strip())
                cost      = int(parts[1].strip())
                edges[from_node].append((to_node, cost))

            # Parse origin: single integer
            elif section == 'origin':
                origin = int(line)

            # Parse destinations: separated by semicolons
            elif section == 'destinations':
                destinations = [int(d.strip()) for d in line.split(';')]

    return nodes, edges, origin, destinations


# ─────────────────────────────────────────────
# HEURISTIC  (straight-line / Euclidean distance)
# ─────────────────────────────────────────────

def heuristic(nodes, node, destinations):
    """
    Returns the minimum straight-line distance from node
    to any destination node.  Used by GBFS, A*, and CUS2.
    """
    nx, ny = nodes[node]
    return min(
        math.sqrt((nx - nodes[d][0])**2 + (ny - nodes[d][1])**2)
        for d in destinations
    )


# ─────────────────────────────────────────────
# RESULT FORMATTER
# ─────────────────────────────────────────────

def format_result(filename, method, goal, num_nodes, path):
    print(f"{filename} {method}")
    print(f"{goal} {num_nodes}")
    print(' -> '.join(str(n) for n in path))


def no_path(filename, method):
    print(f"{filename} {method}")
    print("No path found")


# ─────────────────────────────────────────────
# 1. DFS  –  Depth-First Search  (uninformed)
# ─────────────────────────────────────────────

def dfs(nodes, edges, origin, destinations):
    """
    Uses a LIFO stack.
    Tie-breaking: nodes are pushed in REVERSE ascending order so that
    the smallest-numbered neighbour is explored first.
    """
    dest_set = set(destinations)
    # Stack entries: (current_node, path_so_far)
    stack = [(origin, [origin])]
    visited = set()
    num_nodes = 0

    while stack:
        node, path = stack.pop()

        if node in visited:
            continue
        visited.add(node)
        num_nodes += 1

        if node in dest_set:
            return node, num_nodes, path

        # Sort neighbours ascending, then reverse so smallest goes on top
        neighbours = sorted(edges.get(node, []), key=lambda x: x[0], reverse=True)
        for neighbour, _ in neighbours:
            if neighbour not in visited:
                stack.append((neighbour, path + [neighbour]))

    return None, num_nodes, []


# ─────────────────────────────────────────────
# 2. BFS  –  Breadth-First Search  (uninformed)
# ─────────────────────────────────────────────

def bfs(nodes, edges, origin, destinations):
    """
    Uses a FIFO queue.
    Tie-breaking: neighbours are enqueued in ascending node-number order.
    """
    dest_set = set(destinations)
    queue = deque([(origin, [origin])])
    visited = set([origin])
    num_nodes = 0

    while queue:
        node, path = queue.popleft()
        num_nodes += 1

        if node in dest_set:
            return node, num_nodes, path

        neighbours = sorted(edges.get(node, []), key=lambda x: x[0])
        for neighbour, _ in neighbours:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, path + [neighbour]))

    return None, num_nodes, []


# ─────────────────────────────────────────────
# 3. GBFS  –  Greedy Best-First Search  (informed)
# ─────────────────────────────────────────────

def gbfs(nodes, edges, origin, destinations):
    """
    Priority = straight-line distance to nearest goal (h only).
    Tie-breaking: ascending node number, then insertion order.
    """
    dest_set = set(destinations)
    # heap entry: (h_value, node_id, path)
    counter = 0  # for insertion-order tie-breaking
    h0 = heuristic(nodes, origin, destinations)
    heap = [(h0, counter, origin, [origin])]
    visited = set()
    num_nodes = 0

    while heap:
        h, _, node, path = heapq.heappop(heap)

        if node in visited:
            continue
        visited.add(node)
        num_nodes += 1

        if node in dest_set:
            return node, num_nodes, path

        neighbours = sorted(edges.get(node, []), key=lambda x: x[0])
        for neighbour, _ in neighbours:
            if neighbour not in visited:
                counter += 1
                h_n = heuristic(nodes, neighbour, destinations)
                heapq.heappush(heap, (h_n, neighbour, counter, neighbour, path + [neighbour]))
                # Note: heapq compares tuples element-by-element:
                # (h, node_id, counter, ...) ensures ascending-node tie-break,
                # then insertion-order tie-break.

    return None, num_nodes, []


# Fix GBFS heap tuple to match pop correctly
def gbfs(nodes, edges, origin, destinations):
    dest_set = set(destinations)
    counter = 0
    h0 = heuristic(nodes, origin, destinations)
    heap = [(h0, 0, counter, origin, [origin])]
    visited = set()
    num_nodes = 0

    while heap:
        h, node_id_key, _, node, path = heapq.heappop(heap)

        if node in visited:
            continue
        visited.add(node)
        num_nodes += 1

        if node in dest_set:
            return node, num_nodes, path

        neighbours = sorted(edges.get(node, []), key=lambda x: x[0])
        for neighbour, _ in neighbours:
            if neighbour not in visited:
                counter += 1
                h_n = heuristic(nodes, neighbour, destinations)
                heapq.heappush(heap, (h_n, neighbour, counter, neighbour, path + [neighbour]))

    return None, num_nodes, []


# ─────────────────────────────────────────────
# 4. A*  –  A Star  (informed)
# ─────────────────────────────────────────────

def astar(nodes, edges, origin, destinations):
    """
    Priority = g (path cost so far) + h (straight-line distance to goal).
    Tie-breaking: ascending node number, then insertion order.
    """
    dest_set = set(destinations)
    counter = 0
    h0 = heuristic(nodes, origin, destinations)
    # heap entry: (f, node_id, counter, node, g, path)
    heap = [(h0, origin, counter, origin, 0, [origin])]
    visited = {}  # node -> best g seen
    num_nodes = 0

    while heap:
        f, _, _, node, g, path = heapq.heappop(heap)

        if node in visited and visited[node] <= g:
            continue
        visited[node] = g
        num_nodes += 1

        if node in dest_set:
            return node, num_nodes, path

        neighbours = sorted(edges.get(node, []), key=lambda x: x[0])
        for neighbour, cost in neighbours:
            g_new = g + cost
            if neighbour not in visited or visited[neighbour] > g_new:
                counter += 1
                h_n = heuristic(nodes, neighbour, destinations)
                f_new = g_new + h_n
                heapq.heappush(heap, (f_new, neighbour, counter, neighbour, g_new, path + [neighbour]))

    return None, num_nodes, []


# ─────────────────────────────────────────────
# 5. CUS1  –  Iterative Deepening DFS  (uninformed)
# ─────────────────────────────────────────────

def cus1(nodes, edges, origin, destinations):
    """
    Iterative Deepening Depth-First Search (IDDFS).
    Combines DFS's space efficiency with BFS's completeness.
    Gradually increases the depth limit until a goal is found.
    """
    dest_set = set(destinations)
    total_nodes = 0

    def depth_limited_dfs(node, path, depth_limit, visited_in_path):
        nonlocal total_nodes
        total_nodes += 1

        if node in dest_set:
            return node, path

        if depth_limit == 0:
            return None, []

        # Sort neighbours ascending for consistent tie-breaking
        neighbours = sorted(edges.get(node, []), key=lambda x: x[0])
        for neighbour, _ in neighbours:
            if neighbour not in visited_in_path:
                visited_in_path.add(neighbour)
                result, result_path = depth_limited_dfs(
                    neighbour, path + [neighbour], depth_limit - 1, visited_in_path
                )
                visited_in_path.discard(neighbour)
                if result is not None:
                    return result, result_path

        return None, []

    # Increase depth limit from 0 upward
    for limit in range(len(nodes) + 1):
        total_nodes = 0
        result, path = depth_limited_dfs(origin, [origin], limit, {origin})
        if result is not None:
            return result, total_nodes, path

    return None, total_nodes, []


# ─────────────────────────────────────────────
# 6. CUS2  –  IDA*  (informed)
# ─────────────────────────────────────────────

def cus2(nodes, edges, origin, destinations):
    """
    Iterative Deepening A* (IDA*).
    Uses f = g + h as the cost threshold, expanding it each iteration.
    Finds the optimal (lowest cost) path with minimal memory use.
    """
    dest_set = set(destinations)
    total_nodes = [0]  # use list so inner function can mutate it

    def search(node, g, threshold, path, visited_in_path):
        total_nodes[0] += 1
        h = heuristic(nodes, node, destinations)
        f = g + h

        if f > threshold:
            return None, [], f  # return f as the new minimum threshold

        if node in dest_set:
            return node, path, f

        min_threshold = float('inf')
        neighbours = sorted(edges.get(node, []), key=lambda x: x[0])
        for neighbour, cost in neighbours:
            if neighbour not in visited_in_path:
                visited_in_path.add(neighbour)
                result, result_path, new_t = search(
                    neighbour, g + cost, threshold, path + [neighbour], visited_in_path
                )
                visited_in_path.discard(neighbour)
                if result is not None:
                    return result, result_path, new_t
                min_threshold = min(min_threshold, new_t)

        return None, [], min_threshold

    # Start with h(origin) as the initial threshold
    threshold = heuristic(nodes, origin, destinations)

    while True:
        total_nodes[0] = 0
        result, path, new_threshold = search(origin, 0, threshold, [origin], {origin})
        if result is not None:
            return result, total_nodes[0], path
        if new_threshold == float('inf'):
            return None, total_nodes[0], []  # no path exists
        threshold = new_threshold


# ─────────────────────────────────────────────
# MAIN  –  CLI entry point
# ─────────────────────────────────────────────

METHODS = {
    'DFS':  dfs,
    'BFS':  bfs,
    'GBFS': gbfs,
    'AS':   astar,
    'CUS1': cus1,
    'CUS2': cus2,
}

def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        print(f"Methods: {', '.join(METHODS.keys())}")
        sys.exit(1)

    filename = sys.argv[1]
    method   = sys.argv[2].upper()

    if method not in METHODS:
        print(f"Unknown method '{method}'. Choose from: {', '.join(METHODS.keys())}")
        sys.exit(1)

    nodes, edges, origin, destinations = parse_file(filename)
    search_fn = METHODS[method]
    goal, num_nodes, path = search_fn(nodes, edges, origin, destinations)

    if goal is None:
        no_path(filename, method)
    else:
        format_result(filename, method, goal, num_nodes, path)


if __name__ == '__main__':
    main()