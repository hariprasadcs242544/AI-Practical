from collections import deque

# Shopping Mall Navigation Graph (Adjacency List)
mall_map = {
    'Main Entrance': ['Food Court', 'Clothing Store', 'Electronics Store', 'Supermarket'],

    # Route 1
    'Food Court': ['Cinema'],
    'Cinema': ['Exit'],

    # Route 2
    'Clothing Store': ['Book Store', 'Footwear Store'],
    'Book Store': ['Exit'],
    'Footwear Store': ['Exit'],

    # Route 3
    'Electronics Store': ['Gaming Zone'],
    'Gaming Zone': ['Exit'],

    # Route 4
    'Supermarket': ['Home Decor'],
    'Home Decor': ['Gift Shop'],
    'Gift Shop': ['Exit'],

    # Destination
    'Exit': []
}

# Breadth First Search (BFS)
def bfs_shortest_path(graph, start, destination):
    queue = deque([(start, [start])])
    visited = set([start])
    nodes_explored_count = 0

    while queue:
        current, path = queue.popleft()
        nodes_explored_count += 1

        if current == destination:
            return path, nodes_explored_count

        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None, nodes_explored_count

# Iterative Depth First Search (DFS)
def iterative_dfs_path(graph, start, destination):
    stack = [(start, [start])]
    visited = set()
    nodes_explored_count = 0

    while stack:
        current, path = stack.pop()
        nodes_explored_count += 1

        if current == destination:
            return path, nodes_explored_count

        if current not in visited:
            visited.add(current)

            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

    return None, nodes_explored_count

# Execution
start_node = 'Main Entrance'
end_node = 'Exit'

bfs_path, bfs_count = bfs_shortest_path(mall_map, start_node, end_node)
dfs_path, dfs_count = iterative_dfs_path(mall_map, start_node, end_node)

print("--- BFS Results ---")
print(f"Path Found: {' -> '.join(bfs_path)}")
print(f"Total Steps (Edges): {len(bfs_path) - 1}")
print(f"Total Nodes Visited/Checked: {bfs_count}\n")

print("--- Iterative DFS Results ---")
print(f"Path Found: {' -> '.join(dfs_path)}")
print(f"Total Steps (Edges): {len(dfs_path) - 1}")
print(f"Total Nodes Visited/Checked: {dfs_count}")
