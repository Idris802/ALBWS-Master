#!/usr/bin/env python3
import math
import random
import itertools
import numpy as np
import time

###############################
# Helper: Euclidean Distance  #
###############################
def euclidean_distance(p, q):
    return math.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)

#############################################
# 1. Nearest Neighbor Heuristic for TSP     #
#############################################
def nearest_neighbor_tsp(points):
    n = len(points)
    if n == 0: 
        return []
    unvisited = list(range(n))
    tour = []
    current = unvisited.pop(0)  # starting from the first point
    tour.append(current)
    while unvisited:
        next_city = min(unvisited, key=lambda i: euclidean_distance(points[current], points[i]))
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    return tour

def tour_cost(tour, points):
    cost = 0
    for i in range(len(tour)):
        cost += euclidean_distance(points[tour[i]], points[tour[(i+1) % len(tour)]])
    return cost

######################################################
# 2. Simulated Annealing TSP (Metaheuristic Approach)  #
######################################################
def simulated_annealing_tsp(points, initial_temp=1000, cooling_rate=0.995, iterations=10000):
    current_tour = nearest_neighbor_tsp(points)
    current_cost = tour_cost(current_tour, points)
    best_tour = current_tour[:]
    best_cost = current_cost
    T = initial_temp
    for i in range(iterations):
        # Generate a neighbor by swapping two cities
        a, b = random.sample(range(len(points)), 2)
        new_tour = current_tour[:]
        new_tour[a], new_tour[b] = new_tour[b], new_tour[a]
        new_cost = tour_cost(new_tour, points)
        delta = new_cost - current_cost
        # Accept new tour if cost is lower or with a probability if worse
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_tour = new_tour
            current_cost = new_cost
            if new_cost < best_cost:
                best_tour = new_tour
                best_cost = new_cost
        T *= cooling_rate
    return best_tour

##################################
# 3. Christofides Algorithm      #
##################################
def prim_mst(points):
    n = len(points)
    in_mst = [False] * n
    key = [float('inf')] * n
    parent = [-1] * n
    key[0] = 0
    for _ in range(n):
        u = min(range(n), key=lambda i: key[i] if not in_mst[i] else float('inf'))
        in_mst[u] = True
        for v in range(n):
            d = euclidean_distance(points[u], points[v])
            if not in_mst[v] and d < key[v]:
                key[v] = d
                parent[v] = u
    mst_edges = []
    for v in range(1, n):
        mst_edges.append((parent[v], v))
    return mst_edges

def find_odd_vertices(mst_edges, n):
    degree = [0] * n
    for u, v in mst_edges:
        degree[u] += 1
        degree[v] += 1
    odd = [i for i, d in enumerate(degree) if d % 2 == 1]
    return odd

def min_weight_perfect_matching(odd, points):
    if not odd:
        return []
    best = None
    best_cost = float('inf')
    first = odd[0]
    for i in range(1, len(odd)):
        pair = (first, odd[i])
        rest = odd[1:i] + odd[i+1:]
        matching_rest = min_weight_perfect_matching(rest, points)
        matching = [pair] + matching_rest
        cost = sum(euclidean_distance(points[u], points[v]) for (u, v) in matching)
        if cost < best_cost:
            best_cost = cost
            best = matching
    return best

def combine_mst_matching(mst_edges, matching):
    # Build a multigraph as an adjacency list
    graph = {}
    def add_edge(u, v):
        graph.setdefault(u, []).append(v)
    for u, v in mst_edges:
        add_edge(u, v)
        add_edge(v, u)
    for u, v in matching:
        add_edge(u, v)
        add_edge(v, u)
    return graph

def eulerian_tour(graph, start=0):
    # Hierholzer's algorithm
    graph_copy = {u: list(neighbors) for u, neighbors in graph.items()}
    tour = []
    stack = [start]
    while stack:
        v = stack[-1]
        if graph_copy.get(v):
            w = graph_copy[v].pop()
            stack.append(w)
            graph_copy[w].remove(v)
        else:
            tour.append(stack.pop())
    return tour

def shortcut_tour(euler_tour):
    visited = set()
    circuit = []
    for v in euler_tour:
        if v not in visited:
            circuit.append(v)
            visited.add(v)
    circuit.append(circuit[0])
    return circuit

def christofides_tsp(points):
    n = len(points)
    mst_edges = prim_mst(points)
    odd_vertices = find_odd_vertices(mst_edges, n)
    matching = min_weight_perfect_matching(odd_vertices, points)
    graph = combine_mst_matching(mst_edges, matching)
    euler_t = eulerian_tour(graph, 0)
    hamiltonian = shortcut_tour(euler_t)
    return hamiltonian

##################################
# 4. Branch and Bound TSP         #
##################################
def branch_and_bound_tsp(points):
    n = len(points)
    best_tour = None
    best_cost = float('inf')
    
    def recurse(tour, cost, remaining):
        nonlocal best_tour, best_cost
        if not remaining:
            total_cost = cost + euclidean_distance(points[tour[-1]], points[tour[0]])
            if total_cost < best_cost:
                best_cost = total_cost
                best_tour = tour.copy()
        else:
            # Lower bound heuristic: add min edge from each remaining vertex to any visited vertex
            bound = cost
            for v in remaining:
                bound += min(euclidean_distance(points[u], points[v]) for u in tour)
            if bound >= best_cost:
                return
            for v in list(remaining):
                tour.append(v)
                remaining.remove(v)
                recurse(tour, cost + euclidean_distance(points[tour[-2]], points[v]), remaining)
                tour.pop()
                remaining.add(v)
    
    remaining = set(range(n))
    start = 0
    remaining.remove(start)
    recurse([start], 0, remaining)
    return best_tour, best_cost


def benchmark_algorithm(algorithm_func, cost_func, *args, runs=30):
    times = []
    costs = []
    for _ in range(runs):
        start = time.time()
        tour = algorithm_func(*args)
        elapsed = time.time() - start
        times.append(elapsed)
        costs.append(cost_func(tour, args[0]))  # assuming first arg is 'points'
    return {
        "avg_time": np.mean(times),
        "std_time": np.std(times),
        "avg_cost": np.mean(costs),
        "std_cost": np.std(costs),
    }

def print_results(name, result):
    print(f"{name}:")
    print(f"  Avg Time  = {result['avg_time']:.6f}s ± {result['std_time']:.6f}s")
    print(f"  Avg Cost  = {result['avg_cost']:.6f} ± {result['std_cost']:.6f}\n")

###############################
# Example Usage and Comparison#
###############################
if __name__ == "__main__":
    # Example set of points (2D coordinates)
    points = [
        (-0.3738729799137606, -0.29980112826021416), (-0.2473369637210849, -0.17858211467646673),
        (-0.45812922704991776, -0.14543003269053853), (-0.3343023651160801, -0.009476697317997872),
        (-0.29961311623030135, 0.21202017955604452), (-0.44859124935707373, 0.17697553961075976)
    ]

    """
    
    print("Nearest Neighbor TSP:")
    nn_tour = nearest_neighbor_tsp(points)
    print("Tour:", nn_tour, "Cost:", tour_cost(nn_tour, points))
    sorted_points_nn = [points[i] for i in nn_tour]
    print(sorted_points_nn)

    
    print("\nSimulated Annealing TSP:")
    sa_tour = simulated_annealing_tsp(points, initial_temp=1000, cooling_rate=0.995, iterations=10000)
    print("Tour:", sa_tour, "Cost:", tour_cost(sa_tour, points))
    sorted_points_sa = [points[i] for i in sa_tour]
    print(sorted_points_sa)
    
    print("\nChristofides TSP:")
    ch_tour = christofides_tsp(points)
    print("Tour:", ch_tour, "Cost:", tour_cost(ch_tour, points))
    if ch_tour[0] == ch_tour[-1]:
        ch_tour = ch_tour[:-1] 
    sorted_points_ch = [points[i] for i in ch_tour]
    print(sorted_points_ch)
    
    print('\nBranch and Bound TSP (Exact):')
    bb_tour, bb_cost = branch_and_bound_tsp(points)
    print("Tour:", bb_tour, "Cost:", bb_cost)
    sorted_points_bb = [points[i] for i in bb_tour]
    print(sorted_points_bb)
    """

    nn_results = benchmark_algorithm(nearest_neighbor_tsp, tour_cost, points)
    sa_results = benchmark_algorithm(simulated_annealing_tsp, tour_cost, points, 1000, 0.995, 10000)
    ch_results = benchmark_algorithm(christofides_tsp, tour_cost, points)
    bb_results = benchmark_algorithm(lambda pts: branch_and_bound_tsp(pts)[0], tour_cost, points)


    print_results("Nearest Neighbor", nn_results)
    print_results("Simulated Annealing", sa_results)
    print_results("Christofides", ch_results)
    print_results("Branch and Bound", bb_results)
