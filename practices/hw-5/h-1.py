# =========================================
# Assignment 5 – TSP with Neighborhoods 
# # Dünya haritası denemesi
# =========================================

import random
import math
import folium
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# -----------------------------
# 1. Neighborhood Model
# -----------------------------

def generate_neighborhoods(n_regions, area_size=100, radius=8):
    neighborhoods = []
    for i in range(n_regions):
        cx = random.uniform(0, area_size)
        cy = random.uniform(0, area_size)
        neighborhoods.append({
            "id": i,
            "center": (cx, cy),
            "radius": radius
        })
    return neighborhoods


def sample_point_in_circle(center, radius):
    angle = random.uniform(0, 2 * math.pi)
    r = radius * math.sqrt(random.random())
    x = center[0] + r * math.cos(angle)
    y = center[1] + r * math.sin(angle)
    return (x, y)


def select_points_from_neighborhoods(neighborhoods):
    return [sample_point_in_circle(nb["center"], nb["radius"])
            for nb in neighborhoods]


# -----------------------------
# 2. Distance Utilities
# -----------------------------

def euclidean(a, b):
    return math.dist(a, b)


def distance_matrix(points):
    n = len(points)
    mat = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            mat[i][j] = euclidean(points[i], points[j])
    return mat


# -----------------------------
# 3. Nearest Neighbor Solver
# -----------------------------

def nearest_neighbor(dist_mat, start=0):
    n = len(dist_mat)
    visited = [start]
    unvisited = set(range(n))
    unvisited.remove(start)
    total = 0.0
    cur = start

    while unvisited:
        nxt = min(unvisited, key=lambda x: dist_mat[cur][x])
        total += dist_mat[cur][nxt]
        visited.append(nxt)
        unvisited.remove(nxt)
        cur = nxt

    total += dist_mat[cur][start]
    visited.append(start)
    return visited, total


# -----------------------------
# 4. Genetic Algorithm Solver
# -----------------------------

def create_random_tour(n):
    tour = list(range(1, n))
    random.shuffle(tour)
    return [0] + tour + [0]


def tour_length(tour, dist):
    return sum(dist[tour[i]][tour[i+1]] for i in range(len(tour)-1))


def tournament_selection(pop, dist, k=3):
    candidates = random.sample(pop, k)
    candidates.sort(key=lambda t: tour_length(t, dist))
    return candidates[0]


def crossover(p1, p2):
    size = len(p1) - 2
    a, b = sorted(random.sample(range(1, size + 1), 2))
    child = [None]*(size+2)
    child[0] = child[-1] = 0
    child[a:b] = p1[a:b]

    fill = [x for x in p2 if x not in child]
    idx = 1
    for x in fill:
        while child[idx] is not None:
            idx += 1
        child[idx] = x
    return child


def mutate(tour, rate=0.1):
    if random.random() < rate:
        i, j = random.sample(range(1, len(tour)-1), 2)
        tour[i], tour[j] = tour[j], tour[i]


def solve_ga(dist, pop_size=50, generations=200):
    n = len(dist)
    population = [create_random_tour(n) for _ in range(pop_size)]
    best_tour = None
    best_len = float("inf")

    for _ in range(generations):
        new_pop = []
        for _ in range(pop_size):
            p1 = tournament_selection(population, dist)
            p2 = tournament_selection(population, dist)
            child = crossover(p1, p2)
            mutate(child)
            l = tour_length(child, dist)
            if l < best_len:
                best_len = l
                best_tour = child
            new_pop.append(child)
        population = new_pop

    return best_tour, best_len


# -----------------------------
# 5. OR-Tools Solver
# -----------------------------

def solve_ortools(dist):
    n = len(dist)
    scale = 1000
    int_dist = [[int(dist[i][j]*scale) for j in range(n)] for i in range(n)]

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def callback(i, j):
        return int_dist[manager.IndexToNode(i)][manager.IndexToNode(j)]

    cb_index = routing.RegisterTransitCallback(callback)
    routing.SetArcCostEvaluatorOfAllVehicles(cb_index)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.time_limit.seconds = 5

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return None, float("inf")

    idx = routing.Start(0)
    route = []
    while not routing.IsEnd(idx):
        route.append(manager.IndexToNode(idx))
        idx = solution.Value(routing.NextVar(idx))
    route.append(0)

    length = sum(dist[route[i]][route[i+1]] for i in range(len(route)-1))
    return route, length


# -----------------------------
# 6. Visualization (Folium)
# -----------------------------

def visualize(neighborhoods, points, tours):
    center = neighborhoods[0]["center"]
    m = folium.Map(location=[center[1], center[0]], zoom_start=12)

    # Draw neighborhoods
    for nb in neighborhoods:
        folium.Circle(
            location=[nb["center"][1], nb["center"][0]],
            radius=nb["radius"]*1000,
            color="blue",
            fill=False
        ).add_to(m)

    # Draw selected points
    for p in points:
        folium.CircleMarker(
            location=[p[1], p[0]],
            radius=4,
            color="black",
            fill=True
        ).add_to(m)

    colors = {"NN": "red", "GA": "green", "ORT": "purple"}

    for name, tour in tours.items():
        path = [[points[i][1], points[i][0]] for i in tour]
        folium.PolyLine(path, color=colors[name], tooltip=name).add_to(m)

    return m


# -----------------------------
# 7. Main Execution
# -----------------------------

random.seed(42)
N = 8

neighborhoods = generate_neighborhoods(N)
points = select_points_from_neighborhoods(neighborhoods)
dist = distance_matrix(points)

nn_tour, _ = nearest_neighbor(dist)
ga_tour, _ = solve_ga(dist)
ort_tour, _ = solve_ortools(dist)

map_result = visualize(
    neighborhoods,
    points,
    {
        "NN": nn_tour,
        "GA": ga_tour,
        "ORT": ort_tour
    }
)

map_result.save("hw5_tsp_with_neighborhoods.html")
print("✅ HW5 completed. Map saved as hw5_tsp_with_neighborhoods.html")
