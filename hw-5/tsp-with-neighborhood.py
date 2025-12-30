# Assignment 5 – TSP with Neighborhoods

# Objective: Extend the problem to continuous regions and finalize comparisons.
# • Instead of visiting fixed nodes, each “city” is a region (circle/polygon).
# • Implement a solver that selects a point within each region to minimize total tour length.
# • Apply your three approaches (heuristic, OR-Tools/other, AI technique).
# • Visualize tours on a map (leaflet/folium).
# • Compare and discuss results in a short (2–3 page) report.
# Bursa – İznik Case Study
# ==============================

import random
import time
import math
import folium
import matplotlib.pyplot as plt
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# -------------------------------------------------
# 1. Neighborhood Tanımı
# -------------------------------------------------
CENTER_LAT = 40.4289   # İznik
CENTER_LON = 29.7215

N_NEIGHBORHOODS = 10
RADIUS_KM = 0.4

random.seed(42)

def random_point_in_circle(lat, lon, radius_km):
    r = radius_km * math.sqrt(random.random())
    theta = random.random() * 2 * math.pi

    dx = r * math.cos(theta)
    dy = r * math.sin(theta)

    return (
        lat + dy / 111,
        lon + dx / (111 * math.cos(math.radians(lat)))
    )

# Neighborhood merkezleri
neighborhoods = []
for _ in range(N_NEIGHBORHOODS):
    neighborhoods.append(
        random_point_in_circle(CENTER_LAT, CENTER_LON, 8)
    )

# -------------------------------------------------
# 2. Neighborhood içinden nokta seçme
# -------------------------------------------------
def sample_points_from_neighborhoods():
    points = []
    for (lat, lon) in neighborhoods:
        points.append(
            random_point_in_circle(lat, lon, RADIUS_KM)
        )
    return points

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def distance_matrix(points):
    n = len(points)
    return [[dist(points[i], points[j]) for j in range(n)] for i in range(n)]

# -------------------------------------------------
# 3. Nearest Neighbor
# -------------------------------------------------
def solve_nn(points):
    start = 0
    visited = [start]
    unvisited = set(range(1, len(points)))
    total = 0
    cur = start

    while unvisited:
        nxt = min(unvisited, key=lambda x: dist(points[cur], points[x]))
        total += dist(points[cur], points[nxt])
        cur = nxt
        visited.append(cur)
        unvisited.remove(cur)

    total += dist(points[cur], points[start])
    visited.append(start)
    return visited, total

# -------------------------------------------------
# 4. OR-Tools
# -------------------------------------------------
def solve_ortools(points):
    mat = distance_matrix(points)
    scale = 1000
    int_mat = [[int(m*scale) for m in row] for row in mat]

    manager = pywrapcp.RoutingIndexManager(len(points), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def callback(i, j):
        return int_mat[manager.IndexToNode(i)][manager.IndexToNode(j)]

    transit = routing.RegisterTransitCallback(callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.time_limit.seconds = 3

    sol = routing.SolveWithParameters(params)
    route = []
    index = routing.Start(0)

    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = sol.Value(routing.NextVar(index))
    route.append(0)

    total = sum(dist(points[route[i]], points[route[i+1]]) for i in range(len(route)-1))
    return route, total

# -------------------------------------------------
# 5. Genetic Algorithm (Basit)
# -------------------------------------------------
def solve_ga(points, pop=40, gen=150):
    n = len(points)

    def random_tour():
        t = list(range(1, n))
        random.shuffle(t)
        return [0]+t+[0]

    def length(t):
        return sum(dist(points[t[i]], points[t[i+1]]) for i in range(len(t)-1))

    population = [random_tour() for _ in range(pop)]
    best = min(population, key=length)

    for _ in range(gen):
        population.sort(key=length)
        population = population[:pop//2]

        while len(population) < pop:
            a, b = random.sample(population, 2)
            cut = random.randint(1, n-2)
            child = a[:cut] + [x for x in b if x not in a[:cut]]
            child.append(0)
            population.append(child)

        best = min(population+[best], key=length)

    return best, length(best)

# -------------------------------------------------
# 6. Çalıştır & Ölç
# -------------------------------------------------
results = {}

for name, solver in {
    "NN": solve_nn,
    "OR-Tools": solve_ortools,
    "GA": solve_ga
}.items():
    pts = sample_points_from_neighborhoods()
    t0 = time.perf_counter()
    tour, length_val = solver(pts)
    t = time.perf_counter() - t0

    results[name] = (pts, tour, length_val, t)

# -------------------------------------------------
# 7. Harita
# -------------------------------------------------
m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=11)

# Neighborhoods
for (lat, lon) in neighborhoods:
    folium.Circle(
        location=(lat, lon),
        radius=RADIUS_KM*1000,
        color="blue",
        fill=False
    ).add_to(m)

colors = {"NN":"red", "OR-Tools":"green", "GA":"purple"}

for name, (pts, tour, _, _) in results.items():
    coords = [(pts[i][0], pts[i][1]) for i in tour]
    folium.PolyLine(coords, color=colors[name], weight=3, tooltip=name).add_to(m)

m.save("hw5_tsp_neighborhoods.html")

# -------------------------------------------------
# 8. Grafikler
# -------------------------------------------------
labels = list(results.keys())
lengths = [results[k][2] for k in labels]
times = [results[k][3] for k in labels]
times_s = [results[k][3] for k in labels]   # saniye
times_ms = [t * 1000 for t in times_s] # milisaniye

plt.figure()
plt.bar(labels, lengths)
plt.ylabel("Total Tour Length")
plt.title("Tour Length Comparison")
plt.show()

plt.figure()
plt.bar(labels, times_ms)
plt.yscale("log")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime Comparison")
plt.show()
