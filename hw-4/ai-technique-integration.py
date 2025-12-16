# ==============================
# Assignment 4 – AI Technique Integration
# Objective: Bring an AI method into the pipeline.
# • Choose one AI technique taught in the course (Genetic Programming, Fuzzy Logic, Neural Network, Reinforcement Learning, etc.).
# • Implement it as a third solver for the TSP instances.
# • Compare performance with your previous two approaches using the same 30-instance protocol.
# • Submit: Scripts, performance plots, and short analysis.
# ==============================

import os
import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# ------------ Başlangıç Parametreleri ------------
OUT_DIR = "hw-4"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_INSTANCES = 30        # En az 30 topology isteği
N_NODES = 15              # Her instance'taki şehir sayısı
AREA_SIZE = 100           # Noktaların üretileceği kare alan (0..AREA_SIZE)
BASE_SEED = 1000          # Repeatable seeds başlangıcı

# Genetik Algoritma Parametreleri
GA_POP_SIZE = 50
GA_GENERATIONS = 200
GA_MUTATION_RATE = 0.1
GA_SEED = 42 # Rastgelelik için sabit bir seed

# OR-Tools için zaman limiti (saniye)
ORTOOLS_TIME_LIMIT_SEC = 5

# Nearest Neighbor zamanını daha güvenilir ölçmek için kaç defa tekrarlayacağımız 
# (Çok hızlı olduğu için ölçümü daha hassas yapmam gerekti.)
REPEAT_NN = 50

# ------------ Utility functions ------------
def generate_points(seed, n=N_NODES, area=AREA_SIZE):
    # Belirli seed ile n adet rastgele (x,y) nokta üretir.
    rnd = random.Random(seed)
    pts = [(rnd.uniform(0, area), rnd.uniform(0, area)) for _ in range(n)]
    return pts

def euclidean_distance(a, b): # İki nokta arasındaki Öklid uzaklığı.
    return math.hypot(a[0]-b[0], a[1]-b[1])

def compute_distance_matrix(points): # Gerçek (float) mesafe matrisi oluşturur.
    n = len(points)
    mat = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i][j] = 0.0
            else:
                mat[i][j] = euclidean_distance(points[i], points[j])
    return mat

# Nearest Neighbor heuristic
def nearest_neighbor_from_matrix(dist_mat, start=0):
    n = len(dist_mat)
    visited = [start]
    unvisited = set(range(n))
    unvisited.remove(start)
    total = 0.0
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda x: dist_mat[cur][x])
        total += dist_mat[cur][nxt]
        cur = nxt
        visited.append(cur)
        unvisited.remove(cur)
    # return to start
    total += dist_mat[cur][start]
    visited.append(start)
    return visited, total

# OR-Tools TSP çözümü
def solve_tsp_ortools(dist_matrix, time_limit_sec=ORTOOLS_TIME_LIMIT_SEC):
    """OR-Tools ile TSP çözümü:
       - dist_matrix: float mesafe matrisi
       - time_limit_sec: her instance için arama süresi (saniye)
       Çıktı: (route_list, total_length) veya (None, inf) eğer çözüm yoksa.
    """
    n = len(dist_matrix)
    # OR-Tools integer kullanır o yüzden dönüştürüldü
    scale = 1000  # mm hassasiyeti
    int_matrix = [[int(round(dist_matrix[i][j] * scale)) for j in range(n)] for i in range(n)]

    # Manager: düğüm indekslerini yönetir (iç indeks <-> gerçek node)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # single vehicle, depot 0
    routing = pywrapcp.RoutingModel(manager)

    # OR-Tools için mesafe callback'i kaydet
    def distance_callback(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int_matrix[i][j]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    
    # Tüm araçlar için maliyet değerlendiricisini ayarla
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    # search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.time_limit.seconds = time_limit_sec
    # Hızlı başlangıç stratejisi
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    # Yerel arama iyileştiricisi
    search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.log_search = False

    # Çözümü çalıştır
    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        return None, float('inf')

    # Çözümden rota çıkartma
    index = routing.Start(0)
    route = []
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        route.append(node)
        index = solution.Value(routing.NextVar(index))
    route.append(manager.IndexToNode(index))  # back to depot

    # Orijinal float matrisinden gerçek uzunluğu hesapla
    total = 0.0
    for k in range(len(route)-1):
        total += dist_matrix[route[k]][route[k+1]]
    return route, total

# GA Çözümü
def create_random_tour_ga(n_nodes):
    tour = list(range(1, n_nodes))
    random.shuffle(tour)
    return [0] + tour + [0]

def tour_length(tour, dist_matrix):
    length = 0.0
    for i in range(len(tour) - 1):
        length += dist_matrix[tour[i]][tour[i+1]]
    return length

def initial_population(pop_size, n_nodes):
    return [create_random_tour_ga(n_nodes) for _ in range(pop_size)]

def tournament_selection(population, dist_matrix, k=3):
    contenders = random.sample(population, k)
    contenders.sort(key=lambda t: tour_length(t, dist_matrix))
    return contenders[0]

def crossover(parent1, parent2):
    size = len(parent1) - 2
    # Başlangıç ve bitiş hariç aralığı seç
    a, b = sorted(random.sample(range(1, size + 1), 2))

    child = [None] * (size + 2)
    child[0] = child[-1] = 0
    # İlk ebeveynden parçayı kopyala
    child[a:b] = parent1[a:b]

    # İkinci ebeveyndeki eksik şehirleri sıra ile doldur
    fill = [x for x in parent2 if x not in child]
    idx = 1
    for x in fill:
        while child[idx] is not None:
            idx = (idx + 1) % (size + 1)
            if idx == 0: idx = 1 # 0'ı atla
        child[idx] = x

    return child

def mutate(tour, mutation_rate=0.1):
    if random.random() < mutation_rate:
        # 0 (depo) hariç rastgele iki şehir seç
        i, j = random.sample(range(1, len(tour)-1), 2)
        tour[i], tour[j] = tour[j], tour[i]

def solve_tsp_ga(dist_matrix, pop_size=GA_POP_SIZE, generations=GA_GENERATIONS, mutation_rate=GA_MUTATION_RATE):
    """GA ile TSP çözümü."""
    n_nodes = len(dist_matrix)
    population = initial_population(pop_size, n_nodes)

    best_tour = None
    best_length = float('inf')

    for gen in range(generations):
        new_population = []

        # Elitizm: En iyi çözümü koru 
        current_best = min(population, key=lambda t: tour_length(t, dist_matrix))
        current_best_len = tour_length(current_best, dist_matrix)
        if current_best_len < best_length:
            best_length = current_best_len
            best_tour = current_best

        # Yeni popülasyonu oluştur
        for _ in range(pop_size):
            p1 = tournament_selection(population, dist_matrix)
            p2 = tournament_selection(population, dist_matrix)

            child = crossover(p1, p2)
            mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

    return best_tour, best_length

# ------------ Deney Döngüsü ------------
records = []
for inst in range(NUM_INSTANCES):
    seed = BASE_SEED + inst
    
    random.seed(seed)

    points = generate_points(seed, n=N_NODES, area=AREA_SIZE)
    dist_mat = compute_distance_matrix(points)

    # --- 1. Nearest Neighbor ---
    t_ns_total = 0
    nn_tour = None
    nn_length = None
    for _ in range(REPEAT_NN):
        t0_ns = time.perf_counter_ns()
        tour_tmp, length_tmp = nearest_neighbor_from_matrix(dist_mat, start=0)
        t_ns_total += (time.perf_counter_ns() - t0_ns)
        nn_tour = tour_tmp
        nn_length = length_tmp
    avg_nn_time_s = (t_ns_total / REPEAT_NN) / 1e9

    # --- 2. OR-Tools ---
    t0_ort = time.perf_counter()
    ort_tour, ort_length = solve_tsp_ortools(dist_mat, time_limit_sec=ORTOOLS_TIME_LIMIT_SEC)
    ort_time_s = time.perf_counter() - t0_ort
    if ort_tour is None:
        ort_length = float('inf')

    # --- 3. Genetic Algorithm (GA) ---
    # solve_tsp_ga içinde random.seed() kullanmadık, ana döngüde ayarlandı.
    t0_ga = time.perf_counter()
    ga_tour, ga_length = solve_tsp_ga(dist_mat)
    ga_time_s = time.perf_counter() - t0_ga
    
    # record results
    records.append({
        "instance": inst,
        "seed": seed,
        "n_nodes": N_NODES,
        "nn_length": nn_length,
        "nn_time_s": avg_nn_time_s,
        "ort_length": ort_length,
        "ort_time_s": ort_time_s,
        "ga_length": ga_length,
        "ga_time_s": ga_time_s
    })
    
    print(f"Inst {inst+1}/{NUM_INSTANCES}: NN_len={nn_length:.2f} NN_t={avg_nn_time_s*1e6:.2f} µs | ORT_len={ort_length:.2f} ORT_t={ort_time_s:.3f}s | GA_len={ga_length:.2f} GA_t={ga_time_s:.3f}s")

# ------------ Analysis & Plots ------------
# Ortalama uzunluk ve zaman hesapları (OR-Tools inf ise NaN yap)
df = pd.DataFrame.from_records(records)
avg_nn_len = df["nn_length"].mean()
avg_ort_len = df["ort_length"].replace(np.inf, np.nan).mean()  # ignore inf if any
avg_nn_time_s = df["nn_time_s"].mean()
avg_ort_time_s = df["ort_time_s"].mean()
avg_ga_len = df["ga_length"].mean()
avg_ga_time_s = df["ga_time_s"].mean()

print("\n--- Özet ---")
print(f"Ortalama tur uzunlukları: NN={avg_nn_len:.2f}, OR-Tools={avg_ort_len:.2f}, GA={avg_ga_len:.2f}")
print(f"Ortalama çalışma süreleri: NN={avg_nn_time_s*1e6:.2f} µs, OR-Tools={avg_ort_time_s:.4f}s, GA={avg_ga_time_s:.4f}s")
print("------------")

# 1. Bar chart - average tour length
labels = ["NN", "OR-Tools", "GA"]
avg_lengths = [avg_nn_len, avg_ort_len, avg_ga_len]

plt.figure(figsize=(8, 5))
plt.bar(labels, avg_lengths, color=["tab:blue", "tab:orange", "tab:green"])
plt.ylabel("Ortalama Tur Uzunluğu")
plt.title(f"{NUM_INSTANCES} Instance Üzerinden Ortalama Tur Uzunlukları (N={N_NODES})")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "avg_lengths_3way.png"), dpi=200)

# 2. Bar chart - average runtime
avg_times_s = [avg_nn_time_s, avg_ort_time_s, avg_ga_time_s]
avg_times_ms = [t * 1000 for t in avg_times_s]

plt.figure(figsize=(8, 5))
plt.bar(labels, avg_times_ms, color=["tab:blue", "tab:orange", "tab:green"])
plt.ylabel("Ortalama Çalışma Süresi (ms)")
plt.title(f"{NUM_INSTANCES} Instance Üzerinden Ortalama Çalışma Süreleri (N={N_NODES})")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "avg_runtimes_3way.png"), dpi=200)

print(f"\n Grafikler {os.path.join(os.getcwd(), OUT_DIR)} klasörüne kaydedildi.")
print("\nBitti.")