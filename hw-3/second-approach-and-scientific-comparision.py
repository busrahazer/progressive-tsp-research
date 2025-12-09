# ==============================
# Assignment 3 – Second Approach & Scientific Comparison

# Objective: Introduce an advanced solver and compare under research-style conditions.
# • Implement or call a second TSP approach (e.g., Google OR-Tools).
# • Compare the two approaches (your heuristic vs. OR-Tools) using scientific standards:
#     o Fix random seeds for repeatability.
#     o Generate at least 30 different topologies.
#     o Compute average tour length and runtime.
# • Visualize results in tables/plots
# • Submit: Scripts, plots, and a short (1–2 page) comparison report.
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
OUT_DIR = "hw-3"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_INSTANCES = 30        # En az 30 topology isteği
N_NODES = 15              # Her instance'taki şehir sayısı
AREA_SIZE = 100           # Noktaların üretileceği kare alan (0..AREA_SIZE)
BASE_SEED = 1000          # Repeatable seeds başlangıcı

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

# ------------ Deney Döngüsü ------------
records = []
for inst in range(NUM_INSTANCES):
    seed = BASE_SEED + inst
    points = generate_points(seed, n=N_NODES, area=AREA_SIZE)
    dist_mat = compute_distance_matrix(points)

        # NN çok hızlı olabilir; o yüzden REPEAT_NN kere çalıştırıp ortalama ns cinsinden alıyoruz.
    t_ns_total = 0
    nn_tour = None
    nn_length = None
    for _ in range(REPEAT_NN):
        t0_ns = time.perf_counter_ns()
        tour_tmp, length_tmp = nearest_neighbor_from_matrix(dist_mat, start=0)
        t_ns_total += (time.perf_counter_ns() - t0_ns)
        # son tekrardan rota ve uzunluğu al
        nn_tour = tour_tmp
        nn_length = length_tmp
    avg_nn_time_s = (t_ns_total / REPEAT_NN) / 1e9  # saniyeye çevir

    # --- OR-Tools: zaman ölçümü (perf_counter) ---
    t0 = time.perf_counter()
    ort_tour, ort_length = solve_tsp_ortools(dist_mat, time_limit_sec=ORTOOLS_TIME_LIMIT_SEC)
    ort_time_s = time.perf_counter() - t0

    if ort_tour is None:
        ort_length = float('inf')

    # record results
    records.append({
        "instance": inst,
        "seed": seed,
        "n_nodes": N_NODES,
        "nn_length": nn_length,
        "nn_time_s": avg_nn_time_s,
        "ort_length": ort_length,
        "ort_time_s": ort_time_s
    })
    print(f"Inst {inst+1}/{NUM_INSTANCES}: NN_len={nn_length:.2f} NN_t={avg_nn_time_s*1e6:.2f} µs | ORT_len={ort_length:.2f} ORT_t={ort_time_s:.3f}s")

# ------------ Analysis & Plots ------------
# Ortalama uzunluk ve zaman hesapları (OR-Tools inf ise NaN yap)
df = pd.DataFrame.from_records(records)
avg_nn_len = df["nn_length"].mean()
avg_ort_len = df["ort_length"].replace(np.inf, np.nan).mean()  # ignore inf if any
avg_nn_time_s = df["nn_time_s"].mean()
avg_ort_time_s = df["ort_time_s"].mean()

print(f"\nOrtalama tur uzunlukları: NN={avg_nn_len:.2f}, OR-Tools={avg_ort_len:.2f}")
print(f"Ortalama çalışma süreleri (s): NN={avg_nn_time_s*1e6:.2f} µs, OR-Tools={avg_ort_time_s:.4f}s")

# Bar chart - average tour length
plt.figure(figsize=(6,4))
plt.bar(["Nearest Neighbor", "OR-Tools"], [avg_nn_len, avg_ort_len], color=["tab:blue", "tab:orange"])
plt.ylabel("Average tour length")
plt.title("Average tour length over instances")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "avg_lengths.png"), dpi=200)

# Bar chart - average runtime
plt.figure(figsize=(6,4))
avg_nn_time_us = avg_nn_time_s * 1e6
avg_ort_time_us = avg_ort_time_s * 1e6
plt.bar(["Nearest Neighbor", "OR-Tools"], [avg_nn_time_us, avg_ort_time_us], color=["tab:blue", "tab:orange"])
plt.ylabel("Average runtime (µs)")
plt.title("Average runtime (microseconds)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "avg_runtimes.png"), dpi=200)

# Boxplot of tour lengths to see spread
plt.figure(figsize=(8,5))
plt.boxplot([df["nn_length"].values, df["ort_length"].replace(np.inf, np.nan).values], labels=["NN", "OR-Tools"])
plt.ylabel("Tour length")
plt.title("Distribution of tour lengths")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "lengths_boxplot.png"), dpi=200)

print(f" Plots saved in {os.path.join(OUT_DIR, 'hw-3')}")
print("\nDone.")
