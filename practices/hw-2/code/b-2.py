# ==============================
# Gerçek haritada İznik şehrinde seçilen 10 nokta için En Yakın Komşu algoritması uyguladım.
# Düğümlerin id'leri ile ziyaret edilmelerini ve toplam mesafeyi konsolda yazdırdım. (b-2.png)
# ==============================

import osmnx as ox
import random
import networkx as nx

# 1. Bursa, İznik şehrinin yollar ağını indirir
G = ox.graph_from_place("İznik, Bursa, Turkey", network_type="drive")

# 2. Grafikteki düğümleri listeler
nodes = list(G.nodes)

# 3. Rastgele 10 düğüm seçme
random.seed(42)
selected_nodes = random.sample(nodes, 10)

# 4. Gerçek mesafeleri hesaplayan yeni alt grafik
subG = nx.Graph()

for i in range(len(selected_nodes)):
    for j in range(i + 1, len(selected_nodes)):
        try:
            # Gerçek yol uzunluğu (metre)
            length = nx.shortest_path_length(G, selected_nodes[i], selected_nodes[j], weight="length")
            subG.add_edge(selected_nodes[i], selected_nodes[j], weight=length)
        except nx.NetworkXNoPath:
            pass  # Eğer iki nokta arasında yol yoksa, geç

# 4. En Yakın Komşu (Nearest Neighbor) algoritması
def nearest_neighbor_tsp(G, start_node):
    visited = [start_node]
    current = start_node
    total_dist = 0

    while len(visited) < len(G.nodes):
        neighbors = [(n, G[current][n]['weight']) for n in G.neighbors(current) if n not in visited]
        if not neighbors: # komsu yoksa geç
            break
        next_node, dist = min(neighbors, key=lambda x: x[1])
        visited.append(next_node)
        total_dist += dist
        current = next_node

    # Başlangıç noktasına geri dön
    total_dist += G[current][start_node]['weight']
    visited.append(start_node)
    return visited, total_dist

# 5. Algoritmayı çalıştır
start = selected_nodes[0]
tour, total_distance = nearest_neighbor_tsp(subG, start)
print("Tur sırası (node ID):", tour)
print(f"Toplam Mesafe: {total_distance:.2f} metre")