# ==============================
# Gri çizgiler tüm bağlantıları gösterir, kırmızı çizgiler ise satıcının izlediği yolu gösterir.
# Satıcı rastgele bir noktadan başladı ve en yakın komşuya gitti. Tüm noktaları ziyaret edip başlangıca döndü.
# Sonuç en optimal çözüm değil ancak en hızlı çözüm sayılabilir.
# Satıcının ziyareti 5 komşudan 0-4-1-3-2-0 şeklinde ve toplam mesafe 118'dir. (Konsolda çıktısı mevcut) 
# Bu koddan üretilen grafik, graphs klasörüne eklendi. (m-3.png)
# Bu alıştırma dosyası homework-1'in tamamlanmış halidir.
# ==============================

import random
import math
import matplotlib.pyplot as plt
import networkx as nx 

# 1. Parametreler 
NUM_POINTS = 5
AREA_SIZE = 50

# 2. Seed Değeri (Her zaman aynı rastgele diziyi üretir.)
random.seed(42)

# 3. Rastgele noktalar oluştur
points = []
for i in range(NUM_POINTS): # Noktaları (x, y) şeklinde saklıyorum
    x = random.uniform(0, AREA_SIZE)
    y = random.uniform(0, AREA_SIZE)
    points.append((x,y))

# 4. NetworkX grafiği oluşturma
G = nx.Graph()    

# Düğümleri ekleme
for i, p in enumerate(points):
    G.add_node(i,  pos=p)  # Her düğümün (x,y) pozisyonu

# Kenarları ekleme - her nokta diğer tüm noktalara bağlı
for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        distance = math.dist(points[i], points[j]) # İki nokta arasındaki Öklid mesafesi 
        G.add_edge(i,j, weight=distance)   

# 5. En Yakın Komşu Algoritması
def nearest_neighbor_tsp(G, start=0):
    visited = [start]
    current = start
    total_dist = 0

    while len(visited) < len(G.nodes):
        # Daha ziyaret edilmemiş komşular ve uzaklıkları
        neighbors = [(n, G[current][n]['weight']) for n in G.neighbors(current) if n not in visited]
        # En kısa mesafeli komşuyu bulma
        next_node, dist = min(neighbors, key=lambda x: x[1])
        visited.append(next_node)
        total_dist += dist
        current = next_node

    # En son başlangıç noktasına geri dön
    total_dist += G[current][start]['weight'] 
    visited.append(start)

    return visited, total_dist   

# 6. Algoritmayı çalıştırma
tour, total_distance = nearest_neighbor_tsp(G)
print("Tur Sırası:", tour)
print("Toplam Mesafe:", total_distance)

# 7. Görselleştirme
pos = nx.get_node_attributes(G, 'pos') # node pozisyonlarını al
plt.figure(figsize=(5,5))

# Tur sırasını kırmızı çizgiyle göster
path_edges = list(zip(tour[:-1], tour[1:]))
nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)

# Düğümlerin çizimi
nx.draw_networkx_nodes(G, pos, node_color='pink', node_size=100)
# Kenarların çizimi
nx.draw_networkx_edges(G, pos, edge_color='gray')
# Node  isimleri
nx.draw_networkx_labels(G, pos, font_color='black', font_size=10)

# 8. Grafik ayarları
plt.title(f"En Yakın Komşu TSP Turu (Mesafe = {total_distance:.2f})")
plt.xlabel('X Ekseni')
plt.ylabel('Y Ekseni')
plt.grid(True)
plt.xlim(0,AREA_SIZE)
plt.ylim(0,AREA_SIZE)
plt.show()
