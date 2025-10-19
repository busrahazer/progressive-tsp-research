# ==============================
# Bir önceki grafiğin üstüne networkx kütüphanesi ile grafik ürettiriyorum.
# Noktalar arasındaki mesafeyi (weight) hesaplıyorum.
# Bu koddan üretilen grafik, grpahs klasörüne eklendi. (m-2.png)
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

# 5. Görselleştirme
pos = nx.get_node_attributes(G, 'pos') # node pozisyonlarını al
plt.figure(figsize=(5,5))

# Düğümlerin çizimi
nx.draw_networkx_nodes(G, pos, node_color='pink', node_size=100)
# Kenarların çizimi
nx.draw_networkx_edges(G, pos, edge_color='gray')
# Node  isimleri
nx.draw_networkx_labels(G, pos, font_color='black', font_size=10)

# 6. Grafik ayarları
plt.title('Bağlantılı Grafik')
plt.xlabel('X Ekseni')
plt.ylabel('Y Ekseni')
plt.grid(True)
plt.xlim(0,AREA_SIZE)
plt.ylim(0,AREA_SIZE)
plt.show()
