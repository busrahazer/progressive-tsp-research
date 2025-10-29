# ==============================
# Gerçek dünya haritasında çalışmak için Bursa, İznik şehrini seçiyorum. Küçük bir alan seçmek,osmnx sorgusunu hızlandırdı.
# Buradan 10 rastgele noktanın belirlenmesi ve görselleştirilmesini yaptım.
# Üretilen görsel graphs klasörüne eklendi. (b-1.png)
# ==============================

import osmnx as ox
import random
import matplotlib.pyplot as plt

# 1. Bursa, İznik şehrinin yollar ağını indirir
G = ox.graph_from_place("İznik, Bursa, Turkey", network_type="drive")

# 2. Grafikteki düğümleri listeler
nodes = list(G.nodes)

# 3. Rastgele 10 düğüm seçme
random.seed(42)
selected_nodes = random.sample(nodes, 10)

# 4. Koordinatları alma
coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in selected_nodes]

# 5. Haritayı çizme
fig, ax = ox.plot_graph(G, node_size=0, edge_color='lightgray', show=False, close=False, figsize=(8, 8))

# 6. Seçilen noktaları kırmızı olarak gösterme
lons = [G.nodes[n]['x'] for n in selected_nodes]
lats = [G.nodes[n]['y'] for n in selected_nodes]
ax.scatter(lons, lats, c='red', s=40, label='Seçilen Noktalar')

# 7. Grafiği oluştur
plt.legend()
plt.title("Bursa İznik Şehri - Rastgele Seçilen 10 Nokta", fontsize=12)
plt.tight_layout()
plt.show()
