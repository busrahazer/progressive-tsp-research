# ==============================
# Burada öncelikle sabit  bir seed  değeri ile rastgele dizi üretiyorum ve bunları grafikleştiriyorum.
# graphs klasörüne bu koddan çalışan grafiğin ekran görüntüsünü ekledim.
# ==============================

import random
import matplotlib.pyplot as plt

# 1. Parametreler 
NUM_POINTS = 5        # 5 nokta üretilecek
AREA_SIZE = 50        # Noktaların bulunduğu alanın boyutu
SEED = 42             # Her zaman aynı rastgele dizi üretilmesi için seed değeri 

# 2. Rastgelelik ayarı 
random.seed(SEED)

# 3. Rastgele noktalar oluştur
points = []  # Noktaları (x, y) şeklinde saklıyorum
for i in range(NUM_POINTS):
    x = random.uniform(0, AREA_SIZE)
    y = random.uniform(0, AREA_SIZE)
    points.append((x, y))

# 4. Görselleştirme
x_coords = [p[0] for p in points]
y_coords = [p[1] for p in points]



plt.figure(figsize=(5, 5))
plt.scatter(x_coords, y_coords, color='pink', s=50)

# Takip etmek için her noktaya numara verdim
for i, (x, y) in enumerate(points):
    plt.text(x + 1, y + 1, str(i), fontsize=12, color='lightpink')

plt.title("Rastgele Noktalar")
plt.xlabel("X koordinatı")
plt.ylabel("Y koordinatı")
plt.grid(True)
plt.xlim(0, AREA_SIZE)
plt.ylim(0, AREA_SIZE)
plt.show()
