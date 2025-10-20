# Progressive TSP Research  
**Course Project – Travelling Salesman Problem (TSP)**  
**Yazar / Author:** MİNE BÜŞRA HAZER  
**Dil / Language:** Python  

---

## Overall Goal / Genel Amaç  
This project series progressively develops solutions to the **Travelling Salesman Problem (TSP)** through five assignments — starting from simple random graphs and ending with **AI-based TSP with Neighborhoods**.  

Bu proje serisi, **Gezgin Satıcı Problemi (TSP)** için artan karmaşıklıkta çözümler geliştirir.  
Rastgele noktalardan başlanarak, gerçek haritalar ve yapay zekâ yöntemleriyle son aşamada **bölgeli TSP** çözümü elde edilir.  

---

## Assignment Summary / Ödev Özeti  

| No | Title | Objective (Amaç) |
|----|--------|------------------|
| **1** | **Basic TSP on Random Points** | Create random points, build a graph (`networkx`), solve with *Nearest Neighbor*, visualize with `matplotlib`. <br> Rastgele noktalar üret, graf oluştur, basit sezgisel yöntem uygula, görselleştir. |
| **2** | **Real Map TSP** | Use `osmnx` to get a real map, select nodes, and solve using the heuristic. <br> Gerçek harita verisiyle aynı yöntemi uygula. |
| **3** | **Comparison with OR-Tools** | Add an advanced solver (e.g., OR-Tools), test 30 instances, compare runtime and distance. <br> Gelişmiş bir çözücü ekle ve karşılaştır. |
| **4** | **AI-Based Solver** | Integrate an AI method (e.g., Genetic Algorithm, Reinforcement Learning). <br> Bir yapay zekâ yöntemi uygula. |
| **5** | **TSP with Neighborhoods** | Extend to regional TSP (each city = area). <br> Noktalar yerine bölgelerle çalış. |

---

## Tools & Environment / Kullanılan Araçlar  

| Category | Libraries |
|-----------|------------|
| Graphs & Visualization | `networkx`, `matplotlib`, `folium` |
| Maps | `osmnx`, `leaflet` |
| Optimization | `Google OR-Tools`, `numpy`, `pandas` |
| AI / ML | `scikit-learn`, `tensorflow` |
| Reports | `matplotlib`, `seaborn`, `pandas` |

---

## Learning Outcomes / Öğrenme Kazanımları  
- Understand **graph and spatial modeling** / Grafik ve mekansal modelleme öğrenimi  
- Apply **heuristic, exact, and AI-based solvers** / Sezgisel, kesin ve yapay zekâ tabanlı çözücüler  
- Perform **reproducible experiments** / Tekrarlanabilir deneyler  
- Visualize and compare algorithmic performance / Algoritma performanslarını karşılaştırma  

---

> **From random points ➜ to real maps ➜ to AI-powered optimization**  
> **Rastgele noktalardan ➜ gerçek haritalara ➜ yapay zekâ destekli optimizasyona geçiş**
