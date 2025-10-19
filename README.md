# 🧭 Homework Series: Progressive TSP Research  
**Course Project – Travelling Salesman Problem (TSP)** 
 
**Author: MİNE BÜŞRA HAZER**

**Language: Python** 

---

## 🎯 Overall Goal

This project series progressively develops solutions to the **Travelling Salesman Problem (TSP)** across five assignments.  
Starting with random synthetic data, we gradually move toward real-world maps, AI-driven solvers, and finally the **TSP with Neighborhoods** (continuous-region optimization).

Each step introduces new levels of abstraction, algorithmic complexity, and visualization methods — ultimately combining **heuristics, exact solvers, and AI techniques** for performance comparison.

---

## 🧩 Assignment Overview

### **Assignment 1 – Basic TSP on Random Points**
**Objective:**  
Practice graph abstraction, random instance generation, simple heuristics, and visualization.

**Tasks:**  
- Generate random 2D points and represent them as a graph (`networkx`).  
- Implement one simple TSP heuristic (e.g., *Nearest Neighbor* or *Greedy Insertion*).  
- Visualize the tour using `matplotlib`.  

**Deliverables:**  
- Python script  
- Graph screenshot  
- Short explanation of the heuristic 

---

### **Assignment 2 – Real Map TSP**
**Objective:**  
Move from synthetic data to real-world geography.

**Tasks:**  
- Use `osmnx` to download a real street network (e.g., a city).  
- Choose multiple locations as TSP nodes.  
- Solve using your heuristic from Assignment 1.  
- Visualize the route using `folium` or `leaflet`.  

**Deliverables:**  
- Python script + map visualization (HTML or image)  

---

### **Assignment 3 – Second Approach & Scientific Comparison**
**Objective:**  
Introduce a more advanced TSP solver and compare it with your heuristic using **research-style methodology**.

**Tasks:**  
- Add a second solver (e.g., **Google OR-Tools**).  
- Generate 30 random topologies with fixed random seeds.  
- Compare both methods in terms of **average tour length** and **runtime**.  
- Visualize results in plots and tables.  

**Deliverables:**  
- Scripts  
- Performance plots  
- Short (1–2 page) comparison report  

---

### **Assignment 4 – AI Technique Integration**
**Objective:**  
Integrate an AI technique into the TSP pipeline.

**Tasks:**  
- Choose an AI-based approach (e.g., **Genetic Algorithm**, **Fuzzy Logic**, **Neural Network**, or **Reinforcement Learning**).  
- Implement it as a third solver.  
- Compare against previous two methods using the same dataset protocol (30 instances).  
- Visualize performance and analyze results.  

**Deliverables:**  
- Python scripts  
- Performance charts  
- Short analytical report  

---

### **Assignment 5 – TSP with Neighborhoods**
**Objective:**  
Extend the problem to *continuous regions* instead of fixed nodes.

**Tasks:**  
- Define each “city” as a region (circle or polygon).  
- Implement a solver that selects one optimal point per region to minimize total tour length.  
- Apply all three approaches (heuristic, OR-Tools, AI).  
- Visualize tours on a map using `folium`.  
- Compare and discuss the final results.  

**Deliverables:**  
- Full scripts  
- Map visualizations  
- Final comparative report (2–3 pages)  

---

## ⚙️ Tools & Environment

| Category | Libraries / Tools |
|-----------|------------------|
| **Graphs** | `networkx`, `matplotlib` |
| **Real Maps** | `osmnx`, `folium`, `leaflet` |
| **Optimization** | `Google OR-Tools`, `numpy`, `pandas` |
| **AI / ML** | `scikit-learn`, `tensorflow` or custom implementation |
| **Visualization** | `matplotlib`, `seaborn`, `folium` |

---

## 📊 Expected Learning Outcomes

- Understand **graph representations** and **spatial data modeling**.  
- Implement **heuristic**, **exact**, and **AI-based** optimization methods.  
- Learn **experiment reproducibility** and **statistical comparison techniques**.  
- Gain experience in **map-based visualization** and **research-style reporting**.  

---