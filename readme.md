# NYC Taxi & Traffic Flow Network Analysis

This Python project processes New York City transportation data (taxi trips, census data, and for-hire vehicle trips) to build a zone-based network model, estimate parameters, and simulate traffic flow scenarios—particularly focusing on flows related to Manhattan.

---

## 📌 Project Workflow


The main processing pipeline is composed of sequential steps:


## Step 0: Prepare Raw Data

**Dataset Name**: NYC Taxi Zones  
**Source**: [NYC Open Data - Taxi Zones]
**Description**:  
This dataset provides the official taxi service zone boundaries (polygon shapefile) for New York City, along with a unique `LocationID` (Zone ID) for each zone.

**Dataset Name**: NYC Census Blocks  
**Source**: [U.S. Census Bureau - TIGER/Line Shapefiles]
**Description**:  
This dataset provides the geographic boundaries for Census Blocks in New York City. Each block is uniquely identified and can be used to aggregate population, employment, and other demographic data.

**Dataset Name**: S0801 – Commuting/Workplace Data (ACS 5-Year Estimates)  
**Source**: [U.S. Census Bureau API], API key is needed here.
        Visit [Census API Key Registration](https://api.census.gov/data/key_signup.html)

**Dataset Name**: 出租车运营数据

**Dataset Name**: od数据

**Dataset Name**: free travel time data
**Source**: google map API is needed.
Visit [Google Cloud Console - Get API Key](https://console.cloud.google.com/apis/credentials) 

### **Step 1 – Merge Taxi Zones**
Merge the original 200+ NYC taxi zones into **130 aggregated zones**.

**Outputs:**
- Shapefile: `shapfile/merged_taxi_zones.shp`
- Map: `merged_taxi_zones.pdf`
- Zone mapping: `merged_taxi_zones.pkl` / `merged_taxi_zones.csv`

---

### **Step 2 – Download Census Data**
Download data from **S0801** and store it into `GreYelHir.db`, table `SensusData_NewYork`.

---

### **Step 3 – Taxi & For-Hire Vehicles Data Cleaning**
Download **2022 monthly data** for:
- Green Taxi
- Yellow Taxi
- For-Hire Vehicles

**Data cleaning:**
- Keep only complete rows.
- Keep only rows where pickup & dropoff times are **between 6:00 AM and 10:00 AM**.

**Outputs:**
- `taxi` dataset (Green + Yellow)
- `for_hire_vehicles` dataset

---

### **Step 4 – Map Census Blocks to Merged Zones**
Overlay merged taxi zone shapefile with census block shapefile to **map census blocks to merged zones**.

**Output:**  
`grouped_taxi_zones.pkl`

---

### **Step 5 – Map Taxi Data to New Zones**
Map taxi trips (originally between **200+ zones**) to the **new 130-zone dataset**.

---

### **Step 6 – Find Adjacent Zones**
Find adjacency relationships — zones that are **geographically connected** and have a path.

**Output:**  
`adjacent_districts` dictionary

---

### **Step 7 – Map OD Data to New Zones**
Map original **Census Block OD data** to the **new zones**.

**Outputs:**
- `ZoneOutNodes` – destinations from each zone
- `ZoneInNodes` – origins to each zone

---

### **Step 8 – Map Census S0801 Data to New Zones**
Create `ZoneSensusData_NewYork` table containing **S0801 data** under **new zone definitions**.

---

### **Step 9 – Generate Annual Network Graphs**
Generate the **final network graph** for each year.  
Uses `Graph_NewYork_2010.pkl` if available for faster processing.

---

### **Step 10 – Calculate Road Capacities**
Calculate **traffic capacity** of each road per year (**2022 as example**) and update the network graph.

---

### **Step 11 – Parameter Estimation Preparation**
- **Step 11_0** – Prepare data for parameter estimation.
- **Step 11_1** – Estimate parameters for the entire NYC network.

---

### **Step 12 – Export Network Graphs as PDF**
Generate **undirected**, **directed**, and **road graphs** from saved `.pkl` files.

---

### **Step 13–14 – Manhattan-Only Network Processing**
- Extract Manhattan-related traffic flows.
- Adjust population so it only relates to Manhattan traffic.
- Prepare simulation data.
- Draw Manhattan-only traffic network maps.

---

### **Step 15–18 – Simulation & Congestion Pricing Analysis**
- **Step 15** – Prepare iteration data.
- **Step 16** – Simulate Manhattan-only traffic flows.
- **Step 17** – Simulate without congestion pricing.
- **Step 18** – Analyze scenarios with different congestion pricing levels.

---

## ⚙️ Requirements

- Python 3.8+
- Pandas, NumPy, GeoPandas, Matplotlib, NetworkX
- SQLite (via `sqlite3` or custom Database class)

---

## 🚀 How to Run

```bash
python main.py
