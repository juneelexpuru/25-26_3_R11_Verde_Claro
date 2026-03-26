# 25-26_3_R11_Verde_Claro
# Reto 11: Digital Transformation in Electric Motor Manufacturing

## 1. Project Overview
[cite_start]This project focuses on the digital transformation of electric motor manufacturing and maintenance, aligning with **Industry 4.0** and **Industry 5.0** principles[cite: 3, 26]. [cite_start]The goal is to optimize motor design, simulate production lines, and implement advanced predictive maintenance systems through mechatronics and data science[cite: 4, 304].

### Core Objectives
* [cite_start]**Design Optimization:** Propose the 5 best motor designs using Multi-Objective Evolutionary Algorithms (MOEAs) based on 6 geometric parameters[cite: 247, 318].
* [cite_start]**Manufacturing Simulation:** Model assembly lines and manage stocks (copper/aluminum) using **SymPy** to identify bottlenecks[cite: 23, 24].
* [cite_start]**Intelligent Control:** Use **Reinforcement Learning (RL)** to minimize active power consumption in electric drives[cite: 9, 250].
* [cite_start]**Predictive Maintenance:** Identify bearing anomalies (ball, cage, and race faults) using **Signal Analysis** (FFT/DSP)[cite: 18, 252].

## 2. Repository Structure
[cite_start]Following the project requirements, the code is divided into functional units[cite: 263]:

```text
├── 01_Preprocessing/
│   └── signal_preprocessing.ipynb    # Detrending, Hamming Windowing, and Quality Control
├── 02_Feature_Extraction/
│   └── extraction_pipeline.ipynb     # Time and Frequency domain feature extraction
├── 03_Modeling/
│   ├── design_optimization_moo.ipynb # MOEA implementation (NSGA-II, NSGA-III)
│   ├── rl_power_optimization.ipynb  # Q-Learning agent for active power stabilization
│   └── fault_diagnosis_model.ipynb   # Hybrid classification and severity regression
├── 04_Visualization/
│   └── dashboard_logic.py            # Logic for Node-RED and Looker Studio integration
└── README.md                         # Project documentation (this file)
