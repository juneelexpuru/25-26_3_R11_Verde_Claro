# Reto 11: Digital Transformation in Electric Motor Manufacturing

## 1. Project Overview
This project focuses on the digital transformation of electric motor manufacturing and maintenance, aligning with **Industry 4.0** and **Industry 5.0** principles. The goal is to optimize motor design, simulate production lines, and implement advanced predictive maintenance systems through mechatronics and data science.

### Core Objectives
* **Design Optimization:** Propose the 5 best motor designs using Multi-Objective Evolutionary Algorithms (MOEAs) based on 6 geometric parameters.
* **Manufacturing Simulation:** Model assembly lines and manage stocks (copper/aluminum) using **SymPy** to identify bottlenecks.
* **Intelligent Control:** Use **Reinforcement Learning (RL)** to minimize active power consumption in electric drives.
* **Predictive Maintenance:** Identify bearing anomalies (ball, cage, and race faults) using **Signal Analysis** (FFT/DSP).

## 2. Repository Structure
Following the project requirements, the code is divided into functional units:

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
