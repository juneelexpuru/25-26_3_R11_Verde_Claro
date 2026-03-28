# Reto 11: Digital Transformation in Electric Motor Manufacturing

## 👥 Equipo Verde Claro:
* **Libe Arana Carrascal**
* **Jon Ayala Lecea**
* **June Elexpuru Domínguez**
* **Markel Jorge Gomez**
* **Vega Lopez De Lapuente**
* **Martin Martinez Orive**


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
├── Datos/
├── Graficos/
├── NodeRed/
├── Scripts/
├── 01_Optimization/
│   └── moea_optimization.ipynb 
│   └── surrogate_model.ipynb    
├── 02_RL/
│   └── agente_RL_V1.ipynb
│   └── agente_RL_V2.ipynb
│   └── env_V1.ipynb
│   └── env_V2.ipynb
├── 03_Validacion/
│   ├── 01_Procesamiento_Senal
│   ├── 02_Extraccion_caracteristicas
│   └── 03_Modelado_Predictivo_Base
│   └── 04_Optimizacion_y_Seleccion_variables
├── 04_Simulacion/
│   └── 04_SimulacionSimpy.py            
└── README.md                         # Project documentation (this file)
