# PINNs-Infused Nanobot Path Planning for In-Silico Cardiovascular Disease Detection

## Abstract
This project presents a Physics-Informed Neural Network (PINN)–based framework
for intelligent nanobot path planning in a simulated cardiovascular environment.
By embedding the incompressible Navier–Stokes equations into the learning process,
the system ensures physically consistent blood flow prediction and enables
autonomous nanobot navigation for identifying disease-specific hemodynamic
abnormalities in an in-silico setting.

## Project Scope
- Fully simulation-based (in-silico)
- No real patient or clinical data
- Focus on physics-consistent modeling and intelligent navigation
- Intended for academic and research purposes

## Methodology
1. Physics-Informed Neural Network (PINN) formulation using Navier–Stokes equations  
2. Hemodynamic field prediction (velocity and pressure)  
3. Nanobot modeling as a virtual agent  
4. Path planning using an optimization-based strategy  
5. Identification of abnormal flow regions during navigation  

## Repository Structure
- `PINNS_MODEL.py` – PINN architecture and physics-based loss formulation  
- `Path_Planner.py` – Nanobot path planning and optimization logic  
- `Train.py` – Training and execution script  

## Technologies Used
- Programming Language: Python  
- Deep Learning Framework: PyTorch  
- Scientific Computing: NumPy, SciPy  
- Visualization: Matplotlib  

## How to Run
```bash
pip install torch numpy scipy matplotlib
python Train.py
