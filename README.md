# PINNs-Infused PLGA-Based Nanobot System for In-Silico Cardiovascular Disease Detection

## Overview

This project presents a simulation-based framework for early detection of cardiovascular diseases using intelligent nanobots operating in a virtual bloodstream environment.

The system integrates:

* Physics-Informed Neural Networks (PINNs) for blood flow modeling
* Simulated Annealing for nanobot path planning
* PLGA-based biodegradable nanobot design with biosensors

The goal is to analyze nanobot movement and sensor responses to identify abnormalities such as stenosis, atherosclerosis, aneurysm, and turbulent blood flow.


## Introduction

Cardiovascular diseases (CVDs) are among the leading causes of death worldwide. Conventional diagnostic techniques such as angiography, CT, and MRI are often invasive, expensive, and limited in early detection.

This project proposes a **PLGA-based biodegradable nanobot system** capable of navigating through blood vessels, monitoring physiological conditions, and assisting in early disease detection using a simulation-driven approach. 


## Problem Statement

Existing diagnostic systems:

* Are invasive and costly
* Lack continuous real-time monitoring
* Fail to detect early-stage abnormalities effectively

There is a need for an intelligent system that can:

* Navigate inside blood vessels
* Monitor vascular conditions continuously
* Detect abnormalities at an early stage


## Objectives

* To design PLGA-based biodegradable nanobots for safe vascular navigation
* To model blood flow using Physics-Informed Neural Networks (PINNs)
* To integrate biosensors for detecting biochemical and mechanical changes
* To predict nanobot paths using Simulated Annealing
* To detect cardiovascular diseases in a simulation environment


## Nanobot Model

### Nanobot Description

The nanobot is modeled as a biodegradable micro-device composed of **Poly(lactic-co-glycolic acid) (PLGA)**, ensuring biocompatibility and safe degradation.

### Sensor-Integrated Design

The nanobot is equipped with biosensors to monitor:

* Blood flow velocity and turbulence
* Pressure and vessel wall stress
* Biochemical markers such as cholesterol, calcium ions, and inflammation indicators

### Ca²⁺ Ion Release Mechanism (Conceptual)

The nanobot is assumed to release **calcium ions (Ca²⁺)** to simulate biochemical interaction and assist in identifying plaque formation and calcification.

### Biodegradation

* Operates for a limited duration
* Gradually degrades into non-toxic byproducts
* Eliminates the need for removal


## Biosensors

The system incorporates multiple biosensors (conceptual model):

* Electrochemical sensors – detect biomarkers (CRP, cholesterol)
* Calcium/lipid sensors – detect plaque buildup
* Enzyme-based sensors – monitor tissue remodeling
* MEMS pressure sensors – measure vessel stiffness and stress
* Flow sensors – detect velocity, shear stress, turbulence
* Optical/fluorescent sensors – visualize molecular activity

These sensors enable comprehensive monitoring of vascular health. 

## Methodology and Implementation

### 1. Blood Flow Modeling using PINNs

* Models blood flow using physical laws
* Predicts velocity and pressure fields
* Ensures realistic hemodynamic simulation

### 2. Nanobot Navigation

* Nanobot moves passively with blood flow
* Influenced by velocity, vessel geometry, and disturbances
* Collects sensor data during movement

### 3. Path Planning using Simulated Annealing

* Optimizes nanobot trajectory
* Explores complex vascular regions
* Identifies abnormal flow zones

### 4. Disease Detection

Based on flow patterns and sensor data, the system detects:

* Atherosclerosis (plaque buildup)
* Stenosis (vessel narrowing)
* Aneurysm (vessel dilation)
* Turbulent flow

## Results

The simulation provides:

* Blood flow visualization
* Nanobot trajectory mapping
* Detection of abnormal regions
* Optimization convergence

These results demonstrate the effectiveness of combining PINNs, nanobot modeling, and optimization techniques. 

## Repository Structure

* `PINNS_MODEL.py` – Blood flow modeling using PINNs
* `Path_Planner.py` – Nanobot trajectory optimization
* `Train.py` – Simulation execution

## Technologies Used

* Python
* PyTorch
* NumPy
* SciPy
* Matplotlib

## Dependencies

* Python 3.x
* torch
* numpy
* scipy
* matplotlib

## How to Run

```bash id="final1"
pip install torch numpy scipy matplotlib
python Train.py
```

## Project Scope

* Fully simulation-based system
* No real clinical implementation
* Focus on PLGA nanobot + biosensor modeling + AI-based flow analysis
* Intended for research and academic purposes

## Current Work and Future Hardware Development

### Current Work

The current project is implemented as a **simulation-based system**, where:

* Blood flow is modeled using PINNs
* Nanobot movement is simulated
* Path optimization is performed using Simulated Annealing
* Biosensors are modeled conceptually


### Future Hardware Implementation

As a future extension, the project aims to develop a **real-world nanobot system**, focusing on:

* PLGA-based biodegradable nanobots for safe in-body operation

* Integration of miniaturized biosensors to detect:

  * Blood flow
  * Pressure
  * Biochemical markers

* Controlled Ca²⁺ ion release mechanisms

* Real-time monitoring and data transmission


### Long-Term Vision

* Autonomous nanobots navigating inside blood vessels
* Continuous cardiovascular monitoring
* Early and minimally invasive disease detection systems

⚠️ Hardware implementation is part of future research and not included in the current system.

## Conclusion

This project presents an integrated framework combining **PLGA-based nanobots, biosensors, PINNs, and optimization techniques** for early cardiovascular disease detection.

The system demonstrates:

* Intelligent navigation inside blood vessels
* Detection of abnormalities using flow and sensor data
* A scalable simulation framework for future healthcare technologies


## Future Work

* Extend to full 3D vascular simulations
* Integrate real medical imaging (MRI/CT)
* Improve biosensor modeling
* Move toward real nanobot implementation

  
## Demo

Nanobot path planning simulation (in-silico):

[![Nanobot Demo](https://img.youtube.com/vi/TKSz1DcToIs/0.jpg)](https://youtu.be/TKSz1DcToIs)

▶️ Direct Link: https://youtu.be/TKSz1DcToIs
