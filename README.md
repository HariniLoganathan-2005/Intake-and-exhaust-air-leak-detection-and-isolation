# 🔧 Caterpillar C7 — Air Path Leak Detection System

An industrial-grade Digital Twin and Anomaly Detection pipeline built to detect and isolate air and exhaust leaks in heavy-duty diesel engines during test cell development. 

This system acts as a hybrid intelligence layer, fusing **First-Principles Thermodynamics (Physics Engine)** with **Machine Learning (Autoencoder)** to provide steady-state leak detection without requiring additional physical hardware.

---

## 📖 Problem Statement
During diesel engine development, air and exhaust leaks (between the airflow meter, turbocharger, and aftertreatment systems) cause significant test cell downtime, performance degradation, and hardware risks. 

This software solves this by:
1. Detecting the presence of leaks in real-time.
2. Localizing the leak to specific component groups (Zones A, B, and C).
3. Providing actionable mechanic instructions and a Go/No-Go confidence score.

---

## 🏗️ System Architecture

The codebase is split into 6 decoupled layers. Data flows sequentially from the simulator down to the fusion output.

1. **`simulator.py` (The Digital Twin)**
   - Simulates a 7.2L Caterpillar engine at 10 Hz (100ms steps).
   - Dynamically calculates raw sensor values (MAF, MAP, EBP, Temps) based on the current RPM and Load %.
   - Allows users to inject mathematical "leaks" to simulate broken pipes or clogged filters.

2. **`pipeline.py` (Data Cleaning & Validation)**
   - Ingests raw data and applies a rolling Median Filter to kill sensor noise.
   - **Crucial Feature:** Implements a 10-second Steady-State Detector. It calculates the standard deviation of RPM/Load and pauses detection during transient events (like gear shifts or turbo lag) to prevent false alarms.

3. **`physics_engine.py` (Thermodynamic Rules)**
   - Uses Volumetric Efficiency tables, the Ideal Gas Law, and raw engine displacement to calculate what the sensors *should* be reading.
   - Compares Expected vs. Actual readings to generate a positive or negative `residual_pct`.

4. **`ml_engine.py` (Machine Learning Autoencoder)**
   - A Scikit-Learn `MLPRegressor` trained to reconstruct 7 highly-correlated, dimensionless sensor ratios (e.g., `maf_per_rpm`, `ebp_per_fuel`).
   - Trained across a massive "Grid" of varying RPMs and Loads to understand the full physical envelope of the engine. 
   - If a ratio suddenly breaks physical laws, the reconstruction error spikes, flagging an anomaly.

5. **`fusion.py` (The Brains)**
   - Weights the final decision: **60% Physics Score | 40% ML Score**.
   - Translates raw math into human-readable mechanic actions mapped to specific sub-locations.
   - Handles edge cases (e.g., If the engine is in DPF Regen, the Physics Engine forcefully suppresses the ML engine to prevent false positives from temperature spikes).

6. **`dashboard.py` (Streamlit GUI)**
   - A multi-threaded, real-time frontend. 
   - Runs the engine simulation loop in the background while updating the UI twice a second via a thread-locked JSON state dictionary.

---

## 📍 Leak Detection Zones

The system monitors three primary engine zones:
* **Zone A (Pre-Turbo):** Air filter to Turbocharger inlet. Detected via Mass Air Flow (MAF) drop.
* **Zone B (Post-Turbo / Charge-Air):** Turbo outlet through the Intercooler to Intake Ports. Detected via Manifold Absolute Pressure (MAP) drop.
* **Zone C (Exhaust Passages):** Exhaust manifold through Turbine, DOC, DPF, and SCR tailpipe. Detected via Exhaust Back Pressure (EBP) and Thermal-Cascade temperature drops.

---

## 🚀 How to Run the Project

### 1. Install Dependencies
Ensure you have Python 3.9+ installed. Run the following command to install the required libraries:
```bash
pip install numpy pandas scikit-learn streamlit
```
2. Train the Machine Learning Model (First Run Only)
Before running the dashboard, the ML Autoencoder must learn what a "healthy" engine looks like.
Run the training script. This will simulate a healthy engine across 30 different RPM/Load combinations and save the model weights to the /models/ folder.

```Bash
python ml_engine.py --train
```
Wait for the terminal to print: [ML] Training complete.

3. Launch the Dashboard
Start the real-time Streamlit interface:

```Bash
python -m streamlit run dashboard.py --server.port 8501
```
This will open the interactive UI in your web browser. You can now adjust the RPM/Load sliders, inject leaks, and watch the Hybrid Fusion system detect and isolate the faults in real-time.

