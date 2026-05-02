# ADAR Framework Reproduction 
**University of Vaasa - Advanced Artificial Intelligence**

## Project Overview
This repository contains the rigorous reproduction and validation of the Adaptive Dynamic Attribute and Rule (ADAR) framework for Fuzzy Inference Systems, based on the foundational research [cite: 1]. 

The primary objective of this repository is to demonstrate the viability of dual-weighting mechanisms in managing high-dimensional datasets (the "Curse of Dimensionality"). The implementation specifically tracks structural interpretability via the Average Overlap Index ($I_{ov}$) and Average Fuzzy Set Position Index ($I_{fsp}$).

## Team Roles
- **Daniel Ebrahimzadeh:** Project Lead & Framework Architecture
- **Haleh Adab:** Data Engineering & Normalization logic
- **Amin Sarlak:** Pruning/Growth Algorithm & Optimization
- **Muhammad Abuzar:** Validation & Critical Analysis

## Repository Contents
- `adar_anfis.py`: The core implementation (PyTorch). Contains the full algorithm including K-Means initialization, the `ADARLayer` (Dual Weighting), and the `StructureManager` for structural plasticity. It explicitly executes an ablation study on a 27-feature dataset.
- `Final_Report.docx`: The official academic submission formatted via the University's guidelines, detailing mathematical proofs, novelties, and a critical data engineering analysis.
- `presentation.html`: A standalone interactive HTML/JS slide deck designed for the project defense.
- `hero_chart.png`: Visual evidence of dynamic rule generation intersecting with validation loss.
- `/paper`: Directory containing the original research PDF and instructions.

## Execution Instructions
**Dependencies:**
This framework requires Python 3.10+ and standard numerical/deep learning libraries:
```bash
pip install torch scikit-learn numpy pandas matplotlib
```

**Running the End-to-End Validation:**
```bash
python adar_anfis.py
```
This command will:
1. Synthesize a 27-variable high-dimensional space (equivalent to Appliances Energy dataset).
2. Execute the Baseline ANFIS (Static Structure).
3. Execute the ADAR-ANFIS (Dynamic Structure).
4. Output the definitive RMSE, $I_{ov}$, and $I_{fsp}$ indices into the console.
5. Export the latest `hero_chart.png`.

## References
[1] Ke Liu, Jing Ma, and Edmund M-K Lai. "A Dynamic Fuzzy Rule and Attribute Management framework for Fuzzy Inference Systems in High-Dimensional Data."
