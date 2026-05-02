# ADAR Framework Reproduction 
**University of Vaasa - Computational Intelligence (ICAT3360)**

## Project Overview
This repository contains our reproduction and validation of the Adaptive Dynamic Attribute and Rule (ADAR) framework for Fuzzy Inference Systems, based on the foundational research [cite: 1]. 

Our implementation explores the viability of dual-weighting mechanisms in managing high-dimensional datasets (the "Curse of Dimensionality"). We explicitly tracked structural interpretability via the Average Overlap Index ($I_{ov}$) and Average Fuzzy Set Position Index ($I_{fsp}$).

## Team Roles
- **Daniel Ebrahimzadeh:** Architecture lead and version control.
- **Haleh Adab:** Preprocessing logic and dataset scaling.
- **Amin Sarlak:** Pruning/Growing algorithm and hyperparameter tuning.
- **Muhammad Abuzar:** Position Index ($I_{fsp}$) analysis and critical review.

## Quick Start
**1. Dependencies:**
Our framework requires Python 3.10+ and standard numerical/deep learning libraries. You can install them by running:
```bash
pip install torch scikit-learn numpy pandas matplotlib
```

**2. Running the Full Pipeline:**
To execute the end-to-end ablation validation, simply run:
```bash
python adar_anfis.py
```
This command will:
1. Synthesize a 27-variable high-dimensional space (equivalent to the Appliances Energy dataset).
2. Execute **Trial A:** Baseline ANFIS (Static Structure, growing/pruning disabled).
3. Execute **Trial B:** Full ADAR-ANFIS Framework (Dual weighting and structure scaling enabled).
4. Export the resulting metrics automatically into `comparison_results.json`.
5. Export the convergence and comparison visualization to `hero_chart.png`.

## Hero Chart Description
The generated `hero_chart.png` contains two critical subplots:
*   **Left Subplot (Rule Growth & Loss):** Illustrates the validation and training loss curves over epochs, superimposed with a step-function curve demonstrating precisely when the framework spawned new fuzzy rules to combat validation stagnation.
*   **Right Subplot (RMSE Comparison):** A bar chart contrasting the final Root Mean Square Error of our ADAR framework against the static Baseline ANFIS.

## Repository Contents
- `adar_anfis.py`: The core implementation (PyTorch). Contains the full algorithm including K-Means initialization, the `ADARLayer` (Dual Weighting), and the `StructureManager` for structural plasticity. 
- `Final_Report.md`: Our official academic submission formatted via the University's guidelines, detailing mathematical proofs and critical engineering analysis.
- `comparison_results.json`: A generated JSON file containing the explicit RMSE, $I_{ov}$, $I_{fsp}$, and rule count metrics.
- `presentation.html`: A sleek standalone Reveal.js interactive slide deck designed for the project defense.
- `/paper`: Directory containing the original research PDF and instructions.

## References
[1] Ke Liu, Jing Ma, and Edmund M-K Lai. "A Dynamic Fuzzy Rule and Attribute Management framework for Fuzzy Inference Systems in High-Dimensional Data."
