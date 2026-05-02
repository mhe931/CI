# ADAR Framework Reproduction

This repository contains the reproduction of the ADAR (Adaptive Dynamic Attribute and Rule) framework for Fuzzy Inference Systems as part of the Advanced Artificial Intelligence course at the University of Vaasa.

## Contents
- `adar_anfis.py`: The main PyTorch implementation of the ADAR framework. Includes data synthesis, K-Means initialization, the ADARLayer with dual weighting/pruning, and the StructureManager for dynamic rule scaling.
- `report.docx`: The final academic report detailing the methodology, results, and critical analysis, formatted with the official University template.
- `plot.py`: Generates the loss curve graph for the report.
- `loss_curve.png`: The output chart showing training vs. validation loss.

## Execution Instructions
Ensure you have Python installed and the required dependencies:

```bash
pip install torch scikit-learn numpy pandas matplotlib
```

To run the framework:
```bash
python adar_anfis.py
```
This will automatically generate a synthetic PM2.5 dataset, train the ADAR model, dynamically adjust the rules, and output the final RMSE, Overlap Index, and Position Index metrics. 

To recreate the loss curve chart:
```bash
python plot.py
```
