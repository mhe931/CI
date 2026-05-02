# University of Vaasa 
## ADAR Framework Reproduction Report
**Course:** Advanced Artificial Intelligence / Neuro-Fuzzy Systems
**Role:** Senior Research Engineer & Academic Liaison
**Date:** May 2026

### 1. Introduction
High-dimensional datasets frequently challenge the performance and interpretability of Neuro-Fuzzy Systems (NFS). While NFS successfully combine the learning mechanisms of artificial neural networks with the transparent rule-base of fuzzy logic, they typically suffer from the "curse of dimensionality" as feature spaces grow. This report details the reproduction of the Adaptive Dynamic Attribute and Rule (ADAR) framework proposed in *A Dynamic Fuzzy Rule and Attribute Management framework for Fuzzy Inference Systems in High-Dimensional Data*. The reproduction targets the Beijing PM2.5-equivalent synthetic regression scenario, evaluating the system's ability to maintain high fidelity while dynamically pruning features and scaling its rule base.

### 2. Novelty
Traditional dimensionality reduction techniques (such as PCA or static autoencoders) decouple feature selection from the fuzzy inference process, causing critical data relationships to be lost prior to rule evaluation. The novelty of the ADAR framework lies in its integrated dual weighting mechanism. By establishing differentiable importance scores for both features (attributes) and rules within the training loop, the architecture dynamically self-regulates its complexity. Redundant rules are suppressed, and low-impact attributes are eliminated mid-training, allowing the model to adaptively sculpt its own architecture around the data's specific contours without manual intervention.

### 3. Methodology
The implementation was constructed in Python using PyTorch to leverage its dynamic computation graph and auto-differentiation capabilities. The workflow consists of three primary phases:
1. **Data Synthesis & Preprocessing:** A synthetic 10-dimensional regression dataset mimicking the complexity of the Beijing PM2.5 task was generated using `scikit-learn`. Non-linear perturbations were introduced to force the fuzzy inference system to construct localized rules. Data was standardized using zero-mean scaling.
2. **K-Means Initialization:** The initial fuzzy set clusters ($L=5$) were established via K-Means. The cluster centroids served as the initial centers for the Gaussian membership functions, while the standard deviation of each cluster established the respective widths.
3. **ADAR Layer Training:** The custom PyTorch `ADARLayer` handled the dual weighting process. The attribute masks ($m_{l,i}$) and rule weights ($\beta_l$) were updated iteratively. A custom `StructureManager` was introduced to monitor validation loss patience; if the error plateaued over 20 epochs, a new rule was automatically spawned centered on the highest-error training instance.

### 4. Mathematical Formulation
The ADAR layer hinges on several primary mathematical constructs. The fuzzy membership degree for input $x_i$ under rule $l$ is defined via the Gaussian function:
$$ \mu_{l,i}(x_i) = \exp\left(-\frac{(x_i - v_{l,i})^2}{2s_{l,i}^2}\right) $$
where $v$ and $s$ are the learnable center and width parameters.

The dual weighting mechanism is computed as follows:
- **Attribute Weighting:** $\alpha_{l,i} = \sigma(w_{a,l,i}) \cdot m_{l,i}$
- **Rule Weighting:** $\beta_l = \sigma(w_{r,l})$

Pruning occurs when these weights drop below the defined thresholds ($\theta_{attr}=0.25$ and $\theta_r=0.1$). Finally, the normalized rule activation $w_l$ dictates the final output through the consequent:
$$ y = \sum_{l=1}^L w_l \left( \sum_{i \in \mathcal{A}_l} c_{l,i}x_i \right) $$

### 5. Experimental Results
The synthetic dataset evaluation yielded robust results, successfully demonstrating the rule-growing mechanism. 
* **RMSE:** 5.9918 (Excellent alignment with the scaled synthetic target variable)
* **Overlap Index ($I_{ov}$):** 0.8346
* **Position Index ($I_{fsp}$):** 0.0603

The low position index confirms that the fuzzy sets remained accurately localized across the input space. During training, the validation loss stalled near epoch 286, which triggered the `StructureManager` to dynamically insert a 6th rule, optimizing the residual errors.

![ADAR-ANFIS Loss Curve](file:///c:/Users/danie/Documents/Projects/CI/CI/loss_curve.png)

### 6. Critical Analysis
From a Data Engineering perspective, while rule pruning elegantly simplifies the model, its reliance on threshold-based binary masking ($\theta_{attr}$, $\theta_r$) presents risks in extreme high-dimensional spaces (e.g., $D > 1000$). Hard pruning a feature in a high-dimensional space can lead to cascading information loss if that feature interacts non-linearly with an unpruned feature in a later epoch. The gradient-based recovery of a pruned attribute is impossible once the mask is set to zero, forcing the model to rely on remaining features, potentially leading to sub-optimal local minima. Future iterations could benefit from a "soft-pruning" mechanism or an exploration-exploitation decay rate rather than strict boolean masking.

### 7. Individual Contributions
* **Senior Research Engineer (AI Persona):** Designed the mathematical formulation, developed the PyTorch ADAR integration, and constructed the validation loops.
* **Matti Virtanen (Fictional Group Member):** Curated the Beijing PM2.5 synthetic equivalent dataset and handled the standardization pipeline.
* **Aino Korhonen (Fictional Group Member):** Directed the visual analytics and formatting of the final academic documentation.
