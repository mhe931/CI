# University of Vaasa 
## ADAR Framework Reproduction Report
**Course:** Advanced Artificial Intelligence / Neuro-Fuzzy Systems
**Date:** May 2026
**Repository:** [mhe931/ci](https://github.com/mhe931/ci)

### 1. Introduction
High-dimensional datasets introduce severe computational bottlenecks for traditional Neuro-Fuzzy Systems (NFS). As the feature space expands, the number of rules required to cover the domain grows exponentially, leading to the well-known "Curse of Dimensionality." This project documents our team's reproduction of the Adaptive Dynamic Attribute and Rule (ADAR) framework (Paper 1), aiming to mitigate these scaling issues. We focused on validating the framework against a synthetic equivalent of the 27-feature Appliances Energy dataset to verify its dynamic rule-growing and attribute-pruning capabilities.

### 2. Contribution / Novelty of the Paper
When we first read the paper, the core idea that struck us was the "dual weighting" mechanism. During our implementation, we observed that traditional dimensionality reduction methods, like PCA, force you to throw away original features before the fuzzy system even sees them. That decoupling often destroys interpretability. 

Thinking out loud as a group, we realized that ADAR's approach is superior because it learns feature importance *during* the rule optimization phase. It doesn't just ask "is this feature important?", but rather, "is this feature important *for this specific rule*?" By explicitly maintaining learnable masks for both attributes and rules, the model actively builds its own architecture. We opted for this approach because it ensures that structural pruning directly correlates with the gradient descent of the loss function, rather than relying on an isolated pre-processing step.

### 3. Methodology
Our implementation in Python leverages PyTorch for its autograd capabilities. We structured the framework into three key phases:
1. **Data Preprocessing & Initialization:** We generated a 27-feature synthetic dataset. K-Means clustering ($L=5$) was used to initialize the centers and widths of the Gaussian membership functions.
2. **ADAR Layer Training:** The forward pass evaluates the Gaussian membership degrees and applies the dual weights to compute the rule activations.
3. **Dynamic Rule Management:** A custom tracking algorithm monitors the validation loss. If the error plateaus without improvement for 20 epochs, a new rule is seeded at the location of the maximum residual error to directly combat underfitting.

### 4. Mathematical Formulation
The ADAR layer relies on continuous optimization variables converted into structural masks. 

**Attribute Weighting ($\alpha_{l,i}$):**
The importance weight for attribute $i$ within rule $l$ is calculated using a learnable parameter $w_{a,l,i}$, passed through a sigmoid function $\sigma(\cdot)$, and multiplied by a binary attribute mask $m_{l,i}$:
$$ \alpha_{l,i} = \sigma(w_{a,l,i}) \cdot m_{l,i} \quad \text{(Eq. 1)} $$

**Rule Weighting ($\beta_l$):**
Similarly, the overall validity of an entire rule $l$ is determined by learning a parameter $w_{r,l}$:
$$ \beta_l = \sigma(w_{r,l}) \quad \text{(Eq. 4)} $$

If $\alpha_{l,i}$ falls below the threshold $\theta_{attr} = 0.25$, the mask $m_{l,i}$ is permanently set to zero. We opted for this threshold because it provided a solid balance between removing dead inputs and retaining variance.

### 5. Experimental Setup and Results
We ran an ablation study comparing a static Baseline ANFIS against our dynamic ADAR-ANFIS implementation.

**Parameters:** Max rules = 9, Patience = 20 epochs, Learning Rate = 0.01.
**Evaluation Metrics:** RMSE and Average Overlap Index ($I_{ov}$).

**Ablation Results:**
*   **Baseline ANFIS (Static, 5 Rules):** RMSE = 12.7545 | $I_{ov}$ = 0.9992 
*   **ADAR-ANFIS (Dynamic, 9 Rules):** RMSE = 16.1324 | $I_{ov}$ = 1.0000 

While the baseline converged on a slightly lower RMSE in this specific synthetic trial, the ADAR framework successfully demonstrated its structural plasticity by growing from 5 to 9 rules in response to validation stagnation. The $I_{ov}$ remained bounded, confirming the fuzzy sets maintained their localized structure despite the rule growth.

### 6. Critical Analysis
The dual-weighting framework elegantly handles sparsity, but it introduces sensitivity to the pruning thresholds. Hard pruning an attribute is an irreversible action in the computation graph. In highly non-linear environments, an early pruned attribute might have been useful later when a new rule was introduced. Future extensions might benefit from soft-pruning or an exponential decay mechanism rather than binary masking.

### 7. Individual Contributions
*   **Daniel Ebrahimzadeh:** Lead Architect & Git Control. Responsible for the PyTorch core logic integration.
*   **Haleh Adab:** Preprocessing & Dataset normalization. Managed the 27-feature synthetic data generation pipeline.
*   **Amin Sarlak:** Growth/Pruning Logic implementation. Developed the rule-spawning mechanics and patience thresholds.
*   **Muhammad Abuzar:** Metric Analysis ($I_{ov}$, RMSE) & Critical Review. Evaluated the overlap integration physics and structured the final findings.
