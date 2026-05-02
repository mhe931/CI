# Mathematical Formulation of the ADAR Framework

## 1. Gaussian Membership Function Initialization
The fuzzy set for attribute $i$ in rule $l$ uses a Gaussian membership function, initialized via K-means clustering. The centroids provide the centers $v_{l,i}$, and the cluster standard deviations provide the widths $s_{l,i}$.
$$ \mu_{l,i}(x_i) = \exp\left(-\frac{(x_i - v_{l,i})^2}{2s_{l,i}^2}\right) $$

## 2. Dual Weighting Mechanisms
### 2.1 Attribute Weighting ($\alpha_{l,i}$)
The importance weight for attribute $i$ within rule $l$ is calculated using a learnable parameter $w_{a,l,i}$, passed through a sigmoid function $\sigma(\cdot)$, and multiplied by a binary mask $m_{l,i}$:
$$ \alpha_{l,i} = \sigma(w_{a,l,i}) \cdot m_{l,i} $$

### 2.2 Rule Weighting ($\beta_l$)
The overall importance of rule $l$ is learned via the parameter $w_{r,l}$:
$$ \beta_l = \sigma(w_{r,l}) $$

## 3. Pruning Thresholds
### 3.1 Attribute Pruning ($\theta_{attr}$)
Attributes with importance weights consistently below the threshold $\theta_{attr}$ are pruned by setting their mask to $0$:
$$ m_{l,i} = 0 \quad \text{if} \quad \alpha_{l,i} < \theta_{attr} $$

### 3.2 Rule Pruning ($\theta_r$)
Rules with an importance weight below the rule pruning threshold $\theta_r$ for a specified duration are marked for removal:
$$ \text{Prune rule } l \quad \text{if} \quad \beta_l < \theta_r $$

## 4. Rule Growing Criteria
When validation error stalls, new rules are generated to capture complex data regions.
**Condition:**
If there is no improvement in validation error for $p$ epochs AND $L < L_{max}$, select high-error samples from the training set to initialize the membership functions of a new rule.

## 5. Fuzzy Inference Module
The final output computation integrates the fuzzy sets and the weighting mechanisms.

**Firing Strength:**
$$ \text{Firing Strength}(R_l) = \prod_{i=1}^D \left[ \mu_{l,i}(x_i) \cdot \alpha_{l,i} \right] $$

**Rule Activation:**
$$ \tilde{f}_l = \text{Firing Strength}(R_l) \cdot \beta_l $$

**Normalized Activation:**
$$ w_l = \frac{\tilde{f}_l}{\sum_{m=1}^L \tilde{f}_m + \epsilon} $$

**Rule Output (Consequent):**
$$ y_l = \sum_{i \in \mathcal{A}_l} c_{l,i}x_i $$
*(where $\mathcal{A}_l$ is the set of active attributes in rule $l$, and $c_{l,i}$ are learnable consequent parameters)*

**Final Output:**
$$ y = \sum_{l=1}^L w_l y_l $$
