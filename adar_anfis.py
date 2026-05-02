import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import math

class ADARLayer(nn.Module):
    def __init__(self, num_rules, num_features, centers, widths):
        """
        Initializes the ADAR (Adaptive Dynamic Attribute and Rule) layer.
        Args:
            num_rules: Initial number of fuzzy rules (L).
            num_features: Number of input attributes (D).
            centers: K-means cluster centroids.
            widths: Standard deviations of clusters.
        """
        super(ADARLayer, self).__init__()
        self.num_rules = num_rules
        self.num_features = num_features
        
        # Fuzzy membership function parameters (Gaussian)
        self.v = nn.Parameter(torch.tensor(centers, dtype=torch.float32))
        self.s = nn.Parameter(torch.tensor(widths, dtype=torch.float32))
        
        # Attribute weighting logits
        self.w_a = nn.Parameter(torch.randn(num_rules, num_features, dtype=torch.float32) * 0.1)
        # Attribute mask (m_l,i) - not updated by autograd
        self.register_buffer('m_a', torch.ones(num_rules, num_features, dtype=torch.float32))
        
        # Rule weighting logits
        self.w_r = nn.Parameter(torch.zeros(num_rules, dtype=torch.float32))
        
        # Consequent parameters (c_l,i and bias)
        self.c = nn.Parameter(torch.randn(num_rules, num_features, dtype=torch.float32) * 0.1)
        self.c_0 = nn.Parameter(torch.zeros(num_rules, dtype=torch.float32))
        
        # Pruning thresholds
        self.theta_attr = 0.25 # we opted for a 0.25 threshold here to balance interpretability with accuracy
        self.theta_r = 0.1
        
    def forward(self, x):
        """
        Forward pass for the fuzzy inference module.
        x: (batch_size, num_features)
        """
        batch_size = x.shape[0]
        
        # Expand x for rules computation: (batch_size, 1, num_features)
        x_exp = x.unsqueeze(1)
        
        # 1. Gaussian Membership Function
        # mu: (batch_size, num_rules, num_features)
        mu = torch.exp(- (x_exp - self.v.unsqueeze(0))**2 / (2 * self.s.unsqueeze(0)**2 + 1e-8))
        
        # 2. Attribute Weighting
        alpha = torch.sigmoid(self.w_a) * self.m_a # (num_rules, num_features)
        
        # 3. Rule Weighting
        beta = torch.sigmoid(self.w_r) # (num_rules)
        
        # Firing Strength (Eq 7)
        # Note: We use product across features. 
        # Adding a tiny epsilon to prevent 0 firing strength issues.
        firing_strength = torch.prod(mu * alpha.unsqueeze(0) + 1e-6, dim=2) # (batch_size, num_rules)
        
        # Rule Activation (Eq 8)
        f_tilde = firing_strength * beta.unsqueeze(0) # (batch_size, num_rules)
        
        # Normalized Activation (Eq 9)
        w_l = f_tilde / (torch.sum(f_tilde, dim=1, keepdim=True) + 1e-8) # (batch_size, num_rules)
        
        # Rule Output (Consequent) (Eq 10)
        # y_l = sum(c_l,i * x_i) + c_l,0
        # Only active attributes contribute
        active_c = self.c * self.m_a
        y_l = torch.sum(x_exp * active_c.unsqueeze(0), dim=2) + self.c_0.unsqueeze(0) # (batch_size, num_rules)
        
        # Final Output (Eq 11)
        y = torch.sum(w_l * y_l, dim=1) # (batch_size)
        return y
        
    def prune_attributes(self):
        """Prunes attributes that fall below the threshold."""
        with torch.no_grad():
            alpha = torch.sigmoid(self.w_a) * self.m_a
            # If alpha < theta_attr, set mask to 0
            pruned_mask = (alpha >= self.theta_attr).float()
            self.m_a.copy_(pruned_mask)

class StructureManager:
    def __init__(self, model, patience=15, max_rules=15):
        self.model = model
        self.patience = patience
        self.max_rules = max_rules
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        
    def check_and_grow(self, val_loss, x_train, y_train, y_pred_train):
        """Checks if validation loss is stalling and adds a rule based on highest error sample."""
        if val_loss < self.best_val_loss - 1e-4:
            self.best_val_loss = val_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
            
        if self.epochs_no_improve >= self.patience and self.model.num_rules < self.max_rules:
            # We hit patience, time to grow a rule. I'm choosing the highest error point to seed the new rule
            # to directly tackle the region where the model struggles most.
            errors = torch.abs(y_train - y_pred_train)
            idx = torch.argmax(errors)
            new_center = x_train[idx:idx+1]
            
            # Use average width for stability
            new_width = self.model.s.mean(dim=0, keepdim=True)
            
            with torch.no_grad():
                self.model.v = nn.Parameter(torch.cat([self.model.v, new_center], dim=0))
                self.model.s = nn.Parameter(torch.cat([self.model.s, new_width], dim=0))
                
                # new logits init
                self.model.w_a = nn.Parameter(torch.cat([self.model.w_a, torch.randn(1, self.model.num_features)*0.1], dim=0))
                
                # new mask
                new_mask = torch.ones(1, self.model.num_features)
                self.model.register_buffer('m_a', torch.cat([self.model.m_a, new_mask], dim=0))
                
                self.model.w_r = nn.Parameter(torch.cat([self.model.w_r, torch.zeros(1)], dim=0))
                self.model.c = nn.Parameter(torch.cat([self.model.c, torch.randn(1, self.model.num_features)*0.1], dim=0))
                self.model.c_0 = nn.Parameter(torch.cat([self.model.c_0, torch.zeros(1)], dim=0))
                
                self.model.num_rules += 1
            
            self.epochs_no_improve = 0
            return True # Indicates structure grew, might need optimizer reinit
        return False

def calculate_metrics(model, x_tensor):
    """Calculates I_ov and I_fsp to evaluate structural interpretability."""
    # This is a simplified estimation of the metrics to keep computation feasible.
    v = model.v.detach()
    s = model.s.detach()
    L, D = v.shape
    
    # 1. Overlap Index (I_ov)
    I_ov_d = []
    for d in range(D):
        overlap_sum = 0
        pairs = 0
        for i in range(L):
            for j in range(L):
                if i != j:
                    # simplified overlap based on intersection of Gaussians approx
                    dist = torch.abs(v[i, d] - v[j, d])
                    width_sum = s[i, d] + s[j, d]
                    # if they are far apart, overlap is minimal
                    overlap = torch.exp(-(dist / width_sum)**2)
                    overlap_sum += overlap.item()
                    pairs += 1
        I_ov_d.append(overlap_sum / max(1, pairs))
    I_ov = np.mean(I_ov_d)
    
    # 2. Position Index (I_fsp)
    I_fsp_d = []
    for d in range(D):
        # Sort centers for adjacent rule calculation
        v_sorted, indices = torch.sort(v[:, d])
        s_sorted = s[indices, d]
        
        fsp_sum = 0
        for l in range(L - 1):
            v_l, v_lp = v_sorted[l], v_sorted[l+1]
            s_l, s_lp = s_sorted[l], s_sorted[l+1]
            
            phi = torch.exp(-0.5 * ((v_l - v_lp) / (s_l + s_lp))**2)
            psi = torch.exp(-0.5 * ((v_l - v_lp) / (s_l - s_lp + 1e-8))**2)
            fsp_sum += 2 * torch.abs(0.5 - phi + psi).item()
        
        I_fsp_d.append(fsp_sum)
    I_fsp = np.mean(I_fsp_d) / max(1, L * D)
    
    return I_ov, I_fsp

def main():
    # Phase 2: Dataset Acquisition & Preprocessing
    # We generate a dataset mathematically mimicking Beijing PM2.5 in scale (10 variables)
    print("Generating synthetic PM2.5-like dataset...")
    X, y = make_regression(n_samples=2000, n_features=10, n_informative=6, noise=5.0, random_state=42)
    
    # Adding non-linear relationships to mimic real-world complexity
    y += 10 * np.sin(X[:, 0]) + 5 * X[:, 1]**2
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Initialization using K-means
    initial_rules = 5
    kmeans = KMeans(n_clusters=initial_rules, random_state=42, n_init=10)
    kmeans.fit(X_train)
    centers = kmeans.cluster_centers_
    
    # Estimate standard deviation per cluster for initialization
    widths = np.ones_like(centers) * 0.5
    for i in range(initial_rules):
        cluster_points = X_train[kmeans.labels_ == i]
        if len(cluster_points) > 1:
            widths[i] = np.std(cluster_points, axis=0) + 1e-4

    # Model Setup
    model = ADARLayer(num_rules=initial_rules, num_features=10, centers=centers, widths=widths)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    struct_manager = StructureManager(model, patience=20, max_rules=9)
    
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_te_t = torch.tensor(X_test, dtype=torch.float32)
    y_te_t = torch.tensor(y_test, dtype=torch.float32)
    
    epochs = 300
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        y_pred = model(X_tr_t)
        loss = criterion(y_pred, y_tr_t)
        
        # Add L1 regularization on weights for sparsity
        l1_loss = 0.001 * (torch.norm(model.w_a, 1) + torch.norm(model.w_r, 1))
        total_loss = loss + l1_loss
        
        total_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        
        # Periodic pruning
        if epoch % 50 == 0 and epoch > 0:
            model.prune_attributes()
            
        # Structure Check
        grew = struct_manager.check_and_grow(val_loss, X_tr_t, y_tr_t, y_pred.detach())
        if grew:
            print(f"Epoch {epoch}: Rule added! Total rules now: {model.num_rules}")
            # Re-init optimizer because parameters changed
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

    # Final Evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(X_te_t)
        # Inverse transform to calculate real RMSE
        test_pred_real = scaler_y.inverse_transform(test_pred.numpy().reshape(-1, 1)).flatten()
        y_te_real = scaler_y.inverse_transform(y_te_t.numpy().reshape(-1, 1)).flatten()
        rmse = np.sqrt(np.mean((test_pred_real - y_te_real)**2))
        
    I_ov, I_fsp = calculate_metrics(model, X_te_t)
    
    print("\n--- Final Results ---")
    print(f"Final Rule Count: {model.num_rules}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Overlap Index (I_ov): {I_ov:.4f}")
    print(f"Position Index (I_fsp): {I_fsp:.4f}")
    
    with open("results.txt", "w") as f:
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"I_ov: {I_ov:.4f}\n")
        f.write(f"I_fsp: {I_fsp:.4f}\n")
        f.write(f"Final Rules: {model.num_rules}\n")

if __name__ == "__main__":
    main()
