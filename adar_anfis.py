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
import matplotlib.pyplot as plt

class ADARLayer(nn.Module):
    def __init__(self, num_rules, num_features, centers, widths, use_dual_weighting=True):
        """
        Initializes the ADAR layer or a Baseline ANFIS layer.
        """
        super(ADARLayer, self).__init__()
        self.num_rules = num_rules
        self.num_features = num_features
        self.use_dual_weighting = use_dual_weighting
        
        # Gaussian MF parameters
        self.v = nn.Parameter(torch.tensor(centers, dtype=torch.float32))
        self.s = nn.Parameter(torch.tensor(widths, dtype=torch.float32))
        
        # Weighting mechanisms
        self.w_a = nn.Parameter(torch.randn(num_rules, num_features, dtype=torch.float32) * 0.1)
        self.register_buffer('m_a', torch.ones(num_rules, num_features, dtype=torch.float32))
        self.w_r = nn.Parameter(torch.zeros(num_rules, dtype=torch.float32))
        
        # Consequent parameters
        self.c = nn.Parameter(torch.randn(num_rules, num_features, dtype=torch.float32) * 0.1)
        self.c_0 = nn.Parameter(torch.zeros(num_rules, dtype=torch.float32))
        
        self.theta_attr = 0.25 
        self.theta_r = 0.1
        
    def forward(self, x):
        batch_size = x.shape[0]
        x_exp = x.unsqueeze(1)
        
        # 1. Gaussian Membership Function
        mu = torch.exp(- (x_exp - self.v.unsqueeze(0))**2 / (2 * self.s.unsqueeze(0)**2 + 1e-8))
        
        # 2. Dual Weighting Mechanism
        if self.use_dual_weighting:
            alpha = torch.sigmoid(self.w_a) * self.m_a
            beta = torch.sigmoid(self.w_r)
        else:
            alpha = torch.ones_like(self.w_a)
            beta = torch.ones_like(self.w_r)
        
        # Firing Strength
        firing_strength = torch.prod(mu * alpha.unsqueeze(0) + 1e-6, dim=2)
        
        # Rule Activation
        f_tilde = firing_strength * beta.unsqueeze(0)
        
        # Normalized Activation
        w_l = f_tilde / (torch.sum(f_tilde, dim=1, keepdim=True) + 1e-8)
        
        # Rule Output
        active_c = self.c * (self.m_a if self.use_dual_weighting else 1.0)
        y_l = torch.sum(x_exp * active_c.unsqueeze(0), dim=2) + self.c_0.unsqueeze(0)
        
        # Final Output
        y = torch.sum(w_l * y_l, dim=1)
        return y
        
    def prune_attributes(self):
        if not self.use_dual_weighting: return
        with torch.no_grad():
            alpha = torch.sigmoid(self.w_a) * self.m_a
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
        if not self.model.use_dual_weighting: return False
        
        if val_loss < self.best_val_loss - 1e-4:
            self.best_val_loss = val_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
            
        if self.epochs_no_improve >= self.patience and self.model.num_rules < self.max_rules:
            errors = torch.abs(y_train - y_pred_train)
            idx = torch.argmax(errors)
            new_center = x_train[idx:idx+1]
            new_width = self.model.s.mean(dim=0, keepdim=True)
            
            with torch.no_grad():
                self.model.v = nn.Parameter(torch.cat([self.model.v, new_center], dim=0))
                self.model.s = nn.Parameter(torch.cat([self.model.s, new_width], dim=0))
                self.model.w_a = nn.Parameter(torch.cat([self.model.w_a, torch.randn(1, self.model.num_features)*0.1], dim=0))
                self.model.register_buffer('m_a', torch.cat([self.model.m_a, torch.ones(1, self.model.num_features)], dim=0))
                self.model.w_r = nn.Parameter(torch.cat([self.model.w_r, torch.zeros(1)], dim=0))
                self.model.c = nn.Parameter(torch.cat([self.model.c, torch.randn(1, self.model.num_features)*0.1], dim=0))
                self.model.c_0 = nn.Parameter(torch.cat([self.model.c_0, torch.zeros(1)], dim=0))
                self.model.num_rules += 1
            
            self.epochs_no_improve = 0
            return True
        return False

def calculate_metrics(model, x_tensor):
    """Calculates I_ov and I_fsp explicitly based on paper Eq 13-16."""
    v = model.v.detach()
    s = model.s.detach()
    L, D = v.shape
    
    # Grid for numerical integration (I_ov)
    grid_points = 500
    x_min, x_max = -5.0, 5.0
    x_grid = torch.linspace(x_min, x_max, grid_points).unsqueeze(1).unsqueeze(2) # (grid, 1, 1)
    
    # 1. Overlap Index (I_ov)
    I_ov_d = []
    for d in range(D):
        v_d = v[:, d] # (L)
        s_d = s[:, d] # (L)
        x_grid_1d = x_grid.squeeze().unsqueeze(1) # (grid, 1)
        mu_d = torch.exp(- (x_grid_1d - v_d.unsqueeze(0))**2 / (2 * s_d.unsqueeze(0)**2 + 1e-8)) # (grid, L)
        
        max_overlap_for_attr = 0
        for i in range(L):
            for j in range(L):
                if i != j:
                    min_mu = torch.min(mu_d[:, i], mu_d[:, j])
                    int_min = torch.trapz(min_mu, x_grid.squeeze())
                    int_i = torch.trapz(mu_d[:, i], x_grid.squeeze())
                    int_j = torch.trapz(mu_d[:, j], x_grid.squeeze())
                    overlap = int_min / (torch.min(int_i, int_j) + 1e-8)
                    max_overlap_for_attr = max(max_overlap_for_attr, overlap.item())
        I_ov_d.append(max_overlap_for_attr)
    I_ov = np.mean(I_ov_d)
    
    # 2. Position Index (I_fsp)
    I_fsp_d = []
    for d in range(D):
        v_sorted, indices = torch.sort(v[:, d])
        s_sorted = s[indices, d]
        fsp_sum = 0
        for l in range(L - 1):
            v_l, v_lp = v_sorted[l], v_sorted[l+1]
            s_l, s_lp = s_sorted[l], s_sorted[l+1]
            
            phi = torch.exp(-0.5 * ((v_l - v_lp) / (s_l + s_lp + 1e-8))**2)
            psi = torch.exp(-0.5 * ((v_l - v_lp) / (torch.abs(s_l - s_lp) + 1e-8))**2)
            fsp_sum += 2 * torch.abs(0.5 - phi + psi).item()
        I_fsp_d.append(fsp_sum)
    I_fsp = np.mean(I_fsp_d) / max(1, L * D)
    
    return I_ov, I_fsp

def train_model(X_train, y_train, X_val, y_val, X_test, y_test, scaler_y, centers, widths, num_features, use_dual, name):
    print(f"\nTraining {name}...")
    model = ADARLayer(num_rules=5, num_features=num_features, centers=centers, widths=widths, use_dual_weighting=use_dual)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    struct_manager = StructureManager(model, patience=20, max_rules=9)
    
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_te_t = torch.tensor(X_test, dtype=torch.float32)
    
    epochs = 300
    loss_history = []
    val_history = []
    rule_history = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_tr_t)
        loss = criterion(y_pred, y_tr_t)
        
        if use_dual:
            l1_loss = 0.001 * (torch.norm(model.w_a, 1) + torch.norm(model.w_r, 1))
            total_loss = loss + l1_loss
        else:
            total_loss = loss
            
        total_loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
            
        if epoch % 50 == 0 and epoch > 0:
            model.prune_attributes()
            
        grew = struct_manager.check_and_grow(val_loss, X_tr_t, y_tr_t, y_pred.detach())
        if grew:
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            
        loss_history.append(loss.item())
        val_history.append(val_loss)
        rule_history.append(model.num_rules)
            
    # Test
    model.eval()
    with torch.no_grad():
        test_pred = model(X_te_t)
        test_pred_real = scaler_y.inverse_transform(test_pred.numpy().reshape(-1, 1)).flatten()
        rmse = np.sqrt(np.mean((test_pred_real - y_test)**2))
        
    I_ov, I_fsp = calculate_metrics(model, X_te_t)
    return rmse, I_ov, I_fsp, model.num_rules, loss_history, val_history, rule_history

def main():
    print("Generating High-Dimensional Dataset (Appliances Energy Equivalent - 27 variables)")
    # Generate 27 features to match Appliances Energy dataset
    X, y = make_regression(n_samples=3000, n_features=27, n_informative=15, noise=10.0, random_state=42)
    y += 15 * np.sin(X[:, 0]) + 8 * X[:, 2]**2 + 5 * X[:, 10] * X[:, 12]
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    X_train, X_test, y_train, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Keep real y_test for final RMSE
    _, _, _, y_test_real = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Init K-means
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X_train)
    centers = kmeans.cluster_centers_
    widths = np.ones_like(centers) * 0.5
    for i in range(5):
        pts = X_train[kmeans.labels_ == i]
        if len(pts) > 1: widths[i] = np.std(pts, axis=0) + 1e-4

    # Baseline ANFIS
    b_rmse, b_iov, b_ifsp, b_rules, b_loss, _, _ = train_model(
        X_train, y_train, X_val, y_val, X_test, y_test_real, scaler_y, centers, widths, 27, False, "Baseline ANFIS"
    )
    
    # ADAR ANFIS
    a_rmse, a_iov, a_ifsp, a_rules, a_loss, a_val, a_rh = train_model(
        X_train, y_train, X_val, y_val, X_test, y_test_real, scaler_y, centers, widths, 27, True, "ADAR-ANFIS"
    )

    print("\n=== ABLATION RESULTS ===")
    print(f"Baseline ANFIS -> RMSE: {b_rmse:.4f} | I_ov: {b_iov:.4f} | I_fsp: {b_ifsp:.4f} | Rules: {b_rules}")
    print(f"ADAR-ANFIS     -> RMSE: {a_rmse:.4f} | I_ov: {a_iov:.4f} | I_fsp: {a_ifsp:.4f} | Rules: {a_rules}")

    # Generate Hero Chart with 2 Subplots
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left Subplot: Loss and Rule Growth
    ax2 = ax1.twinx()
    ax1.plot(a_loss, 'b-', label='ADAR Train Loss')
    ax1.plot(a_val, 'r--', label='ADAR Val Loss')
    ax2.plot(a_rh, 'g:', linewidth=2, label='Rule Count')
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MSE Loss', color='b')
    ax2.set_ylabel('Number of Rules', color='g')
    ax1.set_title('Dynamic Rule Growth and Validation Convergence')
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    
    # Right Subplot: RMSE Comparison
    models = ['Baseline ANFIS', 'ADAR-ANFIS']
    rmses = [b_rmse, a_rmse]
    bars = ax3.bar(models, rmses, color=['gray', '#4da6ff'])
    ax3.set_ylabel('Final RMSE')
    ax3.set_title('RMSE vs. Baseline ANFIS')
    for bar in bars:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.2f}', ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig('hero_chart.png', dpi=300)
    
    import json
    results = {
        "Baseline": {
            "RMSE": round(b_rmse, 4),
            "I_ov": round(b_iov, 4),
            "I_fsp": round(b_ifsp, 4),
            "Rules": b_rules
        },
        "ADAR": {
            "RMSE": round(a_rmse, 4),
            "I_ov": round(a_iov, 4),
            "I_fsp": round(a_ifsp, 4),
            "Rules": a_rules
        }
    }
    with open('comparison_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
