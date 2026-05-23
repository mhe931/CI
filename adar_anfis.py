import torch # Import PyTorch framework for tensor operations and autograd
import torch.nn as nn # Import neural network modules from PyTorch
import torch.optim as optim # Import optimization algorithms (like Adam) from PyTorch
import numpy as np # Import NumPy for standard numerical and array calculations
import pandas as pd # Import Pandas for data manipulation (if needed)
from sklearn.datasets import make_regression # Import regression dataset generator from scikit-learn
from sklearn.preprocessing import StandardScaler # Import standard scaler to normalize inputs/outputs
from sklearn.cluster import KMeans # Import K-Means clustering for antecedent parameter initialization
from sklearn.model_selection import train_test_split # Import utility to partition data into train/test sets
import math # Import Python math module for basic scientific calculations
import matplotlib.pyplot as plt # Import Matplotlib for plotting loss and accuracy convergence

class ADARLayer(nn.Module): # Define the main ADAR fuzzy layer inheriting from PyTorch nn.Module
    def __init__(self, num_rules, num_features, initial_centers, initial_widths, use_dual_weighting=True): # Constructor method
        """
        Initializes the ADAR layer or a Baseline ANFIS layer.
        """
        super(ADARLayer, self).__init__() # Call the parent class (nn.Module) constructor
        self.num_rules = num_rules # Save the total number of fuzzy rules in the architecture
        self.num_features = num_features # Save the total number of input features (dimensions)
        self.use_dual_weighting = use_dual_weighting # Boolean flag to toggle between dynamic ADAR and static Baseline ANFIS
        
        # 1. Antecedent parameters: Gaussian Membership Function parameters (Mean and Width)
        # self.fuzzy_centers stores the mean (v) of the Gaussian membership functions for each rule and feature.
        # It is registered as a learnable PyTorch Parameter, meaning backpropagation will adapt these centers.
        self.fuzzy_centers = nn.Parameter(torch.tensor(initial_centers, dtype=torch.float32)) 
        
        # self.fuzzy_widths stores the standard deviation (s) of the Gaussian membership functions.
        # It is registered as a learnable parameter so that widths adapt dynamically during gradient descent.
        self.fuzzy_widths = nn.Parameter(torch.tensor(initial_widths, dtype=torch.float32)) 
        
        # 2. Weighting mechanisms: Learnable parameters for dynamic structural pruning
        # self.attribute_weights stores w_a, the real-valued parameters optimized via gradient descent.
        # Passed through a sigmoid function, it yields the attribute importance value alpha.
        self.attribute_weights = nn.Parameter(torch.randn(num_rules, num_features, dtype=torch.float32) * 0.1) 
        
        # self.attribute_pruning_mask stores m_a, a binary buffer mask (1 = active feature, 0 = pruned feature).
        # We register it as a buffer instead of a Parameter because it is updated manually during pruning, not via gradients.
        self.register_buffer('attribute_pruning_mask', torch.ones(num_rules, num_features, dtype=torch.float32)) 
        
        # self.rule_weights stores w_r, the real-valued parameters determining the validity of an entire rule.
        # Passed through a sigmoid function, it yields the rule weighting parameter beta.
        self.rule_weights = nn.Parameter(torch.zeros(num_rules, dtype=torch.float32)) 
        
        # 3. Consequent parameters: Linear coefficients for Takagi-Sugeno-Kang (TSK) fuzzy output calculation
        # self.consequent_coefficients stores linear multipliers (c) for the input variables in each rule.
        self.consequent_coefficients = nn.Parameter(torch.randn(num_rules, num_features, dtype=torch.float32) * 0.1) 
        
        # self.consequent_intercepts stores the constant intercept (c_0) for each fuzzy rule output.
        self.consequent_intercepts = nn.Parameter(torch.zeros(num_rules, dtype=torch.float32)) 
        
        # 4. Pruning Threshold parameters
        self.attribute_pruning_threshold = 0.25 # Threshold below which an attribute importance alpha is permanently pruned
        self.rule_pruning_threshold = 0.1 # Threshold below which an entire rule beta is flagged for pruning
        
    def forward(self, input_tensor): # Forward propagation method (evaluates network output for input batch)
        batch_size = input_tensor.shape[0] # Extract the number of samples (batch size) in the current input batch
        expanded_input = input_tensor.unsqueeze(1) # Add a singleton dimension: changes shape from (batch, features) to (batch, 1, features)
        
        # 1. Antecedent Evaluation: Gaussian Membership Function
        # Equation: mu = exp( - (x - v)^2 / (2 * s^2) )
        # expanded_input shape: (batch_size, 1, num_features)
        # self.fuzzy_centers.unsqueeze(0) shape: (1, num_rules, num_features)
        # self.fuzzy_widths.unsqueeze(0) shape: (1, num_rules, num_features)
        squared_difference = (expanded_input - self.fuzzy_centers.unsqueeze(0))**2 # Numerator: squared difference (x - v)^2
        variance = 2 * (self.fuzzy_widths.unsqueeze(0)**2) + 1e-8 # Denominator: 2 * s^2 (1e-8 prevents division by zero)
        membership_degrees = torch.exp(- squared_difference / variance) # Full Gaussian MF evaluation
        
        # 2. Dual Weighting Mechanism implementation
        if self.use_dual_weighting: # If ADAR is enabled:
            # Map attribute weights through Sigmoid to force range [0, 1] and multiply by binary pruning mask
            # Equation 1: alpha_l,i = sigmoid(w_a,l,i) * m_l,i
            sigmoid_attribute_weights = torch.sigmoid(self.attribute_weights) * self.attribute_pruning_mask 
            # Map rule weights through Sigmoid to force range [0, 1]
            # Equation 4: beta_l = sigmoid(w_r,l)
            sigmoid_rule_weights = torch.sigmoid(self.rule_weights) 
        else: # If Baseline ANFIS is enabled:
            # Freeze attribute weights to a constant tensor of ones (no attribute weighting)
            sigmoid_attribute_weights = torch.ones_like(self.attribute_weights) 
            # Freeze rule weights to a constant tensor of ones (no rule weighting)
            sigmoid_rule_weights = torch.ones_like(self.rule_weights) 
        
        # 3. Rule Activation: Firing Strength Calculation
        # Multiply Gaussian degrees by learned attribute weights, sum 1e-6 for numerical stability, and multiply across features.
        # Equation: firing_strength = Product_over_features( mu * alpha + 1e-6 )
        raw_firing_strengths = torch.prod(membership_degrees * sigmoid_attribute_weights.unsqueeze(0) + 1e-6, dim=2) 
        
        # Weight firing strength by rule-level validity (beta)
        # Equation: weighted_firing = firing_strength * beta
        weighted_firing_strengths = raw_firing_strengths * sigmoid_rule_weights.unsqueeze(0) 
        
        # Normalization of Firing Strengths (ensures all rule activations sum up to 1 for defuzzification)
        # Equation: normalized_strength = weighted_firing / sum(weighted_firing)
        normalized_firing_strengths = weighted_firing_strengths / (torch.sum(weighted_firing_strengths, dim=1, keepdim=True) + 1e-8) 
        
        # 4. Consequent Evaluation: Takagi-Sugeno-Kang (TSK) Linear Output
        # Multiply linear consequent coefficients by the binary pruning mask so pruned features don't contribute to the output
        masked_consequent_coefficients = self.consequent_coefficients * (self.attribute_pruning_mask if self.use_dual_weighting else 1.0) 
        # Calculate rule output: y_l = sum(x_i * c_l,i) + c_l,0
        # expanded_input: (batch_size, 1, num_features) | masked_consequent: (1, num_rules, num_features)
        individual_rule_outputs = torch.sum(expanded_input * masked_consequent_coefficients.unsqueeze(0), dim=2) + self.consequent_intercepts.unsqueeze(0) 
        
        # 5. Defuzzification: Weighted sum of normalized activations and TSK outputs
        # Equation: y = sum( normalized_strength * rule_output )
        final_defuzzified_output = torch.sum(normalized_firing_strengths * individual_rule_outputs, dim=1) 
        return final_defuzzified_output # Return the final scalar prediction tensor of the network
        
    def prune_attributes(self): # Method to execute hard structural pruning on attributes
        if not self.use_dual_weighting: return # If Baseline ANFIS is enabled, skip pruning
        with torch.no_grad(): # Disable gradient calculations to perform raw parameter updates
            # Compute current active weights: alpha = sigmoid(w_a) * m_a
            sigmoid_attribute_weights = torch.sigmoid(self.attribute_weights) * self.attribute_pruning_mask 
            # Create a binary threshold mask: 1.0 if weight is >= threshold (0.25), 0.0 otherwise
            updated_pruning_mask = (sigmoid_attribute_weights >= self.attribute_pruning_threshold).float() 
            # Overwrite the binary mask buffer with the newly updated threshold mask
            self.attribute_pruning_mask.copy_(updated_pruning_mask) 

class StructureManager: # Define the StructureManager to control dynamic rule growth
    def __init__(self, neuro_fuzzy_model, patience=15, max_rules=15): # Constructor method
        self.neuro_fuzzy_model = neuro_fuzzy_model # Bind the ADAR model instance to this manager
        self.patience = patience # Stagnation epoch threshold (validation patience) before spawning a new rule
        self.max_rules = max_rules # Suffix constraint limiting the rule explosion
        self.best_validation_loss = float('inf') # Set initial best validation loss to positive infinity
        self.epochs_no_improvement = 0 # Counter tracking continuous epochs without error reduction
        
    def check_and_grow(self, validation_loss, training_inputs, training_targets, predicted_training_outputs): # Rule Growth assessment
        if not self.neuro_fuzzy_model.use_dual_weighting: return False # Skip growing if Baseline ANFIS is active
        
        # Check if validation loss has improved by a threshold of 1e-4
        if validation_loss < self.best_validation_loss - 1e-4: 
            self.best_validation_loss = validation_loss # Update the best validation record
            self.epochs_no_improvement = 0 # Reset the stagnation counter to zero
        else: 
            self.epochs_no_improvement += 1 # Increment the stagnation counter
            
        # Check if stagnation counter has reached the patience threshold and rule count is under the maximum limit
        if self.epochs_no_improvement >= self.patience and self.neuro_fuzzy_model.num_rules < self.max_rules: 
            # Calculate absolute residual error for each training sample: error = |y - y_pred|
            absolute_residual_errors = torch.abs(training_targets - predicted_training_outputs) 
            # Identify index of the training sample with the highest prediction error
            max_error_sample_index = torch.argmax(absolute_residual_errors) 
            # Extract the raw input coordinates of this high-error sample to seed the new rule center
            new_fuzzy_center = training_inputs[max_error_sample_index:max_error_sample_index+1] 
            # Initialize the new rule width as the mean of all currently active rule widths
            new_fuzzy_width = self.neuro_fuzzy_model.fuzzy_widths.mean(dim=0, keepdim=True) 
            
            with torch.no_grad(): # Disable gradients to structurally expand parameters
                # 1. Cat/Append the new center coordinates to self.fuzzy_centers
                self.neuro_fuzzy_model.fuzzy_centers = nn.Parameter(torch.cat([self.neuro_fuzzy_model.fuzzy_centers, new_fuzzy_center], dim=0)) 
                # 2. Cat/Append the calculated mean widths to self.fuzzy_widths
                self.neuro_fuzzy_model.fuzzy_widths = nn.Parameter(torch.cat([self.neuro_fuzzy_model.fuzzy_widths, new_fuzzy_width], dim=0)) 
                # 3. Append random small weight parameters for the new rule's attribute weight w_a
                self.neuro_fuzzy_model.attribute_weights = nn.Parameter(torch.cat([self.neuro_fuzzy_model.attribute_weights, torch.randn(1, self.neuro_fuzzy_model.num_features)*0.1], dim=0)) 
                # 4. Append ones to the binary mask buffer for the new rule (all features initially active)
                self.neuro_fuzzy_model.attribute_pruning_mask = torch.cat([self.neuro_fuzzy_model.attribute_pruning_mask, torch.ones(1, self.neuro_fuzzy_model.num_features)], dim=0) 
                # 5. Append a zero weight parameter for the new rule's validity w_r
                self.neuro_fuzzy_model.rule_weights = nn.Parameter(torch.cat([self.neuro_fuzzy_model.rule_weights, torch.zeros(1)], dim=0)) 
                # 6. Append random small parameters for TSK consequent coefficients c
                self.neuro_fuzzy_model.consequent_coefficients = nn.Parameter(torch.cat([self.neuro_fuzzy_model.consequent_coefficients, torch.randn(1, self.neuro_fuzzy_model.num_features)*0.1], dim=0)) 
                # 7. Append a zero for the TSK consequent intercept c_0
                self.neuro_fuzzy_model.consequent_intercepts = nn.Parameter(torch.cat([self.neuro_fuzzy_model.consequent_intercepts, torch.zeros(1)], dim=0)) 
                # 8. Increment the model's total rule count tracker
                self.neuro_fuzzy_model.num_rules += 1 
            
            self.epochs_no_improvement = 0 # Reset the stagnation counter since we expanded the structure
            return True # Return True to flag that a rule expansion successfully occurred
        return False # Return False if no growing was triggered

def calculate_metrics(neuro_fuzzy_model, test_inputs): # Quality Assurance evaluation: computes I_ov and I_fsp
    """Calculates I_ov and I_fsp explicitly based on paper Eq 13-16."""
    fuzzy_centers = neuro_fuzzy_model.fuzzy_centers.detach() # Extract centers without tracking gradients
    fuzzy_widths = neuro_fuzzy_model.fuzzy_widths.detach() # Extract widths without tracking gradients
    num_rules, num_features = fuzzy_centers.shape # Extract dimensions: total rules (L) and features (D)
    
    # Grid initialization for numerical integration of membership overlaps
    numerical_integration_steps = 500 # Set 500 steps for numerical precision
    integration_min, integration_max = -5.0, 5.0 # Integration limits matching our standardized input domain
    # Generate 500 evenly spaced points and add dimensions: shape (500, 1, 1)
    integration_grid = torch.linspace(integration_min, integration_max, numerical_integration_steps).unsqueeze(1).unsqueeze(2) 
    
    # 1. Calculate Overlap Index (I_ov)
    # Evaluates the maximum overlap ratio between pairwise membership functions along each feature dimension
    overlap_indices_per_feature = [] # List to store overlap values per feature variable
    for d in range(num_features): # Loop through every input feature
        centers_d = fuzzy_centers[:, d] # Extract centers of all rules for feature d: shape (num_rules)
        widths_d = fuzzy_widths[:, d] # Extract widths of all rules for feature d: shape (num_rules)
        grid_1d = integration_grid.squeeze().unsqueeze(1) # Reshape grid to shape (500, 1) for broadcasting
        # Compute membership value of each grid point across all rules: shape (500, num_rules)
        membership_degrees_d = torch.exp(- (grid_1d - centers_d.unsqueeze(0))**2 / (2 * widths_d.unsqueeze(0)**2 + 1e-8)) 
        
        max_overlap_for_feature = 0 # Track the worst-case overlap for this feature variable
        for i in range(num_rules): # Outer loop for rule i
            for j in range(num_rules): # Inner loop for rule j
                if i != j: # Only check pairs of different rules
                    min_membership = torch.min(membership_degrees_d[:, i], membership_degrees_d[:, j]) # Intersection of MFs
                    int_minimum = torch.trapz(min_membership, integration_grid.squeeze()) # Area of intersection
                    int_i = torch.trapz(membership_degrees_d[:, i], integration_grid.squeeze()) # Area of MF i
                    int_j = torch.trapz(membership_degrees_d[:, j], integration_grid.squeeze()) # Area of MF j
                    # Overlap ratio: intersection area divided by the minimum of the two individual areas
                    overlap = int_minimum / (torch.min(int_i, int_j) + 1e-8) 
                    max_overlap_for_feature = max(max_overlap_for_feature, overlap.item()) # Update worst-case tracker
        overlap_indices_per_feature.append(max_overlap_for_feature) # Append feature overlap to list
    average_overlap_index = np.mean(overlap_indices_per_feature) # Average overlap index (I_ov) across all features
    
    # 2. Calculate Position Index (I_fsp)
    # Measures how uniformly spaced adjacent fuzzy membership sets are on the domain
    position_indices_per_feature = [] # List to store spacing indicators per feature variable
    for d in range(num_features): # Loop through every input feature
        sorted_centers, sort_indices = torch.sort(fuzzy_centers[:, d]) # Sort rule centers in ascending order
        sorted_widths = fuzzy_widths[sort_indices, d] # Sort widths in the same order as their centers
        fsp_sum_for_feature = 0 # Initialize sum of spacing deviations
        for l in range(num_rules - 1): # Loop through adjacent rule pairs
            center_l, center_next = sorted_centers[l], sorted_centers[l+1] # Center of current and next set
            width_l, width_next = sorted_widths[l], sorted_widths[l+1] # Width of current and next set
            
            # Phi: overlap scaling term based on centers distance over sum of widths
            phi = torch.exp(-0.5 * ((center_l - center_next) / (width_l + width_next + 1e-8))**2) 
            # Psi: overlap scaling term based on centers distance over difference of widths
            psi = torch.exp(-0.5 * ((center_l - center_next) / (torch.abs(width_l - width_next) + 1e-8))**2) 
            # Spacing deviation indicator equation (Equation 14)
            fsp_sum_for_feature += 2 * torch.abs(0.5 - phi + psi).item() 
        position_indices_per_feature.append(fsp_sum_for_feature) # Save feature spacing summation
    # Normalize Position Index (I_fsp) across all variables and rules
    average_position_index = np.mean(position_indices_per_feature) / max(1, num_rules * num_features) 
    
    return average_overlap_index, average_position_index # Return final calculated interpretability indices

def train_model(X_train, y_train, X_val, y_val, X_test, y_test, scaler_y, initial_centers, initial_widths, num_features, use_dual, name): # Training pipeline
    print(f"\nTraining {name}...") # Display execution header to console
    # Initialize the model using 5 initial rules, 27 features, and K-Means centers/widths
    model = ADARLayer(num_rules=5, num_features=num_features, initial_centers=initial_centers, initial_widths=initial_widths, use_dual_weighting=use_dual) 
    optimizer = optim.Adam(model.parameters(), lr=0.01) # Bind Adam optimizer with a learning rate of 0.01
    criterion = nn.MSELoss() # Define standard Mean Squared Error loss function
    # Bind StructureManager with rule patience = 20 epochs and maximum rule count = 9
    struct_manager = StructureManager(model, patience=20, max_rules=9) 
    
    # Convert numpy training, validation, and test datasets to PyTorch float32 Tensors
    X_tr_t = torch.tensor(X_train, dtype=torch.float32) 
    y_tr_t = torch.tensor(y_train, dtype=torch.float32) 
    X_val_t = torch.tensor(X_val, dtype=torch.float32) 
    y_val_t = torch.tensor(y_val, dtype=torch.float32) 
    X_te_t = torch.tensor(X_test, dtype=torch.float32) 
    
    epochs = 300 # Define total training epochs (cycles) = 300
    loss_history = [] # History list to record training loss progress
    val_history = [] # History list to record validation loss progress
    rule_history = [] # History list to track rule growth step steps
    
    for epoch in range(epochs): # Loop through 300 epochs
        model.train() # Put PyTorch model in training mode
        optimizer.zero_grad() # Clear previous optimization gradients to prevent accumulation
        y_pred = model(X_tr_t) # Forward pass: calculate predictions for training batch
        loss = criterion(y_pred, y_tr_t) # Calculate raw Mean Squared Error loss
        
        if use_dual: # If ADAR framework is active, add Sparsity regularization:
            # Enforce L1 penalty on learned weight tensors: L1 = 0.001 * (||w_a||_1 + ||w_r||_1)
            l1_loss = 0.001 * (torch.norm(model.attribute_weights, 1) + torch.norm(model.rule_weights, 1)) 
            total_loss = loss + l1_loss # Sum L1 penalty and raw MSE loss
        else: 
            total_loss = loss # Only use raw MSE loss for static ANFIS
            
        total_loss.backward() # Backward pass: compute parameter gradients via autograd
        optimizer.step() # Optimization step: update weights using Adam gradients
        
        model.eval() # Put PyTorch model in evaluation mode for validation check
        with torch.no_grad(): # Disable gradients for speed during validation propagation
            val_pred = model(X_val_t) # Forward pass: calculate predictions for validation batch
            val_loss = criterion(val_pred, y_val_t).item() # Extract scalar MSE validation loss
            
        if epoch % 50 == 0 and epoch > 0: # Attribute Pruning interval: triggers every 50 epochs
            model.prune_attributes() # Call hard pruning on attribute weight tensors
            
        # StructureManager growth check: grows rule if validation loss stagnated for 20 epochs
        grew = struct_manager.check_and_grow(val_loss, X_tr_t, y_tr_t, y_pred.detach()) 
        if grew: # If a new rule was born, optimizer parameters must re-initialize:
            optimizer = optim.Adam(model.parameters(), lr=0.01) # Re-bind optimizer to include expanded parameters
            
        loss_history.append(loss.item()) # Log training loss
        val_history.append(val_loss) # Log validation loss
        rule_history.append(model.num_rules) # Log rule count progress
            
    # Evaluation on the partition Test Set
    model.eval() # Put model in evaluation mode
    with torch.no_grad(): # Disable gradient tracking
        test_pred = model(X_te_t) # Calculate predictions on test inputs
        # Denormalize output scaling to compute RMSE in real-world units (e.g. Watts/kWh)
        test_pred_real = scaler_y.inverse_transform(test_pred.numpy().reshape(-1, 1)).flatten() 
        # Calculate Root Mean Squared Error (RMSE) against actual targets
        rmse = np.sqrt(np.mean((test_pred_real - y_test)**2)) 
        
    I_ov, I_fsp = calculate_metrics(model, X_te_t) # Calculate final quality indices Overlap and Position
    return rmse, I_ov, I_fsp, model.num_rules, loss_history, val_history, rule_history # Return all tracked parameters

def main(): # Main execution pipeline
    print("Generating High-Dimensional Dataset (Appliances Energy Equivalent - 27 variables)")
    # Generate a high-dimensional regression dataset (3000 samples, 27 features)
    X, y = make_regression(n_samples=3000, n_features=27, n_informative=15, noise=10.0, random_state=42) 
    # Add non-linear complexity using trigonometry, quadratic variables, and interactions to match IoT patterns
    y += 15 * np.sin(X[:, 0]) + 8 * X[:, 2]**2 + 5 * X[:, 10] * X[:, 12] 
    
    scaler_x = StandardScaler() # Create standard scaler for inputs
    scaler_y = StandardScaler() # Create standard scaler for outputs
    
    X_scaled = scaler_x.fit_transform(X) # Normalize features to have mean=0 and variance=1
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten() # Normalize target variables
    
    # Partition dataset into 80% Training and 20% Testing subsets
    X_train, X_test, y_train, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42) 
    # Partition Training subset into 80% Training and 20% Validation partitions
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42) 
    
    # Save the original unscaled target variables for calculating final real-world RMSE
    _, _, _, y_test_real = train_test_split(X, y, test_size=0.2, random_state=42) 
    
    # Antecedent Initialization: Initialize Centers using K-Means Clustering (L=5)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X_train) 
    centers = kmeans.cluster_centers_ # Extract calculated cluster centroids to seed centers
    widths = np.ones_like(centers) * 0.5 # Initialize widths with a default value of 0.5
    for i in range(5): # Loop through each of the 5 initial clusters
        pts = X_train[kmeans.labels_ == i] # Get training points belonging to cluster i
        if len(pts) > 1: widths[i] = np.std(pts, axis=0) + 1e-4 # Assign standard deviation of points as cluster widths

    # Run Trial A: Static Baseline ANFIS (Dual weighting and rule growth disabled)
    b_rmse, b_iov, b_ifsp, b_rules, b_loss, _, _ = train_model(
        X_train, y_train, X_val, y_val, X_test, y_test_real, scaler_y, centers, widths, 27, False, "Baseline ANFIS"
    )
    
    # Run Trial B: Dynamic ADAR-FIS Framework (All pruning and rule growing enabled)
    a_rmse, a_iov, a_ifsp, a_rules, a_loss, a_val, a_rh = train_model(
        X_train, y_train, X_val, y_val, X_test, y_test_real, scaler_y, centers, widths, 27, True, "ADAR-ANFIS"
    )

    print("\n=== ABLATION RESULTS ===") # Log results to console
    print(f"Baseline ANFIS -> RMSE: {b_rmse:.4f} | I_ov: {b_iov:.4f} | I_fsp: {b_ifsp:.4f} | Rules: {b_rules}")
    print(f"ADAR-ANFIS     -> RMSE: {a_rmse:.4f} | I_ov: {a_iov:.4f} | I_fsp: {a_ifsp:.4f} | Rules: {a_rules}")

    # Generate the Hero Chart with 2 Subplots to visualize results
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 5)) 
    
    # Left Subplot: ADAR training MSE loss, validation loss, and step-wise rule growth
    ax2 = ax1.twinx() # Create a shared-x twin-y axis for rule count
    ax1.plot(a_loss, 'b-', label='ADAR Train Loss') # Plot training loss curve in solid blue
    ax1.plot(a_val, 'r--', label='ADAR Val Loss') # Plot validation loss curve in dashed red
    ax2.plot(a_rh, 'g:', linewidth=2, label='Rule Count') # Plot rule count steps in dotted green
    
    ax1.set_xlabel('Epochs') # Set x-axis label
    ax1.set_ylabel('MSE Loss', color='b') # Set y-axis label for loss
    ax2.set_ylabel('Number of Rules', color='g') # Set twin y-axis label for rule count
    ax1.set_title('Dynamic Rule Growth and Validation Convergence') # Subplot title
    
    lines_1, labels_1 = ax1.get_legend_handles_labels() # Extract legend metrics from axis 1
    lines_2, labels_2 = ax2.get_legend_handles_labels() # Extract legend metrics from axis 2
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right') # Render legend
    
    # Right Subplot: Bar chart comparing final RMSE of Baseline ANFIS and ADAR-ANFIS
    models = ['Baseline ANFIS', 'ADAR-ANFIS'] # Bar labels
    rmses = [b_rmse, a_rmse] # Bar values
    bars = ax3.bar(models, rmses, color=['gray', '#4da6ff']) # Plot bars (Grey for Baseline, Blue for ADAR)
    ax3.set_ylabel('Final RMSE') # Set y-axis label
    ax3.set_title('RMSE vs. Baseline ANFIS') # Subplot title
    for bar in bars: # Loop through bars to draw target labels
        yval = bar.get_height() # Get height of current bar
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.2f}', ha='center', va='bottom') # Draw label
        
    plt.tight_layout() # Compress spacing to prevent clipping
    plt.savefig('hero_chart.png', dpi=300) # Save image file
    
    import json # Import Python json library
    results = { # Structure results inside a nested dictionary
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
    with open('comparison_results.json', 'w') as f: # Open results file for writing
        json.dump(results, f, indent=4) # Dump structured results to comparison_results.json

if __name__ == "__main__": # Entry point check
    main() # Call main pipeline execution
