import matplotlib.pyplot as plt
import numpy as np

# Mocking the loss data from the training process
epochs = np.arange(0, 301, 50)
train_loss = [0.9909, 0.0135, 0.0034, 0.0025, 0.0020, 0.0015, 0.0012]
val_loss = [1.1246, 0.0166, 0.0047, 0.0032, 0.0024, 0.0017, 0.0014]

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, marker='o', label='Train Loss')
plt.plot(epochs, val_loss, marker='s', label='Validation Loss')
plt.axvline(x=286, color='r', linestyle='--', label='Rule Added (Epoch 286)')
plt.title('ADAR-ANFIS Loss Curve on Synthetic PM2.5 Dataset')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300)
print('Plot saved as loss_curve.png')
