from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch import nn
import matplotlib.pyplot as plt
import torch

# Make N samples
n_samples = 1000

# Create circles dataset
X, y = make_circles(n_samples=n_samples, noise=0.03, random_state=42)

# Visualize the dataset
print(X.shape, y.shape)
print('First 5 samples of X:\n', X[:5])
print('First 5 labels of y:\n', y[:5])

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.show()

# Convert to tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Training samples: {X_train.shape}, Training labels: {y_train.shape}')
print(f'Test samples: {X_test.shape}, Test labels: {y_test.shape}')

# Build the model
# 1. Setup the device-agnostic model
# 2. Create the model instance (by subclassing nn.Module)
# 3. Create the loss function and optimizer
# 4. Setup a train/test loop to see how the model performs

# Agnostic device check
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Create the model instance  
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=8),
    nn.Linear(in_features=8, out_features=1)
).to(device)
print(model_0)

# Loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.1)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100

# Training loop
epochs = 100
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    model_0.train()
    
    # Forward pass
    y_logits = model_0(X_train)
    y_pred = torch.round(torch.sigmoid(y_logits))
    
    # Compute loss and accuracy
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_train, y_pred)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Testing loop
    with torch.inference_mode():
        model_0.eval()
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_pred)
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch} | Loss: {loss.item():.4f} | Accuracy: {acc:.2f}% | Test Loss: {test_loss.item():.4f} | Test Accuracy: {test_acc:.2f}%')