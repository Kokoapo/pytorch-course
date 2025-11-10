import torch
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt

# Linear Regression Model 
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float32, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float32, requires_grad=True))
    
    def forward(self, x:torch.Tensor):
        return self.weights * x + self.bias

# Function to plot predictions
def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
    plt.figure(figsize=(10, 7))

    # Plot training/test data in blue/green
    plt.scatter(train_data, train_labels, c='b', s=4, label='Training Data')
    plt.scatter(test_data, test_labels, c='g', s=4, label='Testing Data')

    # Plot Predictions in red
    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s=4, label='Predictions')
    
    # Show the Legend
    plt.legend(prop={'size': 14})

# Create known parameters
weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# Create train/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(f'Train and Test Splits sizes: {(len(X_train), len(y_train), len(X_test), len(y_test))}')
plot_predictions(X_train, y_train, X_test, y_test)

# Create instance of Model
torch.manual_seed(42)

model_0 = LinearRegressionModel()
print(f'Model: {list(model_0.parameters())}')

# Define Loss and Optimizer
loss_fn = nn.L1Loss() # MAE Loss
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.034) #Stocastic Gradient Descend

# Make Predictions with Model
# Note: in older PyTorch code you might also see torch.no_grad()
with torch.inference_mode():
    y_preds = model_0(X_test)
plot_predictions(X_train, y_train, X_test, y_test, y_preds)

# Define Epoch Loop
epochs = 100
epoch_count, train_loss_values, test_loss_values = [], [], []
for epoch in range(epochs):
    # Define Train Loop
    model_0.train() # Turn Train Mode on

    y_preds = model_0(X_train)              # Forward Propagation
    train_loss = loss_fn(y_preds, y_train)  # Calculate MAE Loss
    optimizer.zero_grad()                   # Gradients accumulate, zero them to start fresh each forward pass
    train_loss.backward()                   # Backpropagation on Loss
    optimizer.step()                        # Update Model's parameters with respect to the gradients

    # Define Test Loop
    model_0.eval() # Turn Evaluation Mode on

    with torch.inference_mode(): # Disable functionalities such as gradient tracking
        y_preds = model_0(X_test)               # Forward Propagation
        test_loss = loss_fn(y_preds, y_test)    # Calculate MAE Loss
    
    if epoch % 10 == 0: # Display information output every 10 epochs 
        epoch_count.append(epoch)
        train_loss_values.append(train_loss.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())
        print(f'Epoch {epoch} - MAE Train Loss {train_loss} - MAE Test Loss {test_loss}')

# Plot the loss curves
# plt.plot(epoch_count, train_loss_values, label="Train loss")
# plt.plot(epoch_count, test_loss_values, label="Test loss")
# plt.title("Training and test loss curves")
# plt.ylabel("Loss")
# plt.xlabel("Epochs")
# plt.legend();

# Make Predictions with Model
model_0.eval() # 1. Set the model in evaluation mode

with torch.inference_mode(): # 2. Setup the inference mode context manager
    # 3. Make sure the calculations are done with the model and data on the same device
    # in our case, we haven't setup device-agnostic code yet so our data and model are
    # on the CPU by default.
    # model_0.to(device)
    # X_test = X_test.to(device)
    y_preds = model_0(X_test)
plot_predictions(X_train, y_train, X_test, y_test, y_preds)

# Save/Load Model
MODEL_PATH = Path('models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = '01_workflow_fundamentals.pth'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

loaded_model_1 = LinearRegressionModel()
loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))

# Show all plots in the end
plt.show()