"""This is a sample neural ODE got from gemini. I've been looking at and playing with it"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint

# 1. Define the Neural Network representing the derivative: dy/dt = f(y, t)
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        # The network takes the current state (2D) and maps it to the velocity vector (2D)
        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, t, y):
        # t is the scalar time, y is the current state tensor
        return self.net(y)

# 2. Generate dummy ground-truth trajectory data
class TrueDynamics(nn.Module):
    def forward(self, t, y):
        # Matrix rotation vector field
        r = 1.5 #growth rate parameter
        return r * y * (1 - y) # Logistic growth dynamics

true_y0 = torch.tensor([[0.1]])
t = torch.linspace(0., 5., 20)

with torch.no_grad():
    true_y = odeint(TrueDynamics(), true_y0, t) # Shape: [20, 1, 1]

# 3. Initialize the Neural ODE Model, Optimizer, and Loss
func = ODEFunc()
optimizer = optim.Adam(func.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 4. Training Loop
print("Training Neural ODE...")
for epoch in range(1, 101):
    optimizer.zero_grad()
    
    # Forward pass: Evaluate the neural ODE solver over the time sequence
    # odeint integrates dy/dt = func(t, y) starting from true_y0
    pred_y = odeint(func, true_y0, t) 
    
    # Calculate loss against the target trajectory
    loss = criterion(pred_y, true_y)
    
    # Backward pass & Optimize
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f}")

print("\nTraining Complete!")
