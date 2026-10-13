"""This is a sample neural ODE got from gemini. I've been looking at and playing with it"""
"""With generous aid form google gemini, I have set it up to run with the derivatives from Hoa's derivatives as a map of sample data. 
It spits out a plot of the approximations vs the integral on the graph."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import hoa_derivatives
from hoa_derivatives import derivatives, STATE_VARS, pack, zero_feed

# 1. Define the Neural Network representing the derivative: dy/dt = f(y, t)
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        # The network takes the current state (2D) and maps it to the velocity vector (2D)
        self.net = nn.Sequential(
            nn.Linear(35, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 35)
        )

    def forward(self, t, y):
        # t is the scalar time, y is the current state tensor
        return self.net(y)

# 2. Ground-Truth Dynamics Wrapper (Bridges PyTorch tensor -> NumPy -> PyTorch tensor)
class TrueDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = dict(
            F=1.0, V=3000.0,
            k_i1=1e-2, k_i2=1e-2,
            k11=1e-2, k12=1e-2, k13=1e-2,
            k21=1e-2, k22=1e-2,
            k31=1e-2,
            k_a=1e-2, k_x=1e-3, k_x2=1e-3, k_x3=1e-3,
            k_t=1e-2, k_t2=1e-3, k_t3=1e-3,
            k_tr=1e-3, k_tr1=1e-3, k_trM2=1e-3,
        )
        self.feed = zero_feed()
        self.feed.update(C_M1=2.0, C_M2=1.0, C_M3=0.05, C_C1=1e-3, C_H2=1e-3, C_Al=1e-2)

    def forward(self, t, y):
        # 1. Convert PyTorch tensor to a 1D NumPy array
        y_np = y.detach().cpu().numpy()
        is_batched = (y_np.ndim == 2)
        y_flat = y_np[0] if is_batched else y_np
        
        # 2. Extract scalar time and call your external NumPy derivative function
        t_val = t.item() if hasattr(t, 'item') else t
        dydt_np = derivatives(t_val, y_flat, self.p, feed=self.feed)
        
        # 3. Convert the NumPy result back into a PyTorch tensor matching the input shape
        dydt_val = np.expand_dims(dydt_np, 0) if is_batched else dydt_np
        return torch.tensor(dydt_val, dtype=y.dtype, device=y.device)

initial_concentrations = {var: 0.1 for var in STATE_VARS} 
true_y0_np = pack(initial_concentrations) # packs into 35-element array
true_y0 = torch.tensor(true_y0_np, dtype=torch.float32).unsqueeze(0) # Shape: [1, 35]

t = torch.linspace(0., 1., 20)

with torch.no_grad():
    true_y = odeint(TrueDynamics(), true_y0, t) # Shape: [20, 1, 35]

# 3. Initialize the Neural ODE Model, Optimizer, and Loss
func = ODEFunc()
optimizer = optim.Adam(func.parameters(), lr=0.005)
criterion = nn.MSELoss()

# 4. Training Loop
print("Training Neural ODE...")
for epoch in range(1, 201):
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

# 5. Graph the Results
# Generate final predictions from our trained model
with torch.no_grad():
    final_pred_y = odeint(func, true_y0, t)

t_np = t.numpy()
state_index = 3
state_name = STATE_VARS[state_index]

true_y_np = true_y[:, 0, state_index].numpy()
pred_y_np = final_pred_y[:, 0, state_index].numpy()

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(t_np, true_y_np, label=f"True {state_name}", color="blue", linewidth=2, linestyle="--")
plt.plot(t_np, pred_y_np, label=f"Neural ODE {state_name}", color="red", linewidth=2)

plt.xlabel("Time ($t$)", fontsize=12)
plt.ylabel(f"Concentration ({state_name}) ($y$)", fontsize=12)
plt.title(f"Neural ODE vs Ground Truth for {state_name}", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, linestyle=":", alpha=0.7)
plt.show()
