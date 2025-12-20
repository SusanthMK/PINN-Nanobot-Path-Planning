
"""
Physics-Informed Neural Networks with Simulated Annealing for Nanobot Path Planning
Complete Implementation based on the PDF methodology

Requirements:
- torch >= 1.9.0
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, splrep, splev
import random
from typing import Tuple, List, Dict, Optional

# Set device and random seeds
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

class PhysicsInformedNN(nn.Module):
    """
    Physics-Informed Neural Network for Nanobot Dynamics
    Implements the PINN architecture from the PDF
    """

    def __init__(self, input_dim: int = 3, hidden_dims: List[int] = [50, 50, 50], 
                 output_dim: int = 3):
        super(PhysicsInformedNN, self).__init__()

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())  # Tanh activation as per PDF
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        # Physics parameters
        self.reynolds = 1e-4  # Low Reynolds number
        self.viscosity = 0.004  # Pa·s
        self.density = 1060  # kg/m³

        # Loss weights from PDF
        self.w_physics = 1.0
        self.w_boundary = 10.0
        self.w_data = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.network(x)

    def physics_loss(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute physics-informed loss based on Navier-Stokes equations
        Implements L_pde from the PDF equation
        """
        # Combine inputs
        inputs = torch.cat([x, y, t], dim=1)
        inputs.requires_grad_(True)

        # Forward pass
        outputs = self.forward(inputs)
        u, v, p = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]

        # Compute gradients using autograd
        u_x = torch.autograd.grad(u, inputs, torch.ones_like(u), create_graph=True)[0][:, 0:1]
        u_y = torch.autograd.grad(u, inputs, torch.ones_like(u), create_graph=True)[0][:, 1:2]
        u_t = torch.autograd.grad(u, inputs, torch.ones_like(u), create_graph=True)[0][:, 2:3]

        v_x = torch.autograd.grad(v, inputs, torch.ones_like(v), create_graph=True)[0][:, 0:1]
        v_y = torch.autograd.grad(v, inputs, torch.ones_like(v), create_graph=True)[0][:, 1:2]
        v_t = torch.autograd.grad(v, inputs, torch.ones_like(v), create_graph=True)[0][:, 2:3]

        p_x = torch.autograd.grad(p, inputs, torch.ones_like(p), create_graph=True)[0][:, 0:1]
        p_y = torch.autograd.grad(p, inputs, torch.ones_like(p), create_graph=True)[0][:, 1:2]

        # Second derivatives for viscosity term
        u_xx = torch.autograd.grad(u_x, inputs, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, inputs, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
        v_xx = torch.autograd.grad(v_x, inputs, torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y, inputs, torch.ones_like(v_y), create_graph=True)[0][:, 1:2]

        # Navier-Stokes equations (low Reynolds approximation)
        ns_x = u_t + u * u_x + v * u_y + p_x/self.density - self.viscosity/self.density * (u_xx + u_yy)
        ns_y = v_t + u * v_x + v * v_y + p_y/self.density - self.viscosity/self.density * (v_xx + v_yy)
        continuity = u_x + v_y

        # Physics loss
        physics_loss = torch.mean(ns_x**2 + ns_y**2 + continuity**2)
        return physics_loss

    def boundary_loss(self, x_bc: torch.Tensor, y_bc: torch.Tensor, t_bc: torch.Tensor,
                     u_bc: torch.Tensor, v_bc: torch.Tensor) -> torch.Tensor:
        """Boundary condition loss"""
        inputs_bc = torch.cat([x_bc, y_bc, t_bc], dim=1)
        outputs_bc = self.forward(inputs_bc)
        u_pred, v_pred = outputs_bc[:, 0:1], outputs_bc[:, 1:2]

        bc_loss = torch.mean((u_pred - u_bc)**2 + (v_pred - v_bc)**2)
        return bc_loss

    def total_loss(self, collocation_points: Dict, boundary_points: Dict, 
                   data_points: Optional[Dict] = None) -> torch.Tensor:
        """Total PINN loss from PDF equation"""
        # Physics loss
        phys_loss = self.physics_loss(
            collocation_points['x'], 
            collocation_points['y'], 
            collocation_points['t']
        )

        # Boundary loss
        bc_loss = self.boundary_loss(
            boundary_points['x'], boundary_points['y'], boundary_points['t'],
            boundary_points['u'], boundary_points['v']
        )

        # Data loss (if available)
        data_loss = torch.tensor(0.0, device=device)
        if data_points is not None:
            inputs_data = torch.cat([data_points['x'], data_points['y'], data_points['t']], dim=1)
            outputs_data = self.forward(inputs_data)
            data_loss = torch.mean((outputs_data - data_points['values'])**2)

        total = self.w_physics * phys_loss + self.w_boundary * bc_loss + self.w_data * data_loss
        return total

class NanobotPathPlanner:
    """
    Simulated Annealing Path Planner using PINN for cost evaluation
    Implements the hybrid PINN-SA algorithm from the PDF
    """

    def __init__(self, pinn_model: PhysicsInformedNN, domain_bounds: Tuple[float, float, float, float]):
        self.pinn = pinn_model
        self.x_min, self.x_max, self.y_min, self.y_max = domain_bounds

        # SA parameters from PDF
        self.T_initial = 0.1
        self.T_min = 1e-6
        self.cooling_rate = 0.95
        self.max_iterations = 2000

    def generate_bspline_path(self, control_points: np.ndarray, num_points: int = 100) -> np.ndarray:
        """Generate smooth B-spline path from control points"""
        if len(control_points) < 4:
            return control_points

        t = np.linspace(0, 1, len(control_points))

        try:
            tck_x, _ = splrep(t, control_points[:, 0], k=3)
            tck_y, _ = splrep(t, control_points[:, 1], k=3)

            t_new = np.linspace(0, 1, num_points)
            x_new = splev(t_new, tck_x)
            y_new = splev(t_new, tck_y)

            return np.column_stack([x_new, y_new])
        except:
            # Fallback to linear interpolation
            t_new = np.linspace(0, 1, num_points)
            x_new = np.interp(t_new, t, control_points[:, 0])
            y_new = np.interp(t_new, t, control_points[:, 1])
            return np.column_stack([x_new, y_new])

    def evaluate_path_cost(self, path: np.ndarray) -> float:
        """
        Evaluate path cost using PINN dynamics simulation
        Implements multi-objective cost function from PDF
        """
        if len(path) < 2:
            return float('inf')

        x = torch.tensor(path[:, 0:1], dtype=torch.float32, device=device)
        y = torch.tensor(path[:, 1:2], dtype=torch.float32, device=device)
        t = torch.tensor(np.linspace(0, 1, len(path)).reshape(-1, 1), dtype=torch.float32, device=device)

        with torch.no_grad():
            inputs = torch.cat([x, y, t], dim=1)
            outputs = self.pinn(inputs)
            u, v, p = outputs[:, 0], outputs[:, 1], outputs[:, 2]

        # Convert back to numpy
        u_np = u.cpu().numpy()
        v_np = v.cpu().numpy()
        p_np = p.cpu().numpy()
        path_np = path

        # Multi-objective cost function components
        # J(p) = w1*Jlength + w2*Jenergy + w3*Jcollision + w4*Jsafety + w5*Jchemo

        # Path length
        path_length = np.sum(np.linalg.norm(np.diff(path_np, axis=0), axis=1))

        # Energy consumption (drag forces)
        velocity_mag = np.sqrt(u_np**2 + v_np**2)
        energy_cost = np.trapz(velocity_mag**2)

        # Collision avoidance (example obstacles)
        collision_cost = 0
        obstacles = [(0.3, 0.3, 0.1), (0.7, 0.7, 0.1)]  # (x, y, radius)

        for x_pt, y_pt in path_np:
            for ox, oy, r in obstacles:
                dist = np.sqrt((x_pt - ox)**2 + (y_pt - oy)**2)
                if dist < r:
                    collision_cost += 1000  # High penalty
                else:
                    collision_cost += np.exp(-dist/0.05)

        # Safety margins
        safety_cost = np.sum(np.exp(-velocity_mag))

        # Chemical gradient following (example)
        chemo_reward = -np.sum(path_np[:, 0] + path_np[:, 1])  # Move toward upper right

        # Weighted combination
        w1, w2, w3, w4, w5 = 1.0, 0.1, 5.0, 0.5, -0.1
        total_cost = (w1 * path_length + w2 * energy_cost + 
                     w3 * collision_cost + w4 * safety_cost + w5 * chemo_reward)

        return total_cost

    def simulated_annealing(self, start_point: np.ndarray, goal_point: np.ndarray,
                          num_control_points: int = 8) -> Tuple[np.ndarray, float, List[float]]:
        """
        Main SA optimization loop implementing PDF Algorithm
        """
        # Initialize path
        control_points = self._initialize_path(start_point, goal_point, num_control_points)
        current_path = self.generate_bspline_path(control_points)
        current_cost = self.evaluate_path_cost(current_path)

        best_control_points = control_points.copy()
        best_cost = current_cost

        temperature = self.T_initial
        cost_history = []

        for iteration in range(self.max_iterations):
            # Generate candidate path
            candidate_control_points = self._perturb_path(control_points)
            candidate_path = self.generate_bspline_path(candidate_control_points)
            candidate_cost = self.evaluate_path_cost(candidate_path)

            # Metropolis criterion
            if self._accept_candidate(current_cost, candidate_cost, temperature):
                control_points = candidate_control_points
                current_cost = candidate_cost

                if current_cost < best_cost:
                    best_control_points = control_points.copy()
                    best_cost = current_cost

            cost_history.append(current_cost)
            temperature *= self.cooling_rate

            if temperature < self.T_min:
                break

        best_path = self.generate_bspline_path(best_control_points)
        return best_path, best_cost, cost_history

    def _initialize_path(self, start: np.ndarray, goal: np.ndarray, n_points: int) -> np.ndarray:
        """Initialize random path from start to goal"""
        points = [start]
        for i in range(1, n_points - 1):
            alpha = i / (n_points - 1)
            base = (1 - alpha) * start + alpha * goal
            noise = np.random.normal(0, 0.05, 2)
            point = np.clip(base + noise, [self.x_min, self.y_min], [self.x_max, self.y_max])
            points.append(point)
        points.append(goal)
        return np.array(points)

    def _perturb_path(self, control_points: np.ndarray, strength: float = 0.02) -> np.ndarray:
        """Perturb control points for neighbor generation"""
        new_points = control_points.copy()
        for i in range(1, len(control_points) - 1):
            noise = np.random.normal(0, strength, 2)
            new_points[i] = np.clip(new_points[i] + noise, 
                                  [self.x_min, self.y_min], [self.x_max, self.y_max])
        return new_points

    def _accept_candidate(self, current_cost: float, candidate_cost: float, 
                         temperature: float) -> bool:
        """Metropolis acceptance criterion"""
        if candidate_cost < current_cost:
            return True
        prob = np.exp(-(candidate_cost - current_cost) / temperature)
        return np.random.random() < prob

def train_pinn(model: PhysicsInformedNN, num_epochs: int = 5000, lr: float = 1e-3):
    """
    Train PINN on physics and boundary conditions
    Implements training algorithm from PDF
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    # Generate training data
    n_collocation = 1000
    n_boundary = 200

    # Collocation points for physics loss
    x_col = torch.rand(n_collocation, 1, device=device) * 2 - 1  # [-1, 1]
    y_col = torch.rand(n_collocation, 1, device=device) * 2 - 1
    t_col = torch.rand(n_collocation, 1, device=device)

    # Boundary points (example: walls at x=±1, y=±1)
    x_bc = torch.cat([torch.ones(n_boundary//4, 1), -torch.ones(n_boundary//4, 1),
                      torch.rand(n_boundary//2, 1) * 2 - 1], dim=0).to(device)
    y_bc = torch.cat([torch.rand(n_boundary//4, 1) * 2 - 1, torch.rand(n_boundary//4, 1) * 2 - 1,
                      torch.ones(n_boundary//4, 1), -torch.ones(n_boundary//4, 1)], dim=0).to(device)
    t_bc = torch.rand(n_boundary, 1, device=device)
    u_bc = torch.zeros(n_boundary, 1, device=device)  # No-slip boundary
    v_bc = torch.zeros(n_boundary, 1, device=device)

    collocation_points = {'x': x_col, 'y': y_col, 't': t_col}
    boundary_points = {'x': x_bc, 'y': y_bc, 't': t_bc, 'u': u_bc, 'v': v_bc}

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        loss = model.total_loss(collocation_points, boundary_points)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    return model

# Usage example
if __name__ == "__main__":
    # Initialize PINN
    pinn = PhysicsInformedNN().to(device)

    # Train PINN
    print("Training PINN...")
    pinn = train_pinn(pinn)

    # Initialize path planner
    planner = NanobotPathPlanner(pinn, domain_bounds=(-1, 1, -1, 1))

    # Plan path
    start = np.array([-0.8, -0.8])
    goal = np.array([0.8, 0.8])

    print("Planning optimal path...")
    optimal_path, cost, history = planner.simulated_annealing(start, goal)

    print(f"Optimal path found with cost: {cost:.4f}")
    print(f"Path contains {len(optimal_path)} points")
