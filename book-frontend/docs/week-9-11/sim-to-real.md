---
sidebar_position: 3
title: Sim-to-Real Transfer for Physical AI
---

# Sim-to-Real Transfer for Physical AI

## Introduction to Sim-to-Real Transfer

Sim-to-Real transfer is a critical challenge in Physical AI, addressing the gap between simulation environments and real-world robotic systems. While simulation provides a safe, cost-effective, and controllable environment for developing and testing robotic algorithms, the ultimate goal is to deploy these systems in the real world. The sim-to-real transfer problem encompasses the techniques and methodologies needed to bridge this gap effectively.

### The Reality Gap

The reality gap refers to the differences between simulated and real environments that can cause algorithms trained in simulation to fail when deployed on physical robots:

- **Visual differences**: Lighting conditions, textures, and appearance variations
- **Physics discrepancies**: Friction, dynamics, and material properties
- **Sensor noise**: Real sensors have noise, latency, and imperfections
- **Actuator limitations**: Real actuators have delays, power limits, and wear
- **Environmental factors**: Temperature, humidity, and electromagnetic interference

### Importance in Physical AI

Sim-to-Real transfer is particularly important for Physical AI because:

- **Safety**: Testing dangerous scenarios in simulation first
- **Cost**: Reducing hardware experimentation costs
- **Speed**: Accelerating development cycles
- **Reproducibility**: Ensuring consistent testing conditions
- **Scalability**: Training on multiple simulated robots simultaneously

## Domain Randomization

### Concept and Implementation

Domain randomization is a technique that introduces random variations in simulation parameters to make learned policies more robust to real-world variations:

```python
import numpy as np
import random

class DomainRandomizer:
    def __init__(self):
        self.randomization_params = {
            'visual': {
                'lighting': {
                    'intensity_range': [0.5, 1.5],
                    'color_temperature_range': [5000, 8000],
                    'position_variance': [0.1, 0.1, 0.1]
                },
                'textures': {
                    'roughness_range': [0.1, 0.9],
                    'metallic_range': [0.0, 0.2],
                    'normal_map_strength_range': [0.0, 1.0]
                }
            },
            'physics': {
                'dynamics': {
                    'mass_multiplier_range': [0.8, 1.2],
                    'friction_range': [0.1, 0.9],
                    'restitution_range': [0.0, 0.3]
                },
                'actuators': {
                    'torque_range': [0.8, 1.2],
                    'delay_range': [0.0, 0.05],
                    'noise_std_range': [0.001, 0.01]
                }
            },
            'sensors': {
                'camera': {
                    'noise_std_range': [0.001, 0.01],
                    'distortion_range': [0.0, 0.1],
                    'exposure_range': [0.5, 1.5]
                },
                'imu': {
                    'noise_density_range': [1e-4, 1e-3],
                    'bias_range': [1e-5, 1e-4]
                }
            }
        }

    def randomize_visual_properties(self, light, material):
        """Randomize visual properties for domain randomization"""
        # Randomize lighting
        intensity = random.uniform(
            self.randomization_params['visual']['lighting']['intensity_range'][0],
            self.randomization_params['visual']['lighting']['intensity_range'][1]
        )
        color_temp = random.uniform(
            self.randomization_params['visual']['lighting']['color_temperature_range'][0],
            self.randomization_params['visual']['lighting']['color_temperature_range'][1]
        )

        light.set_intensity(intensity)
        light.set_color_temperature(color_temp)

        # Randomize material properties
        roughness = random.uniform(
            self.randomization_params['visual']['textures']['roughness_range'][0],
            self.randomization_params['visual']['textures']['roughness_range'][1]
        )
        metallic = random.uniform(
            self.randomization_params['visual']['textures']['metallic_range'][0],
            self.randomization_params['visual']['textures']['metallic_range'][1]
        )

        material.set_roughness(roughness)
        material.set_metallic(metallic)

    def randomize_physics_properties(self, robot):
        """Randomize physics properties for domain randomization"""
        # Randomize link masses
        for link in robot.get_links():
            mass_multiplier = random.uniform(
                self.randomization_params['physics']['dynamics']['mass_multiplier_range'][0],
                self.randomization_params['physics']['dynamics']['mass_multiplier_range'][1]
            )
            new_mass = link.get_mass() * mass_multiplier
            link.set_mass(new_mass)

        # Randomize friction coefficients
        for joint in robot.get_joints():
            friction = random.uniform(
                self.randomization_params['physics']['dynamics']['friction_range'][0],
                self.randomization_params['physics']['dynamics']['friction_range'][1]
            )
            joint.set_friction(friction)

    def randomize_sensor_properties(self, sensor):
        """Randomize sensor properties for domain randomization"""
        if sensor.type == 'camera':
            noise_std = random.uniform(
                self.randomization_params['sensors']['camera']['noise_std_range'][0],
                self.randomization_params['sensors']['camera']['noise_std_range'][1]
            )
            sensor.set_noise_std(noise_std)

        elif sensor.type == 'imu':
            noise_density = random.uniform(
                self.randomization_params['sensors']['imu']['noise_density_range'][0],
                self.randomization_params['sensors']['imu']['noise_density_range'][1]
            )
            sensor.set_noise_density(noise_density)

    def apply_randomization(self, episode_count):
        """Apply domain randomization at the beginning of each episode"""
        # Apply randomization based on episode count
        if episode_count % 10 == 0:  # Randomize every 10 episodes
            # Get all objects in the scene
            lights = self.get_all_lights()
            materials = self.get_all_materials()
            robots = self.get_all_robots()
            sensors = self.get_all_sensors()

            # Apply randomization to each object
            for light in lights:
                material = self.get_corresponding_material(light)
                self.randomize_visual_properties(light, material)

            for robot in robots:
                self.randomize_physics_properties(robot)

            for sensor in sensors:
                self.randomize_sensor_properties(sensor)
```

### Advanced Domain Randomization Techniques

#### Systematic Domain Randomization

```python
class SystematicDomainRandomization:
    def __init__(self, param_space):
        self.param_space = param_space  # Dictionary defining parameter ranges
        self.grid_points = self.create_grid()
        self.current_point = 0

    def create_grid(self):
        """Create a grid of parameter combinations"""
        from itertools import product

        # Create parameter value lists
        param_values = {}
        for param_name, param_range in self.param_space.items():
            if 'values' in param_range:
                param_values[param_name] = param_range['values']
            else:
                # Create discrete values within range
                start, end, num_points = param_range['range']
                param_values[param_name] = np.linspace(start, end, num_points)

        # Create all combinations
        param_combinations = list(product(*param_values.values()))

        # Convert back to dictionary format
        grid_points = []
        param_names = list(param_values.keys())
        for combo in param_combinations:
            point = {}
            for i, param_name in enumerate(param_names):
                point[param_name] = combo[i]
            grid_points.append(point)

        return grid_points

    def get_next_params(self):
        """Get next parameter set in systematic randomization"""
        params = self.grid_points[self.current_point]
        self.current_point = (self.current_point + 1) % len(self.grid_points)
        return params

    def randomize_systematically(self, simulation_env):
        """Apply systematic domain randomization"""
        params = self.get_next_params()

        for param_name, param_value in params.items():
            self.set_simulation_parameter(simulation_env, param_name, param_value)

    def set_simulation_parameter(self, env, param_name, value):
        """Set a specific simulation parameter"""
        if param_name == 'gravity':
            env.set_gravity([0, 0, value])
        elif param_name == 'friction':
            env.set_global_friction(value)
        elif param_name == 'restitution':
            env.set_global_restitution(value)
        # Add more parameter types as needed
```

#### Adaptive Domain Randomization

```python
class AdaptiveDomainRandomization:
    def __init__(self, initial_params, performance_threshold=0.8):
        self.current_params = initial_params.copy()
        self.performance_history = []
        self.performance_threshold = performance_threshold
        self.param_adaptation_rate = 0.1

    def update_params_based_on_performance(self, current_performance):
        """Adapt domain randomization parameters based on performance"""
        self.performance_history.append(current_performance)

        # Calculate recent performance average
        recent_avg = np.mean(self.performance_history[-10:]) if len(self.performance_history) >= 10 else 1.0

        # If performance is good, increase randomization range
        if recent_avg > self.performance_threshold:
            self.increase_randomization_range()
        else:
            # If performance is poor, decrease randomization range
            self.decrease_randomization_range()

    def increase_randomization_range(self):
        """Increase the range of randomization parameters"""
        for param_name, param_range in self.current_params.items():
            if isinstance(param_range, dict) and 'range' in param_range:
                current_range = param_range['range']
                center = (current_range[0] + current_range[1]) / 2
                width = current_range[1] - current_range[0]

                # Increase range by adaptation rate
                new_width = width * (1 + self.param_adaptation_rate)
                new_range = [center - new_width/2, center + new_width/2]

                param_range['range'] = new_range

    def decrease_randomization_range(self):
        """Decrease the range of randomization parameters"""
        for param_name, param_range in self.current_params.items():
            if isinstance(param_range, dict) and 'range' in param_range:
                current_range = param_range['range']
                center = (current_range[0] + current_range[1]) / 2
                width = current_range[1] - current_range[0]

                # Decrease range by adaptation rate (with minimum bounds)
                new_width = max(width * (1 - self.param_adaptation_rate), 0.1)
                new_range = [center - new_width/2, center + new_width/2]

                param_range['range'] = new_range
```

## System Identification

### Physics Parameter Calibration

System identification involves calibrating simulation parameters to match real-world behavior:

```python
import scipy.optimize as opt
import numpy as np

class SystemIdentifier:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.simulation = None
        self.real_data = []
        self.sim_data = []

    def collect_system_data(self, input_signal, real_robot, simulation):
        """Collect data from both real and simulated systems"""
        # Reset both systems
        real_robot.reset()
        simulation.reset()

        # Apply the same input signal to both systems
        real_positions = []
        sim_positions = []

        for t, input_t in enumerate(input_signal):
            # Apply input to real robot
            real_robot.apply_action(input_t)
            real_pos = real_robot.get_joint_positions()
            real_positions.append(real_pos)

            # Apply input to simulation
            simulation.apply_action(input_t)
            sim_pos = simulation.get_joint_positions()
            sim_positions.append(sim_pos)

        self.real_data = np.array(real_positions)
        self.sim_data = np.array(sim_positions)

    def objective_function(self, params):
        """Objective function to minimize difference between real and sim"""
        # Set simulation parameters
        self.set_simulation_params(params)

        # Run simulation with current parameters
        sim_output = self.run_simulation_with_params(params)

        # Calculate error between real and simulated outputs
        error = np.mean((self.real_data - sim_output) ** 2)
        return error

    def identify_parameters(self, initial_params, bounds):
        """Identify optimal simulation parameters"""
        result = opt.minimize(
            self.objective_function,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds
        )

        return result.x

    def set_simulation_params(self, params):
        """Set simulation parameters from optimization result"""
        # Example: set link masses, friction coefficients, etc.
        param_idx = 0

        # Set link masses
        for link in self.robot_model.links:
            if param_idx < len(params):
                link.mass = params[param_idx]
                param_idx += 1

        # Set joint friction
        for joint in self.robot_model.joints:
            if param_idx < len(params):
                joint.friction = params[param_idx]
                param_idx += 1

        # Set other parameters as needed
        if param_idx < len(params):
            self.robot_model.damping = params[param_idx]

    def run_simulation_with_params(self, params):
        """Run simulation with given parameters"""
        # Temporarily set parameters
        original_params = self.get_current_params()
        self.set_simulation_params(params)

        # Run simulation
        output = self.simulation.run()

        # Restore original parameters
        self.set_simulation_params(original_params)

        return output

    def get_current_params(self):
        """Get current simulation parameters"""
        params = []
        for link in self.robot_model.links:
            params.append(link.mass)
        for joint in self.robot_model.joints:
            params.append(joint.friction)
        params.append(self.robot_model.damping)
        return params
```

### Dynamic Parameter Estimation

```python
class DynamicParameterEstimator:
    def __init__(self, robot_dynamics):
        self.robot_dynamics = robot_dynamics
        self.parameter_history = []
        self.uncertainty_estimates = []

    def estimate_dynamics_parameters(self, joint_positions, joint_velocities,
                                   joint_torques, dt=0.001):
        """Estimate dynamics parameters using inverse dynamics"""
        # Use the equation: τ = M(q)q̈ + C(q, q̇)q̇ + g(q)
        # Rearrange to: τ - g(q) = M(q)q̈ + C(q, q̇)q̇

        # Calculate joint accelerations
        joint_accelerations = self.compute_accelerations(
            joint_velocities, dt)

        # Form the regression equation: Y * θ = τ
        Y = self.form_regression_matrix(joint_positions, joint_velocities,
                                      joint_accelerations)
        tau = joint_torques

        # Solve for parameters using least squares
        try:
            params = np.linalg.lstsq(Y, tau, rcond=None)[0]
            return params
        except np.linalg.LinAlgError:
            # Use regularized least squares if matrix is singular
            reg_param = 1e-6
            params = np.linalg.solve(Y.T @ Y + reg_param * np.eye(Y.shape[1]),
                                   Y.T @ tau)
            return params

    def form_regression_matrix(self, q, q_dot, q_ddot):
        """Form the regression matrix for dynamics identification"""
        # This is a simplified example - in practice, this would involve
        # complex dynamics calculations

        n_joints = len(q)
        Y = np.zeros((n_joints, 10))  # Example: 10 parameters

        # Fill regression matrix based on robot dynamics
        # Column 1: q_ddot (inertia terms)
        Y[:, :n_joints] = np.eye(n_joints)

        # Column 2: q_dot^2 (Coriolis terms)
        Y[:, n_joints:2*n_joints] = np.diag(q_dot**2)

        # Additional columns for gravity, friction, etc.
        # ... more complex dynamics terms would go here

        return Y

    def compute_accelerations(self, velocities, dt):
        """Compute accelerations from velocities"""
        if len(self.velocity_history) < 2:
            return np.zeros_like(velocities)

        # Use finite difference to compute acceleration
        prev_vel = self.velocity_history[-2]
        curr_vel = self.velocity_history[-1]

        acceleration = (curr_vel - prev_vel) / dt
        return acceleration

    def update_parameter_estimate(self, new_data):
        """Update parameter estimate with new data"""
        # Use recursive least squares or other adaptive estimation technique
        pass
```

## Robust Control Design

### Robust Control for Sim-to-Real Transfer

```python
import control  # python-control library
import numpy as np

class RobustController:
    def __init__(self, nominal_model, uncertainty_bounds):
        self.nominal_model = nominal_model
        self.uncertainty_bounds = uncertainty_bounds
        self.controller = None

    def design_robust_controller(self):
        """Design a robust controller using H-infinity synthesis"""
        # Define weighting functions for robust control
        W1 = self.get_performance_weighting()
        W2 = self.get_control_weighting()
        W3 = self.get_uncertainty_weighting()

        # Augment the plant with weighting functions
        augmented_plant = self.augment_plant(
            self.nominal_model, W1, W2, W3)

        # Synthesize H-infinity controller
        self.controller, _, _ = control.mixsyn(
            augmented_plant,
            sensing_outputs=[0],  # sensor outputs
            control_inputs=[0]    # control inputs
        )

        return self.controller

    def get_performance_weighting(self):
        """Define performance weighting function"""
        # Weighting for tracking performance
        # Typically low-pass filter to emphasize low frequencies
        wn = 10.0  # Natural frequency
        zeta = 0.7  # Damping ratio

        num = [1, 2*zeta*wn]
        den = [1, 2*zeta*wn, wn**2]

        return control.TransferFunction(num, den)

    def get_control_weighting(self):
        """Define control weighting function"""
        # Weighting for control effort
        # Typically high-pass filter to penalize high-frequency control
        wn = 50.0  # Natural frequency

        num = [wn]
        den = [1, wn]

        return control.TransferFunction(num, den)

    def get_uncertainty_weighting(self):
        """Define uncertainty weighting function"""
        # Weighting for model uncertainty
        # Captures expected modeling errors
        uncertainty_gain = 0.1

        num = [uncertainty_gain]
        den = [1]

        return control.TransferFunction(num, den)

    def augment_plant(self, plant, W1, W2, W3):
        """Augment plant with weighting functions for mixed-sensitivity"""
        # This creates an augmented plant for H-infinity synthesis
        # The exact implementation depends on the specific control structure

        # Simplified example - in practice, this would be more complex
        P_augmented = control.append(plant, W1, W2, W3)

        # Connect inputs and outputs appropriately
        # This is a conceptual example
        return P_augmented

class AdaptiveController:
    def __init__(self, initial_params):
        self.params = initial_params
        self.param_history = []
        self.learning_rate = 0.01

    def update_parameters(self, error, regressor):
        """Update controller parameters based on tracking error"""
        # Gradient descent parameter update
        gradient = error * regressor
        self.params -= self.learning_rate * gradient

        # Store parameter history
        self.param_history.append(self.params.copy())

    def get_control_signal(self, state, reference):
        """Get control signal using adaptive parameters"""
        # Example: simple adaptive control law
        error = reference - state
        control_signal = np.dot(self.params, np.concatenate([state, reference, error]))

        return control_signal
```

### Model Predictive Control for Robustness

```python
import cvxpy as cp
import numpy as np

class RobustMPC:
    def __init__(self, A, B, Q, R, N, state_constraints, input_constraints):
        self.A = A  # State matrix
        self.B = B  # Input matrix
        self.Q = Q  # State cost matrix
        self.R = R  # Input cost matrix
        self.N = N  # Prediction horizon
        self.state_constraints = state_constraints
        self.input_constraints = input_constraints

    def solve_robust_mpc(self, x0, reference_trajectory, model_uncertainty):
        """Solve robust MPC problem with model uncertainty"""
        # Define variables
        X = cp.Variable((self.A.shape[0], self.N+1))  # State trajectory
        U = cp.Variable((self.B.shape[1], self.N))    # Input trajectory

        # Objective function
        cost = 0
        for k in range(self.N):
            cost += cp.quad_form(X[:, k] - reference_trajectory[k], self.Q)
            cost += cp.quad_form(U[:, k], self.R)

        # Add terminal cost
        cost += cp.quad_form(X[:, self.N] - reference_trajectory[-1], self.Q)

        # Constraints
        constraints = []

        # Initial state
        constraints.append(X[:, 0] == x0)

        # Dynamics with uncertainty
        for k in range(self.N):
            # Robust constraint: worst-case over uncertainty set
            A_uncertain = self.A + model_uncertainty['A'] * np.random.uniform(-1, 1, self.A.shape)
            B_uncertain = self.B + model_uncertainty['B'] * np.random.uniform(-1, 1, self.B.shape)

            # Use scenario approach for robustness
            for scenario in range(5):  # Multiple scenarios for robustness
                A_scen = A_uncertain + np.random.normal(0, model_uncertainty['A_std'], self.A.shape)
                B_scen = B_uncertain + np.random.normal(0, model_uncertainty['B_std'], self.B.shape)

                constraints.append(X[:, k+1] == A_scen @ X[:, k] + B_scen @ U[:, k])

        # State constraints
        for k in range(self.N+1):
            constraints.append(X[:, k] >= self.state_constraints[0])
            constraints.append(X[:, k] <= self.state_constraints[1])

        # Input constraints
        for k in range(self.N):
            constraints.append(U[:, k] >= self.input_constraints[0])
            constraints.append(U[:, k] <= self.input_constraints[1])

        # Solve the problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        if problem.status not in ["infeasible", "unbounded"]:
            return U[:, 0].value  # Return first control input
        else:
            # Fallback to safe control if optimization fails
            return np.zeros(self.B.shape[1])
```

## Transfer Learning Techniques

### Domain Adaptation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainAdaptationNetwork(nn.Module):
    def __init__(self, input_dim, feature_dim, num_classes):
        super(DomainAdaptationNetwork, self).__init__()

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

        # Label classifier
        self.label_classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 2 domains: sim and real
        )

        self.grl = GradientReversalLayer()

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)

        # Label prediction
        label_pred = self.label_classifier(features)

        # Domain prediction (with gradient reversal)
        domain_features = self.grl(features, alpha)
        domain_pred = self.domain_classifier(domain_features)

        return label_pred, domain_pred

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.alpha = alpha
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

def train_domain_adaptation(model, sim_loader, real_loader, epochs=100):
    """Train model with domain adaptation"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    label_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for (sim_data, sim_labels), (real_data, _) in zip(sim_loader, real_loader):
            optimizer.zero_grad()

            # Sim data (domain label = 0)
            sim_label_pred, sim_domain_pred = model(sim_data, alpha=1.0)
            sim_label_loss = label_criterion(sim_label_pred, sim_labels)
            sim_domain_loss = domain_criterion(sim_domain_pred,
                                            torch.zeros(sim_data.size(0)).long())

            # Real data (domain label = 1)
            real_label_pred, real_domain_pred = model(real_data, alpha=1.0)
            real_domain_loss = domain_criterion(real_domain_pred,
                                             torch.ones(real_data.size(0)).long())

            # Total loss: label prediction + domain confusion
            total_loss = (sim_label_loss +
                         0.5 * (sim_domain_loss + real_domain_loss))

            total_loss.backward()
            optimizer.step()
```

### Sim-to-Real Transfer with Meta-Learning

```python
import torch
import torch.nn as nn

class MAMLLearner(nn.Module):
    """Model-Agnostic Meta-Learning for Sim-to-Real"""
    def __init__(self, model):
        super(MAMLLearner, self).__init__()
        self.model = model
        self.meta_lr = 0.001
        self.task_lr = 0.01

    def forward(self, sim_support, sim_query, real_support, real_query):
        # Inner loop: adapt to sim task
        fast_weights = self.model.parameters()

        # Forward pass on support set
        sim_support_pred = self.model.functional_forward(
            sim_support, fast_weights)
        sim_loss = F.mse_loss(sim_support_pred, sim_support_labels)

        # Compute gradients
        grads = torch.autograd.grad(sim_loss, fast_weights)

        # Update weights
        updated_weights = []
        for param, grad in zip(fast_weights, grads):
            updated_weights.append(param - self.task_lr * grad)

        # Evaluate on query set
        sim_query_pred = self.model.functional_forward(
            sim_query, updated_weights)
        sim_query_loss = F.mse_loss(sim_query_pred, sim_query_labels)

        # Also train on real data
        real_support_pred = self.model.functional_forward(
            real_support, updated_weights)
        real_loss = F.mse_loss(real_support_pred, real_support_labels)

        total_loss = sim_query_loss + real_loss
        return total_loss

class PhysicalAIController(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PhysicalAIController, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, state):
        return torch.tanh(self.network(state))

    def functional_forward(self, state, weights):
        """Forward pass with custom weights for meta-learning"""
        x = F.linear(state, weights[0], weights[1])
        x = F.relu(x)
        x = F.linear(x, weights[2], weights[3])
        x = F.relu(x)
        x = torch.tanh(F.linear(x, weights[4], weights[5]))
        return x
```

## Reality Gap Quantification

### Gap Measurement Techniques

```python
import numpy as np
from scipy import stats

class RealityGapQuantifier:
    def __init__(self):
        self.gap_metrics = {}

    def measure_visual_gap(self, sim_images, real_images):
        """Measure visual gap between simulation and reality"""
        # Calculate statistical differences
        sim_mean = np.mean(sim_images, axis=(1, 2, 3))
        real_mean = np.mean(real_images, axis=(1, 2, 3))

        sim_std = np.std(sim_images, axis=(1, 2, 3))
        real_std = np.std(real_images, axis=(1, 2, 3))

        # Compute Bhattacharyya distance
        bhattacharyya_dist = self.compute_bhattacharyya_distance(
            sim_mean, real_mean, sim_std, real_std)

        # Compute Earth Mover's Distance (Wasserstein)
        wasserstein_dist = self.compute_wasserstein_distance(
            sim_images.flatten(), real_images.flatten())

        self.gap_metrics['visual'] = {
            'bhattacharyya_distance': bhattacharyya_dist,
            'wasserstein_distance': wasserstein_dist,
            'mean_difference': np.mean(np.abs(sim_mean - real_mean))
        }

        return self.gap_metrics['visual']

    def measure_dynamics_gap(self, sim_trajectories, real_trajectories):
        """Measure dynamics gap between simulation and reality"""
        # Calculate trajectory similarity
        trajectory_distances = []

        for sim_traj, real_traj in zip(sim_trajectories, real_trajectories):
            # Dynamic Time Warping distance
            dtw_dist = self.dynamic_time_warping(sim_traj, real_traj)
            trajectory_distances.append(dtw_dist)

        # Calculate statistical measures
        gap_mean = np.mean(trajectory_distances)
        gap_std = np.std(trajectory_distances)
        gap_max = np.max(trajectory_distances)

        self.gap_metrics['dynamics'] = {
            'dtw_mean': gap_mean,
            'dtw_std': gap_std,
            'dtw_max': gap_max,
            'consistency': 1.0 - (gap_std / (gap_mean + 1e-8))
        }

        return self.gap_metrics['dynamics']

    def compute_bhattacharyya_distance(self, mean1, mean2, std1, std2):
        """Compute Bhattacharyya distance between two distributions"""
        # Simplified for 1D case
        term1 = 0.25 * np.log(0.25 * (std1**2 / std2**2 + std2**2 / std1**2 + 2))
        term2 = 0.25 * (mean1 - mean2)**2 / (std1**2 + std2**2)
        return term1 + term2

    def compute_wasserstein_distance(self, dist1, dist2):
        """Compute 1D Wasserstein (Earth Mover's) distance"""
        return stats.wasserstein_distance(dist1, dist2)

    def dynamic_time_warping(self, seq1, seq2):
        """Compute Dynamic Time Warping distance"""
        n, m = len(seq1), len(seq2)
        dtw_matrix = np.zeros((n + 1, m + 1))

        # Initialize matrix
        dtw_matrix[0, 1:] = np.inf
        dtw_matrix[1:, 0] = np.inf

        # Fill matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(seq1[i-1] - seq2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],    # insertion
                    dtw_matrix[i, j-1],    # deletion
                    dtw_matrix[i-1, j-1]   # match
                )

        return dtw_matrix[n, m]

    def assess_transfer_readiness(self):
        """Assess if system is ready for real-world deployment"""
        if 'visual' not in self.gap_metrics or 'dynamics' not in self.gap_metrics:
            return False, "Insufficient data for assessment"

        # Define thresholds for acceptable gaps
        visual_threshold = 0.5  # Adjust based on application
        dynamics_threshold = 1.0  # Adjust based on application

        visual_gap_ok = (self.gap_metrics['visual']['bhattacharyya_distance'] < visual_threshold and
                        self.gap_metrics['visual']['wasserstein_distance'] < visual_threshold)

        dynamics_gap_ok = (self.gap_metrics['dynamics']['dtw_mean'] < dynamics_threshold and
                          self.gap_metrics['dynamics']['consistency'] > 0.7)

        ready = visual_gap_ok and dynamics_gap_ok

        return ready, {
            'visual_gap_acceptable': visual_gap_ok,
            'dynamics_gap_acceptable': dynamics_gap_ok,
            'overall_readiness': ready
        }
```

## Practical Implementation Strategies

### Gradual Domain Transfer

```python
class GradualDomainTransfer:
    def __init__(self, simulation_env, real_env):
        self.sim_env = simulation_env
        self.real_env = real_env
        self.transfer_stage = 0
        self.max_stages = 5

    def increment_transfer_stage(self):
        """Move to next transfer stage with increasing realism"""
        if self.transfer_stage < self.max_stages:
            self.transfer_stage += 1
            self.update_simulation_to_stage(self.transfer_stage)

    def update_simulation_to_stage(self, stage):
        """Update simulation parameters based on transfer stage"""
        if stage == 0:  # Pure simulation
            self.sim_env.set_realistic_parameters({
                'friction': 0.0,  # Ideal conditions
                'noise': 0.0,
                'delay': 0.0
            })
        elif stage == 1:  # Basic realism
            self.sim_env.set_realistic_parameters({
                'friction': 0.1,
                'noise': 0.01,
                'delay': 0.001
            })
        elif stage == 2:  # Moderate realism
            self.sim_env.set_realistic_parameters({
                'friction': 0.2,
                'noise': 0.02,
                'delay': 0.005
            })
        elif stage == 3:  # High realism
            self.sim_env.set_realistic_parameters({
                'friction': 0.3,
                'noise': 0.05,
                'delay': 0.01
            })
        elif stage == 4:  # Near-real conditions
            self.sim_env.set_realistic_parameters({
                'friction': 0.4,
                'noise': 0.1,
                'delay': 0.02
            })
        elif stage == 5:  # Real-world deployment
            # Transition to real environment
            pass

    def evaluate_at_stage(self, policy, num_episodes=10):
        """Evaluate policy performance at current transfer stage"""
        total_reward = 0
        success_count = 0

        for episode in range(num_episodes):
            state = self.sim_env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = policy.act(state)
                next_state, reward, done, info = self.sim_env.step(action)
                episode_reward += reward
                state = next_state

            total_reward += episode_reward
            if info.get('success', False):
                success_count += 1

        avg_reward = total_reward / num_episodes
        success_rate = success_count / num_episodes

        return avg_reward, success_rate

    def execute_gradual_transfer(self, policy):
        """Execute gradual transfer process"""
        results = []

        for stage in range(self.max_stages):
            self.update_simulation_to_stage(stage)

            # Evaluate policy at current stage
            avg_reward, success_rate = self.evaluate_at_stage(policy)

            results.append({
                'stage': stage,
                'avg_reward': avg_reward,
                'success_rate': success_rate,
                'transfer_ready': success_rate > 0.8  # 80% success threshold
            })

            # If performance is good, proceed to next stage
            if success_rate > 0.8:
                self.increment_transfer_stage()
            else:
                # Retrain or adapt policy for current stage
                self.adapt_policy_to_stage(policy, stage)

        return results

    def adapt_policy_to_stage(self, policy, stage):
        """Adapt policy to current stage conditions"""
        # Fine-tune policy with current stage parameters
        # This could involve additional training or domain adaptation
        pass
```

### Simulation Fidelity Tuning

```python
class SimulationFidelityTuner:
    def __init__(self, base_simulation):
        self.sim = base_simulation
        self.fidelity_levels = {
            'low': {
                'physics': {'solver_iterations': 10, 'substeps': 1},
                'rendering': {'resolution': (320, 240), 'shadows': False},
                'sensors': {'noise': 0.05, 'update_rate': 30}
            },
            'medium': {
                'physics': {'solver_iterations': 50, 'substeps': 2},
                'rendering': {'resolution': (640, 480), 'shadows': True},
                'sensors': {'noise': 0.02, 'update_rate': 60}
            },
            'high': {
                'physics': {'solver_iterations': 100, 'substeps': 4},
                'rendering': {'resolution': (1280, 720), 'shadows': True},
                'sensors': {'noise': 0.01, 'update_rate': 100}
            }
        }

    def set_fidelity_level(self, level):
        """Set simulation fidelity level"""
        if level in self.fidelity_levels:
            params = self.fidelity_levels[level]

            # Update physics parameters
            self.sim.set_physics_params(
                solver_iterations=params['physics']['solver_iterations'],
                substeps=params['physics']['substeps']
            )

            # Update rendering parameters
            self.sim.set_rendering_params(
                resolution=params['rendering']['resolution'],
                shadows=params['rendering']['shadows']
            )

            # Update sensor parameters
            self.sim.set_sensor_params(
                noise_std=params['sensors']['noise'],
                update_rate=params['sensors']['update_rate']
            )

    def adaptive_fidelity_control(self, performance_monitor):
        """Adjust fidelity based on performance requirements"""
        current_performance = performance_monitor.get_current_performance()
        target_performance = performance_monitor.get_target_performance()

        if current_performance < target_performance * 0.9:  # Below 90% of target
            # Increase fidelity
            current_level = self.get_current_fidelity_level()
            if current_level == 'low':
                self.set_fidelity_level('medium')
            elif current_level == 'medium':
                self.set_fidelity_level('high')
        elif current_performance > target_performance * 1.1:  # Above 110% of target
            # Decrease fidelity to save computational resources
            current_level = self.get_current_fidelity_level()
            if current_level == 'high':
                self.set_fidelity_level('medium')
            elif current_level == 'medium':
                self.set_fidelity_level('low')

    def get_current_fidelity_level(self):
        """Determine current fidelity level based on parameters"""
        # This would compare current parameters to known levels
        # Simplified implementation
        if self.sim.get_physics_param('solver_iterations') >= 100:
            return 'high'
        elif self.sim.get_physics_param('solver_iterations') >= 50:
            return 'medium'
        else:
            return 'low'
```

## Evaluation and Validation

### Transfer Success Metrics

```python
class TransferEvaluator:
    def __init__(self):
        self.metrics = {}

    def evaluate_transfer_success(self, sim_policy_performance, real_policy_performance):
        """Evaluate the success of sim-to-real transfer"""
        # Calculate transfer efficiency
        transfer_efficiency = real_policy_performance / sim_policy_performance

        # Calculate performance drop
        performance_drop = sim_policy_performance - real_policy_performance

        # Calculate success metrics
        self.metrics = {
            'transfer_efficiency': transfer_efficiency,
            'performance_drop': performance_drop,
            'transfer_ratio': real_policy_performance / sim_policy_performance if sim_policy_performance > 0 else 0,
            'success_threshold_met': real_policy_performance > 0.7 * sim_policy_performance  # 70% threshold
        }

        return self.metrics

    def validate_safety_in_real_world(self, policy, real_env, num_trials=100):
        """Validate safety of transferred policy in real world"""
        safe_executions = 0
        total_trials = num_trials

        for trial in range(num_trials):
            state = real_env.reset()
            done = False
            is_safe = True

            while not done:
                action = policy.act(state)

                # Check for safety constraints before executing
                if self.check_safety_constraints(state, action, real_env):
                    next_state, reward, done, info = real_env.step(action)
                    state = next_state
                else:
                    # Safety violation - mark as unsafe
                    is_safe = False
                    break

            if is_safe:
                safe_executions += 1

        safety_rate = safe_executions / total_trials
        self.metrics['safety_validation'] = {
            'safety_rate': safety_rate,
            'safe_trials': safe_executions,
            'total_trials': total_trials,
            'safety_passed': safety_rate > 0.95  # 95% safety threshold
        }

        return self.metrics['safety_validation']

    def check_safety_constraints(self, state, action, env):
        """Check if action violates safety constraints"""
        # This would implement specific safety checks for the robot
        # Example checks:
        joint_limits_ok = self.check_joint_limits(state, action, env)
        collision_free = self.check_collision_avoidance(state, action, env)
        velocity_limits_ok = self.check_velocity_limits(state, action, env)

        return joint_limits_ok and collision_free and velocity_limits_ok

    def check_joint_limits(self, state, action, env):
        """Check if action respects joint limits"""
        # Implementation would depend on specific robot
        return True  # Placeholder

    def check_collision_avoidance(self, state, action, env):
        """Check if action leads to collision"""
        # Implementation would depend on specific robot and environment
        return True  # Placeholder

    def check_velocity_limits(self, state, action, env):
        """Check if action respects velocity limits"""
        # Implementation would depend on specific robot
        return True  # Placeholder
```

## Best Practices and Guidelines

### Systematic Approach to Sim-to-Real Transfer

1. **Characterize the Reality Gap**: Quantify differences between simulation and reality
2. **Start Simple**: Begin with basic tasks and gradually increase complexity
3. **Use Domain Randomization**: Make policies robust to variations
4. **Validate Safety First**: Ensure safe operation before performance optimization
5. **Iterative Refinement**: Continuously improve based on real-world feedback
6. **Monitor Performance**: Track metrics to detect degradation over time

### Common Pitfalls to Avoid

- **Overfitting to Simulation**: Policies that work only in specific simulation conditions
- **Ignoring Sensor Limitations**: Not accounting for real sensor noise and delays
- **Insufficient Randomization**: Randomizing too few parameters
- **Skipping Safety Validation**: Deploying without proper safety checks
- **No Performance Monitoring**: Not tracking real-world performance

## Summary

Sim-to-Real transfer is a critical component of Physical AI development, enabling the safe and efficient deployment of robotic systems. By using techniques like domain randomization, system identification, robust control, and careful evaluation, we can bridge the gap between simulation and reality. The key is to develop policies that are not only effective in simulation but also robust and safe when deployed on physical robots.

In the next module, we'll explore Vision-Language-Action (VLA) systems that integrate perception, language understanding, and physical action for more sophisticated Physical AI applications.