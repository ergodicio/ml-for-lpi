# Laser-Plasma Instability Minimization using Differentiable Simulators
This repo contains code used for gradient-based minimization of laser plasma instabilities (LPI) using ADEPT-LPSE

### The repo
The content is of 3 different categories
1. Python scripts that run `ADEPT` in an optimization loop or parameter scan
2. Configuration `yaml` files for `ADEPT`
3. Module files that extend the `ADEPT` functionality by providing parameterized inputs, loss functions, and postprocessing functions

### The physics
We solve the slowly-varying envelope approximation for modeling electron plasma waves driven at a quarter critical surface by a laser beam.

### The optimization problem
We want to minimize the LPI that occurs in a simulation. The free parameters are those that parameterize the bandwidth of the driving laser. Because our simulation is differentiable, we can take a gradient of the simulation with respect to the free parameters. 

### Generative Neural Reparameterization
Rather than find just one set of optimal bandwidth parameters, we can choose to learn a generative function that learns the distribution of optimal parameters. This method is described in `Joglekar, A. S. Generative Neural Reparameterization for Differentiable PDE-constrained Optimization. Preprint at http://arxiv.org/abs/2410.12683 (2024).` This repo provides the code for this method.

### ADEPT
`ADEPT` is a differentiable plasma physics simulation tool. It can be found at https://github.com/ergodicio/adept. This particular set of solvers is a JAX adaptation of the Laser-Plasma Simulation Environment developed at UR-LLE.