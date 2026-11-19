# **D**imensional **A**nnealing--based **Hy**perbolic **E**mbedding (DAHyE)

**D**imensional **A**nnealing--based **Hy**perbolic **E**mbedding (DAHyE) is a C++/Python library that combines dimensional annealing with the Barnes–Hut approximation for efficient hyperbolic embedding. Each node is represented by coordinates (x, κ, z), where x is a periodic coordinate with period N, κ is a positive hidden-degree parameter, and z is an auxiliary dimension that is gradually suppressed during annealing. The Barnes–Hut approximation accelerates the computation of repulsive forces.

## Features

- **Barnes–Hut**: O(N log N) repulsive force via a hyperbolic quadtree.

- **Exact mode**: Optional O(N²) all-pairs repulsive force for small graphs or validation.

- **Adam optimizer**: Gradient-based updates with configurable learning rate and clipping.

## Requirements

- C++17 compiler

- Python 3.6+

- [pybind11](https://github.com/pybind/pybind11) (`pip install pybind11`)

- NumPy, SciPy, and NetworkX

## Build (Python extension)

```bash
python -m pip install setuptools pybind11 numpy scipy networkx
python setup.py
```

This produces `_hyperbolic_embedding*.so` in the project root. Import the Python wrapper with:

```python
import hyperbolic_embedding
```

## Usage

Install the example dependencies, then generate a hyperbolic network:

```bash
python -m pip install numba matplotlib
python generate_hyperbolic.py
```

This generates edgelist.txt and coordinate.txt. Then, optimize the hyperbolic embedding by:

```bash
python example.py
```

## Example code

```python
import numpy as np
import hyperbolic_embedding

gs = hyperbolic_embedding.GradState()

edgelist = np.loadtxt("edgelist.txt", dtype=int, ndmin=2)
gs.set_edgelist(edgelist)
N = gs.N

gs.set_mu(0.1)     # Connection scale parameter
gs.set_beta(4.0)   # Hyperbolic model temperature
gs.set_theta(0.6)  # Barnes–Hut opening criterion

# Initialize embedding position
for i in range(N):
    x = np.random.rand() * N
    kappa = 1.0 + np.random.rand() * 4.0
    z = np.random.rand() * 2 - 1
    gs.set_pos(i, [x, kappa, z])

# Optimization loop
lr = 0.01
for epoch in range(500):
    gs.compute_repulsive_force_bh()   # or compute_repulsive_force_exact()
    gs.add_attractive_force()
    gs.step_adam(lr)

positions = gs.get_positions()
```

### Methods: hyperbolic_embedding.GradState

| Method / attribute                               | Description                                                                         |
| ------------------------------------------------ | ----------------------------------------------------------------------------------- |
| `GradState(N=1)`                                 | Create state for N nodes.                                                           |
| `set_edgelist(edges)`                            | Set edges from list of (u,v) or (E,2) numpy array.                                  |
| `init_embedding("random")`                       | Initialize (x, κ, z) randomly.                                                      |
| `set_pos(i, x, kappa, z)`                        | Set position of node i using positional arguments.                                  |
| `get_positions()`                                | Return (N, 3) NumPy array (x, κ, z).                                                |
| `set_positions(positions)`                       | Set all positions from (N, 3) array.                                                |
| `set_mu(mu)`, `set_beta(beta)`, `set_theta(theta)` | Set layout parameters.                                                              |
| `compute_repulsive_force_exact()`                | All-pairs repulsive forces.                                                         |
| `compute_repulsive_force_bh()`                   | Barnes–Hut repulsive forces (returns interaction count).                            |
| `add_attractive_force()`                         | Add attractive forces along edges.                                                  |
| `add_l1_penalty_z(penalty_z)`                    | Add L1 penalty on z: adds force -penalty_z*sign(z) to the z-component.              |
| `add_l2_penalty_z(penalty_z)`                    | Add L2 penalty on z: adds force -penalty_z*z to the z-component (pulls z toward 0). |
| `step_adam(lr)`                                  | Perform one Adam step.                                                              |
| `get_lnp_exact()`                                | Log-probability of the embedding (exact, all pairs).                                |
| `get_lnp_bh()`                                   | Log-probability of the embedding (Barnes–Hut approximation).                        |
| `get_forces()`                                   | Current forces (N x 3).                                                             |
| `project_equator()`                              | Set z = 0 for all nodes (sets the z-momentums in adam optimizer as well).           |
