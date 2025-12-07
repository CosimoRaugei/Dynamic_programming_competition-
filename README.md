# Dynamic Programming Flight Controller (Competition Winner)

Winner of the ETH Zurich **Dynamic Programming and Optimal Control** yearly programming exercise by Prof. Raffaello D'Andrea. 
methods and highlights: 
- carefully vectorized build of transition matrices and per-action costs. 
- implemented hybrid solver , fine-tuned strategy using Bayesian Optimisation to develop robust euristics to different problem istances. 



## What’s inside
- `Solver.py`: hybrid value-iteration + policy-iteration solver using sparse matrices for fast convergence.
- `ComputeTransitionProbabilities.py` and `ComputeExpectedStageCosts.py`: model dynamics and per-action costs.
- `simulation.py`: pygame renderer and step-by-step simulator for the learned policy.
- `Const.py`: single source of problem parameters and state/input spaces.
- `main.py`: entry point that builds/loads the optimal policy and can launch a simulation.



## Highlights
- **Competition-winning solution**: best-performing controller in the course challenge.
- **Efficient state pruning**: only valid obstacle configurations are enumerated, keeping the state space compact.
- **Sparse linear algebra**: transition matrices are built as CSR blocks; the hybrid solver converges quickly even with strong-flap uncertainty.
- **Reproducible setup**: environment locked in `environment.yml`; workspace caching avoids recomputation between runs.


