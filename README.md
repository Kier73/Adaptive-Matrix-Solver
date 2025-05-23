# Adaptive Matrix Solver (MAUS)

MAUS (Meta-heuristic Adaptive Universal Solver) is a Python-based solver designed for tackling various complex matrix problems. It employs a population of "solution candidates," each acting as an autonomous agent, to explore the solution space. The system adaptively adjusts its strategy based on the problem's characteristics and the candidates' progress.

## Key Features:

-   **Problem Types:**
    -   Eigenvalue Problems (Ax = λx)
    -   Linear Systems (Ax = b)
    -   Singular Value Decomposition (SVD)
-   **Adaptive Core:**
    -   **Population-Based:** Uses a dynamic population of solution candidates.
    -   **Psi Regularization:** Applies adaptive regularization (Ψ-Regularization) to handle ill-conditioned or singular matrices, dynamically adjusting its strength.
    -   **Dynamic Solver Choice:** The internal `InverseIterateSolver` can switch between direct dense/sparse solvers and iterative methods (GMRES) based on problem characteristics and candidate state.
    -   **Candidate Self-Adaptation:** Each candidate adapts its own step size (`alpha_local_step`) and can be retired or reinitialized if stuck.
    -   **Global Strategy Adaptation:** MAUS monitors overall progress (landscape energy, average stuckness) and adjusts global parameters like Psi aggression, candidate spawn rates, and convergence thresholds.
-   **Complex Matrix Optimizations:**
    -   **Hermitian Eigenvalue Problems:** Detects Hermitian matrices for eigenvalue problems and utilizes specialized, more efficient solvers (`scipy.linalg.eigh` for dense, `scipy.sparse.linalg.eigsh` for sparse) where appropriate.
    -   **GMRES Preconditioning:** For iterative GMRES solves, a diagonal (Jacobi) preconditioner is applied if a candidate appears to be stuck, aiming to improve convergence for ill-conditioned systems.
-   **Sparse Matrix Handling:**
    -   Automatically detects and converts matrices to sparse formats (CSC) for relevant internal computations if sparsity is beneficial.
    -   Utilizes SciPy's sparse linear algebra routines.

## Approach:

The solver can be described as a "brute force" exploration guided by meta-heuristics. It doesn't rely on a single deterministic algorithm but rather on a cooperative and competitive multi-agent system that learns and adapts to the problem's landscape.

## How to Use (Example):

(This section would typically include a code snippet showing how to initialize and run the solver for a sample problem. For now, please refer to the test scenarios at the end of `Adaptive_Matrix_Solver_0.1.py`.)
