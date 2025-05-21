import numpy as np
import scipy.linalg as sla # For dense linear algebra (e.g., np.linalg.solve)
import scipy.sparse as sp # For sparse matrix objects
import scipy.sparse.linalg as spla # For sparse linear algebra solvers (e.g., GMRES, sparse SVD)
import cmath
from enum import Enum
import random

# --- Enumerations for Problem Types ---
class ProblemType(Enum):
    EIGENVALUE = 1
    SOLVE_LINEAR_SYSTEM = 2
    SVD = 3 

# --- Global Configuration Parameters (Informing MAUS's Heuristics) ---
# These are adjustable hyperparameters that MAUS uses in its internal decision-making.
GLOBAL_DEFAULT_PSI_EPSILON_BASE = np.complex128(1e-20) # Base regularization magnitude (multiplied by aggression factor)
GLOBAL_DEFAULT_ALPHA_V_INITIAL = np.complex128(0.01) # Initial learning rate for candidates' steps
GLOBAL_MAX_PSI_ATTEMPTS = 25 # Max attempts for InverseIterateSolver per candidate update
GLOBAL_MAX_STUCK_FOR_RETIREMENT = 8 # Times a candidate can repeatedly fail before being retired (population management)

GLOBAL_MIN_WEIGHT_TO_SURVIVE_PRUNE = 1e-10 # Minimum confidence weight for a candidate to stay in population
GLOBAL_VECTOR_SIMILARITY_TOL = 0.999 # Cosine similarity threshold for considering two vectors "the same"
GLOBAL_LAMBDA_SIMILARITY_TOL = 1e-5 # Absolute difference for eigenvalue uniqueness
GLOBAL_SIGMA_SIMILARITY_TOL_ABS = 1e-6 # Absolute threshold for singular value uniqueness (below this, it's considered ~0)
GLOBAL_SIGMA_SIMILARITY_TOL_REL = 1e-4 # Relative threshold for singular value uniqueness (e.g., 1e-4 of max sigma)

# --- InverseIterateSolver: Adaptive Local Solver for Ax=b-like Problems ---
# This class encapsulates the robust linear system solving using direct_solve or iterative_gmres,
# dynamically choosing or falling back, and applying Ψ regularization.
class InverseIterateSolver:
    def __init__(self, N, base_psi_epsilon, max_attempts, preferred_method='direct_solve', is_sparse=False):
        self.N = N # Dimension of the matrix for solve
        self.base_psi_epsilon = base_psi_epsilon # Base magnitude for Ψ
        self.max_attempts = max_attempts # Max internal retries for a single solve call
        self.preferred_method = preferred_method # 'direct_solve' (sla.solve or spsolve) or 'iterative_gmres'
        self.fallback_method = 'iterative_gmres' if preferred_method == 'direct_solve' else 'direct_solve' # Auto-determines fallback
        self.is_sparse = is_sparse # True if problem is sparse

    def solve(self, A_target, b_rhs, candidate_stuck_counter):
        """
        Attempts to solve A_target @ x = b_rhs robustly with Psi regularization,
        potentially trying fallback solvers.
        """
        num_psi_attempts = 0
        current_method_for_try = self.preferred_method # Start with preferred method

        while num_psi_attempts < self.max_attempts:
            # Scale PSI by base and attempt count for increasing aggression based on history
            psi_scalar_magnitude = self.base_psi_epsilon * (10**(num_psi_attempts / 2.0)) * (10**(candidate_stuck_counter / 3.0)) 
            
            # Create regularization term (Psi): dynamically chooses sparse identity or dense random matrix
            if self.is_sparse:
                regularization_term = sp.identity(self.N, dtype=A_target.dtype, format='csc') * psi_scalar_magnitude 
                # Note: `A_target` should already be in a sparse format for addition.
            else: # Dense matrix: adds random noise component to Psi
                random_perturb = (np.random.rand(self.N, self.N) - 0.5 + 1j * (np.random.rand(self.N, self.N) - 0.5)) * psi_scalar_magnitude * 0.15 
                regularization_term = psi_scalar_magnitude * np.eye(self.N, dtype=A_target.dtype) + random_perturb 

            # Add regularization to the target matrix for solving
            H_solve = A_target + regularization_term
            
            try:
                # Core solving logic based on `current_method_for_try`
                if current_method_for_try == 'direct_solve':
                    if self.is_sparse:
                        result_vec = spla.spsolve(H_solve.tocsc(), b_rhs) # scipy.sparse.linalg.spsolve for sparse direct solve
                    else:
                        result_vec = sla.solve(H_solve, b_rhs, assume_a='general') # np.linalg.solve for dense direct solve

                elif current_method_for_try == 'iterative_gmres':
                    # GMRES (Generalized Minimal Residual): robust for non-symmetric systems, can handle near-singularity by finding least-squares sol.
                    # It accepts both dense NumPy arrays and sparse SciPy matrices.
                    # x0: initial guess for solution. tol: relative tolerance. maxiter: max iterations.
                    x0_init = b_rhs if b_rhs.shape == H_solve.shape[1:] else np.zeros_like(b_rhs) # Use RHS as initial guess or zeros
                    result_vec, info = spla.gmres(H_solve, b_rhs, x0=x0_init, tol=1e-8, maxiter=50) 
                    if info != 0: raise np.linalg.LinAlgError(f"GMRES did not converge cleanly (info={info}).")

                else:
                    raise ValueError(f"Unknown solver method: {current_method_for_try}")

                if not np.all(np.isfinite(result_vec)): # Critical check for NaN/Inf in result
                    raise ValueError("Solution vector not finite after solve.")
                
                return result_vec, num_psi_attempts # Successful solve and number of attempts

            except (np.linalg.LinAlgError, ValueError, TypeError) as e:
                # If solve fails: first time, switch to fallback method. Subsequent times, retry with stronger Psi.
                if current_method_for_try == self.preferred_method and self.preferred_method != self.fallback_method and num_psi_attempts == 0:
                    current_method_for_try = self.fallback_method # Switch once to fallback on first failure
                    num_psi_attempts = 0 # Reset PSI attempts for the newly chosen solver
                    continue 
                
                num_psi_attempts += 1 

        # If all attempts exhausted without success
        raise RuntimeError(f"InverseIterateSolver failed all {self.max_attempts} attempts for {self.preferred_method} and {self.fallback_method}.")


# --- Solution Candidate Class (Represents a single hypothesis/solution candidate) ---
# Each candidate is an autonomous agent making local progress based on MAUS's global strategy.
class SolutionCandidate:
    _candidate_id_counter = 0
    # Internal states define candidate behavior
    class State(Enum):
        EXPLORING = 1  # In search phase, might take larger/randomized steps
        REFINING = 2   # Has found a promising region, focusing on tighter convergence
        STUCK = 3      # Repeatedly failed or diverged locally. Needs intervention or retirement.
        CONVERGED = 4  # Has met convergence criteria
        RETIRED = 5    # Has been pruned from population due to redundancy or persistent failure

    def __init__(self, problem_matrix, problem_type, N_diag, initial_lambda=None, initial_v=None, initial_x=None, initial_u=None, initial_sigma=None, initial_weight=0.01):
        self.id = SolutionCandidate._candidate_id_counter
        SolutionCandidate._candidate_id_counter += 1

        self.N_diag = N_diag # Dimension for square operations (e.g., N for Eigen)
        self.M_rows, self.M_cols = problem_matrix.shape # Actual dimensions of input matrix
        self.problem_type = problem_type
        self.problem_matrix = problem_matrix 
        self.b_vector = None 
        
        # Solution parameters (type-specific containers)
        self.lambda_k = initial_lambda 
        self.v_k = initial_v        
        self.x_k = initial_x        
        self.sigma_k = initial_sigma 
        self.u_k = initial_u         
        self.right_v_k = initial_v 

        # Candidate State and confidence tracking
        self.state = SolutionCandidate.State.EXPLORING # Initial state
        self.w_k = initial_weight # Confidence/weight
        self.residual_k = float('inf') # Current residual (lower is better)
        self.prev_residual = float('inf') # Residual from previous step (for adaptation)
        self.alpha_local_step = GLOBAL_DEFAULT_ALPHA_V_INITIAL # Individual step size for reactive gradient

        self.stuck_counter = 0 # Counts how many consecutive times the candidate needed brute-force intervention
        self.local_psi_retries_needed = 0 # Records retries needed by InverseIterateSolver for last update
        self.num_resets = 0 # Counts total times its internal state was randomly re-initialized due to failures

        # History (for debugging and learning over time)
        self.param_history = []
        self.residual_history = []

        self.initialize_random_solution() # Set initial state of solution parameters


    def initialize_random_solution(self):
        # Helper to create a random normalized complex vector
        rand_vec_init = lambda N: (np.random.rand(N) + 1j * np.random.rand(N)).astype(np.complex128)
        norm_rand_vec = lambda v: v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-10 else rand_vec_init(v.shape[0]) / np.linalg.norm(rand_vec_init(v.shape[0]))

        if self.problem_type == ProblemType.EIGENVALUE:
            self.v_k = norm_rand_vec(rand_vec_init(self.N_diag))
            self.lambda_k = (random.random() * 5 - 2.5 + 1j * (random.random() * 5 - 2.5)) # Random complex lambda
            
        elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
            self.x_k = norm_rand_vec(rand_vec_init(self.N_diag)) * random.uniform(0.1, 10.0) # Random magnitude initial solution
            
        elif self.problem_type == ProblemType.SVD:
            self.u_k = norm_rand_vec(rand_vec_init(self.M_rows))
            self.right_v_k = norm_rand_vec(rand_vec_init(self.M_cols))
            self.sigma_k = 1.0 

        # Store initial (possibly inf) residual and solution for history.
        self.param_history.append(self.get_current_solution_params()) 
        self.residual_history.append(self.residual_k) 

    def update_solution_step(self, current_matrix_A, b_vector=None, strat_params=None, global_knowledge=None):
        self.b_vector = b_vector # `b` vector for Ax=b problems
        self.prev_residual = self.residual_k # Store last residual for step-size adaptation

        # Parameters for local solver instance from global strategy & knowledge
        overall_psi_aggression_factor = strat_params.get('overall_psi_aggression_factor', 1.0)
        max_psi_retries_global = strat_params.get('max_psi_retries', GLOBAL_MAX_PSI_ATTEMPTS)
        local_solver_preference = global_knowledge.get('local_solver_preference', 'direct_solve') # 'direct_solve' or 'iterative_gmres'
        is_matrix_sparse = global_knowledge.get('is_sparse_problem', False)

        solver_instance = InverseIterateSolver(self.N_diag, GLOBAL_DEFAULT_PSI_EPSILON_BASE * overall_psi_aggression_factor, 
                                                max_psi_retries_global, local_solver_preference, is_matrix_sparse)
        
        # Branch based on problem type for specific update logic
        if self.problem_type == ProblemType.SVD: 
            try:
                # SVD works via alternating matrix-vector products (like power method variants).
                # Psi's aggressive component is handled by `current_psi_mag` but the main logic is different from direct solves.
                # If a vector's norm is tiny, we might add noise or reinitialize.
                if np.linalg.norm(self.right_v_k) < 1e-10:
                    self.right_v_k = (np.random.rand(self.M_cols) + 1j * np.random.rand(self.M_cols)); self.right_v_k /= np.linalg.norm(self.right_v_k)
                    self.stuck_counter += 1; self.num_resets += 1;
                    raise ValueError("SVD right_v_k collapsed. Reinitializing.") 
                
                temp_u_k = current_matrix_A @ self.right_v_k
                self.sigma_k = np.linalg.norm(temp_u_k) # Best singular value estimate
                self.u_k = temp_u_k / (self.sigma_k if self.sigma_k > 1e-10 else 1.0) # Normalize `u`

                if np.linalg.norm(self.u_k) < 1e-10: # Check if u also collapsed (potential error propagation)
                     self.u_k = (np.random.rand(self.M_rows)+1j*np.random.rand(self.M_rows)); self.u_k /= np.linalg.norm(self.u_k)
                     self.stuck_counter += 1; self.num_resets += 1;
                     raise ValueError("SVD u_k collapsed. Reinitializing.")

                temp_v_k = current_matrix_A.conj().T @ self.u_k
                self.sigma_k = max(self.sigma_k, np.linalg.norm(temp_v_k)) # Take the maximum sigma from both updates
                self.right_v_k = temp_v_k / (np.linalg.norm(temp_v_k) if np.linalg.norm(temp_v_k) > 1e-10 else 1.0)

                # Small sigma might indicate convergence to zero singular value, not necessarily a failure.
                if self.sigma_k < GLOBAL_SIGMA_SIMILARITY_TOL_ABS / 100 : # If sigma is tiny even lower than default (often implies it's really 0)
                    self.residual_k = strat_params.get('current_convergence_threshold', 1e-6) * 0.1 # Set very small residual to acknowledge "convergence to zero sigma"
                    self.state = SolutionCandidate.State.CONVERGED # It found a very small sigma and solved for it
                    self.stuck_counter = 0 # No longer stuck
                    # Ensure u and v are well-defined for downstream usage if sigma is zero
                    if np.linalg.norm(self.u_k) < 1e-10: self.u_k = np.ones(self.M_rows, dtype=np.complex128)/np.sqrt(self.M_rows)
                    if np.linalg.norm(self.right_v_k) < 1e-10: self.right_v_k = np.ones(self.M_cols, dtype=np.complex128)/np.sqrt(self.M_cols)

                else: # Otherwise, standard processing
                    self.stuck_counter = max(0, self.stuck_counter - 1) # Reduce stuck counter on successful SVD step

            except (RuntimeError, ValueError, np.linalg.LinAlgError) as e: # Catch any errors from SVD path or its internal vector normalization
                self.stuck_counter += 1; self.w_k *= 0.001; self.alpha_local_step *= 0.5 
                self.state = SolutionCandidate.State.STUCK 
                if self.stuck_counter >= GLOBAL_MAX_STUCK_FOR_RETIREMENT: self.state = SolutionCandidate.State.RETIRED
                # If SVD method explicitly threw error, re-randomize for brute-force exploration
                self.u_k = (np.random.rand(self.M_rows)+1j*np.random.rand(self.M_rows))/np.sqrt(self.M_rows)
                self.right_v_k = (np.random.rand(self.M_cols)+1j*np.random.rand(self.M_cols))/np.sqrt(self.M_cols)
                self.sigma_k = 1.0 
        
        # --- Common update block for Eigenvalue and SolveLinearSystem problems (using InverseIterateSolver) ---
        else: 
            target_A_for_solve = current_matrix_A
            rhs_for_solve = None 
            current_main_vec_ref = None 

            if self.problem_type == ProblemType.EIGENVALUE:
                if np.linalg.norm(self.v_k) < 1e-10: 
                    self.v_k = (np.random.rand(self.N_diag) + 1j*np.random.rand(self.N_diag)); self.v_k /= np.linalg.norm(self.v_k)
                    self.stuck_counter += 1; self.num_resets += 1;
                    raise ValueError("Eigenvector collapsed. Reinitializing random vector for a fresh start.") # This restarts `try` block, potentially with a new `Psi`
                self.lambda_k = (self.v_k.conj().T @ current_matrix_A @ self.v_k) / (self.v_k.conj().T @ self.v_k) # Reactive lambda update
                target_A_for_solve = current_matrix_A - self.lambda_k * np.eye(self.N_diag, dtype=current_matrix_A.dtype)
                rhs_for_solve = self.v_k # `v_k` serves as the right-hand side for (A-λI)z = v
                current_main_vec_ref = self.v_k 

            elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
                target_A_for_solve = current_matrix_A
                rhs_for_solve = self.b_vector
                current_main_vec_ref = self.x_k 

            try:
                new_vec_raw, self.local_psi_retries_needed = solver_instance.solve(target_A_for_solve, rhs_for_solve, self.stuck_counter)
                
                # Apply alpha_local_step for controlled blend/step. This prevents overshooting and aids stability.
                if self.problem_type == ProblemType.EIGENVALUE:
                    # Blends the old `v_k` with the `new_vec_raw` in the direction of the solution
                    self.v_k = (1.0 - self.alpha_local_step) * self.v_k + self.alpha_local_step * new_vec_raw 
                    self.v_k /= np.linalg.norm(self.v_k) if np.linalg.norm(self.v_k) > 1e-10 else (np.random.rand(self.N_diag)+1j*np.random.rand(self.N_diag))/np.sqrt(self.N_diag) # Normalize and protect against 0-norm
                elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
                    # For Ax=b, `new_vec_raw` IS the candidate `X`. Alpha blends current `x_k` with this newly calculated `X`.
                    self.x_k = (1.0 - self.alpha_local_step) * current_main_vec_ref + self.alpha_local_step * new_vec_raw 

                self.stuck_counter = max(0, self.stuck_counter - 1) # Success means reduction in stuckness

            except (RuntimeError, ValueError) as e: # Catch InverseIterateSolver failure (ran out of PSI/solver types)
                self.stuck_counter += 1
                self.w_k *= 0.001 # Penalize candidate weight
                self.alpha_local_step = max(self.alpha_local_step * 0.5, 1e-6) # Aggressive step size reduction
                if self.stuck_counter >= GLOBAL_MAX_STUCK_FOR_RETIREMENT: # Candidate failed too many times, retire it
                    self.state = SolutionCandidate.State.RETIRED 
                    self.num_resets += 1 # Count how many were completely reset and retired
                else: # Otherwise, mark as stuck for now and retry with random state next
                    self.state = SolutionCandidate.State.STUCK 
                    self.initialize_random_solution() # Reset vector/params, retaining `stuck_counter`


        # --- Common Residual Calculation & History Logging (Regardless of previous branch) ---
        A = self.problem_matrix # Get the current (potentially updated, e.g., dynamic A(t)) matrix for residual calc
        if self.problem_type == ProblemType.EIGENVALUE:
            self.residual_k = np.linalg.norm(A @ self.v_k - self.lambda_k * self.v_k)
        elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
            self.residual_k = np.linalg.norm(A @ self.x_k - self.b_vector)
        # SVD residual is calculated in its update path; just verify it's not infinite now
        elif self.problem_type == ProblemType.SVD:
            self.residual_k = np.linalg.norm(A @ self.right_v_k - self.sigma_k * self.u_k) + \
                              np.linalg.norm(A.conj().T @ self.u_k - self.sigma_k * self.right_v_k)

        # Append to history, now guaranteed to have all parameters from whichever branch
        self.param_history.append(self.get_current_solution_params())
        self.residual_history.append(self.residual_k)

        # Adaptive alpha_local_step and candidate State Transition (Common)
        if self.prev_residual > 1e-10: 
            if self.residual_k < self.prev_residual * 0.9: # Significant improvement (reward)
                self.alpha_local_step = min(self.alpha_local_step * 1.1, 1.0)
                if self.state != SolutionCandidate.State.CONVERGED: self.state = SolutionCandidate.State.REFINING
            elif self.residual_k > self.prev_residual * 1.5 and self.prev_residual > 1e-5: # Diverging (only if had prior good value)
                self.alpha_local_step = max(self.alpha_local_step * 0.5, 1e-6) # Dampen
                if self.state != SolutionCandidate.State.CONVERGED: self.state = SolutionCandidate.State.STUCK 
            else: # Stagnant or minor progress (decay learning rate)
                self.alpha_local_step = max(self.alpha_local_step * 0.95, 1e-6) 
                if self.state not in [SolutionCandidate.State.CONVERGED, SolutionCandidate.State.STUCK, SolutionCandidate.State.RETIRED]:
                     self.state = SolutionCandidate.State.EXPLORING # Continue searching

        # Final Convergence check (can switch state to CONVERGED)
        if self.residual_k < strat_params.get('current_convergence_threshold', GLOBAL_CONVERGENCE_RESIDUAL_TOL) and \
           np.all(np.isfinite(self.get_current_solution_params()[-1])): # Final check for numerical stability of result
            self.state = SolutionCandidate.State.CONVERGED
            self.w_k = 1.0 # Max confidence for converged solutions
            self.stuck_counter = 0 # Reset
            self.alpha_local_step = 0.0 # Halt stepping for converged solutions

    def get_current_solution_params(self):
        # Returns the relevant solution parameters as a tuple
        if self.problem_type == ProblemType.EIGENVALUE: return (self.lambda_k, self.v_k)
        elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM: return (self.x_k,)
        elif self.problem_type == ProblemType.SVD: return (self.sigma_k, self.u_k, self.right_v_k)
        return None 

# --- MAUS: The Universal Adaptive Matrix Solver (Main Class) ---
class MAUS_Solver:
    def __init__(self, problem_matrix, problem_type, b_vector=None, initial_num_candidates=None, global_convergence_tol=1e-8):
        # Initialize matrix: Convert to sparse if needed, else to dense complex.
        if isinstance(problem_matrix, (sp.spmatrix,)):
            self.M = problem_matrix.copy() 
        else:
            self.M = problem_matrix.astype(np.complex128) 

        self.N_rows, self.N_cols = self.M.shape 
        self.N_diag = self.N_rows # General diagonal dimension placeholder

        self.problem_type = problem_type
        self.b = b_vector.astype(np.complex128) if b_vector is not None else None 

        # Initial problem diagnosis to set up `problem_knowledge`
        self.diag_info = self._diagnose_matrix_initial(self.M) 
        self.is_sparse_problem_init = self.diag_info['is_sparse_init'] # Is initial matrix sparse (from initial threshold)?
        self.cond_number = self.diag_info['condition_number'] # Initial condition number for dense matrix

        # MAUS's internal "Cognitive State" for the problem: informs all strategy decisions
        self.problem_knowledge = {
            'matrix_type': 'Dense', # Becomes 'Sparse' if converted.
            'spectrum_hint': 'Unknown', 
            'numerical_stability_state': 'Stable', # 'Stable', 'Fragile', 'Critical'
            'local_solver_preference': 'direct_solve', # Local solver mode: 'direct_solve' or 'iterative_gmres'
            'effective_rank_SVD': min(self.N_rows, self.N_cols), # SVD rank, estimated dynamically
            'true_matrix_is_singular': self.diag_info['is_singular'], # If initial dense matrix is truly singular
            'is_sparse_problem': self.is_sparse_problem_init # Track actual internal state for sparse solving
        }

        # Convert M to a usable sparse format (CSC for solves) if deemed sparse enough OR is already sparse obj.
        if self.problem_knowledge['is_sparse_problem'] and not isinstance(self.M, (sp.csc_matrix, sp.csr_matrix, sp.coo_matrix)):
             self.M = sp.csc_matrix(self.M) 
             print(f"  MAUS converting input matrix to sparse CSC format for efficient compute.")
             self.problem_knowledge['matrix_type'] = 'Sparse' # Update cognitive state
        
        # Adaptive strategy parameters: These are dynamically tuned by MAUS
        self.strat_params = {
            'overall_psi_aggression_factor': 1.0, 
            'max_psi_retries': GLOBAL_MAX_PSI_ATTEMPTS, 
            'min_survival_weight': GLOBAL_MIN_WEIGHT_TO_SURVIVE_PRUNE, 
            'spawn_rate_multiplier': 1.0, 
            'convergence_tolerance': global_convergence_tol, 
            'current_convergence_threshold': 1.0, # This threshold changes based on overall progress
        }
        
        self._set_initial_strategy() # Sets up initial strategy based on matrix diagnosis
        
        # Candidate population initialization
        initial_num_candidates = initial_num_candidates if initial_num_candidates is not None else (self.N_diag * 3)
        if self.problem_type == ProblemType.SVD:
            initial_num_candidates = max(initial_num_candidates, min(self.N_rows, self.N_cols) * 3) 

        self.candidates = []
        for _ in range(initial_num_candidates):
            self.candidates.append(SolutionCandidate(self.M, self.problem_type, self.N_diag)) # Pass sparse matrix
        
        SolutionCandidate._candidate_id_counter = initial_num_candidates 
        print(f"MAUS Initialized with {initial_num_candidates} candidates for {self.problem_type.name} (Dims={self.N_rows}x{self.N_cols}).")
        print(f"Initial matrix diagnostics: Cond={self.cond_number:.2e}, MatrixType={self.problem_knowledge['matrix_type']}. Initial MAUS Knowledge: {self.problem_knowledge['numerical_stability_state']}.")

        # Global metrics for MAUS's internal awareness of overall problem state
        self.landscape_energy = 1.0 # Global objective: minimize this
        self.avg_residual = 1.0
        self.avg_stuckness = 0.0
        self.num_distinct_converged_solutions = 0
        self.converged_solutions = [] # Stores final unique converged solutions found


    def _diagnose_matrix_initial(self, matrix):
        """Initial static diagnosis of the matrix at MAUS initialization."""
        is_sparse_init = False
        if isinstance(matrix, (np.ndarray,)): # Check if a dense NumPy array is sparse enough for conversion
            is_sparse_init = np.count_nonzero(matrix) < 0.25 * matrix.size 
        elif isinstance(matrix, (sp.spmatrix,)): # Check if it's already a SciPy sparse matrix object
            is_sparse_init = True 

        cond_num = np.inf
        matrix_is_singular = False
        try: 
            # Condition number check, but only if matrix is not initially flagged as sparse (costly for large N)
            if not is_sparse_init: 
                cond_num = np.linalg.cond(matrix)
                if np.isinf(cond_num) or cond_num > 1e15: matrix_is_singular = True
            else: # For sparse matrix, assume condition number from behavior, or specialized norms.
                pass 
        except np.linalg.LinAlgError: # Catches errors during condition calculation itself
            cond_num = np.inf 
            matrix_is_singular = True 

        return {'is_sparse_init': is_sparse_init, 'condition_number': cond_num, 'is_singular': matrix_is_singular}

    def _set_initial_strategy(self):
        """Sets MAUS's initial global strategy based on initial matrix diagnosis."""

        # Determine initial `numerical_stability_state` and solver preference based on matrix properties.
        # This decision flow dictates how aggressive MAUS starts its "brute-force" exploration.
        if self.cond_number > 1e12: 
            self.problem_knowledge['numerical_stability_state'] = 'Critical'
            self.strat_params['overall_psi_aggression_factor'] = 50.0 # High aggression from the start
            self.strat_params['max_psi_retries'] = GLOBAL_MAX_PSI_ATTEMPTS * 2 # Double local retry attempts
            self.strat_params['current_convergence_threshold'] = 1e-2 # Loose initial global convergence target
            self.problem_knowledge['local_solver_preference'] = 'iterative_gmres' # Critical problems start with GMRES
        elif self.cond_number > 1e6: # Moderately ill-conditioned at start (Fragile)
             self.problem_knowledge['numerical_stability_state'] = 'Fragile'
             self.strat_params['overall_psi_aggression_factor'] = 10.0
             self.strat_params['max_psi_retries'] = GLOBAL_MAX_PSI_ATTEMPTS 
             self.problem_knowledge['local_solver_preference'] = 'iterative_gmres' # Fragile problems also start with GMRES

        else: # Well-conditioned
             self.problem_knowledge['numerical_stability_state'] = 'Stable'
             self.problem_knowledge['local_solver_preference'] = 'direct_solve' # Use fastest direct solve


        # Specific adaptations for SOLVE_LINEAR_SYSTEM, particularly for dynamic singular matrices
        if self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM and self.diag_info['is_singular']:
            self.problem_knowledge['true_matrix_is_singular'] = True
            self.problem_knowledge['local_solver_preference'] = 'iterative_gmres' # GMRES for robust pseudo-inverse
            self.strat_params['overall_psi_aggression_factor'] = max(self.strat_params['overall_psi_aggression_factor'], 20.0) 

        # Specific adaptations for SVD problems, which often need aggressive PSI and can involve very tiny singular values.
        if self.problem_type == ProblemType.SVD: 
            if self.problem_knowledge['numerical_stability_state'] == 'Stable': # If general cond-num makes it seem stable
                self.strat_params['overall_psi_aggression_factor'] = max(self.strat_params['overall_psi_aggression_factor'], 2.0) # Nudge aggression up.
            self.strat_params['current_convergence_threshold'] = 1e-4 # Tighter starting threshold for SVD (small values common)

    def _update_global_diagnostics(self, iteration):
        """
        Phase 1: Global Information Acquisition & Matrix Understanding (Dynamic)
        Re-evaluates overall problem status based on current candidate behavior.
        Updates problem_knowledge and calculates landscape_energy.
        """
        total_active_candidates = len(self.candidates)
        sum_residuals = 0.0; sum_stuck_counters = 0; sum_confidence = 0.0; num_converged_all_types = 0 
        
        self.num_distinct_converged_solutions = 0 # Count unique converged solutions based on similarity metrics
        self.converged_solutions = [] # List to hold unique converged solutions
        current_sigma_magnitudes = [] # For SVD rank detection heuristic

        for c in self.candidates:
            if c.state == SolutionCandidate.State.CONVERGED:
                num_converged_all_types += 1 
                current_tuple = c.get_current_solution_params()
                is_distinct = True # Assume distinct until proven redundant

                # Robust Uniqueness Check (Adapting based on Problem Type)
                if self.problem_type == ProblemType.EIGENVALUE:
                    for s_item in self.converged_solutions: # Compare current `c` with already accepted `s_item` in `converged_solutions`
                        s_lam, s_vec = s_item[0], s_item[1]
                        effective_tol = GLOBAL_LAMBDA_SIMILARITY_TOL + np.abs(s_lam) * 1e-6 # Absolute + relative tolerance
                        if np.abs(current_tuple[0] - s_lam) < effective_tol and \
                           np.abs(np.vdot(current_tuple[1], s_vec)) > GLOBAL_VECTOR_SIMILARITY_TOL:
                            is_distinct = False; break
                elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM: 
                    # For Ax=b, there is usually only one unique solution `X`. So, if one is found, others are redundant.
                    if len(self.converged_solutions) > 0 and np.linalg.norm(current_tuple[0] - self.converged_solutions[0][0]) < self.strat_params['convergence_tolerance'] * 100:
                        is_distinct = False
                elif self.problem_type == ProblemType.SVD:
                    # SVD: first filter for significance, then for distinctness
                    max_current_sigma = max(c.sigma_k.real for c in self.candidates if c.sigma_k.real > 0) if len(self.candidates)>0 else 1.0
                    # Filter: if singular value is tiny compared to max, it's effectively zero, not a "distinct" rank.
                    if current_tuple[0].real / max_current_sigma < GLOBAL_SIGMA_SIMILARITY_TOL_REL : 
                         is_distinct = False 

                    if is_distinct: # If still distinct after significance check, then compare with already-found solutions
                        for s_item in self.converged_solutions: 
                            s_sigma, s_u, s_v = s_item[0], s_item[1], s_item[2]
                            effective_abs_tol = GLOBAL_SIGMA_SIMILARITY_TOL_ABS
                            effective_rel_tol = s_sigma * GLOBAL_SIGMA_SIMILARITY_TOL_REL
                            if np.abs(current_tuple[0] - s_sigma) < max(effective_abs_tol, effective_rel_tol) and \
                               np.abs(np.vdot(current_tuple[1], s_u)) > GLOBAL_VECTOR_SIMILARITY_TOL and \
                               np.abs(np.vdot(current_tuple[2], s_v)) > GLOBAL_VECTOR_SIMILARITY_TOL:
                                is_distinct = False; break
                    current_sigma_magnitudes.append(current_tuple[0].real) # Collect for overall rank detection heuristic

                if is_distinct:
                    self.converged_solutions.append(current_tuple)
                    self.num_distinct_converged_solutions += 1

            # Sum metrics ONLY for candidates that are NOT converged AND NOT retired
            if c.state not in [SolutionCandidate.State.CONVERGED, SolutionCandidate.State.RETIRED]:
                sum_residuals += c.residual_k
                sum_stuck_counters += c.stuck_counter
                sum_confidence += c.w_k

        # Normalize metrics to contribute to Landscape Energy calculation
        self.avg_residual = sum_residuals / max(1, total_active_candidates - num_converged_all_types)
        self.avg_stuckness = sum_stuck_counters / max(1, total_active_candidates - num_converged_all_types)
        self.avg_confidence_active = sum_confidence / max(1, total_active_candidates - num_converged_all_types)

        # Dynamic Landscape Energy (MAUS's primary global objective to minimize)
        norm_avg_res = self.avg_residual / (self.strat_params['current_convergence_threshold'] * 10) # Normalize by a multiple of current tolerance
        norm_avg_stuck = self.avg_stuckness / (GLOBAL_MAX_STUCK_FOR_RETIREMENT * 2) # Penalizes historical failures heavily
        
        target_sols_N_global = self.N_diag # Default for Eigenvalue problems
        if self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM: target_sols_N_global = 1 # Linear system expects 1 unique solution
        elif self.problem_type == ProblemType.SVD: 
            # Dynamic SVD rank detection heuristic (updates problem_knowledge['effective_rank_SVD'])
            if len(current_sigma_magnitudes) > 1: # Only estimate rank if at least 2 sigmas exist
                current_sigma_magnitudes_sorted = sorted([s for s in current_sigma_magnitudes if s > 0], reverse=True) # Filter out noise and sort
                if len(current_sigma_magnitudes_sorted) > 0:
                    max_sigma = current_sigma_magnitudes_sorted[0]
                    rank_detected = 0
                    for s_val in current_sigma_magnitudes_sorted:
                        if s_val / max_sigma > GLOBAL_SIGMA_SIMILARITY_TOL_REL: # Only count if value is significant relative to max
                            rank_detected += 1
                else: rank_detected = 0 # All sigmas appear to be 0 or too small.

                # MAUS learns the rank dynamically by picking the *minimum* rank detected that is consistent.
                self.problem_knowledge['effective_rank_SVD'] = min(rank_detected, min(self.N_rows, self.N_cols), max(1, self.problem_knowledge['effective_rank_SVD'])) # Cap at actual dim, never drop below 1
            target_sols_N_global = self.problem_knowledge['effective_rank_SVD'] 

        norm_missing_sols = (target_sols_N_global - self.num_distinct_converged_solutions) / max(1, target_sols_N_global) 
        
        self.landscape_energy = (norm_avg_res * 0.4) + (norm_avg_stuck * 0.3) + \
                                (norm_missing_sols * 0.3) 
        self.landscape_energy = max(0.0, min(1.0, self.landscape_energy)) # Clamp energy between 0 and 1

        # MAUS's internal "Cognitive State" update (inference about stability)
        if self.avg_stuckness > GLOBAL_MAX_STUCK_FOR_RETIREMENT * 0.5: # Many candidates are stuck globally
            self.problem_knowledge['numerical_stability_state'] = 'Critical'
        elif self.avg_stuckness > GLOBAL_MAX_STUCK_FOR_PRUNING * 0.5: # Lower level of stuckness indicates `Fragile` state
             self.problem_knowledge['numerical_stability_state'] = 'Fragile'
        else: # Mostly progressing well
             self.problem_knowledge['numerical_stability_state'] = 'Stable'


    def _adjust_global_strategy(self, iteration):
        """
        Phase 2: Meta-Adaptive Strategy Adjustment (MAUS's Brain)
        Orchestrates MAUS's overall behavior by tuning strat_params based on `landscape_energy` and `problem_knowledge`.
        """

        # General tendency towards refinement or aggressive exploration/Psi application
        if self.landscape_energy > 0.6 and self.problem_knowledge['numerical_stability_state'] == 'Critical':
            # Critical Exploration Mode: problem is very hard, go extremely aggressive
            self.problem_knowledge['local_solver_preference'] = 'iterative_gmres' # Force robust iterative solver (GMRES)
            self.strat_params['overall_psi_aggression_factor'] = min(200.0, self.strat_params['overall_psi_aggression_factor'] * 1.1)
            self.strat_params['spawn_rate_multiplier'] = min(10.0, self.strat_params['spawn_rate_multiplier'] * 1.2) # High spawn to brute force explore
            self.strat_params['current_convergence_threshold'] = max(self.strat_params['convergence_tolerance'] * 50, self.strat_params['current_convergence_threshold'] * 1.05) # Loose target to find any sol

        elif self.landscape_energy > 0.4 and self.problem_knowledge['numerical_stability_state'] == 'Fragile':
            # Fragile Exploration Mode: problem is challenging, try iterative
            self.problem_knowledge['local_solver_preference'] = 'iterative_gmres' # Prefer GMRES
            self.strat_params['overall_psi_aggression_factor'] = min(50.0, self.strat_params['overall_psi_aggression_factor'] * 1.05)
            self.strat_params['spawn_rate_multiplier'] = min(5.0, self.strat_params['spawn_rate_multiplier'] * 1.1)
            self.strat_params['current_convergence_threshold'] = max(self.strat_params['convergence_tolerance'] * 5, self.strat_params['current_convergence_threshold'] * 1.02) 

        elif self.landscape_energy < 0.2 and self.problem_knowledge['numerical_stability_state'] == 'Stable':
            # Refinement Mode: Problem is largely solved or stable, focus on tightening convergence
            self.problem_knowledge['local_solver_preference'] = 'direct_solve' # Return to faster direct_solve for refinement
            self.strat_params['overall_psi_aggression_factor'] = max(1.0, self.strat_params['overall_psi_aggression_factor'] * 0.9)
            self.strat_params['spawn_rate_multiplier'] = max(0.01, self.strat_params['spawn_rate_multiplier'] * 0.9)
            self.strat_params['current_convergence_threshold'] = max(self.strat_params['convergence_tolerance'] * 0.1, self.strat_params['current_convergence_threshold'] * 0.9) # Tighten to global_conv_tol

        # Clamp global strategy parameters to remain within defined operational bounds
        self.strat_params['overall_psi_aggression_factor'] = max(1.0, min(200.0, self.strat_params['overall_psi_aggression_factor']))
        self.strat_params['spawn_rate_multiplier'] = max(0.01, min(10.0, self.strat_params['spawn_rate_multiplier']))

    def _manage_candidates(self, iteration):
        """
        Phase 4: Population Management (Weighted Partitioning & Resource Allocation)
        Controls which candidates survive, and how many new ones are spawned.
        """
        initial_candidate_count = len(self.candidates)
        survivors = []
        # Sort candidates by weight to prioritize high-confidence solutions
        for c in sorted(self.candidates, key=lambda x: -x.w_k):
            is_redundant_in_survivors = False
            for s_c in survivors:
                # Compare `c` against already chosen `s_c` (s_c implies higher/equal weight)
                if s_c.state == SolutionCandidate.State.CONVERGED and c.state == SolutionCandidate.State.CONVERGED: 
                    res_tuple_c = c.get_current_solution_params()
                    res_tuple_s_c = s_c.get_current_solution_params()

                    if self.problem_type == ProblemType.EIGENVALUE:
                        effective_tol_sim = GLOBAL_LAMBDA_SIMILARITY_TOL + np.abs(res_tuple_s_c[0]) * 1e-6 # Adaptive absolute/relative lambda similarity
                        if np.abs(res_tuple_c[0] - res_tuple_s_c[0]) < effective_tol_sim and \
                           np.abs(np.vdot(res_tuple_c[1], res_tuple_s_c[1])) > GLOBAL_VECTOR_SIMILARITY_TOL:
                            is_redundant_in_survivors = True; break
                    elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM: 
                        if np.linalg.norm(res_tuple_c[0] - res_tuple_s_c[0]) < self.strat_params['convergence_tolerance'] * 10: # Loose check for solve redundancy
                            is_redundant_in_survivors = True; break
                    elif self.problem_type == ProblemType.SVD:
                        effective_abs_tol_sim = GLOBAL_SIGMA_SIMILARITY_TOL_ABS
                        effective_rel_tol_sim = res_tuple_s_c[0] * GLOBAL_SIGMA_SIMILARITY_TOL_REL
                        # Special SVD redundancy: Acknowledge that numerically zero singular values aren't strictly unique.
                        if res_tuple_s_c[0].real < effective_abs_tol_sim / 100: is_redundant_in_survivors = False # Don't aggressively prune tiny/zero sigmas unless they have very high magnitude error too.
                        elif np.abs(res_tuple_c[0] - res_tuple_s_c[0]) < max(effective_abs_tol_sim, effective_rel_tol_sim) and \
                           np.abs(np.vdot(res_tuple_c[1], res_tuple_s_c[1])) > GLOBAL_VECTOR_SIMILARITY_TOL and \
                           np.abs(np.vdot(res_tuple_c[2], res_tuple_s_c[2])) > GLOBAL_VECTOR_SIMILARITY_TOL:
                         is_redundant_in_survivors = True; break
                
            # Decision flow for pruning/keeping: prioritize non-redundant and improving candidates
            if is_redundant_in_survivors: # Mark redundant for removal later (but not yet removed from current list)
                c.state = SolutionCandidate.State.RETIRED 
            elif c.state == SolutionCandidate.State.RETIRED: # Filter explicitly marked `RETIRED` candidates (from internal update logic or prior management)
                pass 
            elif (c.w_k < self.strat_params['min_survival_weight'] and c.state != SolutionCandidate.State.CONVERGED) or \
                 (c.stuck_counter >= GLOBAL_MAX_STUCK_FOR_RETIREMENT and c.state != SolutionCandidate.State.CONVERGED):
                pass # This candidate is considered permanently "unfruitful" - filter it out.
            else: # All other conditions passed: this candidate is a survivor!
                survivors.append(c)

        self.candidates = survivors # Update actual population of candidates
        num_pruned = initial_candidate_count - len(self.candidates)
        if num_pruned > 0: print(f"  Population Management: Pruned/Retired {num_pruned} candidates.")


        # Determine number of new candidates to spawn (Intelligent Spawning)
        target_distinct_count_for_spawn = self.N_diag
        if self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM: target_distinct_count_for_spawn = 1
        elif self.problem_type == ProblemType.SVD: target_distinct_count_for_spawn = self.problem_knowledge['effective_rank_SVD'] # Spawn to target effective rank


        desired_population_base = max(5, self.N_diag * 2 if self.problem_type != ProblemType.SOLVE_LINEAR_SYSTEM else self.N_diag * 1.5)
        if self.problem_type == ProblemType.SVD: desired_population_base = max(desired_population_base, self.problem_knowledge['effective_rank_SVD'] * 4)

        # How many candidates to add to meet desired population base AND find missing distinct solutions.
        num_to_spawn = max(0, desired_population_base - len(self.candidates)) 
        num_to_spawn += max(0, target_distinct_count_for_spawn - self.num_distinct_converged_solutions)
        
        num_to_spawn = int(num_to_spawn * self.strat_params['spawn_rate_multiplier']) # Apply global spawn rate multiplier
        num_to_spawn = min(num_to_spawn, 10) # Cap maximum spawns per iteration for performance

        if num_to_spawn > 0:
            print(f"  SPAWNING: Adding {num_to_spawn} new candidates (Current:{len(self.candidates)}, Distinct Found:{self.num_distinct_converged_solutions}/{target_distinct_count_for_spawn}).")
            for _ in range(num_to_spawn):
                new_candidate_init_vals = {}
                # Directed Spawning (if suitable for more efficient exploration)
                # If we have some converged solutions, spawn nearby (refinement). Else, use full randomness (exploration/brute-force discovery).
                if self.num_distinct_converged_solutions > 0 and self.landscape_energy < 0.8: # We have strong starting points and not complete chaos
                    base_sol_tuple = random.choice(self.converged_solutions) # Pick a random strong converged solution as base
                    
                    if self.problem_type == ProblemType.EIGENVALUE:
                        new_candidate_init_vals['initial_lambda'] = base_sol_tuple[0] + (random.random() * 0.1 - 0.05 + 1j * (random.random() * 0.1 - 0.05)) # Small lambda perturbation
                        new_candidate_init_vals['initial_v'] = base_sol_tuple[1] + (np.random.rand(self.N_diag) - 0.5 + 1j * (np.random.rand(self.N_diag) - 0.5)) * 0.1 # Small vector perturbation
                        new_candidate_init_vals['initial_v'] /= np.linalg.norm(new_candidate_init_vals['initial_v'])
                    elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
                        new_candidate_init_vals['initial_x'] = base_sol_tuple[0] + (np.random.rand(self.N_diag) - 0.5 + 1j * (np.random.rand(self.N_diag) - 0.5)) * 0.1
                    elif self.problem_type == ProblemType.SVD: 
                        new_candidate_init_vals['initial_sigma'] = base_sol_tuple[0] + (random.random() * 0.1 - 0.05)
                        new_candidate_init_vals['initial_u'] = base_sol_tuple[1] + (np.random.rand(self.M_rows) - 0.5 + 1j * (np.random.rand(self.M_rows) - 0.5)) * 0.1
                        new_candidate_init_vals['initial_u'] /= np.linalg.norm(new_candidate_init_vals['initial_u'])
                        new_candidate_init_vals['initial_v'] = base_sol_tuple[2] + (np.random.rand(self.M_cols)) - 0.5 + 1j * (random.random() * 2 - 1).astype(np.complex128)[0, :self.M_cols] * 0.1 # Corrected for potentially dense/random matrix
                        new_candidate_init_vals['initial_v'] /= np.linalg.norm(new_candidate_init_vals['initial_v'])
                        
                else: # Fallback to fully random initialization (broad exploration in "pure chaos" or initial phase)
                    pass # Default initialization (random values) is done by Candidate constructor.
                
                new_candidate = SolutionCandidate(self.M, self.problem_type, self.N_diag, **new_candidate_init_vals, initial_weight=0.01)
                new_candidate.alpha_local_step = GLOBAL_DEFAULT_ALPHA_V_INITIAL * self.strat_params['overall_psi_aggression_factor'] # New candidates get alpha proportional to global aggression
                self.candidates.append(new_candidate)
                
    def evolve(self, max_iterations=100):
        """
        Main evolution loop of MAUS. Orchestrates global strategy and candidate updates.
        """
        print(f"\n--- Starting MAUS Evolution for {max_iterations} iterations ({self.problem_type.name}) ---")
        
        # Calculate true solution via NumPy for comparison (only once)
        self.true_solution = None
        try:
            if self.problem_type == ProblemType.EIGENVALUE:
                self.true_solution = np.sort(sla.eigvals(self.M).real) + 1j*np.sort(sla.eigvals(self.M).imag)
                print(f"True Eigenvalues (by NumPy for reference): {self.true_solution}")
            elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
                self.true_solution = sla.solve(self.M, self.b, assume_a='general')
                print(f"True solution X (by NumPy for reference):\n{self.true_solution}")
            elif self.problem_type == ProblemType.SVD:
                if isinstance(self.M, (sp.spmatrix,)): # Use sparse SVD for sparse matrices
                    U, s, Vh = spla.svds(self.M, k=min(self.M.shape)-1) # k = min(dim)-1 to get all singular values.
                else: # Use dense SVD for dense matrices
                    U, s, Vh = sla.svd(self.M) 
                self.true_solution = sorted(s.tolist(), reverse=True) 
                print(f"True Singular Values (by NumPy for reference): {self.true_solution}")
        except Exception as e:
            print(f"NumPy reference calculation failed: {e}. Cannot compare results directly.")


        # Main MAUS Iteration Loop
        for i in range(max_iterations):
            # Phase 1&2: Global Diagnostics & Strategy Adjustment (MAUS's Brain)
            self._update_global_diagnostics(i + 1)
            self._adjust_global_strategy(i + 1)

            # Phase 3: Candidate Evolution (Local Execution with MAUS knowledge)
            # Only update candidates that are not yet converged or not retired.
            for candidate in self.candidates:
                if candidate.state != SolutionCandidate.State.CONVERGED and candidate.state != SolutionCandidate.State.RETIRED: 
                    candidate.update_solution_step(self.M, self.b, self.strat_params, self.problem_knowledge)
            
            # Phase 4: Population Management (Pruning and Spawning)
            self._manage_candidates(i + 1)
            
            # --- Global Reporting and Termination Check ---
            print(f"\n--- MAUS Iteration {i+1} Summary ({self.problem_type.name}) ---")
            print(f"  Strategy: PSI Aggro={self.strat_params['overall_psi_aggression_factor']:.1f}, Spawn Rate={self.strat_params['spawn_rate_multiplier']:.1f}, Cand Conv Thresh={self.strat_params['current_convergence_threshold']:.1e}")
            print(f"  Metrics: Energy={self.landscape_energy:.2f}, Avg Stuck={self.avg_stuckness:.2f}, Candidates={len(self.candidates)}, Distinct Found={self.num_distinct_converged_solutions}/{ (self.N_diag if self.problem_type==ProblemType.EIGENVALUE else (1 if self.problem_type==ProblemType.SOLVE_LINEAR_SYSTEM else self.problem_knowledge['effective_rank_SVD']) ) }")

            # Summarize top 3 converged candidates
            converged_candidates_report = sorted([c for c in self.candidates if c.state == SolutionCandidate.State.CONVERGED], key=lambda x: -x.w_k)[:3]
            for c in converged_candidates_report:
                 val_str = ""
                 if self.problem_type == ProblemType.EIGENVALUE: val_str = f"λ={c.lambda_k:.5f}"
                 elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM: val_str = f"X_real_0={c.x_k[0].real:.4f}"
                 elif self.problem_type == ProblemType.SVD: val_str = f"σ={c.sigma_k:.5f}"
                 print(f"  [C] ID {c.id}: {val_str}, Resid={c.residual_k:.2e}, W={c.w_k:.3f}")
            if not converged_candidates_report: print("  No candidates currently converged.")
            
            # Check overall problem completion criteria for early termination
            should_terminate = False
            target_distinct_sols_final = self.N_diag # Expected number of solutions for current problem
            if self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM: target_distinct_sols_final = 1
            elif self.problem_type == ProblemType.SVD: target_distinct_sols_final = self.problem_knowledge['effective_rank_SVD']

            if self.num_distinct_converged_solutions >= target_distinct_sols_final and self.landscape_energy < 0.1: 
                should_terminate = True
            
            # Warning if max iterations reached without full convergence
            if i == max_iterations -1 and self.num_distinct_converged_solutions < target_distinct_sols_final:
                 print(f"WARNING: Max iterations reached but not all solutions found ({self.num_distinct_converged_solutions}/{target_distinct_sols_final}). Landscape energy: {self.landscape_energy:.2f}.")

            if should_terminate:
                print(f"MAUS successfully identified target number of solutions and achieved low landscape energy. Terminating early at iteration {i+1}.")
                break

        # Final Report after evolution loop completes
        print("\n--- MAUS Evolution COMPLETE ---")
        print("Final Report:")
        
        # Sort and print discovered solutions for clear comparison with NumPy truth
        if self.problem_type == ProblemType.EIGENVALUE:
            discovered_final_print = sorted(self.converged_solutions, key=lambda x: (x[0].real, x[0].imag))
        elif self.problem_type == ProblemType.SVD:
             discovered_final_print = sorted(self.converged_solutions, key=lambda x: -x[0].real) # Sort sigmas descending
        else: # For Ax=b, only one solution is typically expected
            discovered_final_print = self.converged_solutions

        for solution_tuple in discovered_final_print:
            final_res = None
            if self.problem_type == ProblemType.EIGENVALUE:
                l, v = solution_tuple
                final_res = np.linalg.norm(self.M @ v - l * v)
                print(f"  λ={l:.8f}, Final Residual={final_res:.2e}")
            elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
                x = solution_tuple[0]
                final_res = np.linalg.norm(self.M @ x - self.b)
                print(f"  X_first_el={x[0]:.8f}, Final Residual={final_res:.2e}")
            elif self.problem_type == ProblemType.SVD:
                s, u, v_right = solution_tuple
                final_res = np.linalg.norm(self.M @ v_right - s * u) + np.linalg.norm(self.M.conj().T @ u - s * v_right)
                print(f"  σ={s:.8f}, Final Residual={final_res:.2e}")
        
        # Comparison with NumPy truth (if available)
        if self.true_solution is not None and self.num_distinct_converged_solutions > 0:
            print("\n--- Comparison to NumPy's True Solution ---")
            if self.problem_type == ProblemType.EIGENVALUE:
                discovered_eigs = np.array([s[0] for s in discovered_final_print]).real + 1j * np.array([s[0] for s in discovered_final_print]).imag 
                print(f"MAUS Eigs (sorted): {discovered_eigs}")
                error_abs_sum = np.sum(np.abs(discovered_eigs - self.true_solution[:len(discovered_eigs)]))
                print(f"Absolute sum error to true solution (for found eigs): {error_abs_sum:.2e}") 
            elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
                final_x = self.converged_solutions[0][0]
                error = np.linalg.norm(final_x - self.true_solution) / np.linalg.norm(self.true_solution)
                print(f"Relative error ||X_MAUS - X_true|| / ||X_true||: {error:.2e}")
            elif self.problem_type == ProblemType.SVD:
                discovered_sigmas = np.array([s[0].real for s in discovered_final_print]) 
                error_sigmas = np.linalg.norm(discovered_sigmas - np.array(self.true_solution[:len(discovered_sigmas)])) / np.linalg.norm(self.true_solution[:len(discovered_sigmas)]) 
                print(f"MAUS Sigmas (sorted): {discovered_sigmas}")
                print(f"Relative sigma error: {error_sigmas:.2e}")


# --- TEST SCENARIOS (UNCHANGED from previous attempt, will show if current MAUS passes) ---

# Scenario 1: Solve Ax=b (N=5, Dynamically Ill-conditioned Hilbert-like) 
def create_dynamic_solve_matrix_and_b(N, t_step, initial_cond_mult=1.0, singularity_period=20, noise_freq=15, time_max_iter=100):
    t_norm = t_step / time_max_iter 
    
    H_base = sla.hilbert(N).astype(np.complex128)
    H_diag_boost = np.diag(np.ones(N) * N * 0.1) 
    
    singular_inductor = np.zeros((N, N), dtype=np.complex128)
    singular_inductor[0, N-1] = 1.0 
    singular_inductor[N-1, 0] = -1.0 
    
    M_val = H_base + H_diag_boost
    M_val += np.sin(t_step * 2 * np.pi / singularity_period) * singular_inductor * (10.0 + t_norm*20.0) 
    M_val += np.cos(t_step * 2 * np.pi / noise_freq) * (np.random.rand(N,N) + 1j * np.random.rand(N,N)) * 1e-4

    b_vec = np.array([1, -1, 0.5, -0.5, 0.1], dtype=np.complex128)
    
    return M_val, b_vec


# Scenario 2: Eigenvalue (N=8, Complex, Clustered, Degenerate Laplace-like) 
def create_laplace_like_complex_eigen_for_MAUS(N):
    M_val = np.zeros((N, N), dtype=np.complex128)

    for i in range(N):
        M_val[i, i] = -2.0 
        if i > 0: M_val[i, i-1] = 1.0
        if i < N-1: M_val[i, i+1] = 1.0
    
    M_val[0,2] = 0.5; M_val[2,0] = 0.5j 
    M_val[N-1,N-3] = 0.8j; M_val[N-3,N-1] = 0.8 
    
    M_val[N//2-1, N//2] = 1.5 + 0.5j 
    M_val[N//2, N//2-1] = -1.5 + 0.5j 
    
    M_val += (np.random.rand(N,N) * 2 - 1) * 1e-3 + 1j * (np.random.rand(N,N) * 2 - 1) * 1e-3
    M_val[0, N-1] += 0.2 
    M_val[N-1, 0] += 0.2j

    M_val[N-1,N-1] = M_val[N-2,N-2] + 1e-6 
    
    return M_val


# Scenario 3: SVD (N=5x4, Near-Low-rank with defined ranks) 
def create_low_rank_svd_matrix_for_MAUS(M_rows, N_cols, target_rank=2):
    u1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])[:M_rows].astype(np.complex128)
    u2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0])[:M_rows].astype(np.complex128)
    u1 = u1 / np.linalg.norm(u1); u2 = u2 / np.linalg.norm(u2) 

    v1 = np.array([1.0, 0.0, 0.0, 0.0])[:N_cols].astype(np.complex128)
    v2 = np.array([0.0, 1.0, 0.0, 0.0])[:N_cols].astype(np.complex128)
    v1 = v1 / np.linalg.norm(v1); v2 = v2 / np.linalg.norm(v2)

    s_vals = [5.0, 0.5] + [1e-10 + random.random()*1e-10 for _ in range(min(M_rows, N_cols)-2)]
    s_vals = s_vals[:min(M_rows, N_cols)] 

    U_base = np.zeros((M_rows, M_rows), dtype=np.complex128)
    U_base[:,0] = u1
    U_base[:,1] = u2
    U_base[:,2:] = np.eye(M_rows, M_rows-2, k=2) 
    Q_U, _ = sla.qr(U_base) 
    
    V_base = np.zeros((N_cols, N_cols), dtype=np.complex128)
    V_base[:,0] = v1
    V_base[:,1] = v2
    V_base[:,2:] = np.eye(N_cols, N_cols-2, k=2)
    Q_V, _ = sla.qr(V_base)
    
    Sigma = np.zeros((M_rows, N_cols), dtype=np.complex128)
    for i, s_val in enumerate(s_vals): Sigma[i, i] = s_val

    A_true_low_rank = Q_U @ Sigma @ Q_V.conj().T

    A_noise = (np.random.randn(M_rows, N_cols) + 1j * np.random.randn(M_rows, N_cols)) * 1e-4

    return A_true_low_rank + A_noise


# SCENARIO 1: SOLVE Ax=b (N=5, Dynamically Ill-conditioned Hilbert-like) 
MAX_ITER_SOLVE_LINEAR = 100 

print("\n\n##################### SCENARIO 1: SOLVE AX=B (N=5, DYNAMICALLY ILL-CONDITIONED) #####################")
print(f"--- Matrix A(t) changes over {MAX_ITER_SOLVE_LINEAR} time steps. ---")
print("MAUS must adapt to track X and handle time-varying near-singularities.")

maus_solver_solve = MAUS_Solver(np.eye(5), problem_type=ProblemType.SOLVE_LINEAR_SYSTEM, b_vector=np.ones(5), initial_num_candidates=10) # Init with dummy A
all_solved_x_guile = []
all_true_x = []

for t_step in range(MAX_ITER_SOLVE_LINEAR):
    A_t, b_t = create_dynamic_solve_matrix_and_b(N=5, t_step=t_step, time_max_iter=MAX_ITER_SOLVE_LINEAR)
    maus_solver_solve.M = A_t 
    maus_solver_solve.b = b_t 
    
    maus_solver_solve._update_global_diagnostics(t_step + 1)
    maus_solver_solve._adjust_global_strategy(t_step + 1)
    for candidate in maus_solver_solve.candidates:
        candidate.update_solution_step(maus_solver_solve.M, maus_solver_solve.b, maus_solver_solve.strat_params, maus_solver_solve.problem_knowledge)
    maus_solver_solve._manage_candidates(t_step + 1) 
    
    current_best_x = None
    if maus_solver_solve.num_distinct_converged_solutions > 0:
        current_best_x = maus_solver_solve.converged_solutions[0][0]
        all_solved_x_guile.append(current_best_x)
    else:
        all_solved_x_guile.append(np.full(5, np.nan, dtype=np.complex128)) 

    try:
        true_x_t = sla.solve(A_t, b_t, assume_a='general')
        all_true_x.append(true_x_t)
    except np.linalg.LinAlgError:
        all_true_x.append(np.full(5, np.nan, dtype=np.complex128)) 
        
    if (t_step + 1) % 10 == 0:
        cond_A_t = 0.0
        try: cond_A_t = np.linalg.cond(A_t)
        except: cond_A_t = np.inf
        
        guile_res = np.linalg.norm(A_t @ current_best_x - b_t) if current_best_x is not None and not np.any(np.isnan(current_best_x)) else float('inf')
        print(f"\nTime Step {t_step+1}: Cond(A)={cond_A_t:.2e}, MAUS Sol Res={guile_res:.2e}, Candidates={len(maus_solver_solve.candidates)}")
        print(f"  PSI Aggro={maus_solver_solve.strat_params['overall_psi_aggression_factor']:.1f}, Energy={maus_solver_solve.landscape_energy:.2f}")

print("\n--- SCENARIO 1 COMPLETE ---")
num_nans_guile_sol = np.sum(np.isnan(all_solved_x_guile)) 
num_nans_true_sol = np.sum(np.isnan(all_true_x)) 
print(f"MAUS failed to converge (NaNs) in {num_nans_guile_sol} steps out of {MAX_ITER_SOLVE_LINEAR}.")
print(f"True solution was singular in {num_nans_true_sol} steps out of {MAX_ITER_SOLVE_LINEAR}.")

valid_errors = []
for i in range(MAX_ITER_SOLVE_LINEAR):
    if not np.any(np.isnan(all_solved_x_guile[i])) and not np.any(np.isnan(all_true_x[i])):
        true_norm = np.linalg.norm(all_true_x[i])
        if true_norm > 1e-10: 
            valid_errors.append(np.linalg.norm(all_solved_x_guile[i] - all_true_x[i]) / true_norm)

if valid_errors:
    mean_relative_error = np.nanmean(valid_errors)
    print(f"MAUS tracked solution across {len(valid_errors)} non-singular steps. Mean relative error: {mean_relative_error:.2e}")
else:
    print("No valid comparisons possible for tracking problem.")


# Scenario 2: Eigenvalue (N=8, Complex, Clustered, Degenerate Laplace-like)
M_eigen_test_for_MAUS = create_laplace_like_complex_eigen_for_MAUS(8)
print("\n\n##################### SCENARIO 2: EIGENVALUE (N=8, COMPLEX, CLUSTERED, LAPLACE-LIKE) #####################")
maus_solver_eigen = MAUS_Solver(M_eigen_test_for_MAUS, problem_type=ProblemType.EIGENVALUE, initial_num_candidates=30, global_convergence_tol=1e-7)
maus_solver_eigen.evolve(max_iterations=150) 


# Scenario 3: SVD (N=5x4, Near-Low-rank with defined ranks)
M_svd_test_for_MAUS = create_low_rank_svd_matrix_for_MAUS(5, 4, target_rank=2) 
print("\n\n##################### SCENARIO 3: SVD (N=5x4, NEAR-LOW-RANK) #####################")
maus_solver_svd = MAUS_Solver(M_svd_test_for_MAUS, problem_type=ProblemType.SVD, initial_num_candidates=20, global_convergence_tol=1e-7)
maus_solver_svd.evolve(max_iterations=200)
