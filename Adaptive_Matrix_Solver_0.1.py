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
GLOBAL_DEFAULT_PSI_EPSILON_BASE = np.complex128(1e-20) 
GLOBAL_DEFAULT_ALPHA_V_INITIAL = np.complex128(0.01) 
GLOBAL_MAX_PSI_ATTEMPTS = 25 
GLOBAL_MAX_STUCK_FOR_RETIREMENT = 8
GLOBAL_MIN_WEIGHT_TO_SURVIVE_PRUNE = 1e-10 
GLOBAL_VECTOR_SIMILARITY_TOL = 0.999 
GLOBAL_LAMBDA_SIMILARITY_TOL = 1e-5 
GLOBAL_SIGMA_SIMILARITY_TOL_ABS = 1e-6
GLOBAL_SIGMA_SIMILARITY_TOL_REL = 1e-4
GLOBAL_CONVERGENCE_RESIDUAL_TOL = 1e-8 
GLOBAL_MAX_STUCK_FOR_PRUNING = 4    


# --- InverseIterateSolver: Adaptive Local Solver for Ax=b-like Problems ---
class InverseIterateSolver:
    def __init__(self, N, base_psi_epsilon, max_attempts, preferred_method='direct_solve', is_sparse=False):
        self.N = N 
        self.base_psi_epsilon = base_psi_epsilon 
        self.max_attempts = max_attempts 
        self.preferred_method = preferred_method 
        self.fallback_method = 'iterative_gmres' if preferred_method == 'direct_solve' else 'direct_solve' 
        self.is_sparse = is_sparse 

    def solve(self, A_target, b_rhs, candidate_stuck_counter):
        num_psi_attempts = 0
        current_method_for_try = self.preferred_method 

        while num_psi_attempts < self.max_attempts:
            psi_scalar_magnitude = self.base_psi_epsilon * (10**(num_psi_attempts / 2.0)) * (10**(candidate_stuck_counter / 3.0)) 
            
            if self.is_sparse:
                regularization_term = sp.identity(self.N, dtype=A_target.dtype, format='csc') * psi_scalar_magnitude 
            else: 
                random_perturb = (np.random.rand(self.N, self.N) - 0.5 + 1j * (np.random.rand(self.N, self.N) - 0.5)) * psi_scalar_magnitude * 0.15 
                regularization_term = psi_scalar_magnitude * np.eye(self.N, dtype=A_target.dtype) + random_perturb 

            H_solve = A_target + regularization_term
            
            try:
                if current_method_for_try == 'direct_solve':
                    if self.is_sparse:
                        result_vec = spla.spsolve(H_solve.tocsc(), b_rhs) 
                    else:
                        result_vec = sla.solve(H_solve, b_rhs, assume_a='general') 
                elif current_method_for_try == 'iterative_gmres':
                    x0_init = b_rhs if b_rhs.shape == H_solve.shape[1:] else np.zeros_like(b_rhs)
                    
                    # MODIFICATION START: GMRES Preconditioner
                    preconditioner = None
                    if candidate_stuck_counter > 1 and self.N > 0: # Only try preconditioning if candidate is stuck
                        try:
                            diag_H = H_solve.diagonal()
                            if diag_H.size == self.N: # Ensure diagonal is conformant
                                # Inverse of diagonal elements for Jacobi preconditioner
                                inv_diag_H = 1.0 / diag_H
                                # Check for NaNs or Infs that could arise from division by zero or near-zero
                                if np.all(np.isfinite(inv_diag_H)) and np.all(np.abs(diag_H) > 1e-12):
                                    if self.is_sparse:
                                        preconditioner = sp.diags(inv_diag_H, format="csc")
                                    else: # Dense
                                        preconditioner = np.diag(inv_diag_H) # This will be a 2D diag matrix
                                    # print(f"GMRES: Using preconditioner (stuck_count={candidate_stuck_counter})")
                                else:
                                    # print(f"Warning: Invalid diagonal for preconditioner (zeros/NaNs/Infs). Skipping for N={self.N}.")
                                    preconditioner = None # Explicitly set to None
                            else:
                                # print(f"Warning: Diagonal size mismatch for preconditioner. Skipping for N={self.N}, diag_size={diag_H.size}.")
                                preconditioner = None # Explicitly set to None
                        except Exception as e:
                            print(f"Error creating preconditioner: {e}. Proceeding without.")
                            preconditioner = None
                    # MODIFICATION END

                    result_vec, info = spla.gmres(H_solve, b_rhs, x0=x0_init, tol=1e-8, maxiter=50, M=preconditioner) 
                    if info != 0: raise np.linalg.LinAlgError(f"GMRES did not converge cleanly (info={info}). Preconditioned: {'Yes' if preconditioner is not None else 'No'}")
                else:
                    raise ValueError(f"Unknown solver method: {current_method_for_try}")

                if not np.all(np.isfinite(result_vec)): 
                    raise ValueError("Solution vector not finite after solve.")
                
                return result_vec, num_psi_attempts 
            except (np.linalg.LinAlgError, ValueError, TypeError) as e:
                if current_method_for_try == self.preferred_method and self.preferred_method != self.fallback_method and num_psi_attempts == 0:
                    current_method_for_try = self.fallback_method 
                    num_psi_attempts = 0 
                    continue 
                num_psi_attempts += 1 
        raise RuntimeError(f"InverseIterateSolver failed all {self.max_attempts} attempts for {self.preferred_method} and {self.fallback_method}.")

# --- Solution Candidate Class ---
class SolutionCandidate:
    _candidate_id_counter = 0
    class State(Enum):
        EXPLORING = 1; REFINING = 2; STUCK = 3; CONVERGED = 4; RETIRED = 5

    def __init__(self, problem_matrix, problem_type, N_diag, initial_lambda=None, initial_v=None, initial_x=None, initial_u=None, initial_sigma=None, initial_weight=0.01):
        self.id = SolutionCandidate._candidate_id_counter
        SolutionCandidate._candidate_id_counter += 1
        self.N_diag = N_diag
        self.M_rows, self.M_cols = problem_matrix.shape
        self.problem_type = problem_type
        self.problem_matrix = problem_matrix 
        self.b_vector = None
        self.lambda_k = initial_lambda; self.v_k = initial_v; self.x_k = initial_x
        self.sigma_k = initial_sigma; self.u_k = initial_u; self.right_v_k = initial_v
        self.state = SolutionCandidate.State.EXPLORING; self.w_k = initial_weight
        self.residual_k = float('inf'); self.prev_residual = float('inf')
        self.alpha_local_step = GLOBAL_DEFAULT_ALPHA_V_INITIAL
        self.stuck_counter = 0; self.local_psi_retries_needed = 0; self.num_resets = 0
        self.param_history = []; self.residual_history = []
        self.initialize_random_solution()

    def initialize_random_solution(self):
        rand_vec_init = lambda N: (np.random.rand(N) + 1j * np.random.rand(N)).astype(np.complex128)
        norm_rand_vec = lambda v: v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-10 else rand_vec_init(v.shape[0]) / np.linalg.norm(rand_vec_init(v.shape[0]))

        if self.problem_type == ProblemType.EIGENVALUE:
            self.v_k = norm_rand_vec(rand_vec_init(self.N_diag))
            self.lambda_k = (random.random() * 5 - 2.5 + 1j * (random.random() * 5 - 2.5))
        elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
            self.x_k = norm_rand_vec(rand_vec_init(self.N_diag)) * random.uniform(0.1, 10.0)
        elif self.problem_type == ProblemType.SVD:
            self.u_k = norm_rand_vec(rand_vec_init(self.M_rows))
            self.right_v_k = norm_rand_vec(rand_vec_init(self.M_cols))
            self.sigma_k = 1.0 
        self.param_history.append(self.get_current_solution_params()) 
        self.residual_history.append(self.residual_k) 

    def update_solution_step(self, current_matrix_A, b_vector=None, strat_params=None, global_knowledge=None):
        self.b_vector = b_vector
        self.prev_residual = self.residual_k

        overall_psi_aggression_factor = strat_params.get('overall_psi_aggression_factor', 1.0)
        max_psi_retries_global = strat_params.get('max_psi_retries', GLOBAL_MAX_PSI_ATTEMPTS)
        local_solver_preference = global_knowledge.get('local_solver_preference', 'direct_solve')
        is_matrix_sparse = global_knowledge.get('is_sparse_problem', False)

        # MODIFICATION START: Specialized Hermitian Eigenvalue Solver
        if self.problem_type == ProblemType.EIGENVALUE and global_knowledge.get('is_hermitian', False):
            hermitian_solver_succeeded = False
            N_eigen = current_matrix_A.shape[0]

            if not is_matrix_sparse: # Dense Hermitian matrix
                try:
                    eigenvalues, eigenvectors = sla.eigh(current_matrix_A)
                    if self.v_k is not None and eigenvectors.shape[1] > 0:
                        # Corrected similarity scores for complex vectors
                        if eigenvectors.shape[1] == self.v_k.shape[0]: # Ensure proper dimensions for dot product
                             similarity_scores = np.abs(self.v_k.conj().T @ eigenvectors)
                        else: # Default to iterating if broadcast fails
                             similarity_scores = np.array([np.abs(np.vdot(self.v_k, eigenvectors[:, i])) for i in range(eigenvectors.shape[1])])

                        best_match_idx = np.argmax(similarity_scores)
                        
                        self.lambda_k = eigenvalues[best_match_idx]
                        self.v_k = eigenvectors[:, best_match_idx]
                        self.v_k /= np.linalg.norm(self.v_k)
                        
                        self.residual_k = np.linalg.norm(current_matrix_A @ self.v_k - self.lambda_k * self.v_k)
                        self.state = SolutionCandidate.State.CONVERGED
                        self.stuck_counter = 0
                        self.local_psi_retries_needed = 0
                        self.w_k = 1.0
                        # print(f"Candidate {self.id}: Dense Hermitian solver (eigh) successful. Lambda_k={self.lambda_k:.4e}, Resid={self.residual_k:.2e}")
                        hermitian_solver_succeeded = True
                except np.linalg.LinAlgError as e:
                    print(f"Candidate {self.id}: Dense Hermitian solver (eigh) failed: {e}. Falling back.")
                except Exception as e:
                    print(f"Candidate {self.id}: Unexpected error during dense Hermitian solve: {e}. Falling back.")
            else: # Sparse Hermitian matrix
                try:
                    k_eigsh = min(6, N_eigen - 1)
                    if k_eigsh < 1 and N_eigen >=1: k_eigsh = 1
                    elif N_eigen < 1: raise ValueError("Matrix dimension is less than 1 for eigsh.")
                    
                    v0_init = self.v_k if self.v_k is not None and self.v_k.shape[0] == N_eigen and np.linalg.norm(self.v_k) > 1e-8 else None

                    eigenvalues, eigenvectors = spla.eigsh(current_matrix_A, k=k_eigsh, which='LM', v0=v0_init, tol=strat_params.get('convergence_tolerance', 1e-8)/100)
                    
                    if self.v_k is not None and eigenvectors.shape[1] > 0:
                        similarity_scores = np.array([np.abs(np.vdot(self.v_k, eigenvectors[:, i])) for i in range(eigenvectors.shape[1])])
                        best_match_idx = np.argmax(similarity_scores)
                        
                        self.lambda_k = eigenvalues[best_match_idx]
                        self.v_k = eigenvectors[:, best_match_idx]
                        self.v_k /= np.linalg.norm(self.v_k)
                        
                        self.residual_k = np.linalg.norm(current_matrix_A @ self.v_k - self.lambda_k * self.v_k)
                        self.state = SolutionCandidate.State.CONVERGED
                        self.stuck_counter = 0
                        self.local_psi_retries_needed = 0
                        self.w_k = 1.0
                        # print(f"Candidate {self.id}: Sparse Hermitian solver (eigsh) successful. Lambda_k={self.lambda_k:.4e}, Resid={self.residual_k:.2e}")
                        hermitian_solver_succeeded = True
                except spla.ArpackNoConvergence as e:
                    print(f"Candidate {self.id}: Sparse Hermitian solver (eigsh) failed to converge: {e}. Falling back.")
                except (np.linalg.LinAlgError, ValueError) as e:
                     print(f"Candidate {self.id}: Sparse Hermitian solver (eigsh) error: {e}. Falling back.")
                except Exception as e:
                    print(f"Candidate {self.id}: Unexpected error during sparse Hermitian solve: {e}. Falling back.")

            if hermitian_solver_succeeded:
                self.param_history.append(self.get_current_solution_params())
                self.residual_history.append(self.residual_k)
                return # Early exit if specialized solver succeeded
        # MODIFICATION END

        solver_instance = InverseIterateSolver(self.N_diag, GLOBAL_DEFAULT_PSI_EPSILON_BASE * overall_psi_aggression_factor, 
                                                max_psi_retries_global, local_solver_preference, is_matrix_sparse)
        
        if self.problem_type == ProblemType.SVD: 
            try:
                if np.linalg.norm(self.right_v_k) < 1e-10:
                    self.right_v_k = (np.random.rand(self.M_cols) + 1j * np.random.rand(self.M_cols)); self.right_v_k /= np.linalg.norm(self.right_v_k)
                    self.stuck_counter += 1; self.num_resets += 1;
                    raise ValueError("SVD right_v_k collapsed.") 
                temp_u_k = current_matrix_A @ self.right_v_k
                self.sigma_k = np.linalg.norm(temp_u_k)
                self.u_k = temp_u_k / (self.sigma_k if self.sigma_k > 1e-10 else 1.0)
                if np.linalg.norm(self.u_k) < 1e-10:
                     self.u_k = (np.random.rand(self.M_rows)+1j*np.random.rand(self.M_rows)); self.u_k /= np.linalg.norm(self.u_k)
                     self.stuck_counter += 1; self.num_resets += 1;
                     raise ValueError("SVD u_k collapsed.")
                temp_v_k = current_matrix_A.conj().T @ self.u_k
                self.sigma_k = max(self.sigma_k, np.linalg.norm(temp_v_k))
                self.right_v_k = temp_v_k / (np.linalg.norm(temp_v_k) if np.linalg.norm(temp_v_k) > 1e-10 else 1.0)
                if self.sigma_k < GLOBAL_SIGMA_SIMILARITY_TOL_ABS / 100 :
                    self.residual_k = strat_params.get('current_convergence_threshold', 1e-6) * 0.1
                    self.state = SolutionCandidate.State.CONVERGED; self.stuck_counter = 0
                    if np.linalg.norm(self.u_k) < 1e-10: self.u_k = np.ones(self.M_rows, dtype=np.complex128)/np.sqrt(self.M_rows)
                    if np.linalg.norm(self.right_v_k) < 1e-10: self.right_v_k = np.ones(self.M_cols, dtype=np.complex128)/np.sqrt(self.M_cols)
                else: self.stuck_counter = max(0, self.stuck_counter - 1)
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
                self.stuck_counter += 1; self.w_k *= 0.001; self.alpha_local_step *= 0.5 
                self.state = SolutionCandidate.State.STUCK 
                if self.stuck_counter >= GLOBAL_MAX_STUCK_FOR_RETIREMENT: self.state = SolutionCandidate.State.RETIRED
                self.u_k = (np.random.rand(self.M_rows)+1j*np.random.rand(self.M_rows))/np.sqrt(self.M_rows)
                self.right_v_k = (np.random.rand(self.M_cols)+1j*np.random.rand(self.M_cols))/np.sqrt(self.M_cols)
                self.sigma_k = 1.0 
        else: 
            target_A_for_solve = current_matrix_A; rhs_for_solve = None; current_main_vec_ref = None 
            if self.problem_type == ProblemType.EIGENVALUE:
                if np.linalg.norm(self.v_k) < 1e-10: 
                    self.v_k = (np.random.rand(self.N_diag) + 1j*np.random.rand(self.N_diag)); self.v_k /= np.linalg.norm(self.v_k)
                    self.stuck_counter += 1; self.num_resets += 1;
                    # Cannot raise here directly as it circumvents fallback, just reinit and proceed to solver
                    print(f"Candidate {self.id}: Eigenvector collapsed, reinitialized randomly before InverseIterateSolver.")
                rayleigh_quotient_denom = np.vdot(self.v_k, self.v_k)
                if np.abs(rayleigh_quotient_denom) < 1e-12: # Avoid division by zero if v_k is tiny
                    self.lambda_k = complex(0.0,0.0) # Default lambda if v_k is near zero
                else:
                    self.lambda_k = np.vdot(self.v_k, current_matrix_A @ self.v_k) / rayleigh_quotient_denom

                target_A_for_solve = current_matrix_A - self.lambda_k * (sp.eye(self.N_diag, dtype=current_matrix_A.dtype) if is_matrix_sparse else np.eye(self.N_diag, dtype=current_matrix_A.dtype))
                rhs_for_solve = self.v_k 
                current_main_vec_ref = self.v_k 
            elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
                target_A_for_solve = current_matrix_A
                rhs_for_solve = self.b_vector
                current_main_vec_ref = self.x_k 
            try:
                new_vec_raw, self.local_psi_retries_needed = solver_instance.solve(target_A_for_solve, rhs_for_solve, self.stuck_counter)
                if self.problem_type == ProblemType.EIGENVALUE:
                    self.v_k = (1.0 - self.alpha_local_step) * self.v_k + self.alpha_local_step * new_vec_raw 
                    norm_v_k = np.linalg.norm(self.v_k)
                    if norm_v_k > 1e-10: self.v_k /= norm_v_k
                    else: self.v_k = (np.random.rand(self.N_diag)+1j*np.random.rand(self.N_diag))/np.sqrt(self.N_diag)
                elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
                    self.x_k = (1.0 - self.alpha_local_step) * current_main_vec_ref + self.alpha_local_step * new_vec_raw 
                self.stuck_counter = max(0, self.stuck_counter - 1)
            except (RuntimeError, ValueError) as e: 
                self.stuck_counter += 1; self.w_k *= 0.001
                self.alpha_local_step = max(self.alpha_local_step * 0.5, 1e-6)
                if self.stuck_counter >= GLOBAL_MAX_STUCK_FOR_RETIREMENT: 
                    self.state = SolutionCandidate.State.RETIRED; self.num_resets += 1
                else: 
                    self.state = SolutionCandidate.State.STUCK; self.initialize_random_solution()
        
        A_res_calc = self.problem_matrix
        if self.problem_type == ProblemType.EIGENVALUE:
            self.residual_k = np.linalg.norm(A_res_calc @ self.v_k - self.lambda_k * self.v_k) if self.v_k is not None else float('inf')
        elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
            self.residual_k = np.linalg.norm(A_res_calc @ self.x_k - self.b_vector) if self.x_k is not None and self.b_vector is not None else float('inf')
        elif self.problem_type == ProblemType.SVD:
            self.residual_k = (np.linalg.norm(A_res_calc @ self.right_v_k - self.sigma_k * self.u_k) +                               np.linalg.norm(A_res_calc.conj().T @ self.u_k - self.sigma_k * self.right_v_k)) if self.right_v_k is not None and self.u_k is not None else float('inf')

        self.param_history.append(self.get_current_solution_params())
        self.residual_history.append(self.residual_k)

        if self.prev_residual > 1e-10: 
            if self.residual_k < self.prev_residual * 0.9: 
                self.alpha_local_step = min(self.alpha_local_step * 1.1, 1.0)
                if self.state != SolutionCandidate.State.CONVERGED: self.state = SolutionCandidate.State.REFINING
            elif self.residual_k > self.prev_residual * 1.5 and self.prev_residual > 1e-5: 
                self.alpha_local_step = max(self.alpha_local_step * 0.5, 1e-6)
                if self.state != SolutionCandidate.State.CONVERGED: self.state = SolutionCandidate.State.STUCK 
            else: 
                self.alpha_local_step = max(self.alpha_local_step * 0.95, 1e-6) 
                if self.state not in [SolutionCandidate.State.CONVERGED, SolutionCandidate.State.STUCK, SolutionCandidate.State.RETIRED]:
                     self.state = SolutionCandidate.State.EXPLORING 
        
        current_conv_tol = strat_params.get('current_convergence_threshold', GLOBAL_CONVERGENCE_RESIDUAL_TOL)
        current_sol_params = self.get_current_solution_params()
        solution_is_finite = False 
        if current_sol_params is not None:
            solution_is_finite = True 
            for p in current_sol_params:
                if p is None: solution_is_finite = False; break
                if isinstance(p, (np.ndarray, sp.spmatrix)): 
                    if not np.all(np.isfinite(p.data if sp.issparse(p) else p)): solution_is_finite = False; break
                elif not np.isfinite(p): solution_is_finite = False; break
        
        if self.residual_k < current_conv_tol and solution_is_finite: 
            self.state = SolutionCandidate.State.CONVERGED
            self.w_k = 1.0; self.stuck_counter = 0; self.alpha_local_step = 0.0

    def get_current_solution_params(self): # Unchanged
        if self.problem_type == ProblemType.EIGENVALUE: return (self.lambda_k, self.v_k)
        elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM: return (self.x_k,)
        elif self.problem_type == ProblemType.SVD: return (self.sigma_k, self.u_k, self.right_v_k)
        return None

# --- MAUS: The Universal Adaptive Matrix Solver (Main Class) ---
class MAUS_Solver: # Mostly unchanged, ensure __init__ and _diagnose_matrix_initial are from previous successful step
    def __init__(self, problem_matrix, problem_type, b_vector=None, initial_num_candidates=None, global_convergence_tol=1e-8):
        if isinstance(problem_matrix, (sp.spmatrix,)): self.M = problem_matrix.copy() 
        else: self.M = problem_matrix.astype(np.complex128) 
        self.N_rows, self.N_cols = self.M.shape; self.N_diag = self.N_rows 
        self.problem_type = problem_type
        self.b = b_vector.astype(np.complex128) if b_vector is not None else None 
        self.diag_info = self._diagnose_matrix_initial(self.M) 
        self.is_sparse_problem_init = self.diag_info['is_sparse_init'] 
        self.cond_number = self.diag_info['condition_number'] 
        self.problem_knowledge = {
            'matrix_type': 'Dense', 'spectrum_hint': 'Unknown', 'numerical_stability_state': 'Stable', 
            'local_solver_preference': 'direct_solve', 'effective_rank_SVD': min(self.N_rows, self.N_cols), 
            'true_matrix_is_singular': self.diag_info['is_singular'], 'is_sparse_problem': self.is_sparse_problem_init, 
            'is_hermitian': self.diag_info.get('is_hermitian', False),
            'is_complex_symmetric': self.diag_info.get('is_complex_symmetric', False)
        }
        if self.problem_knowledge['is_sparse_problem'] and not isinstance(self.M, (sp.csc_matrix, sp.csr_matrix, sp.coo_matrix)):
             self.M = sp.csc_matrix(self.M); self.problem_knowledge['matrix_type'] = 'Sparse' 
        self.strat_params = {
            'overall_psi_aggression_factor': 1.0, 'max_psi_retries': GLOBAL_MAX_PSI_ATTEMPTS, 
            'min_survival_weight': GLOBAL_MIN_WEIGHT_TO_SURVIVE_PRUNE, 'spawn_rate_multiplier': 1.0, 
            'convergence_tolerance': global_convergence_tol, 'current_convergence_threshold': global_convergence_tol, # Initialize current_conv_thresh to global_tol
        }
        self._set_initial_strategy() 
        initial_num_candidates = initial_num_candidates if initial_num_candidates is not None else (self.N_diag * 3)
        if self.problem_type == ProblemType.SVD: initial_num_candidates = max(initial_num_candidates, min(self.N_rows, self.N_cols) * 3) 
        self.candidates = [SolutionCandidate(self.M, self.problem_type, self.N_diag) for _ in range(initial_num_candidates)]
        SolutionCandidate._candidate_id_counter = initial_num_candidates 
        print(f"MAUS Initialized with {initial_num_candidates} candidates for {self.problem_type.name} (Dims={self.N_rows}x{self.N_cols}).")
        print(f"Initial matrix diagnostics: Cond={self.cond_number:.2e}, MatrixType={self.problem_knowledge['matrix_type']}, Hermitian={self.problem_knowledge['is_hermitian']}. Stability: {self.problem_knowledge['numerical_stability_state']}.")
        self.landscape_energy = 1.0; self.avg_residual = 1.0; self.avg_stuckness = 0.0
        self.num_distinct_converged_solutions = 0; self.converged_solutions = []

    def _diagnose_matrix_initial(self, matrix): # From previous successful modification
        diag_info = {
            'is_hermitian': False, 'is_complex_symmetric': False, 'is_sparse_init': False, 
            'condition_number': np.inf, 'is_singular': False
        }
        if isinstance(matrix, np.ndarray): 
            diag_info['is_sparse_init'] = (np.count_nonzero(matrix) / matrix.size) < 0.25 if matrix.size > 0 else False
            try:
                if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
                    if np.allclose(matrix, matrix.conj().T): diag_info['is_hermitian'] = True
                    if np.allclose(matrix, matrix.T): diag_info['is_complex_symmetric'] = True
            except Exception: pass 
        elif isinstance(matrix, sp.spmatrix): 
            diag_info['is_sparse_init'] = True
            try:
                if matrix.shape[0] == matrix.shape[1]:
                    if matrix.shape[0] * matrix.shape[1] > 1e7: 
                        print("Warning: Sparse matrix too large for dense conversion in property diagnosis. Assuming False for Hermitian/Symmetric.")
                    else:
                        M_dense = matrix.todense()
                        if np.allclose(M_dense, M_dense.conj().T): diag_info['is_hermitian'] = True
                        if np.allclose(M_dense, M_dense.T): diag_info['is_complex_symmetric'] = True
            except Exception: pass
        cond_num_val = np.inf; is_singular_val = False
        if not diag_info['is_sparse_init'] and isinstance(matrix, np.ndarray) and            matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] and matrix.size > 0:
            try: 
                cond_num_val = np.linalg.cond(matrix)
                if np.isinf(cond_num_val) or cond_num_val > 1e15: is_singular_val = True
            except np.linalg.LinAlgError: cond_num_val = np.inf; is_singular_val = True
        diag_info['condition_number'] = cond_num_val; diag_info['is_singular'] = is_singular_val
        return diag_info

    def _set_initial_strategy(self): # Unchanged from previous version
        if self.cond_number > 1e12: 
            self.problem_knowledge['numerical_stability_state'] = 'Critical'; self.strat_params['overall_psi_aggression_factor'] = 50.0
            self.strat_params['max_psi_retries'] = GLOBAL_MAX_PSI_ATTEMPTS*2; self.strat_params['current_convergence_threshold'] = 1e-2
            self.problem_knowledge['local_solver_preference'] = 'iterative_gmres'
        elif self.cond_number > 1e6:
             self.problem_knowledge['numerical_stability_state'] = 'Fragile'; self.strat_params['overall_psi_aggression_factor'] = 10.0
             self.problem_knowledge['local_solver_preference'] = 'iterative_gmres'; self.strat_params['current_convergence_threshold'] = 1e-4 # Slightly tighter for fragile
        else: 
             self.problem_knowledge['numerical_stability_state'] = 'Stable'; self.problem_knowledge['local_solver_preference'] = 'direct_solve'
             self.strat_params['current_convergence_threshold'] = self.strat_params['convergence_tolerance'] # Use global for stable
        if self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM and self.diag_info.get('is_singular',False):
            self.problem_knowledge['true_matrix_is_singular'] = True; self.problem_knowledge['local_solver_preference'] = 'iterative_gmres' 
            self.strat_params['overall_psi_aggression_factor'] = max(self.strat_params['overall_psi_aggression_factor'], 20.0) 
        if self.problem_type == ProblemType.SVD: 
            if self.problem_knowledge['numerical_stability_state'] == 'Stable': self.strat_params['overall_psi_aggression_factor'] = max(self.strat_params['overall_psi_aggression_factor'], 2.0)
            self.strat_params['current_convergence_threshold'] = max(1e-5, self.strat_params['convergence_tolerance']) # SVD often needs slightly looser for tiny sigma

    def _update_global_diagnostics(self, iteration): # Mostly unchanged, minor robustness for None
        total_active_candidates = len(self.candidates); sum_residuals = 0.0; sum_stuck_counters = 0; sum_confidence = 0.0; num_converged_all_types = 0
        self.num_distinct_converged_solutions = 0; self.converged_solutions = []; current_sigma_magnitudes = []
        for c in self.candidates:
            if c.state == SolutionCandidate.State.CONVERGED:
                num_converged_all_types += 1; current_tuple = c.get_current_solution_params(); is_distinct = True
                if current_tuple is None or any(p is None for p in current_tuple): continue
                if self.problem_type == ProblemType.EIGENVALUE: # ... (rest of logic as before, ensuring None checks for tuple elements)
                    for s_item in self.converged_solutions:
                        s_lam, s_vec = s_item[0], s_item[1]
                        if s_lam is None or s_vec is None or current_tuple[0] is None or current_tuple[1] is None: continue
                        effective_tol = GLOBAL_LAMBDA_SIMILARITY_TOL + np.abs(s_lam) * 1e-6
                        if np.abs(current_tuple[0] - s_lam) < effective_tol and np.abs(np.vdot(current_tuple[1], s_vec)) > GLOBAL_VECTOR_SIMILARITY_TOL:
                            is_distinct = False; break
                elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM: 
                    if len(self.converged_solutions) > 0 and self.converged_solutions[0][0] is not None and current_tuple[0] is not None and                        np.linalg.norm(current_tuple[0] - self.converged_solutions[0][0]) < self.strat_params['convergence_tolerance'] * 100:
                        is_distinct = False
                elif self.problem_type == ProblemType.SVD:
                    if current_tuple[0] is None: continue 
                    max_s = max((cand.sigma_k.real for cand in self.candidates if cand.sigma_k is not None and cand.sigma_k.real > 0), default=1.0)
                    if current_tuple[0].real / max_s < GLOBAL_SIGMA_SIMILARITY_TOL_REL : is_distinct = False 
                    if is_distinct:
                        for s_item in self.converged_solutions:
                            s_sigma, s_u, s_v = s_item[0], s_item[1], s_item[2]
                            if s_sigma is None or current_tuple[0] is None: continue
                            effective_abs_tol = GLOBAL_SIGMA_SIMILARITY_TOL_ABS; effective_rel_tol = s_sigma * GLOBAL_SIGMA_SIMILARITY_TOL_REL
                            if np.abs(current_tuple[0] - s_sigma) < max(effective_abs_tol, effective_rel_tol) and                                (s_u is None or current_tuple[1] is None or np.abs(np.vdot(current_tuple[1], s_u)) > GLOBAL_VECTOR_SIMILARITY_TOL) and                                (s_v is None or current_tuple[2] is None or np.abs(np.vdot(current_tuple[2], s_v)) > GLOBAL_VECTOR_SIMILARITY_TOL):
                                is_distinct = False; break
                    if current_tuple[0] is not None: current_sigma_magnitudes.append(current_tuple[0].real)
                if is_distinct: self.converged_solutions.append(current_tuple); self.num_distinct_converged_solutions +=1
            if c.state not in [SolutionCandidate.State.CONVERGED, SolutionCandidate.State.RETIRED]:
                sum_residuals += c.residual_k if np.isfinite(c.residual_k) else (self.strat_params['current_convergence_threshold'] * 100)
                sum_stuck_counters += c.stuck_counter; sum_confidence += c.w_k
        non_conv_retired_count = max(1, total_active_candidates - num_converged_all_types)
        self.avg_residual = sum_residuals / non_conv_retired_count; self.avg_stuckness = sum_stuck_counters / non_conv_retired_count
        norm_avg_res = self.avg_residual / (self.strat_params['current_convergence_threshold'] * 10)
        norm_avg_stuck = self.avg_stuckness / (GLOBAL_MAX_STUCK_FOR_RETIREMENT * 2)
        target_sols_N_global = self.N_diag
        if self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM: target_sols_N_global = 1
        elif self.problem_type == ProblemType.SVD: 
            if len(current_sigma_magnitudes) > 1:
                sorted_sigmas = sorted([s for s in current_sigma_magnitudes if s > GLOBAL_SIGMA_SIMILARITY_TOL_ABS], reverse=True)
                if sorted_sigmas:
                    max_sigma_val = sorted_sigmas[0]
                    rank_detected = sum(1 for s_val in sorted_sigmas if s_val / max_sigma_val > GLOBAL_SIGMA_SIMILARITY_TOL_REL)
                    self.problem_knowledge['effective_rank_SVD'] = min(rank_detected if rank_detected > 0 else 1, min(self.N_rows, self.N_cols), max(1, self.problem_knowledge.get('effective_rank_SVD',1)))
            target_sols_N_global = self.problem_knowledge.get('effective_rank_SVD', min(self.N_rows, self.N_cols))
        norm_missing_sols = (target_sols_N_global - self.num_distinct_converged_solutions) / max(1, target_sols_N_global)
        self.landscape_energy = max(0.0, min(1.0, (norm_avg_res*0.4) + (norm_avg_stuck*0.3) + (norm_missing_sols*0.3)))
        if self.avg_stuckness > GLOBAL_MAX_STUCK_FOR_RETIREMENT*0.5: self.problem_knowledge['numerical_stability_state'] = 'Critical'
        elif self.avg_stuckness > GLOBAL_MAX_STUCK_FOR_PRUNING*0.5: self.problem_knowledge['numerical_stability_state'] = 'Fragile'
        else: self.problem_knowledge['numerical_stability_state'] = 'Stable'

    def _adjust_global_strategy(self, iteration): # Unchanged from previous version
        current_stability = self.problem_knowledge['numerical_stability_state']
        psi_factor_adj = 1.0; spawn_adj = 1.0; conv_thresh_adj = 1.0
        if self.landscape_energy > 0.6 and current_stability == 'Critical':
            self.problem_knowledge['local_solver_preference'] = 'iterative_gmres'
            psi_factor_adj = 1.1; spawn_adj = 1.2; conv_thresh_adj = 1.05
            self.strat_params['overall_psi_aggression_factor'] = min(200.0, self.strat_params['overall_psi_aggression_factor'] * psi_factor_adj)
            self.strat_params['spawn_rate_multiplier'] = min(10.0, self.strat_params['spawn_rate_multiplier'] * spawn_adj)
            self.strat_params['current_convergence_threshold'] = max(self.strat_params['convergence_tolerance'] * 50, self.strat_params['current_convergence_threshold'] * conv_thresh_adj)
        elif self.landscape_energy > 0.4 and current_stability == 'Fragile':
            self.problem_knowledge['local_solver_preference'] = 'iterative_gmres'
            psi_factor_adj = 1.05; spawn_adj = 1.1; conv_thresh_adj = 1.02
            self.strat_params['overall_psi_aggression_factor'] = min(50.0, self.strat_params['overall_psi_aggression_factor'] * psi_factor_adj)
            self.strat_params['spawn_rate_multiplier'] = min(5.0, self.strat_params['spawn_rate_multiplier'] * spawn_adj)
            self.strat_params['current_convergence_threshold'] = max(self.strat_params['convergence_tolerance'] * 5, self.strat_params['current_convergence_threshold'] * conv_thresh_adj)
        elif self.landscape_energy < 0.2 and current_stability == 'Stable':
            self.problem_knowledge['local_solver_preference'] = 'direct_solve' # Revert to direct if stable and low energy
            psi_factor_adj = 0.9; spawn_adj = 0.9; conv_thresh_adj = 0.9
            self.strat_params['overall_psi_aggression_factor'] = max(1.0, self.strat_params['overall_psi_aggression_factor'] * psi_factor_adj)
            self.strat_params['spawn_rate_multiplier'] = max(0.01, self.strat_params['spawn_rate_multiplier'] * spawn_adj)
            self.strat_params['current_convergence_threshold'] = max(self.strat_params['convergence_tolerance'], self.strat_params['current_convergence_threshold'] * conv_thresh_adj) # Ensure it doesn't go below base tolerance
        # Clamp params
        self.strat_params['overall_psi_aggression_factor'] = max(1.0, min(200.0, self.strat_params['overall_psi_aggression_factor']))
        self.strat_params['spawn_rate_multiplier'] = max(0.01, min(10.0, self.strat_params['spawn_rate_multiplier']))
        self.strat_params['current_convergence_threshold'] = max(self.strat_params['convergence_tolerance'], min(1.0, self.strat_params['current_convergence_threshold']))


    def _manage_candidates(self, iteration): # Mostly unchanged, minor robustness for None
        initial_candidate_count = len(self.candidates); survivors = []
        sorted_candidates = sorted(self.candidates, key=lambda x: (-x.w_k, x.residual_k if np.isfinite(x.residual_k) else float('inf')))
        for c in sorted_candidates: # ... (rest of logic as before, ensuring None checks)
            is_redundant_in_survivors = False
            if c.state == SolutionCandidate.State.CONVERGED: 
                for s_c in survivors:
                    if s_c.state == SolutionCandidate.State.CONVERGED:
                        res_tuple_c = c.get_current_solution_params(); res_tuple_s_c = s_c.get_current_solution_params()
                        if res_tuple_c is None or res_tuple_s_c is None or any(p is None for p in res_tuple_c) or any(p is None for p in res_tuple_s_c): continue
                        if self.problem_type == ProblemType.EIGENVALUE:
                            if np.abs(res_tuple_c[0] - res_tuple_s_c[0]) < (GLOBAL_LAMBDA_SIMILARITY_TOL + np.abs(res_tuple_s_c[0]) * 1e-6) and                                np.abs(np.vdot(res_tuple_c[1], res_tuple_s_c[1])) > GLOBAL_VECTOR_SIMILARITY_TOL: is_redundant_in_survivors = True; break
                        elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM: 
                            if np.linalg.norm(res_tuple_c[0] - res_tuple_s_c[0]) < self.strat_params['convergence_tolerance'] * 10: is_redundant_in_survivors = True; break
                        elif self.problem_type == ProblemType.SVD:
                            if res_tuple_s_c[0].real < GLOBAL_SIGMA_SIMILARITY_TOL_ABS / 100: is_redundant_in_survivors = False 
                            elif np.abs(res_tuple_c[0] - res_tuple_s_c[0]) < max(GLOBAL_SIGMA_SIMILARITY_TOL_ABS, res_tuple_s_c[0] * GLOBAL_SIGMA_SIMILARITY_TOL_REL) and                                np.abs(np.vdot(res_tuple_c[1], res_tuple_s_c[1])) > GLOBAL_VECTOR_SIMILARITY_TOL and                                np.abs(np.vdot(res_tuple_c[2], res_tuple_s_c[2])) > GLOBAL_VECTOR_SIMILARITY_TOL: is_redundant_in_survivors = True; break
            if is_redundant_in_survivors: c.state = SolutionCandidate.State.RETIRED
            elif c.state == SolutionCandidate.State.RETIRED: pass 
            elif (c.w_k < self.strat_params['min_survival_weight'] and c.state != SolutionCandidate.State.CONVERGED) or                  (c.stuck_counter >= GLOBAL_MAX_STUCK_FOR_RETIREMENT and c.state != SolutionCandidate.State.CONVERGED): c.state = SolutionCandidate.State.RETIRED
            else: survivors.append(c)
        self.candidates = survivors
        num_pruned = initial_candidate_count - len(self.candidates)
        # if num_pruned > 0: print(f"  Population Management: Pruned/Retired {num_pruned} candidates.")
        target_distinct_count_for_spawn = self.N_diag
        if self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM: target_distinct_count_for_spawn = 1
        elif self.problem_type == ProblemType.SVD: target_distinct_count_for_spawn = self.problem_knowledge.get('effective_rank_SVD', min(self.N_rows, self.N_cols))
        desired_pop_base = max(5, int(self.N_diag * 1.5 if self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM else self.N_diag * 2))
        if self.problem_type == ProblemType.SVD: desired_pop_base = max(desired_pop_base, int(target_distinct_count_for_spawn * 2.5))
        num_to_spawn = max(0, desired_pop_base - len(self.candidates)) + max(0, target_distinct_count_for_spawn - self.num_distinct_converged_solutions)
        num_to_spawn = min(int(num_to_spawn * self.strat_params['spawn_rate_multiplier']), self.N_diag * 2, 15)
        if num_to_spawn > 0:
            # print(f"  SPAWNING: Adding {num_to_spawn} new candidates (Current:{len(self.candidates)}, Distinct Found:{self.num_distinct_converged_solutions}/{target_distinct_count_for_spawn}).")
            for _ in range(num_to_spawn): # ... (rest of spawning logic as before)
                new_candidate_init_vals = {}
                if self.num_distinct_converged_solutions > 0 and self.landscape_energy < 0.8 and self.converged_solutions:
                    base_sol_tuple = random.choice(self.converged_solutions)
                    if base_sol_tuple is None or any(p is None for p in base_sol_tuple): continue 
                    if self.problem_type == ProblemType.EIGENVALUE:
                        new_candidate_init_vals['initial_lambda'] = base_sol_tuple[0] + (random.random()*0.1-0.05 + 1j*(random.random()*0.1-0.05)) * (0.1 + self.landscape_energy)
                        v_pert = (np.random.rand(self.N_diag)-0.5 + 1j*(np.random.rand(self.N_diag)-0.5)) * (0.1 + self.landscape_energy)
                        new_v = base_sol_tuple[1] + v_pert; norm_new_v = np.linalg.norm(new_v)
                        new_candidate_init_vals['initial_v'] = new_v / norm_new_v if norm_new_v > 1e-9 else (np.random.rand(self.N_diag)+1j*np.random.rand(self.N_diag))/np.sqrt(self.N_diag)
                new_candidate = SolutionCandidate(self.M, self.problem_type, self.N_diag, **new_candidate_init_vals, initial_weight=0.01)
                new_candidate.alpha_local_step = GLOBAL_DEFAULT_ALPHA_V_INITIAL * (1 + self.strat_params['overall_psi_aggression_factor']/10.0)
                self.candidates.append(new_candidate)

    def evolve(self, max_iterations=100): # Mostly unchanged, minor robustness for None
        print(f"--- Starting MAUS Evolution for {max_iterations} iterations ({self.problem_type.name}) ---")
        self.true_solution = None # Initialize
        try: # ... (true solution calculation as before, ensuring None checks)
            M_dense_for_true = self.M.todense() if sp.issparse(self.M) else self.M
            if M_dense_for_true.size == 0: raise ValueError("Matrix is empty.")
            if self.problem_type == ProblemType.EIGENVALUE:
                if M_dense_for_true.shape[0] != M_dense_for_true.shape[1]: raise ValueError("Non-square matrix for Eigenvalue.")
                self.true_solution = sla.eigvals(M_dense_for_true); self.true_solution.sort()
            elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM:
                if self.b is None: raise ValueError("b_vector is None."); 
                if M_dense_for_true.shape[0] != self.b.shape[0]: raise ValueError("A,b shape mismatch.")
                self.true_solution = sla.solve(M_dense_for_true, self.b, assume_a='general')
            elif self.problem_type == ProblemType.SVD:
                k_svds = min(M_dense_for_true.shape) -1 if min(M_dense_for_true.shape) >1 else 1 # Ensure k is at least 1 or min_dim -1
                if sp.issparse(self.M) and self.M.shape[0]*self.M.shape[1] > 0 and k_svds > 0 : _, s, _ = spla.svds(self.M, k=k_svds, tol=1e-9, return_singular_vectors=False); self.true_solution = sorted(s.tolist(), reverse=True)
                elif not sp.issparse(self.M) and M_dense_for_true.size > 0 : _, s, _ = sla.svd(M_dense_for_true, compute_uv=False); self.true_solution = sorted(s.tolist(), reverse=True)
                elif k_svds <= 0 and M_dense_for_true.size > 0 : _, s, _ = sla.svd(M_dense_for_true, compute_uv=False); self.true_solution = sorted(s.tolist(), reverse=True)
                else: self.true_solution = []
        except (np.linalg.LinAlgError, ValueError, spla.ArpackNoConvergence) as e: print(f"NumPy reference calculation failed: {e}."); self.true_solution = None
        
        for i in range(max_iterations): # ... (main loop as before)
            self._update_global_diagnostics(i + 1); self._adjust_global_strategy(i + 1)
            for candidate in self.candidates:
                if candidate.state not in [SolutionCandidate.State.CONVERGED, SolutionCandidate.State.RETIRED]: 
                    candidate.update_solution_step(self.M, self.b, self.strat_params, self.problem_knowledge)
            self._manage_candidates(i + 1)
            target_sols_disp = self.N_diag
            if self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM: target_sols_disp = 1
            elif self.problem_type == ProblemType.SVD: target_sols_disp = self.problem_knowledge.get('effective_rank_SVD', min(self.N_rows, self.N_cols))
            if (i+1) % 20 == 0 or i == max_iterations -1 : # Print every 20 iter or last iter
                print(f"Iter {i+1}/{max_iterations}: Energy={self.landscape_energy:.2f}, AvgRes={self.avg_residual:.2e}, Conv={self.num_distinct_converged_solutions}/{target_sols_disp}, Stab={self.problem_knowledge['numerical_stability_state']}")
            if self.num_distinct_converged_solutions >= target_sols_final and self.landscape_energy < 0.05 and self.avg_residual < self.strat_params['convergence_tolerance'] : 
                print(f"MAUS converged early at iteration {i+1}."); break
            if i == max_iterations -1 and self.num_distinct_converged_solutions < target_sols_final : print(f"WARNING: Max iterations. Found {self.num_distinct_converged_solutions}/{target_sols_final}.")
        
        print("--- MAUS Evolution COMPLETE ---"); print("Final Report:") # ... (final reporting as before, ensuring None checks)
        final_solutions_print = []
        if self.problem_type == ProblemType.EIGENVALUE: final_solutions_print = sorted(self.converged_solutions, key=lambda x: (x[0].real, x[0].imag) if x[0] is not None else (float('inf'), float('inf')))
        elif self.problem_type == ProblemType.SVD: final_solutions_print = sorted(self.converged_solutions, key=lambda x: -x[0].real if x[0] is not None else float('-inf'))
        else: final_solutions_print = self.converged_solutions
        for sol_idx, sol_tuple in enumerate(final_solutions_print):
            if sol_tuple is None or any(p is None for p in sol_tuple): print(f"  Solution {sol_idx+1}: Invalid"); continue
            if self.problem_type == ProblemType.EIGENVALUE: l, v = sol_tuple; print(f"  Eig {sol_idx+1}: λ={l:.6e}, Res={np.linalg.norm((self.M @ v) - (l * v)):.2e}")
            elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM: x = sol_tuple[0]; print(f"  LinSolve {sol_idx+1}: X_norm1={np.linalg.norm(x,1):.6e}, Res={np.linalg.norm((self.M @ x) - self.b):.2e}")
            elif self.problem_type == ProblemType.SVD: s, u, v_r = sol_tuple; print(f"  SVD {sol_idx+1}: σ={s:.6e}, Res={np.linalg.norm((self.M @ v_r) - (s * u)) + np.linalg.norm((self.M.conj().T @ u) - (s * v_r)):.2e}")
        if self.true_solution is not None and self.num_distinct_converged_solutions > 0 and final_solutions_print:
            print("--- Comparison to NumPy ---") # ... (comparison logic as before, ensuring None checks and correct array slicing)
            if self.problem_type == ProblemType.EIGENVALUE:
                discovered_eigs = np.array(sorted([s[0] for s in final_solutions_print if s[0] is not None], key=lambda x: (x.real, x.imag)))
                true_eigs_s = self.true_solution[:len(discovered_eigs)]
                if discovered_eigs.size > 0 and true_eigs_s.size > 0: print(f"Mean abs error (eigs): {np.sum(np.abs(discovered_eigs - true_eigs_s)) / len(discovered_eigs):.2e}")
            elif self.problem_type == ProblemType.SOLVE_LINEAR_SYSTEM and final_solutions_print[0][0] is not None and self.true_solution is not None:
                true_n = np.linalg.norm(self.true_solution); print(f"Rel error (X): {np.linalg.norm(final_solutions_print[0][0] - self.true_solution) / true_n if true_n > 1e-10 else np.linalg.norm(final_solutions_print[0][0] - self.true_solution):.2e}")
            elif self.problem_type == ProblemType.SVD:
                discovered_sigmas = np.array(sorted([s[0].real for s in final_solutions_print if s[0] is not None], reverse=True))
                true_sigmas_s = np.array(self.true_solution[:len(discovered_sigmas)])
                if discovered_sigmas.size > 0 and true_sigmas_s.size > 0: true_n = np.linalg.norm(true_sigmas_s); print(f"Rel error (sigmas): {np.linalg.norm(discovered_sigmas - true_sigmas_s) / true_n if true_n > 1e-10 else np.linalg.norm(discovered_sigmas - true_sigmas_s):.2e}")

# --- TEST SCENARIOS --- (Mostly unchanged, slight adjustments for robustness if needed)
def create_dynamic_solve_matrix_and_b(N, t_step, time_max_iter=100): # Unchanged
    t_norm = t_step / time_max_iter 
    H_base = sla.hilbert(N).astype(np.complex128); H_diag_boost = np.diag(np.ones(N) * N * 0.1) 
    singular_inductor = np.zeros((N,N),dtype=np.complex128); singular_inductor[0,N-1]=1.0; singular_inductor[N-1,0]=-1.0
    M_val = H_base + H_diag_boost + np.sin(t_step*2*np.pi/20)*(10.0+t_norm*20.0)*singular_inductor +             np.cos(t_step*2*np.pi/15)*(np.random.rand(N,N)+1j*np.random.rand(N,N))*1e-4
    b_vec = np.array([1,-1,0.5,-0.5,0.1][:N],dtype=np.complex128) * (1 + 0.1*np.sin(t_step*np.pi/10))
    return M_val, b_vec

def create_laplace_like_complex_eigen_for_MAUS(N, make_hermitian=False): # Unchanged
    M_val = np.zeros((N,N),dtype=np.complex128)
    for i in range(N): M_val[i,i]=-2.0; 
    for i in range(N-1): M_val[i,i+1]=1.0; M_val[i+1,i]=1.0
    M_val[0,2]=0.5; M_val[2,0]=0.5j; M_val[N-1,N-3]=0.8j; M_val[N-3,N-1]=0.8
    M_val[N//2-1,N//2]=1.5+0.5j; M_val[N//2,N//2-1]=-1.5+0.5j
    M_val += (np.random.rand(N,N)*2-1)*1e-3 + 1j*(np.random.rand(N,N)*2-1)*1e-3
    M_val[0,N-1]+=0.2; M_val[N-1,0]+=0.2j; M_val[N-1,N-1]=M_val[N-2,N-2]+1e-6
    if make_hermitian: M_val = (M_val + M_val.conj().T) / 2.0
    return M_val

def create_low_rank_svd_matrix_for_MAUS(M_rows, N_cols, target_rank=2): # Unchanged
    U = np.random.rand(M_rows, M_rows) + 1j*np.random.rand(M_rows, M_rows); Q_U, _ = sla.qr(U)
    V = np.random.rand(N_cols, N_cols) + 1j*np.random.rand(N_cols, N_cols); Q_V, _ = sla.qr(V)
    true_s_vals = np.zeros(min(M_rows, N_cols), dtype=float)
    for i in range(target_rank): true_s_vals[i] = 5.0 / (i+1)
    for i in range(target_rank, min(M_rows, N_cols)): true_s_vals[i] = 1e-7 * random.random()
    Sigma_mat = np.zeros((M_rows, N_cols), dtype=np.complex128); np.fill_diagonal(Sigma_mat, true_s_vals)
    A_lowrank = Q_U @ Sigma_mat @ Q_V.conj().T
    A_noise = (np.random.randn(M_rows,N_cols)+1j*np.random.randn(M_rows,N_cols))*1e-4
    return A_lowrank + A_noise

if __name__ == '__main__': # Unchanged from previous version
    MAX_ITER_SOLVE_LINEAR = 20 
    print("##################### SCENARIO 1: SOLVE AX=B (N=5, DYNAMIC) #####################")
    maus_solver_solve = MAUS_Solver(np.eye(5), problem_type=ProblemType.SOLVE_LINEAR_SYSTEM, b_vector=np.ones(5), initial_num_candidates=15, global_convergence_tol=1e-7)
    A_final, b_final = create_dynamic_solve_matrix_and_b(N=5, t_step=MAX_ITER_SOLVE_LINEAR-1, time_max_iter=MAX_ITER_SOLVE_LINEAR)
    maus_solver_solve.M = A_final; maus_solver_solve.b = b_final
    maus_solver_solve.diag_info = maus_solver_solve._diagnose_matrix_initial(A_final) 
    maus_solver_solve.problem_knowledge.update({
        'is_hermitian': maus_solver_solve.diag_info.get('is_hermitian', False),
        'is_complex_symmetric': maus_solver_solve.diag_info.get('is_complex_symmetric', False),
        'is_sparse_problem': maus_solver_solve.diag_info.get('is_sparse_init', False)
    })
    maus_solver_solve.evolve(max_iterations=50)
    M_eigen_test = create_laplace_like_complex_eigen_for_MAUS(8, make_hermitian=False)
    print("##################### SCENARIO 2A: EIGENVALUE (N=8, GENERAL COMPLEX) #####################")
    maus_solver_eigen_gen = MAUS_Solver(M_eigen_test, problem_type=ProblemType.EIGENVALUE, initial_num_candidates=30, global_convergence_tol=1e-7)
    maus_solver_eigen_gen.evolve(max_iterations=80) 
    M_eigen_hermitian_test = create_laplace_like_complex_eigen_for_MAUS(8, make_hermitian=True)
    print("##################### SCENARIO 2B: EIGENVALUE (N=8, HERMITIAN COMPLEX) #####################")
    maus_solver_eigen_herm = MAUS_Solver(M_eigen_hermitian_test, problem_type=ProblemType.EIGENVALUE, initial_num_candidates=30, global_convergence_tol=1e-7)
    maus_solver_eigen_herm.evolve(max_iterations=50) 
    M_svd_test = create_low_rank_svd_matrix_for_MAUS(5, 4, target_rank=2) 
    print("##################### SCENARIO 3: SVD (N=5x4, NEAR-LOW-RANK) #####################")
    maus_solver_svd = MAUS_Solver(M_svd_test, problem_type=ProblemType.SVD, initial_num_candidates=25, global_convergence_tol=1e-6)
    maus_solver_svd.evolve(max_iterations=100)
