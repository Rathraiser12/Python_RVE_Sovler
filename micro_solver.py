#
# FILE: micro_solver.py
#
import sys
import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc
from scipy.spatial import KDTree
from tqdm import trange

from mesh_utils import load_3d_mesh
from element import (
    get_tet10_element_properties,
    shape_functions_and_derivatives_tet10,
    calculate_jacobian_and_derivatives,
    assemble_B_matrix_tet10,
    assemble_nonlinear_B_matrix_tet10,
)

from material import (
    get_material_params,
    calculate_stress_from_paper,   # PK2 stress
    calculate_analytic_tangent,    # Mandel 6x6 tangent dS/dE
    _sym_to_mandel6,               # Mandel Voigt mapping
)

NODES_PER_ELEM = 10
DOFS_PER_ELEM = NODES_PER_ELEM * 3  # = 30

_SQ2 = np.sqrt(2.0)


class NonPositiveJError(RuntimeError):
    pass


class PBC_Manager:
    """
    Handles boundary conditions via master–slave elimination.

    mode = "pbc"  -> periodic BCs on fluctuation field u~
    mode = "kubc" -> kinematic uniform BC (u~ = 0 on outer boundary)
    """

    def __init__(self, nodes, n_dof, mode="pbc"):
        """
        nodes : (n_nodes, 3) node coordinates
        n_dof : total DOFs (= 3 * n_nodes)
        mode  : "pbc" or "kubc"
        """
        self.nodes = nodes
        self.n_dof = n_dof
        self.n_nodes = nodes.shape[0]
        self.mode = mode

        self.dims, self.min_coords, self.max_coords = self._get_rve_dimensions()

        # DOF sets
        self.free_dofs = np.arange(self.n_dof, dtype=int)
        self.slave_dofs = np.array([], dtype=int)

        # slave_dof -> (master_dof or None, offset)
        self.constraints = {}
        self.n_free = len(self.free_dofs)
        self.free_dof_map = {dof: i for i, dof in enumerate(self.free_dofs)}

        # Transformation u_all = T @ u_free + u_offset
        self.T = sp.eye(self.n_dof, format="csr")
        self.u_offset = np.zeros(self.n_dof)

    def _get_rve_dimensions(self):
        min_coords = np.min(self.nodes, axis=0)
        max_coords = np.max(self.nodes, axis=0)
        dims = max_coords - min_coords
        if np.any(dims <= 1e-6):
            raise ValueError(f"RVE has zero or negative dimension: {dims}. Check mesh.")
        return dims, min_coords, max_coords

    def build_mapping(self, F_macro, pbc_tol=1e-5, verbose=True):
        """
        Build constraint mapping for either periodic BCs ("pbc") or
        kinematic uniform BCs ("kubc") on the fluctuation field u~.

        After this call we have:
          - self.constraints: dict[slave_dof] = (master_dof or None, offset)
          - self.slave_dofs, self.free_dofs
          - self.T, self.u_offset via _build_transformation()
        """
        # Only used for total-displacement PBCs; here u is fluctuation, so offset=0
        # kept F_minus_I only for compatibility / future use
        F_minus_I = F_macro - np.identity(3)

        # ---- KUBC branch: u~ = 0 on outer boundary ----
        if self.mode == "kubc":
            is_on_min = np.isclose(self.nodes, self.min_coords, atol=pbc_tol)
            is_on_max = np.isclose(self.nodes, self.max_coords, atol=pbc_tol)
            is_boundary = np.any(is_on_min | is_on_max, axis=1)
            boundary_nodes = np.where(is_boundary)[0]

            self.constraints = {}
            slave_dof_list = []

            for node_idx in boundary_nodes:
                for k in range(3):
                    dof = node_idx * 3 + k
                    self.constraints[dof] = (None, 0.0)  # fixed DOF
                    slave_dof_list.append(dof)

            self.slave_dofs = np.array(sorted(set(slave_dof_list)), dtype=int)
            all_dofs = np.arange(self.n_dof, dtype=int)
            self.free_dofs = np.setdiff1d(all_dofs, self.slave_dofs, assume_unique=True)
            self.n_free = len(self.free_dofs)
            self.free_dof_map = {dof: i for i, dof in enumerate(self.free_dofs)}

            if verbose:
                print(f"[KUBC] Fixed {len(self.slave_dofs)} boundary DOFs. Free DOFs: {self.n_free}")

            self._build_transformation()
            return

        # ---- PBC branch (default) ----
        if verbose:
            print("\nApplying Periodic Boundary Conditions for F_macro:")
            print(f"  RVE Dims: {self.dims}")

        self.constraints = {}
        slave_dof_list = []

        # Face classification
        x, y, z = self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2]
        x_min, y_min, z_min = self.min_coords
        x_max, y_max, z_max = self.max_coords

        on_x_min = np.isclose(x, x_min, atol=pbc_tol)
        on_x_max = np.isclose(x, x_max, atol=pbc_tol)
        on_y_min = np.isclose(y, y_min, atol=pbc_tol)
        on_y_max = np.isclose(y, y_max, atol=pbc_tol)
        on_z_min = np.isclose(z, z_min, atol=pbc_tol)
        on_z_max = np.isclose(z, z_max, atol=pbc_tol)

        # Reference corner: (x_min, y_min, z_min) – remove rigid translation
        corner_mask = on_x_min & on_y_min & on_z_min
        corner_indices = np.where(corner_mask)[0]
        if len(corner_indices) == 0:
            dists = np.linalg.norm(self.nodes - self.min_coords, axis=1)
            ref_corner_idx = int(np.argmin(dists))
            if verbose:
                print("  [PBC] Warning: Could not find exact (x_min,y_min,z_min); using closest node.")
        else:
            ref_corner_idx = int(corner_indices[0])

        # Fix all 3 DOFs of reference corner: u~ = 0 there
        for k in range(3):
            dof = ref_corner_idx * 3 + k
            self.constraints[dof] = (None, 0.0)
            slave_dof_list.append(dof)

        if verbose:
            print(f"  [PBC] Reference corner node {ref_corner_idx} fixed (all 3 DOFs).")

        slave_nodes_used = set([ref_corner_idx])

        def build_face_tree(face_mask, proj_axes):
            face_indices = np.where(face_mask)[0]
            if len(face_indices) == 0:
                return None, np.array([], dtype=int)
            coords_proj = self.nodes[face_indices][:, proj_axes]
            tree = KDTree(coords_proj)
            return tree, face_indices

        def pair_direction(dir_name, min_mask, max_mask, proj_axes):
            nonlocal slave_dof_list
            tree, master_indices = build_face_tree(max_mask, proj_axes)
            if tree is None:
                if verbose:
                    print(f"  [PBC] Warning: no master face nodes for direction {dir_name}.")
                return 0

            min_indices = np.where(min_mask)[0]
            n_pairs_dir = 0

            for s_idx in min_indices:
                if s_idx in slave_nodes_used:
                    continue

                s_proj = self.nodes[s_idx, list(proj_axes)]
                dist, j = tree.query(s_proj)
                m_idx = int(master_indices[j])

                if dist > 1e-5 and verbose:
                    print(
                        f"  [PBC] Warning: large projected distance in {dir_name}-pairing: "
                        f"slave {s_idx} -> master {m_idx}, dist={dist:.3e}"
                    )

                # u~_s = u~_m  (purely periodic fluctuation field)
                for k in range(3):
                    s_dof = s_idx * 3 + k
                    m_dof = m_idx * 3 + k
                    self.constraints[s_dof] = (m_dof, 0.0)
                    slave_dof_list.append(s_dof)

                slave_nodes_used.add(s_idx)
                n_pairs_dir += 1

            if verbose:
                print(f"  [PBC] Direction {dir_name}: created {n_pairs_dir} slave–master node pairs.")
            return n_pairs_dir

        pair_direction("x", on_x_min, on_x_max, proj_axes=(1, 2))
        pair_direction("y", on_y_min, on_y_max, proj_axes=(0, 2))
        pair_direction("z", on_z_min, on_z_max, proj_axes=(0, 1))

        self.slave_dofs = np.unique(np.array(slave_dof_list, dtype=int))
        all_dofs = np.arange(self.n_dof, dtype=int)
        self.free_dofs = np.setdiff1d(all_dofs, self.slave_dofs, assume_unique=True)
        self.n_free = len(self.free_dofs)
        self.free_dof_map = {dof: i for i, dof in enumerate(self.free_dofs)}

        if verbose:
            print(f"  [PBC] Total slave DOFs: {len(self.slave_dofs)}, free DOFs: {self.n_free}")

        self._build_transformation()

    def _build_transformation(self):
        """
        Builds the sparse T matrix (n_dof x n_free) and u_offset vector (n_dof)
        """
        T_data, T_rows, T_cols = [], [], []
        u_offset_all = np.zeros(self.n_dof)

        memo = {}

        def find_master_chain(slave_dof):
            """
            Recursively finds ultimate free master and accumulated offset.
            Returns: (scale, root_master_dof, total_offset)
            """
            if slave_dof in memo:
                return memo[slave_dof]

            if slave_dof in self.free_dofs:
                memo[slave_dof] = (1.0, slave_dof, 0.0)
                return 1.0, slave_dof, 0.0

            if slave_dof not in self.constraints:
                print(f"Warning: DOF {slave_dof} not in constraints, treating as free.")
                self.free_dofs = np.append(self.free_dofs, slave_dof)
                self.free_dof_map[slave_dof] = self.n_free
                self.n_free += 1
                memo[slave_dof] = (1.0, slave_dof, 0.0)
                return 1.0, slave_dof, 0.0

            master_dof, offset = self.constraints[slave_dof]

            if master_dof is None:
                memo[slave_dof] = (0.0, None, offset)
                return 0.0, None, offset

            scale, root_master_dof, chain_offset = find_master_chain(master_dof)
            total_offset = scale * chain_offset + offset
            memo[slave_dof] = (scale, root_master_dof, total_offset)
            return scale, root_master_dof, total_offset

        # 1. Free DOFs map to themselves
        for i, dof in enumerate(self.free_dofs):
            T_data.append(1.0)
            T_rows.append(dof)
            T_cols.append(i)
            u_offset_all[dof] = 0.0

        # 2. Slave DOFs map to their free masters
        for s_dof in self.slave_dofs:
            scale, root_dof, total_offset = find_master_chain(s_dof)
            u_offset_all[s_dof] = total_offset
            if root_dof is not None:
                T_data.append(scale)
                T_rows.append(s_dof)
                T_cols.append(self.free_dof_map[root_dof])

        self.T = sp.coo_matrix((T_data, (T_rows, T_cols)), shape=(self.n_dof, self.n_free)).tocsr()
        self.u_offset = u_offset_all


class RVE_Solver:
    def __init__(self, mesh_filepath):
        print(f"Initializing 3D RVE_Solver (Heterogeneous, Quadratic tet10)...")

        self.nodes, self.elements, self.material_index_map = load_3d_mesh(mesh_filepath)
        if self.nodes is None:
            raise IOError("Failed to load mesh.")

        self.n_nodes = self.nodes.shape[0]
        self.n_elems = self.elements.shape[0]
        self.n_dof = self.n_nodes * 3

        print(f"  Mesh: {self.n_nodes} nodes, {self.n_elems} elements, {self.n_dof} DOFs.")

        self.gauss_points, self.gauss_weights = get_tet10_element_properties()
        self.n_gps = len(self.gauss_weights)

        self.element_data = self._precompute_element_data()

        self.u_free = np.zeros(self.n_dof)   # resized by PBC_Manager
        self.u_global = np.zeros(self.n_dof)

        self.F_macro = np.identity(3)
        self.F_macro_old = np.identity(3)

        self.material_params_matrix = get_material_params("Matrix")
        self.material_params_fibers = get_material_params("Fibers")

        mp = self.material_params_matrix
        self.G_init_matrix = 0.5 * np.sum(mp["mu"] * mp["alpha"])
        self.E_initial_approx = max(mp["kappa"], self.G_init_matrix)

        self.bc_mode = "pbc"
        self.pbc_manager = PBC_Manager(self.nodes, self.n_dof, mode=self.bc_mode)
        if not hasattr(self.pbc_manager, 'global_to_reduced'):
            print("Patching PBC Manager with global_to_reduced map...")
            self.num_nodes = len(self.nodes)
            total_dofs = self.num_nodes * 3
            self.pbc_manager.global_to_reduced = np.full(total_dofs, -1, dtype=int)
            # Use 'n_free' or 'free_dofs' depending on your class
            if hasattr(self.pbc_manager, 'free_dofs'):
                frees = self.pbc_manager.free_dofs
                self.pbc_manager.global_to_reduced[frees] = np.arange(len(frees))
            else:
                print("CRITICAL ERROR: PBC Manager has no free_dofs list!")
        self.element_nnz = DOFS_PER_ELEM * DOFS_PER_ELEM  # 900
        self.total_nnz = self.n_elems * self.element_nnz
        print(f"  Pre-allocating for {self.total_nnz} non-zeros (before summation).")

    def set_bc_mode(self, mode: str):
        if mode not in ("pbc", "kubc"):
            raise ValueError(f"Unknown bc mode '{mode}'. Use 'pbc' or 'kubc'.")
        self.bc_mode = mode
        self.pbc_manager = PBC_Manager(self.nodes, self.n_dof, mode=mode)


    def _get_pinned_dofs_indices(self):
            """
            Returns the indices of the reduced system that should be pinned
            to prevent Rigid Body Motion (Drifting).
            We simply pin the first 3 degrees of freedom of the reduced system,
            which correspond to the translation of the 'reference' node.
            """
            # Pin indices 0, 1, 2 of the reduced system (Free DOFs)
            # This constrains the first free node in X, Y, Z.
            return [0, 1, 2]
    # -------------------------------
    # MESH / ELEMENT PRECOMPUTATIONS
    # -------------------------------
    def _precompute_element_data(self):
        print(f"Pre-computing element data over {self.n_gps} Gauss points...")
        element_data = []

        for elem_idx in range(self.n_elems):
            elem_nodes_indices = self.elements[elem_idx]
            elem_nodes_coords = self.nodes[elem_nodes_indices]
            dof_indices = self._get_element_dof_indices(elem_idx)

            # NEW: precompute row/col pattern for this element once
            rows_e, cols_e = np.meshgrid(dof_indices, dof_indices, indexing="ij")
            elem_rows = rows_e.ravel().astype(np.int32)
            elem_cols = cols_e.ravel().astype(np.int32)

            gp_data_list = []
            for gp_idx, (L1, L2, L3, L4) in enumerate(self.gauss_points):
                gp_weight = self.gauss_weights[gp_idx]
                N, dN_d_L = shape_functions_and_derivatives_tet10(L1, L2, L3, L4)
                det_J, dN_d_global = calculate_jacobian_and_derivatives(elem_nodes_coords, dN_d_L)
                B = assemble_B_matrix_tet10(dN_d_global)
                w_int = det_J * gp_weight

                gp_data_list.append({
                    "dN_d_global": dN_d_global,
                    "B": B,
                    "integration_weight": w_int,
                })

            element_data.append({
                "elem_idx": elem_idx,
                "dof_indices": dof_indices,
                "gauss_points": gp_data_list,
                "rows": elem_rows,        # NEW
                "cols": elem_cols,        # NEW
            })

        return element_data

    def _get_element_dof_indices(self, elem_idx):
        node_indices = self.elements[elem_idx]
        dof_indices = np.zeros(DOFS_PER_ELEM, dtype=int)
        for i in range(NODES_PER_ELEM):
            dof_indices[i * 3] = node_indices[i] * 3
            dof_indices[i * 3 + 1] = node_indices[i] * 3 + 1
            dof_indices[i * 3 + 2] = node_indices[i] * 3 + 2
        return dof_indices

    def _get_material_params(self, elem_idx):
        mat_idx = self.material_index_map[elem_idx]
        return self.material_params_matrix if mat_idx == 0 else self.material_params_fibers

    # -------------------------------
    # BOUNDARY CONDITIONS
    # -------------------------------
    def apply_boundary_conditions(self, F_macro, pbc_tol=1e-5, verbose=True):
        self.F_macro_old = self.F_macro.copy()
        self.F_macro = F_macro

        self.pbc_manager.build_mapping(F_macro, pbc_tol, verbose)
        self.u_free = np.zeros(self.pbc_manager.n_free)
        self.u_global = self.pbc_manager.u_offset.copy()

    def update_macro_deformation(self, F_macro_new):
        self.F_macro_old = self.F_macro.copy()
        self.F_macro = F_macro_new

        T = self.pbc_manager.T
        u_offset = self.pbc_manager.u_offset
        n_free = self.pbc_manager.n_free

        if self.u_free is None or self.u_free.shape[0] != n_free:
            self.u_free = np.zeros(n_free)

        self.u_global = T @ self.u_free + u_offset

    def set_initial_guess(self, u_free_init):
        u_free_init = np.asarray(u_free_init, dtype=float)
        n_free = self.pbc_manager.n_free

        if u_free_init.shape[0] != n_free:
            raise ValueError(
                f"Initial guess length {u_free_init.shape[0]} does not match "
                f"current number of free DOFs {n_free}."
            )

        self.u_free = u_free_init.copy()
        self.u_global = self.pbc_manager.T @ self.u_free + self.pbc_manager.u_offset

    # -------------------------------
    # STANDARD NEWTON SOLVER
    # -------------------------------

    def solve(self, F_macro=None, max_iter=30, tol_abs=1e-3, tol_rel=1e-5, reset=False):
            """
            Solves the RVE equilibrium using Newton-Raphson with Line Search.
            Uses Dual Convergence Criteria (Absolute OR Relative tolerance).
            
            Args:
                F_macro: (3,3) Macroscopic Deformation Gradient
                max_iter: Max Newton iterations (increased default to 30)
                tol_abs: Absolute tolerance for residual norm (physical force limit)
                tol_rel: Relative tolerance (reduction factor relative to initial residual)
                reset: If True, resets displacement guess to zero.
            """
            # Update F_macro if provided
            if F_macro is not None:
                self.update_macro_deformation(F_macro)

            print("\n--- Starting RVE Solve (Newton with Elimination + Pinning) ---")

            T = self.pbc_manager.T
            u_offset = self.pbc_manager.u_offset
            n_free = self.pbc_manager.n_free

            if reset or self.u_free is None or self.u_free.shape[0] != n_free:
                self.u_free = np.zeros(n_free)

            self.u_global = T @ self.u_free + u_offset

            # Pre-allocate arrays
            data_arr = np.zeros(self.total_nnz, dtype=float)
            rows_arr = np.zeros(self.total_nnz, dtype=int)
            cols_arr = np.zeros(self.total_nnz, dtype=int)
            
            # Track initial residual for relative convergence check
            res_0 = None 

            for it in range(max_iter):
                print(f"  [Iter {it+1}] Assembling global K and internal force...")

                try:
                    f_internal, _ = self._assemble_global_matrices_and_minJ(
                        self.u_global, self.F_macro, data_arr, rows_arr, cols_arr
                    )
                except (np.linalg.LinAlgError, ValueError, NonPositiveJError) as e:
                    print(f"  ERROR in element computation (iter {it+1}): {e}. Solver failed.")
                    return None, None

                K_global = sp.coo_matrix(
                    (data_arr, (rows_arr, cols_arr)),
                    shape=(self.n_dof, self.n_dof),
                ).tocsr()

                R_red = T.T @ f_internal
                
                # --- CONVERGENCE CHECK (NORMALIZED) ---
                res_norm = float(np.linalg.norm(R_red))
                
                if np.isnan(res_norm):
                    print("  ERROR: Residual is NaN. Solver failed.")
                    return None, None

                # Capture initial residual magnitude
                if it == 0:
                    res_0 = res_norm
                    # Safety for perfect start (already solved)
                    if res_0 < 1e-9: res_0 = 1.0 

                # Calculate Relative Residual
                rel_error = res_norm / res_0
                
                print(f"  Iteration {it+1}: Norm={res_norm:.4e} (Rel={rel_error:.4e})")

                # Check BOTH criteria (stop if EITHER is met)
                # 1. Absolute: Is the force small enough physically?
                # 2. Relative: Have we reduced the error enough?
                if res_norm < tol_abs or rel_error < tol_rel:
                    print(f"  RVE Converged in {it+1} iterations.")
                    P_avg = self.calculate_avg_stress()
                    return self.F_macro, P_avg
                # --------------------------------------

                K_red = (T.T @ K_global @ T).tocsr()

                # Optional: Check for non-finite entries
                if not np.all(np.isfinite(K_red.data)):
                    raise RuntimeError("Non-finite entries in K_red.")

                print(f"  [Iter {it+1}] Solving reduced linear system with PETSc...")

                A = PETSc.Mat().createAIJ(
                    size=K_red.shape,
                    csr=(
                        K_red.indptr.astype(PETSc.IntType),
                        K_red.indices.astype(PETSc.IntType),
                        K_red.data,
                    ),
                )
                # IMPORTANT: Pinning/PBC breaks symmetry, turn off SPD
                A.setOption(PETSc.Mat.Option.SPD, False)
                A.setOption(PETSc.Mat.Option.SYMMETRIC, False)

                b = PETSc.Vec().createWithArray(-R_red)
                x = PETSc.Vec().createSeq(n_free)

                ksp = PETSc.KSP().create()
                ksp.setOperators(A)
                ksp.setType("fgmres")
                try:
                    pc = ksp.getPC()
                    pc.setType("hypre")
                    pc.setHYPREType("boomeramg")
                except Exception:
                    ksp.getPC().setType("ilu")

                ksp.setTolerances(rtol=1e-8, max_it=5000)
                ksp.setFromOptions()
                ksp.solve(b, x)
                reason = ksp.getConvergedReason()
                delta_u_free = x.getArray()

                A.destroy(); b.destroy(); x.destroy(); ksp.destroy()

                if reason < 0:
                    print(f"  ERROR: PETSc KSP failed (reason={reason}).")
                    return None, None

                # --- Line Search ---
                alpha = 1.0
                max_backtrack = 12
                eta = 1e-4
                res_norm_old = res_norm
                accepted = False

                for _ in range(max_backtrack):
                    u_free_trial = self.u_free + alpha * delta_u_free
                    u_global_trial = T @ u_free_trial + u_offset

                    if not self._is_configuration_valid(u_global_trial, self.F_macro, J_min=1e-8):
                        alpha *= 0.5
                        continue

                    try:
                        f_int_trial = self._assemble_residual(u_global_trial, self.F_macro)
                        R_red_trial = T.T @ f_int_trial
                    except (NonPositiveJError, ValueError, np.linalg.LinAlgError):
                        alpha *= 0.5
                        continue

                    res_trial = float(np.linalg.norm(R_red_trial))

                    if res_trial <= (1.0 - eta * alpha) * res_norm_old:
                        if alpha < 1.0:
                            print(f"  Step accepted with alpha = {alpha:.3f}")
                        accepted = True
                        self.u_free = u_free_trial
                        self.u_global = u_global_trial
                        break
                    else:
                        alpha *= 0.5

                if not accepted:
                    print("  ERROR: Line search failed (no admissible + improving step).")
                    return None, None

            print(f"  WARNING: RVE Solver did not converge in {max_iter} iterations.")
            return None, None
    
    # -------------------------------
    # ARC-LENGTH (RIKS) SOLVER
    # -------------------------------
    def solve_arc_length(
        self,
        F_target,
        ds=3e-3,
        steps=10,
        newton_it=30,
        tol=1e-6,
        ds_min=5e-4,
        ds_max=3e-2,
        n_target=10,
        clamp=(0.6, 1.6),
        theta=1.0,
        ):
        """
        Riks/arc-length with master–slave elimination.
        lam goes from 0 → 1; F_macro = I + lam (F_target - I).
        """
        print("\n--- Starting RVE Solve (Arc-Length with Elimination) ---")
        I3 = np.eye(3)
        lam = 0.0
        lam_max = 1.0
        pbc_tol = 1e-5

        # Initial PBC mapping at lam = 0
        self.pbc_manager.build_mapping(self.F_macro, pbc_tol, verbose=False)
        T = self.pbc_manager.T
        u_offset = self.pbc_manager.u_offset
        self.u_free = np.zeros(self.pbc_manager.n_free)
        self.u_global = u_offset.copy()

        u_free = self.u_free.copy()

        data_arr = np.zeros(self.total_nnz, dtype=float)
        rows_arr = np.zeros(self.total_nnz, dtype=int)
        cols_arr = np.zeros(self.total_nnz, dtype=int)

        def build_and_solve_lin(K_red, b_red):
            A = PETSc.Mat().createAIJ(
                size=K_red.shape,
                csr=(
                    K_red.indptr.astype(PETSc.IntType),
                    K_red.indices.astype(PETSc.IntType),
                    K_red.data,
                ),
            )
            ksp = PETSc.KSP().create()
            ksp.setOperators(A)
            ksp.setType("gmres")
            try:
                ksp.getPC().setType("hypre")
                ksp.getPC().setHYPREType("boomeramg")
            except Exception:
                ksp.getPC().setType("ilu")
                print("  [warn] Hypre not available, using ILU preconditioner.")

            try:
                ksp.setGMRESRestart(50)
            except Exception:
                pass

            ksp.setTolerances(rtol=1e-8, max_it=5000)
            ksp.setFromOptions()

            b = PETSc.Vec().createWithArray(-b_red)
            x = PETSc.Vec().createSeq(len(b_red))

            ksp.solve(b, x)
            reason = ksp.getConvergedReason()
            if reason < 0:
                print(f"  [error] PETSc KSP failed (reason={reason}).")
                raise RuntimeError("Linear solve failed in arc-length.")

            du_red = x.getArray()
            A.destroy(); b.destroy(); x.destroy(); ksp.destroy()
            return du_red

        # Initial residual at lam=0, F_macro = I
        R_full, jmin = self._assemble_global_matrices_and_minJ(self.u_global, self.F_macro, data_arr, rows_arr, cols_arr)
        K_full = sp.coo_matrix((data_arr, (rows_arr, cols_arr)), shape=(self.n_dof, self.n_dof)).tocsr()
        R_red = T.T @ R_full
        res0 = np.linalg.norm(R_red)
        print(f"[riks] start: λ={lam:.6f}, ‖R‖={res0:.3e}, min detF={jmin:.3e}, ds0={ds:.3e}")

        u_scale = max(1.0, np.linalg.norm(u_free) / max(self.pbc_manager.n_free, 1))
        xi = theta * (u_scale**2)

        ds_k = float(ds)
        cmin, cmax = clamp

        def get_reduced_G(u_free_loc, lam_loc, R_red_base):
            eps = 1e-6
            F_plus = I3 + (lam_loc + eps) * (F_target - I3)

            self.pbc_manager.build_mapping(F_plus, pbc_tol, verbose=False)
            T_plus = self.pbc_manager.T
            u_offset_plus = self.pbc_manager.u_offset

            u_full_plus = T_plus @ u_free_loc + u_offset_plus

            try:
                R_full_plus = self._assemble_residual(u_full_plus, F_plus)
                R_red_plus = T_plus.T @ R_full_plus
            except (ValueError, np.linalg.LinAlgError, NonPositiveJError):
                print("  [warn] G_red secant assembly failed. Using G=0 fallback.")
                return np.zeros_like(R_red_base)

            G_red = (R_red_plus - R_red_base) / eps

            # Reset PBC manager back to state at lam_loc
            self.pbc_manager.build_mapping(I3 + lam_loc * (F_target - I3), pbc_tol, verbose=False)
            return G_red

        for s in range(1, steps + 1):
            F_macro = I3 + lam * (F_target - I3)
            self.pbc_manager.build_mapping(F_macro, pbc_tol, verbose=False)
            T, u_offset = self.pbc_manager.T, self.pbc_manager.u_offset
            self.u_global = T @ u_free + u_offset

            R_full, jmin = self._assemble_global_matrices_and_minJ(self.u_global, F_macro, data_arr, rows_arr, cols_arr)
            K_full = sp.coo_matrix((data_arr, (rows_arr, cols_arr)), shape=(self.n_dof, self.n_dof)).tocsr()
            R_red = T.T @ R_full
            K_red = (T.T @ K_full @ T).tocsr()

            G_red = get_reduced_G(u_free, lam, R_red)

            Kinvr_red = build_and_solve_lin(K_red, R_red)
            Kinvg_red = build_and_solve_lin(K_red, -G_red)

            t_lambda = 1.0
            t_u_free = Kinvg_red

            nu = np.dot(t_u_free, t_u_free)
            scale = ds_k / np.sqrt(max(nu + xi * t_lambda * t_lambda, 1e-30))

            du_p_free = scale * t_u_free
            dlam_p = scale * t_lambda

            u_free_k = u_free + du_p_free
            lam_k = float(np.clip(lam + dlam_p, 0.0, lam_max))

            print(f"\n[riks] step {s}/{steps}: predict λ -> {lam_k:.6f} (ds={ds_k:.3e})")

            du_tot_free = du_p_free.copy()
            dlam_tot = dlam_p
            hard = False

            for it in range(1, newton_it + 1):
                F_macro_k = I3 + lam_k * (F_target - I3)
                try:
                    self.pbc_manager.build_mapping(F_macro_k, pbc_tol, verbose=False)
                    T_k, u_offset_k = self.pbc_manager.T, self.pbc_manager.u_offset
                    u_global_k = T_k @ u_free_k + u_offset_k

                    R_full_k, jmin = self._assemble_global_matrices_and_minJ(u_global_k, F_macro_k, data_arr, rows_arr, cols_arr)
                    K_full_k = sp.coo_matrix((data_arr, (rows_arr, cols_arr)), shape=(self.n_dof, self.n_dof)).tocsr()
                    R_red_k = T_k.T @ R_full_k
                    K_red_k = (T_k.T @ K_full_k @ T_k).tocsr()
                except (NonPositiveJError, ValueError, np.linalg.LinAlgError):
                    hard = True
                    ds_k = max(ds_min, 0.5 * ds_k)
                    print(f"  [bk] Assembly fail (J<=0?) -> shrink ds={ds_k:.3e} and restart step")
                    break

                resk = np.linalg.norm(R_red_k)
                print(f"  [it {it:02d}] λ={lam_k:.6f}, ‖R‖={resk:.3e}, min detF={jmin:.3e}")
                if resk < tol * max(1.0, res0):
                    print(f"  [it {it:02d}] converged: ‖R‖={resk:.3e}")
                    break

                Phi = float(np.dot(du_tot_free, du_tot_free) + xi * dlam_tot * dlam_tot - ds_k * ds_k)
                Phi_u_f = 2.0 * du_tot_free
                Phi_l = 2.0 * xi * dlam_tot

                G_red_k = get_reduced_G(u_free_k, lam_k, R_red_k)

                Kinvr_red_k = build_and_solve_lin(K_red_k, R_red_k)
                Kinvg_red_k = build_and_solve_lin(K_red_k, -G_red_k)

                a = float(np.dot(Phi_u_f, Kinvg_red_k))
                b = float(np.dot(Phi_u_f, Kinvr_red_k))
                c = float(Phi_l)

                denom = a + c
                if abs(denom) < 1e-30:
                    denom = 1e-30

                dlam = (-b - Phi) / denom
                du_free_corr = Kinvr_red_k - Kinvg_red_k * dlam

                # Line search
                alpha = 1.0
                accepted = False
                eta = 1e-4

                for _ls in range(10):
                    u_free_try = u_free_k + alpha * du_free_corr
                    lam_try = float(np.clip(lam_k + alpha * dlam, 0.0, lam_max))

                    F_macro_try = I3 + lam_try * (F_target - I3)
                    try:
                        self.pbc_manager.build_mapping(F_macro_try, pbc_tol, verbose=False)
                        T_try, u_offset_try = self.pbc_manager.T, self.pbc_manager.u_offset
                        u_global_try = T_try @ u_free_try + u_offset_try

                        R_full_try = self._assemble_residual(u_global_try, F_macro_try)
                        R_red_try = T_try.T @ R_full_try
                    except (NonPositiveJError, ValueError, np.linalg.LinAlgError):
                        alpha *= 0.5
                        continue

                    res_try = np.linalg.norm(R_red_try)

                    if res_try <= (1.0 - eta * alpha) * resk:
                        if alpha < 1.0:
                            print(f"    Line search accepted α = {alpha:.3f}")

                        du_tot_free += alpha * du_free_corr
                        dlam_tot += alpha * dlam
                        u_free_k = u_free_try
                        lam_k = lam_try
                        accepted = True
                        break

                    alpha *= 0.5

                if not accepted:
                    print("  [bk] Line search failed. Restarting step.")
                    hard = True
                    ds_k = max(ds_min, 0.7 * ds_k)
                    break

                if it > newton_it - 2:
                    hard = True

            if hard:
                u_free_k = u_free
                lam_k = lam
                continue

            # Accept step
            u_free, lam = u_free_k.copy(), lam_k
            self.u_free = u_free
            self.F_macro = I3 + lam * (F_target - I3)
            self.pbc_manager.build_mapping(self.F_macro, pbc_tol, verbose=False)
            self.u_global = self.pbc_manager.T @ u_free + self.pbc_manager.u_offset

            print(f"[riks] accept step: λ={lam:.6f}")

            it_used = it
            gain = np.sqrt(max(n_target, 1) / max(it_used, 1))
            ds_k = np.clip(ds_k * gain, ds_min, ds_max)
            if hard:
                ds_k = max(ds_min, 0.9 * ds_k)
            print(f"[riks] ds update -> {ds_k:.3e} (hard={hard})")

            if abs(lam - lam_max) < 1e-12:
                print("\n[riks] Reached target λ=1.0")
                return True

        print(f"\n[riks] Stopped after {steps} steps at λ={lam:.6f}")
        return lam >= 0.99

    # -------------------------------
    # ASSEMBLY HELPERS
    # -------------------------------
    def _is_configuration_valid(self, u_trial, F_macro_current, J_min=1e-8):
        U = u_trial.reshape(-1, 3)

        for elem_data in self.element_data:
            elem_nodes = self.elements[elem_data["elem_idx"]]
            for gp in elem_data["gauss_points"]:
                dN_dx = gp["dN_d_global"]  # 3 x 10
                u_e = U[elem_nodes]        # 10 x 3
                Grad_u = u_e.T @ dN_dx.T   # 3x3

                F = F_macro_current + Grad_u
                J = np.linalg.det(F)
                if not np.isfinite(J) or J < J_min:
                    return False
        return True

    def _safe_residual_and_K(self, u, lam, F_target, pbc_tol, penalty_K=0):
        """
        Assemble full R and K at a given lam. Not used in main paths, kept for debugging.
        """
        I = np.eye(3)
        self.F_macro = I + lam * (F_target - I)

        self.apply_boundary_conditions(self.F_macro, pbc_tol=pbc_tol, verbose=False)

        data = np.zeros(self.total_nnz)
        rows = np.zeros(self.total_nnz, int)
        cols = np.zeros(self.total_nnz, int)
        try:
            R, min_j = self._assemble_global_matrices_and_minJ(u, self.F_macro, data, rows, cols)
        except (ValueError, np.linalg.LinAlgError) as e:
            raise NonPositiveJError(str(e))

        if min_j <= 0.0:
            raise NonPositiveJError(f"Encountered det(F) <= 0 (min={min_j:.3e})")

        K = sp.coo_matrix((data, (rows, cols)), shape=(self.n_dof, self.n_dof)).tocsr()
        return R, K, float(min_j)

    def _assemble_residual(self, u_current, F_macro_current):
        """
        Assembles only the internal force vector f_internal for given u and F_macro.
        """
        f_internal = np.zeros(self.n_dof)

        for elem_data in self.element_data:
            elem_idx = elem_data["elem_idx"]
            dof = elem_data["dof_indices"]
            material_params = self._get_material_params(elem_idx)
            u_e_current = u_current[dof]
            f_elem = np.zeros(DOFS_PER_ELEM)

            for gp in elem_data["gauss_points"]:
                #B_L = gp["B_matrix"]
                dN_global = gp["dN_d_global"]
                w_int = gp["integration_weight"]

                Grad_u_fluc = self._get_element_F(u_e_current, dN_global)
                F_total = F_macro_current + Grad_u_fluc

                detF = np.linalg.det(F_total)
                if detF <= 1e-8:
                    raise NonPositiveJError(f"Non-positive J in residual assembly: J = {detF:.4e}")
                B_NL = assemble_nonlinear_B_matrix_tet10(dN_global, F_total)
                T_stress = calculate_stress_from_paper(F_total, material_params)
                T_voigt = self._tensor_to_voigt_6(T_stress)
                f_elem += B_NL.T @ T_voigt * w_int

            f_internal[dof] += f_elem

        return f_internal

    def _assemble_global_matrices_and_minJ(self, u_global, F_macro_current,
                                           data_arr, rows_arr, cols_arr):
        f_internal = np.zeros(self.n_dof)
        min_j = np.inf
        k = 0  # running nnz index

        for elem_data in self.element_data:
            elem_idx = elem_data["elem_idx"]
            dof_indices = elem_data["dof_indices"]
            u_elem = u_global[dof_indices]

            mat_params = self._get_material_params(elem_idx)
            K_elem = np.zeros((DOFS_PER_ELEM, DOFS_PER_ELEM))
            f_elem = np.zeros(DOFS_PER_ELEM)

            for gp_data in elem_data["gauss_points"]:
                #B_L = gp_data["B_matrix"]      # 6x30 Mandel B
                dN_g = gp_data["dN_d_global"]  # 3x10
                w_int = gp_data["integration_weight"]

                Grad_u_fluc = self._get_element_F(u_elem, dN_g)
                F_total = F_macro_current + Grad_u_fluc

                J = np.linalg.det(F_total)
                if J < min_j:
                    min_j = J
                if J <= 1e-8:
                    raise NonPositiveJError(
                        f"det(F) = {J:.3e} at element {elem_idx} (GP). Configuration invalid."
                    )
                B_NL = assemble_nonlinear_B_matrix_tet10(dN_g, F_total)
                T_stress = calculate_stress_from_paper(F_total, mat_params)
                C_tan_mandel = calculate_analytic_tangent(F_total, mat_params)
                C_tan_voigt = C_tan_mandel  # already Mandel
                T_voigt = _sym_to_mandel6(T_stress)

                K_mat = B_NL.T @ (C_tan_voigt @ B_NL) * w_int
                K_geo = self._get_geometric_stiffness_matrix(T_stress, dN_g) * w_int

                K_elem += K_mat + K_geo
                f_elem += B_NL.T @ T_voigt * w_int
            f_internal[dof_indices] += f_elem

            n_e_nnz = self.element_nnz
            rows_arr[k: k + n_e_nnz] = elem_data["rows"]
            cols_arr[k: k + n_e_nnz] = elem_data["cols"]
            data_arr[k: k + n_e_nnz] = K_elem.ravel()
            k += n_e_nnz

        return f_internal, min_j



    def _get_geometric_stiffness_matrix(self, T_stress, dN_d_global):
            """
            Computes the Geometric Stiffness Matrix (Initial Stress Matrix).
            Args:
                T_stress: (3,3) PK2 stress tensor (symmetric)
                dN_d_global: (3, 10) derivatives of shape functions wrt reference coords
            Returns:
                K_geo: (30, 30) matrix
            """
            # T_stress is PK2 stress S_ij
            # dN_d_global is dN_a / dX_i
            
            # K_geo_ab = (GradN_a . S . GradN_b) * I_3x3
            # The scalar part is k_ab = sum_i sum_j (dN_a/dX_i * S_ij * dN_b/dX_j)
            
            # Efficient computation:
            # 1. Compute vector V = S @ dN_d_global  (3x3) @ (3x10) -> (3x10)
            #    V[:, a] is sum_j S_ij * dN_a/dX_j
            S_dN = T_stress @ dN_d_global
            
            # 2. Compute scalar stiffness block k_scalar (10x10)
            #    k_ab = sum_i (dN_a/dX_i * V_i,b) -> dot product of columns
            k_scalar = dN_d_global.T @ S_dN  # (10x3) @ (3x10) -> (10x10)
            
            # 3. Expand to full (30x30) degrees of freedom
            K_geo = np.zeros((DOFS_PER_ELEM, DOFS_PER_ELEM))
            
            # Populate diagonal 3x3 blocks with scalar value * Identity
            # Node a (row), Node b (col)
            for a in range(NODES_PER_ELEM):
                for b in range(NODES_PER_ELEM):
                    val = k_scalar[a, b]
                    
                    row_start = a * 3
                    col_start = b * 3
                    
                    K_geo[row_start,   col_start]   = val
                    K_geo[row_start+1, col_start+1] = val
                    K_geo[row_start+2, col_start+2] = val
                    
            return K_geo
   
   
    def _get_element_F(self, u_elem, dN_d_global):
        """
        Compute Grad(u~) = sum_a u_a ⊗ ∇N_a at one Gauss point.
        """
        u_mat = u_elem.reshape((NODES_PER_ELEM, 3))

        Grad_u = np.zeros((3, 3))
        for a in range(NODES_PER_ELEM):
            Grad_u[0, :] += u_mat[a, 0] * dN_d_global[:, a]
            Grad_u[1, :] += u_mat[a, 1] * dN_d_global[:, a]
            Grad_u[2, :] += u_mat[a, 2] * dN_d_global[:, a]

        return Grad_u

    def verify_tangent_consistency(self):
        print("\n--- Verifying Tangent Consistency ---")
        # Create a random deformation gradient
        F_test = np.eye(3) 
        F_test[0,0] = 1.2; F_test[1,1] = 0.9; F_test[2,2] = 1.0/(1.2*0.9)
        
        # Get material parameters (Matrix)
        mat_params = self.material_params_matrix
        
        # 1. Analytic
        C_analytic = calculate_analytic_tangent(F_test, mat_params)
        
        # 2. Numerical (Finite Difference) - Requires importing from material.py
        from material import calculate_numerical_tangent
        C_numeric = calculate_numerical_tangent(F_test, mat_params, h=1e-7)
        
        # Compare
        diff = np.linalg.norm(C_analytic - C_numeric) / np.linalg.norm(C_numeric)
        print(f"Tangent Difference Norm (Rel): {diff:.4e}")
        
        if diff > 1e-4:
            print("WARNING: Analytic tangent may be incorrect!")
        else:
            print("Tangent is consistent.")
    # -------------------------------
    # POSTPROCESSING
    # -------------------------------
    def calculate_avg_stress(self):
            print("Calculating homogenized 1st Piola-Kirchhoff stress...")
            P_sum = np.zeros((3, 3))
            V_0_sum = 0.0

            for elem_data in self.element_data:
                elem_dofs = self.u_global[elem_data["dof_indices"]]
                elem_idx = elem_data["elem_idx"]
                
                # Safe material access
                if hasattr(self, '_get_material_params'):
                    material_params = self._get_material_params(elem_idx)
                else:
                    # Fallback if helper doesn't exist
                    mat_id = self.elem_materials[elem_idx]
                    material_params = self.mat_params_fiber if mat_id == 1 else self.mat_params_matrix

                for gp_data in elem_data["gauss_points"]:
                    dN_d_global = gp_data["dN_d_global"]
                    w = gp_data["integration_weight"]

                    Grad_u = self._get_element_F(elem_dofs, dN_d_global)
                    
                   
                    F_total = self.F_macro + Grad_u

                    # Calculate PK2 Stress (S)
                    S_stress = calculate_stress_from_paper(F_total, material_params)
                    
                    # Convert to 1st PK (P = F * S)
                    P_stress = F_total @ S_stress 

                    P_sum += P_stress * w
                    V_0_sum += w

            if V_0_sum <= 0.0:
                raise ValueError("RVE has zero volume.")

            P_avg = P_sum / V_0_sum
            return P_avg

    @staticmethod
    def _tensor_to_voigt_6(T):
        """Consistent Mandel Voigt mapping."""
        return _sym_to_mandel6(T)
