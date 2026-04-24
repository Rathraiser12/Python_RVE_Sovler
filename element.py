#
# FILE: element.py (*** FULL TET10 QUADRATIC ELEMENTS ***)
#
import numpy as np


# --- 1. GAUSS POINTS AND WEIGHTS (4-point rule for TET10 accuracy) ---
NODES_PER_ELEM = 10

def get_tet10_element_properties():
    """
    Returns the 4 standard integration (Gauss) points and weights
    for a 10-node (quadratic) tetrahedron, based on the standard 
    4-point rule (exact for degree 2 polynomials).
    
    Ref: Zienkiewicz & Taylor, Table 9.3 (Quadratic row).
    """
    # Define the two unique base coordinates (alpha, beta) from the reference
    # Alpha (A) = 0.58541020 
    # Beta (B) = 0.13819660
    
    # Using the full precision values ensures accuracy
    alpha = 0.5854101966
    beta = 0.1381966011

    # The four points are permutations of (alpha, beta, beta, beta)
    # We define the first THREE coordinates (L1, L2, L3) for the 4 points
    gauss_points_L1L2L3 = np.array([
        # Point 'a': (alpha, beta, beta, L4)
        [alpha, beta, beta], 
        # Point 'b': (beta, alpha, beta, L4)
        [beta, alpha, beta], 
        # Point 'c': (beta, beta, alpha, L4)
        [beta, beta, alpha], 
        # Point 'd': (beta, beta, beta, L4)
        [beta, beta, beta] 
    ])
    
    # Calculate the fourth barycentric coordinate (L4) for all points
    # L4 = 1 - L1 - L2 - L3
    L4 = 1.0 - np.sum(gauss_points_L1L2L3, axis=1)
    
    # Full barycentric coordinates (L1, L2, L3, L4)
    gauss_points_bary = np.hstack((gauss_points_L1L2L3, L4[:, np.newaxis]))
    
    # All weights are equal in the 4-point rule, W_i = 1/4.
    # The weight is 1/4 (as seen in the table) times the volume of the unit element (1/6).
    # Since the Jacobian det(J) absorbs the physical volume, the normalized integration 
    # weight (gp_weight) often sums to 1 when multiplied by det(J).

    
    # Reverting to the standard interpretation: W_i = 1/4 (as shown in the table)
    # and letting the solver handle the volume scaling with det(J) * (1/6) factor.
    # Since your original code had w_const = 1.0 / 24.0 and multiplied by 6.0, 
    # we maintain that pattern for consistency with the rest of your solver:
    w_const = 1.0 / 24.0
    gauss_weights = np.array([w_const, w_const, w_const, w_const]) * 6.0 

    return gauss_points_bary, gauss_weights

# --- 2. SHAPE FUNCTIONS AND DERIVATIVES (AT A GIVEN POINT) ---

import numpy as np

def shape_functions_and_derivatives_tet10(L1, L2, L3, L4=None):
    """
    Quadratic TET10 shape functions and derivatives in barycentric coords,
    consistent with Gmsh node ordering:

      1:(1,0,0,0)
      2:(0,1,0,0)
      3:(0,0,1,0)
      4:(0,0,0,1)
      5: mid(1,2)
      6: mid(2,3)
      7: mid(1,3)
      8: mid(1,4)
      9: mid(2,4)
      10:mid(3,4)

    Accepts either:
      (L1, L2, L3, L4)  with L1+L2+L3+L4 = 1
      or
      (L1, L2, L3)      and computes L4 = 1 - L1 - L2 - L3

    Returns:
      N      : (10,)  shape functions
      dN_dL  : (3,10) derivatives wrt (L1, L2, L3)
    """
    if L4 is None:
        L4 = 1.0 - L1 - L2 - L3
    else:
        # (optional) tiny consistency check
        s = L1 + L2 + L3 + L4
        if abs(s - 1.0) > 1e-12:
            raise ValueError(
                f"[element] Invalid barycentric coords: L1+L2+L3+L4 = {s}, expected 1."
            )

    # ---- shape functions ----
    N1  = L1 * (2.0*L1 - 1.0)
    N2  = L2 * (2.0*L2 - 1.0)
    N3  = L3 * (2.0*L3 - 1.0)
    N4  = L4 * (2.0*L4 - 1.0)
    N5  = 4.0 * L1 * L2
    N6  = 4.0 * L2 * L3
    N7  = 4.0 * L1 * L3
    N8  = 4.0 * L1 * L4
    N9  = 4.0 * L2 * L4
    N10 = 4.0 * L3 * L4

    N = np.array([N1, N2, N3, N4, N5, N6, N7, N8, N9, N10])

    # dN/d(L1,L2,L3); L4 = 1 - L1 - L2 - L3, so dL4/dLk = -1
    dN1_dL1 = 4.0*L1 - 1.0
    dN1_dL2 = 0.0
    dN1_dL3 = 0.0

    dN2_dL1 = 0.0
    dN2_dL2 = 4.0*L2 - 1.0
    dN2_dL3 = 0.0

    dN3_dL1 = 0.0
    dN3_dL2 = 0.0
    dN3_dL3 = 4.0*L3 - 1.0

    dN4_dL1 = -(4.0*L4 - 1.0)
    dN4_dL2 = -(4.0*L4 - 1.0)
    dN4_dL3 = -(4.0*L4 - 1.0)

    dN5_dL1 = 4.0*L2
    dN5_dL2 = 4.0*L1
    dN5_dL3 = 0.0

    dN6_dL1 = 0.0
    dN6_dL2 = 4.0*L3
    dN6_dL3 = 4.0*L2

    dN7_dL1 = 4.0*L3
    dN7_dL2 = 0.0
    dN7_dL3 = 4.0*L1

    dN8_dL1 = 4.0*(L4 - L1)
    dN8_dL2 = -4.0*L1
    dN8_dL3 = -4.0*L1

    dN9_dL1 = -4.0*L2
    dN9_dL2 = 4.0*(L4 - L2)
    dN9_dL3 = -4.0*L2

    dN10_dL1 = -4.0*L3
    dN10_dL2 = -4.0*L3
    dN10_dL3 = 4.0*(L4 - L3)

    dN_dL = np.array([
        [dN1_dL1,  dN2_dL1,  dN3_dL1,  dN4_dL1,  dN5_dL1,  dN6_dL1,  dN7_dL1,  dN8_dL1,  dN9_dL1,  dN10_dL1],
        [dN1_dL2,  dN2_dL2,  dN3_dL2,  dN4_dL2,  dN5_dL2,  dN6_dL2,  dN7_dL2,  dN8_dL2,  dN9_dL2,  dN10_dL2],
        [dN1_dL3,  dN2_dL3,  dN3_dL3,  dN4_dL3,  dN5_dL3,  dN6_dL3,  dN7_dL3,  dN8_dL3,  dN9_dL3,  dN10_dL3],
    ])

    return N, dN_dL


# --- 3. JACOBIAN AND GLOBAL DERIVATIVES (Changed to NOT raise error on negative detJ) ---

def calculate_jacobian_and_derivatives(elem_nodes_coords, dN_dL, det_tol=1e-14):
    """
    Compute det(J) and global derivatives dN/dx for a TET10 element,
    assuming connectivity matches Gmsh TET10 ordering and the shape
    functions defined in shape_functions_and_derivatives_tet10().

    elem_nodes_coords : (10,3)
    dN_dL             : (3,10)  (wrt L1,L2,L3)

    Returns:
      det_J > 0   (if valid)
      dN_d_global : (3,10)
    """
    X = np.asarray(elem_nodes_coords, dtype=float)
    if X.shape != (10, 3):
        raise ValueError(f"[element] Expected (10,3) coords, got {X.shape}")

    # Geometry is linear in barycentric coordinates -> defined by vertices only
    x1, x2, x3, x4 = X[0], X[1], X[2], X[3]
    J = np.column_stack((x2 - x1, x3 - x1, x4 - x1))

    det_J = np.linalg.det(J)

    if abs(det_J) < det_tol:
        raise ValueError(f"[element] Degenerate Jacobian (det_J = {det_J:.4e}).")

    if det_J <= 0.0:
        # Now: we do NOT "fix" it. If this happens, the mesh orientation is wrong.
        raise ValueError(
            f"[element]Non-positive Jacobian detected (det_J = {det_J:.4e}). "
            f"Element is physically inverted or connectivity ordering is inconsistent."
        )

    # Map derivatives: dN/dx = (J^{-1})^T * dN/dL
    J_inv = np.linalg.inv(J)
    dN_d_global = J_inv.T @ dN_dL

    return det_J, dN_d_global


# --- 4. B-MATRIX ASSEMBLY (No change) ---

def assemble_B_matrix_tet10(dN_d_global):

    """
    Assembles the 3D strain-displacement matrix (B-matrix) 
    for a 10-node element, in Mandel (orthonormal) basis.
    Returns: B (6x30 array)
    """
    B = np.zeros((6, NODES_PER_ELEM * 3))
    SQ2 = np.sqrt(2.0) # <--- ADD THIS
    
    for i in range(NODES_PER_ELEM):
        dN_dx = dN_d_global[0, i]
        dN_dy = dN_d_global[1, i]
        dN_dz = dN_d_global[2, i]
        
        col_u = i * 3
        col_v = i * 3 + 1
        col_w = i * 3 + 2

        # Normal strains (e_xx, e_yy, e_zz)
        B[0, col_u] = dN_dx
        B[1, col_v] = dN_dy
        B[2, col_w] = dN_dz
        
        # Shear strains (Mandel: sqrt(2)*e_xy, sqrt(2)*e_yz, sqrt(2)*e_xz)
        B[3, col_u] = dN_dy * SQ2 # <--- ADD SQ2
        B[3, col_v] = dN_dx * SQ2 # <--- ADD SQ2
        
        B[4, col_v] = dN_dz * SQ2 # <--- ADD SQ2
        B[4, col_w] = dN_dy * SQ2 # <--- ADD SQ2
        
        B[5, col_u] = dN_dz * SQ2 # <--- ADD SQ2
        B[5, col_w] = dN_dx * SQ2 # <--- ADD SQ2
        
    return B


# ADD THIS TO THE END OF element.py

def assemble_nonlinear_B_matrix_tet10(dN_d_global, F):
    """
    Assembles the NONLINEAR (Deformation-Dependent) B-matrix for Total Lagrangian formulation.
    Maps nodal displacement variations (du) to Green-Lagrange strain variations (dE).
    
    dE = 0.5 * (F.T * Grad(du) + Grad(du).T * F)
    
    Args:
        dN_d_global: (3, 10) Shape function derivatives dN/dX
        F:           (3, 3)  Current Deformation Gradient
    Returns:
        B:           (6, 30) B-matrix in Mandel notation
    """
    B = np.zeros((6, NODES_PER_ELEM * 3))
    SQ2 = np.sqrt(2.0)
    
    # Pre-fetch F components for speed
    F00, F01, F02 = F[0,0], F[0,1], F[0,2]
    F10, F11, F12 = F[1,0], F[1,1], F[1,2]
    F20, F21, F22 = F[2,0], F[2,1], F[2,2]

    for a in range(NODES_PER_ELEM):
        # Derivatives dNa/dX, dNa/dY, dNa/dZ
        NX = dN_d_global[0, a]
        NY = dN_d_global[1, a]
        NZ = dN_d_global[2, a]
        
        col_u = a * 3
        col_v = a * 3 + 1
        col_w = a * 3 + 2
        
        # -- Row 0: E_11 (XX) --
        # dE_11 = F_k1 * d(du_k)/dX1
        #       = F00*du_X + F10*dv_X + F20*dw_X
        B[0, col_u] = F00 * NX
        B[0, col_v] = F10 * NX
        B[0, col_w] = F20 * NX
        
        # -- Row 1: E_22 (YY) --
        # dE_22 = F_k2 * d(du_k)/dX2
        B[1, col_u] = F01 * NY
        B[1, col_v] = F11 * NY
        B[1, col_w] = F21 * NY
        
        # -- Row 2: E_33 (ZZ) --
        # dE_33 = F_k3 * d(du_k)/dX3
        B[2, col_u] = F02 * NZ
        B[2, col_v] = F12 * NZ
        B[2, col_w] = F22 * NZ
        
        # -- Row 3: E_12 (XY) -> Mandel: sqrt(2) * E_12 --
        # dE_12 = 0.5 * (F_k1 * d(u_k)/dY + F_k2 * d(u_k)/dX)
        # Scaling by sqrt(2):
        val_u = (F00 * NY + F01 * NX) * (0.5 * SQ2)
        val_v = (F10 * NY + F11 * NX) * (0.5 * SQ2)
        val_w = (F20 * NY + F21 * NX) * (0.5 * SQ2)
        
        B[3, col_u] = val_u
        B[3, col_v] = val_v
        B[3, col_w] = val_w
        
        # -- Row 4: E_23 (YZ) -> Mandel: sqrt(2) * E_23 --
        val_u = (F01 * NZ + F02 * NY) * (0.5 * SQ2)
        val_v = (F11 * NZ + F12 * NY) * (0.5 * SQ2)
        val_w = (F21 * NZ + F22 * NY) * (0.5 * SQ2)
        
        B[4, col_u] = val_u
        B[4, col_v] = val_v
        B[4, col_w] = val_w
        
        # -- Row 5: E_13 (XZ) -> Mandel: sqrt(2) * E_13 --
        val_u = (F00 * NZ + F02 * NX) * (0.5 * SQ2)
        val_v = (F10 * NZ + F12 * NX) * (0.5 * SQ2)
        val_w = (F20 * NZ + F22 * NX) * (0.5 * SQ2)
        
        B[5, col_u] = val_u
        B[5, col_v] = val_v
        B[5, col_w] = val_w
        
    return B