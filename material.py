# material.py
import numpy as np

# ======= Global tolerances =======
EIG_MIN      = 1e-12   # clamp for eigenvalues of C
EIG_SEP_TOL  = 1e-7    # tolerance for coincident eigenvalues
SQ2          = np.sqrt(2.0)

# ---------------------------------------------------------------------------
# Material parameter interface
# ---------------------------------------------------------------------------
def get_material_params(name: str):
    name_l = name.lower()
    if name_l == "matrix":
        mu    = np.array([-26.62, 29.04, 0.0098]) * 1e3 
        alpha = np.array([-5.0,  2.3, 12.0])
        kappa = 800.0 * 1e3
        return dict(mu=mu, alpha=alpha, kappa=kappa, model="ogden_kalina")

    if name_l == "fibers":
        mu    = np.array([1000.0]) * 1e3
        alpha = np.array([2.0])
        kappa = 4666.7 * 1e3
        return dict(mu=mu, alpha=alpha, kappa=kappa, model="ogden_kalina")
    
    raise ValueError(f"Unknown material name '{name}'.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sym_to_mandel6(T: np.ndarray) -> np.ndarray:
    return np.array([T[0,0], T[1,1], T[2,2], SQ2*T[0,1], SQ2*T[1,2], SQ2*T[0,2]])

def _mandel_basis():
    E = []
    for i in range(3):
        T = np.zeros((3,3)); T[i,i] = 1.0; E.append(T)
    # 3:xy, 4:yz, 5:xz
    T = np.zeros((3,3)); T[0,1]=T[1,0]=1.0/SQ2; E.append(T)
    T = np.zeros((3,3)); T[1,2]=T[2,1]=1.0/SQ2; E.append(T)
    T = np.zeros((3,3)); T[0,2]=T[2,0]=1.0/SQ2; E.append(T)
    return E

# ---------------------------------------------------------------------------
# Spectral Decomposition Helper
# ---------------------------------------------------------------------------
def _spectral_decomp(C):
    lam2, V = np.linalg.eigh(0.5 * (C + C.T))
    lam2 = np.clip(lam2, EIG_MIN, None)
    lam = np.sqrt(lam2)
    return lam, lam2, V

# ---------------------------------------------------------------------------
# PK2 Stress Calculation
# ---------------------------------------------------------------------------
def calculate_pk2_stress(F: np.ndarray, material_params: dict) -> np.ndarray:
    mu    = np.asarray(material_params["mu"], dtype=float)
    alpha = np.asarray(material_params["alpha"], dtype=float)
    kappa = float(material_params["kappa"])

    C = F.T @ F
    J = np.linalg.det(F)
    if J <= 0.0: raise ValueError(f"J={J}<=0")

    lam, lam2, V = _spectral_decomp(C)
    
    J_m13 = J**(-1.0/3.0)
    lam_bar = lam * J_m13

    # 1. Volumetric Kirchhoff
    tau_vol = (kappa / 2.0) * (J**2 - 1.0)

    # 2. Isochoric Kirchhoff
    tau_iso = np.zeros(3)
    for p in range(len(mu)):
        mp, ap = mu[p], alpha[p]
        term = lam_bar**ap
        trace_term = np.sum(term) / 3.0
        tau_iso += mp * (term - trace_term)

    tau = tau_vol + tau_iso

    # 3. Reconstruct PK2: S = sum (tau_a / lam_a^2) P_a
    S = np.zeros((3,3))
    for a in range(3):
        Pa = np.outer(V[:,a], V[:,a])
        S_val = tau[a] / lam2[a]
        S += S_val * Pa
        
    return S

calculate_stress_from_paper = calculate_pk2_stress

# ---------------------------------------------------------------------------
# Analytic Tangent (Spectral Formulation)
# C_IJKL = 2 * dS_IJ / dC_KL
# ---------------------------------------------------------------------------
def calculate_analytic_tangent(F: np.ndarray, material_params: dict) -> np.ndarray:
    mu    = np.asarray(material_params["mu"], dtype=float)
    alpha = np.asarray(material_params["alpha"], dtype=float)
    kappa = float(material_params["kappa"])

    C = F.T @ F
    J = np.linalg.det(F)
    lam, lam2, V = _spectral_decomp(C)
    
    J_m13 = J**(-1.0/3.0)
    lam_bar = lam * J_m13

    # --- 1. Kirchhoff Stresses (tau) & Stiffness Moduli (k_ab) ---
    tau = np.zeros(3)
    tau_vol = (kappa / 2.0) * (J**2 - 1.0)
    
    # k_ab = d(tau_a) / d(ln(lam_b))
    k_ab = np.zeros((3,3))

    # Volumetric stiffness
    vol_stiff = kappa * (J**2)
    k_ab += vol_stiff 

    # Isochoric parts
    for p in range(len(mu)):
        mp, ap = mu[p], alpha[p]
        val = lam_bar**ap 
        
        tau += mp * (val - np.mean(val))
        
        term = mp * ap * val
        sum_term = np.sum(term)
        
        for a in range(3):
            for b in range(3):
                delta_ab = 1.0 if a == b else 0.0
                d_part1 = term[a] * (delta_ab - 1.0/3.0)
                d_trace = (term[b] - sum_term/3.0)
                k_ab[a,b] += d_part1 - (1.0/3.0)*d_trace

    tau += tau_vol

    # --- 2. Convert to PK2 Stiffness H_ab ---
    H_ab = np.zeros((3,3))
    for a in range(3):
        for b in range(3):
            delta = 1.0 if a==b else 0.0
            H_ab[a,b] = (k_ab[a,b] - 2.0 * delta * tau[a]) / (lam2[a] * lam2[b])

    # --- 3. Principal Stresses (PK2) ---
    S_eig = tau / lam2 

    # --- 4. Assemble 6x6 Mandel Matrix ---
    C6 = np.zeros((6,6))
    P_vec = [ _sym_to_mandel6(np.outer(V[:,a], V[:,a])) for a in range(3) ]

    # Term 1: Principal Stiffness
    for a in range(3):
        for b in range(3):
            C6 += H_ab[a,b] * np.outer(P_vec[a], P_vec[b])

    # Term 2: Shear Terms (Gamma)
    E = _mandel_basis()
    
    for a in range(3):
        for b in range(a+1, 3): # upper triangle a < b
            
            diff_lam = lam2[a] - lam2[b]
            
            # gamma = (S_a - S_b) / (C_a - C_b)  (Secant modulus)
            if abs(diff_lam) > EIG_SEP_TOL:
                gamma = (S_eig[a] - S_eig[b]) / diff_lam
            else:
                # L'Hopital limit: 0.5 * (H_aa - H_ab)
                gamma = 0.5 * (H_ab[a,a] - H_ab[a,b]) 

            # Add shear contribution.
            # The spectral coefficient is gamma.
            # The geometric tensor is (Pa box Pb + Pb box Pa).
            # In Mandel form, shear components represent sqrt(2) * strain.
            # The shear stiffness term must be 2*gamma for Mandel scaling.
            # The `val` calculation extracts 0.5 of the tensor action.
            # Thus we need 4.0 * gamma.
            
            Pa = np.outer(V[:,a], V[:,a])
            Pb = np.outer(V[:,b], V[:,b])
            
            for i in range(6):
                for j in range(6):
                    # Projection of symmetric geometric tensor onto Mandel basis
                    term = 0.5 * ( (Pa @ E[j] @ Pb) + (Pb @ E[j] @ Pa) )
                    val = np.sum(E[i] * term)
                    
                    # FIX: Multiplier changed from 2.0 to 4.0 to match Mandel shear energy
                    C6[i,j] += 4.0 * gamma * val

    return C6

# ---------------------------------------------------------------------------
# Numerical tangent for verification (Corrected to use calculate_pk2_stress)
# ---------------------------------------------------------------------------
def calculate_numerical_tangent(F: np.ndarray, material_params: dict, h: float = 1e-7) -> np.ndarray:
    # Convert F to C, then to E
    C = F.T @ F
    vals, vecs = np.linalg.eigh(C)
    vals = np.clip(vals, 1e-12, None)
    C = (vecs * vals) @ vecs.T
    
    E_base = 0.5 * (C - np.eye(3))
    
    C6 = np.zeros((6,6))
    basis = _mandel_basis()
    
    for j in range(6):
        dE = basis[j] * h
        
        # Perturb E, get new C
        Cp = 2.0 * (E_base + dE) + np.eye(3)
        Cm = 2.0 * (E_base - dE) + np.eye(3)
        
        # Get F (as pure stretch U) for stress function
        def get_U(C_in):
            v, m = np.linalg.eigh(C_in)
            v = np.clip(v, 1e-12, None)
            return (m * np.sqrt(v)) @ m.T

        Sp = calculate_pk2_stress(get_U(Cp), material_params)
        Sm = calculate_pk2_stress(get_U(Cm), material_params)
        
        dS = (Sp - Sm) / (2.0 * h)
        C6[:, j] = _sym_to_mandel6(dS)
        
    return C6