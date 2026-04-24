#!/usr/bin/env python3
# FILE: generate_working.py
# V6: Grid Sweep + Adaptive Load Stepping (Maximum Stability)

import os
import sys
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool

# --- IMPORT SOLVER ---
try:
    from micro_solver import RVE_Solver
except ImportError:
    print("CRITICAL: 'micro_solver.py' not found.")
    sys.exit(1)

# ================= CONFIGURATION =================
MESH_FILE = "meshes/mesh_100.msh"
OUTPUT_FILE = "output/feann_initial_sweep.csv"

# ================= 1. CORE SOLVER UTILS =================

def apply_step(solver, F_next):
    """
    Tries to solve for F_next starting from the solver's CURRENT state.
    Uses Dual-Convergence (Absolute OR Relative) to handle high residuals.
    Returns homogenized P if converged, else None.
    """
    # Checkpoint current displacement
    u_checkpoint = solver.u_free.copy()

    # Update macro deformation and solve
    solver.update_macro_deformation(F_next)

    # Slightly relaxed tolerances for speed on large meshes
    # tol_abs=1e-2: small physical forces are acceptable
    # tol_rel=1e-4: ~1e4 reduction in residual is enough
    res, P = solver.solve(
        max_iter=25,
        tol_abs=1e-2,
        tol_rel=1e-5,
        reset=False,
    )

    if res is None:
        # Restore state on failure so ramping can retry with smaller dt
        solver.set_initial_guess(u_checkpoint)
        return None

    return P


def ramp_to_deformation(solver, F_target):
    """
    Adaptive Ramping (Continuation Method).
    Moves from F_current -> F_target in steps.
    Uses apply_step(...) for each intermediate configuration.

    Returns:
        P_current (3x3) for F_target if successful, else None.
    """
    # 1. Setup
    F_start = solver.F_macro.copy()
    I = np.eye(3)

    # If Identity (Rest), return zero stress immediately
    if np.allclose(F_target, I, atol=1e-8):
        return np.zeros((3, 3))

    t_current = 0.0
    dt = 0.25       # Start with 4 steps (more aggressive than 0.1)
    dt_min = 1e-2   # Do not go below 1% of the total load path

    max_steps = 40  # Safety cap: avoid infinite ramp loops
    n_steps = 0

    P_current = None

    while t_current < 1.0 and n_steps < max_steps:
        n_steps += 1

        # Cap dt to land exactly on 1.0
        if t_current + dt > 1.0:
            dt = 1.0 - t_current

        t_next = t_current + dt

        # Linear Interpolation: F_start -> F_target
        F_next = F_start + t_next * (F_target - F_start)

        # Try one RVE solve at this intermediate deformation
        P = apply_step(solver, F_next)

        if P is not None:
            # --- SUCCESS ---
            t_current = t_next
            P_current = P

            # Heuristic: If it was easy, accelerate up to dt = 0.5
            if dt < 0.5:
                dt *= 1.5
        else:
            # --- FAILURE ---
            # Step was too large, cut it
            dt *= 0.5

            # If we are already too small, give up on this load case
            if dt < dt_min:
                # print("Ramp: dt below dt_min, aborting this load case.", flush=True)
                return None

    # If we exited because we hit max_steps without reaching t=1.0, fail
    if t_current < 1.0:
        print("Ramp: max_steps reached before t=1.0, aborting this load case.", flush=True)
        return None

    return P_current
# ================= 2. WORKER PROCESS =================

def worker_task(job_data):
    """
    Executes a single simulation.
    F_target is the FINAL desired deformation.
    The ramp_to_deformation function handles the path.
    """
    job_id, F_target = job_data
    
    try:
        # Initialize solver per worker
        solver = RVE_Solver(MESH_FILE)
        
        # Zero initial state
        solver.apply_boundary_conditions(np.eye(3), verbose=False)
        
        # Run Adaptive Simulation
        P_final = ramp_to_deformation(solver, F_target)
        
        if P_final is not None:
            # Check for NaNs
            if np.any(np.isnan(P_final)): return None
            
            # Return flattened data: [F11..F33, P11..P33]
            return np.concatenate([F_target.flatten(), P_final.flatten()])
            
        return None # Failed to converge

    except Exception as e:
        # Catch unexpected errors to keep the pool alive
        return None

# ================= 3. SWEEP GENERATOR LOGIC =================

def generate_sweep_jobs():
    """
    Generates load cases with emphasis on uniaxial tension/compression.

    - UT: λ2 and λ3 are independent (3x3 grid around λ_lat,ideal).
    - ET: reduced sampling (fewer lambdas + fewer transverse values).
    - SS: only one shear mode per plane (no xy / yx duplicates).
    """
    jobs = []

    # 0. Identity (Rest State)
    jobs.append(('ID', np.eye(3)))

    # --- 1. UNIAXIAL SWEEPS (Tension & Compression) ---
    # Same λ-range as before
    scales_ut = np.concatenate([
        np.linspace(0.70, 0.95, 6),   # compression
        np.linspace(1.05, 1.60, 6)   # tension
    ])

    for lam in scales_ut:
        # Ideal lateral stretch for incompressible material: J=1 => λ2*λ3 = 1/λ
        lat_ideal = lam**(-0.5)

        # Make a *3-point* sweep around the ideal lateral stretch
        # (0.9x, 1.0x, 1.1x) to avoid job explosion
        lat_sweep = np.linspace(lat_ideal * 0.9, lat_ideal * 1.1, 3)

        # Axis = direction of applied uniaxial stretch
        for axis in [0, 1, 2]:

            # λ2, λ3 independent: 3 x 3 combinations
            for lat2 in lat_sweep:
                for lat3 in lat_sweep:
                    F = np.eye(3)

                    if axis == 0:
                        # UT-X: λ1 = lam, (λ2, λ3) free
                        F[0, 0] = lam
                        F[1, 1] = lat2
                        F[2, 2] = lat3

                    elif axis == 1:
                        # UT-Y: λ2 = lam, (λ1, λ3) free
                        F[1, 1] = lam
                        F[0, 0] = lat2
                        F[2, 2] = lat3

                    else:
                        # UT-Z: λ3 = lam, (λ1, λ2) free
                        F[2, 2] = lam
                        F[0, 0] = lat2
                        F[1, 1] = lat3

                    jobs.append(('UT', F))

    # --- 2. EQUIBIAXIAL SWEEPS (reduced) ---
    # Slightly reduced ET sampling vs original to keep total jobs reasonable
    scales_et = np.concatenate([
        np.linspace(0.85, 0.95, 3),   # mild compression
        np.linspace(1.05, 1.30, 5)    # tension
    ])

    for lam in scales_et:
        # Ideal transverse stretch for incompressible ET: lat = 1 / lam^2
        trans_ideal = lam**(-2.0)

        # 3-point sweep around ideal transverse stretch
        trans_sweep = np.linspace(trans_ideal * 0.90, trans_ideal * 1.10, 3)

        # Planes: XY (relax Z), XZ (relax Y), YZ (relax X)
        planes = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]

        for ax1, ax2, relax_ax in planes:
            for trans in trans_sweep:
                F = np.eye(3)
                F[ax1, ax1] = lam
                F[ax2, ax2] = lam
                F[relax_ax, relax_ax] = trans
                jobs.append(('ET', F))

    # --- 3. SIMPLE SHEAR (no symmetric duplicates) ---
    gammas = np.linspace(0.05, 0.5, 6)

    # One mode per shear plane: xy, xz, yz
    shear_modes = [(0, 1), (0, 2), (1, 2)]

    for gam in gammas:
        for (i, j) in shear_modes:
            F = np.eye(3)
            F[i, j] = gam
            jobs.append(('SS', F))

    return jobs


# ================= 4. MAIN EXECUTION =================

def append_locked(csv_path, df_row):
    """Safely append to CSV with file locking for parallel workers."""
    try:
        import fcntl
        with open(csv_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            df_row.to_csv(f, header=False, index=False)
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)
    except ImportError:
        # Fallback for non-Unix systems
        df_row.to_csv(csv_path, mode="a", header=False, index=False)

if __name__ == "__main__":
    # 1. Setup checks
    if not os.path.exists(MESH_FILE):
        print("Mesh not found."); sys.exit(1)
    
    try: N_WORKERS = int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count()))
    except: N_WORKERS = 1
    
    print(f"--- Starting ADAPTIVE SWEEP Generator ({N_WORKERS} workers) ---")
    
    # 2. Init CSV
    cols = [f"F{i}{j}" for i in range(1,4) for j in range(1,4)] + \
           [f"P{i}{j}" for i in range(1,4) for j in range(1,4)]
    
    if not os.path.exists(OUTPUT_FILE):
        pd.DataFrame(columns=cols).to_csv(OUTPUT_FILE, index=False)

    # 3. Load Existing Data (Restart Capability)
    existing_hashes = set()
    try:
        df_exist = pd.read_csv(OUTPUT_FILE)
        for i in range(len(df_exist)):
            # Hash deformation gradient only (first 9 cols)
            f_vals = tuple(np.round(df_exist.iloc[i, 0:9].values.astype(float), 4))
            existing_hashes.add(f_vals)
        print(f"Found {len(df_exist)} existing entries.")
    except: 
        print("No existing data found.")

    # 4. Generate Job List
    jobs_to_run = generate_sweep_jobs()
    print(f"Total Sweep Jobs: {len(jobs_to_run)}")

    # 5. Run Parallel
    # We use map instead of imap so we don't start until workers are ready
    with Pool(processes=N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(worker_task, jobs_to_run, chunksize=1)):
            
            status = "Failed"
            if res is not None:
                # Check for duplicates before saving
                f_chk = tuple(np.round(res[0:9], 4))
                
                if f_chk not in existing_hashes:
                    append_locked(OUTPUT_FILE, pd.DataFrame([res], columns=cols))
                    existing_hashes.add(f_chk)
                    status = "Saved"
                else:
                    status = "Duplicate"
            
            # Simple Progress Bar
            if i % 10 == 0:
                print(f"[{i+1}/{len(jobs_to_run)}] Last Status: {status}", flush=True)

    print("--- Done ---")