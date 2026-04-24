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
MESH_FILE = "meshes/mesh_100_60k.msh"
OUTPUT_FILE = "feann_mesh_test.csv"

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
    Minimal diagnostic sweep:
    - ID: identity (rest state)
    - UT-X, UT-Y, UT-Z at lambda = 1.4,
      with symmetric lateral stretch from incompressible guess.

    This is just to check relative stiffness in x1/x2/x3 for the
    fine 100-fiber mesh.
    """
    jobs = []

    # 0. Identity (Rest State)
    jobs.append(("ID", np.eye(3)))

    # 1. Uniaxial tension at lambda = 1.2
    lam = 1.2
    # Incompressible guess: lambda_lat ~ 1/sqrt(lam)
    lat = lam ** (-0.5)

    # UT-X: F11 = lam, F22 = F33 = lat
    F_utx = np.eye(3)
    F_utx[0, 0] = lam
    F_utx[1, 1] = lat
    F_utx[2, 2] = lat
    jobs.append(("UT_X_1.2", F_utx))

    # UT-Y: F22 = lam, F11 = F33 = lat
    F_uty = np.eye(3)
    F_uty[1, 1] = lam
    F_uty[0, 0] = lat
    F_uty[2, 2] = lat
    jobs.append(("UT_Y_1.2", F_uty))

    # UT-Z: F33 = lam, F11 = F22 = lat
    F_utz = np.eye(3)
    F_utz[2, 2] = lam
    F_utz[0, 0] = lat
    F_utz[1, 1] = lat
    jobs.append(("UT_Z_1.2", F_utz))

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