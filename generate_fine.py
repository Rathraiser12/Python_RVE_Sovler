#!/usr/bin/env python3
# FILE: generate.py
# Generates Initial Dataset with Adaptive Stepping and Detailed Logging

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

# Import solver
try:
    from micro_solver import RVE_Solver
except ImportError:
    from micro_solver import RVE_Solver

# ------------------------------
# Configuration
# ------------------------------
RAW_MESH_FILE = "meshes/mesh_30.msh" 
OUTPUT_CSV    = "feann_initial_data_fine.csv"
N_POINTS_PER_PATH = 12 

# ------------------------------
# Helper: Parallel Workers
# ------------------------------
def _default_workers() -> int:
    try:
        scpt = int(os.getenv("SLURM_CPUS_PER_TASK", "0") or 0)
        if scpt > 0: return scpt
    except: pass
    try:
        return os.cpu_count() or 1
    except: return 1

# ------------------------------
# 1. Job Generation (Table 2)
# ------------------------------
def create_deformation_jobs(n_points=12):
    print(f"Generating deformation paths (Table 2) with {n_points} points each...")
    jobs = []
    I = np.eye(3)

    # 1. Uniaxial Tension
    space_ut = np.linspace(1.0, 1.6, n_points)
    for lam in space_ut:
        F = I.copy(); F[0,0] = lam; jobs.append((F, f"UT_X_{lam:.3f}"))
        F = I.copy(); F[1,1] = lam; jobs.append((F, f"UT_Y_{lam:.3f}"))
        F = I.copy(); F[2,2] = lam; jobs.append((F, f"UT_Z_{lam:.3f}"))

    # 2. Uniaxial Compression
    space_uc = np.linspace(1.0, 0.7, n_points)
    for lam in space_uc:
        F = I.copy(); F[0,0] = lam; jobs.append((F, f"UC_X_{lam:.3f}"))
        F = I.copy(); F[1,1] = lam; jobs.append((F, f"UC_Y_{lam:.3f}"))
        F = I.copy(); F[2,2] = lam; jobs.append((F, f"UC_Z_{lam:.3f}"))

    # 3. Equibiaxial Tension
    space_et = np.linspace(1.0, 1.3, n_points)
    for lam in space_et:
        F = I.copy(); F[0,0]=lam; F[1,1]=lam; jobs.append((F, f"ET_XY_{lam:.3f}"))
        F = I.copy(); F[0,0]=lam; F[2,2]=lam; jobs.append((F, f"ET_XZ_{lam:.3f}"))
        F = I.copy(); F[1,1]=lam; F[2,2]=lam; jobs.append((F, f"ET_YZ_{lam:.3f}"))

    # 4. Equibiaxial Compression
    space_ec = np.linspace(1.0, 0.85, n_points)
    for lam in space_ec:
        F = I.copy(); F[0,0]=lam; F[1,1]=lam; jobs.append((F, f"EC_XY_{lam:.3f}"))
        F = I.copy(); F[0,0]=lam; F[2,2]=lam; jobs.append((F, f"EC_XZ_{lam:.3f}"))
        F = I.copy(); F[1,1]=lam; F[2,2]=lam; jobs.append((F, f"EC_YZ_{lam:.3f}"))

    # 5. Simple Shear
    space_sh = np.linspace(0.0, 0.5, n_points)
    for gam in space_sh:
        F = I.copy(); F[0,1]=gam; jobs.append((F, f"SH_XY_{gam:.3f}"))
        F = I.copy(); F[1,0]=gam; jobs.append((F, f"SH_YX_{gam:.3f}"))
        F = I.copy(); F[0,2]=gam; jobs.append((F, f"SH_XZ_{gam:.3f}"))
        F = I.copy(); F[2,0]=gam; jobs.append((F, f"SH_ZX_{gam:.3f}"))
        F = I.copy(); F[1,2]=gam; jobs.append((F, f"SH_YZ_{gam:.3f}"))
        F = I.copy(); F[2,1]=gam; jobs.append((F, f"SH_ZY_{gam:.3f}"))

    # Deduplicate
    unique_jobs = []
    seen_hashes = set()
    for F, label in jobs:
        h = tuple(np.round(F.flatten(), 8))
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_jobs.append((F, label))
            
    print(f"Unique points to simulate: {len(unique_jobs)}")
    return unique_jobs

# ------------------------------
# 2. Simulation Worker (Adaptive)
# ------------------------------
def run_simulation(job_data):
    F_target, label = job_data
    pid = os.getpid()
    
    try:
        rve = RVE_Solver(RAW_MESH_FILE)
        I = np.eye(3)
        
        # If Identity, return zero stress immediately
        if np.allclose(F_target, I, atol=1e-8):
            return np.concatenate([F_target.flatten(), np.zeros(9)])

        # --- Adaptive Stepping Loop ---
        t_current = 0.0
        dt = 0.1  # Initial step size (10%)
        dt_min = 1e-4
        
        # Apply BCs at t=0
        rve.apply_boundary_conditions(I, pbc_tol=1e-5, verbose=False)
        
        while t_current < 1.0:
            # Cap t_next at 1.0
            if t_current + dt > 1.0:
                dt = 1.0 - t_current
            
            t_next = t_current + dt
            F_next = I + t_next * (F_target - I)
            
            # Save state in case we need to rollback
            u_checkpoint = rve.u_free.copy()
            
            # Attempt Step
            rve.update_macro_deformation(F_next)
            converged = rve.solve(max_iter=20, tol=1e-5, reset=False)
            
            if converged:
                # Success: Advance time
                t_current = t_next
                # Heuristic: Increase step size if it was easy
                if dt < 0.2: dt *= 1.2
            else:
                # Failure: Rollback and cut step size
                print(f"[{pid}] Retry: {label} failed at t={t_next:.3f}. Reducing dt to {dt/2:.1e}")
                rve.set_initial_guess(u_checkpoint)
                dt *= 0.5
                
                if dt < dt_min:
                    print(f"[{pid}] FAIL: {label} step too small.")
                    return None

        # Final Stress Calculation
        P_avg = rve.calculate_avg_stress()
        return np.concatenate([F_target.flatten(), P_avg.flatten()])

    except Exception as e:
        print(f"[{pid}] CRASH: {label} - {e}")
        return None

# ------------------------------
# 3. File IO
# ------------------------------
def append_locked(csv_path, df_row):
    try:
        import fcntl
        with open(csv_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            df_row.to_csv(f, header=False, index=False)
            fcntl.flock(f, fcntl.LOCK_UN)
    except ImportError:
        df_row.to_csv(csv_path, mode="a", header=False, index=False)

# ------------------------------
# Main Driver
# ------------------------------
if __name__ == "__main__":
    if not os.path.exists(RAW_MESH_FILE):
        print(f"Error: Mesh {RAW_MESH_FILE} missing.")
        sys.exit(1)

    N_WORKERS = _default_workers()
    print(f"Running with {N_WORKERS} workers.")

    all_jobs = create_deformation_jobs(n_points=N_POINTS_PER_PATH)

    # CSV Setup
    cols =  [f"F{i}{j}" for i in range(1,4) for j in range(1,4)]
    cols += [f"P{i}{j}" for i in range(1,4) for j in range(1,4)]
    
    completed_hashes = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            df = pd.read_csv(OUTPUT_CSV)
            for row in df.iloc[:, 0:9].values:
                completed_hashes.add(tuple(np.round(row, 8)))
            print(f"Found {len(completed_hashes)} completed.")
        except: pass
    else:
        pd.DataFrame(columns=cols).to_csv(OUTPUT_CSV, index=False)

    jobs_to_run = []
    for F, lbl in all_jobs:
        if tuple(np.round(F.flatten(), 8)) not in completed_hashes:
            jobs_to_run.append((F, lbl))
            
    print(f"Jobs remaining: {len(jobs_to_run)}")
    if not jobs_to_run: sys.exit(0)

    with Pool(processes=N_WORKERS) as pool:
        # imap_unordered is faster as it yields results as soon as they finish
        for res in tqdm(pool.imap_unordered(run_simulation, jobs_to_run), total=len(jobs_to_run)):
            if res is not None:
                df_row = pd.DataFrame([res], columns=cols)
                append_locked(OUTPUT_CSV, df_row)