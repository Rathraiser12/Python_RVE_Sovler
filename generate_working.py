#!/usr/bin/env python3
# FILE: generate_working.py
# V4: Robust Step-wise Relaxation & Adaptive Tolerance

import os
import sys
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy import optimize

# --- IMPORT SOLVER ---
try:
    from micro_solver import RVE_Solver
except ImportError:
    print("CRITICAL: 'micro_solver.py' not found.")
    sys.exit(1)

# ================= CONFIGURATION =================
MESH_FILE = "meshes/mesh_simple.msh"
OUTPUT_FILE = "feann_initial_data.csv"
N_POINTS_PER_PATH = 12  

# Tolerance for zero stress (Pa). 
# 50 Pa is approx 0.02% error on a 200kPa load, which is excellent for ML.
# 1.0 Pa was too strict and caused convergence fights.
TOL_RELAX = 4e-3  # 400 Pa         
SIGMA0 = 1e5  # 100 kPa = 100,000 Pa
# ================= 1. CORE SOLVER UTILS =================

def apply_step(solver, F_next):
    """
    Tries to solve for F_next starting from the solver's CURRENT state.
    Returns: P_avg (3x3) or None if diverged.
    """
    u_checkpoint = solver.u_free.copy()
    solver.update_macro_deformation(F_next)

    # Slightly relaxed tolerances for the big 100-fiber mesh:
    # - fewer Newton iterations
    # - still good enough accuracy for ML
    res, P = solver.solve(
        max_iter=35,
        tol_abs=1e-2,
        tol_rel=1e-5,
        reset=False,
    )

    if res is None:
        # Restore state on failure
        solver.set_initial_guess(u_checkpoint)
        return None
    return P


def ramp_to_deformation(solver, F_target):
    """
    Adaptive ramping from solver.F_macro -> F_target.
    Returns P_avg or None.
    """
    F_start = solver.F_macro.copy()
    t_current = 0.0
    dt = 0.25      # start with 4 steps
    dt_min = 1e-2  # don't go below 1% of the path
    max_steps = 40 # safety cap to avoid infinite loops
    n_steps = 0

    while t_current < 1.0 and n_steps < max_steps:
        n_steps += 1

        if t_current + dt > 1.0:
            dt = 1.0 - t_current

        t_next = t_current + dt
        F_next = F_start + t_next * (F_target - F_start)

        P = apply_step(solver, F_next)

        if P is not None:
            t_current = t_next
            # Aggressive step increase if successful
            if dt < 0.5:
                dt *= 1.5
        else:
            dt *= 0.5
            if dt < dt_min:
                return None

    if t_current < 1.0:
        # didn't reach target in max_steps
        return None

    return solver.calculate_avg_stress()

# ================= 2. ROBUST RELAXATION LOGIC =================

def solve_relaxation_at_current_load(solver, active_indices, relax_indices, F_current, guess_lat):
    """
    Optimizes the lateral stretches (relax_indices) to zero out stress
    at the current PRIMARY load (active_indices).
    """
    
    def objective(lat_vars):
        # Construct trial F
        F_try = F_current.copy()
        for i, ax in enumerate(relax_indices):
            F_try[ax, ax] = lat_vars[i]
            
        # Try to solve RVE
        P = apply_step(solver, F_try)
        
        if P is None:
            # High penalty for RVE crash
            return 1e9 
        
        # Calculate stress norm in relaxed directions
        error = 0.0
        for ax in relax_indices:
            error += abs(P[ax, ax]) / SIGMA0
        return error

    # Use Nelder-Mead: Robust, doesn't need gradients, handles noisy functions well.
    # We use a loose tolerance here because we just need to get close enough
    # for the next step, or refine at the end.
    res = optimize.minimize(
        objective, 
        guess_lat, 
        method='Nelder-Mead', 
        tol=TOL_RELAX,
        options={'maxiter': 50, 'xatol': 1e-4, 'fatol': TOL_RELAX}
    )
    
    if res.success or res.fun < TOL_RELAX * 5:
        return res.x
    return None

def attempt_relaxation_sequence(solver, job_type, params, log_tag, num_steps):
    """
    Helper function to run the full relaxation sequence with a specific number of steps.
    Returns (F_final, P_final) or None if failed.
    """
    # ... (Setup Code specific to axis selection) ...
    # --- 1. SETUP AXES (Same as before) ---
    F_target_final = np.eye(3)
    target_val = 1.0
    
    if job_type in ['UT', 'UC']:
        active_ax, val = params
        relax_axes = [0, 1, 2]; relax_axes.remove(active_ax)
        target_val = val
        F_target_final[active_ax, active_ax] = val
    elif job_type in ['ET', 'EC']:
        plane, relax_axis, val = params
        relax_axes = [relax_axis]
        target_val = val
        F_target_final[plane[0], plane[0]] = val
        F_target_final[plane[1], plane[1]] = val
        
    # Initial Guess
    if job_type in ['UT', 'UC']: lat_current = [target_val**(-0.5)] * len(relax_axes)
    else: lat_current = [target_val**(-2.0)] * len(relax_axes)

    # --- 2. STEP-WISE LOOP ---
    active_steps = np.linspace(1.0, target_val, num_steps + 1)[1:] 
    
    # Reset Solver for this attempt
    solver.u_free = np.zeros(solver.pbc_manager.n_free)
    solver.apply_boundary_conditions(np.eye(3), verbose=False)
    
    current_lat_vals = np.array([1.0] * len(relax_axes)) 
    
    for step_idx, sub_target in enumerate(active_steps):
        # Construct F for this step
        F_step_base = np.eye(3)
        if job_type in ['UT', 'UC']: F_step_base[active_ax, active_ax] = sub_target
        else:
            F_step_base[plane[0], plane[0]] = sub_target
            F_step_base[plane[1], plane[1]] = sub_target
            
        if step_idx == 0:
             if job_type in ['UT', 'UC']: current_lat_vals = [sub_target**(-0.5)]*len(relax_axes)
             else: current_lat_vals = [sub_target**(-2.0)]*len(relax_axes)
        
        # Optimize
        def objective(lat_vars):
            F_try = F_step_base.copy()
            for i, ax in enumerate(relax_axes): F_try[ax, ax] = lat_vars[i]
            P = apply_step(solver, F_try) # Make sure apply_step uses the NEW solver call
            if P is None: return 1e9 
            return sum([abs(P[ax, ax]) / SIGMA0 for ax in relax_axes])

        res = optimize.minimize(
            objective, 
            current_lat_vals, 
            method='Nelder-Mead', 
            tol=TOL_RELAX,
            # LOOSER TOLERANCE for intermediate steps to speed up
            options={'maxiter': 20, 'xatol': 1e-3, 'fatol': TOL_RELAX}
        )
        
        if res.fun > TOL_RELAX * 10: # If intermediate step failed to relax
            return None
            
        current_lat_vals = res.x

    # --- 3. FINAL VERIFICATION ---
    F_final = np.eye(3)
    if job_type in ['UT', 'UC']: F_final[active_ax, active_ax] = target_val
    else: 
        F_final[plane[0], plane[0]] = target_val
        F_final[plane[1], plane[1]] = target_val
    for i, ax in enumerate(relax_axes): F_final[ax, ax] = current_lat_vals[i]
        
    P_final = apply_step(solver, F_final)
    if P_final is None: return None
    
    max_err = max([abs(P_final[ax, ax]) / SIGMA0 for ax in relax_axes])
    if max_err > TOL_RELAX * 2.0: return None
        
    return F_final, P_final

def robust_relaxation_solver(solver, job_type, params, log_tag):
    # STRATEGY 1: Try Fast (2 Steps)
    print(f"[{log_tag}] Attempting Fast Relaxation (2 steps)...", flush=True)
    res = attempt_relaxation_sequence(solver, job_type, params, log_tag, num_steps=2)
    
    if res is not None:
        return res
        
    # STRATEGY 2: Fallback to Robust (4 Steps)
    print(f"[{log_tag}] Fast failed. Retrying Robust (4 steps)...", flush=True)
    res = attempt_relaxation_sequence(solver, job_type, params, log_tag, num_steps=4)
    
    if res is not None:
        print(f"[{log_tag}] Robust recovery successful.", flush=True)
        return res
        
    print(f"[{log_tag}] All attempts failed.", flush=True)
    return None

# ================= 3. WORKER PROCESS =================

def worker_task(job_data):
    job_type, params = job_data
    
    # Logging Tag
    if job_type == 'UT': ax, val = params; log_tag = f"UT_{ax}_{val:.2f}"
    elif job_type == 'UC': ax, val = params; log_tag = f"UC_{ax}_{val:.2f}"
    elif job_type == 'ET': _, _, val = params; log_tag = f"ET_{val:.2f}"
    elif job_type == 'EC': _, _, val = params; log_tag = f"EC_{val:.2f}"
    elif job_type == 'SS': idx, val = params; log_tag = f"SS_{idx}_{val:.2f}"
    else: log_tag = "ID"

    try:
        # Initialize solver per worker (required for PETSc)
        solver = RVE_Solver(MESH_FILE)
        
        F_final, P_final = None, None
        
        if job_type == 'ID':
            solver.apply_boundary_conditions(np.eye(3), verbose=False)
            F_final = np.eye(3)
            P_final = solver.calculate_avg_stress()
            
        elif job_type == 'SS':
            # Simple Shear doesn't need relaxation
            idx, val = params
            F_target = np.eye(3); F_target[idx[0], idx[1]] = val
            solver.u_free = np.zeros(solver.pbc_manager.n_free)
            solver.apply_boundary_conditions(np.eye(3), verbose=False)
            P_final = ramp_to_deformation(solver, F_target)
            F_final = F_target
            
        else:
            # Use Robust Step-wise Relaxation
            res = robust_relaxation_solver(solver, job_type, params, log_tag)
            if res: F_final, P_final = res

        if P_final is not None:
            # Final sanity check on NaNs
            if np.any(np.isnan(P_final)): return None
            return np.concatenate([F_final.flatten(), P_final.flatten()])
        return None

    except Exception as e:
        # Catch unexpected errors to keep other workers alive
        # print(f"[{log_tag}] Worker Exception: {e}", flush=True)
        return None

# ================= 4. GENERATOR LOGIC =================

def generate_jobs():
    jobs = []
    jobs.append(('ID', None))
    # 1. Uniaxial Tension (Range 1.05 -> 1.6)
    for lam in np.linspace(1.05, 1.6, N_POINTS_PER_PATH):
        for axis in [0, 1, 2]: jobs.append(('UT', (axis, lam)))
    # 2. Uniaxial Compression (Range 0.95 -> 0.7)
    for lam in np.linspace(0.95, 0.7, N_POINTS_PER_PATH):
        for axis in [0, 1, 2]: jobs.append(('UC', (axis, lam)))
    # 3. Equibiaxial Tension
    for lam in np.linspace(1.05, 1.3, N_POINTS_PER_PATH):
        jobs.append(('ET', ((0,1), 2, lam)))
        jobs.append(('ET', ((0,2), 1, lam)))
        jobs.append(('ET', ((1,2), 0, lam)))
    # 4. Equibiaxial Compression
    for lam in np.linspace(0.95, 0.85, N_POINTS_PER_PATH):
        jobs.append(('EC', ((0,1), 2, lam)))
        jobs.append(('EC', ((0,2), 1, lam)))
        jobs.append(('EC', ((1,2), 0, lam)))
    # 5. Simple Shear
    for gam in np.linspace(0.05, 0.5, N_POINTS_PER_PATH):
        shear_modes = [(0,1), (1,0), (1,2), (2,1), (0,2), (2,0)]
        for idx in shear_modes: jobs.append(('SS', (idx, gam)))
    return jobs

def append_locked(csv_path, df_row):
    try:
        import fcntl
        with open(csv_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            df_row.to_csv(f, header=False, index=False)
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)
    except ImportError:
        df_row.to_csv(csv_path, mode="a", header=False, index=False)

if __name__ == "__main__":
    if not os.path.exists(MESH_FILE):
        print("Mesh not found."); sys.exit(1)
    
    try: N_WORKERS = int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count()))
    except: N_WORKERS = 1
    
    print(f"--- Starting ROBUST FEANN Generator ({N_WORKERS} workers) ---")
    print(f"--- Relaxation Tolerance: {TOL_RELAX} Pa ---")
    
    cols = [f"F{i}{j}" for i in range(1,4) for j in range(1,4)] + \
           [f"P{i}{j}" for i in range(1,4) for j in range(1,4)]
    
    if not os.path.exists(OUTPUT_FILE):
        pd.DataFrame(columns=cols).to_csv(OUTPUT_FILE, index=False)

    # Load existing to avoid duplicates
    try:
        df_exist = pd.read_csv(OUTPUT_FILE)
        existing_hashes = set()
        for i in range(len(df_exist)):
            # Hash deformation gradient only
            f_vals = tuple(np.round(df_exist.iloc[i, 0:9].values.astype(float), 4))
            existing_hashes.add(f_vals)
        print(f"Found {len(df_exist)} existing entries.")
    except: existing_hashes = set()

    jobs_to_run = generate_jobs()
    print(f"Queue size: {len(jobs_to_run)}")

    # Chunksize=1 is important for load balancing heavy RVE jobs
    with Pool(processes=N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(worker_task, jobs_to_run, chunksize=1)):
            if res is not None:
                f_chk = tuple(np.round(res[0:9], 4))
                if f_chk not in existing_hashes:
                    append_locked(OUTPUT_FILE, pd.DataFrame([res], columns=cols))
                    existing_hashes.add(f_chk)
                    print(f"[{i+1}/{len(jobs_to_run)}] Saved.", flush=True)
                else:
                    print(f"[{i+1}/{len(jobs_to_run)}] Duplicate.", flush=True)
            else:
                # Silent fail in log, but progress bar continues
                print(f"[{i+1}/{len(jobs_to_run)}] Failed/Skipped.", flush=True)

    print("--- Done ---")