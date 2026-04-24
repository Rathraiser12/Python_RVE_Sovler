#!/usr/bin/env python3
# FILE: generate.py
# V5.1: Smart Filtering + VERBOSE LOGGING (No Silence)

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
TOL_RELAX = 10.0         # Tolerance for zero stress (Pa)
MAX_RELAX_ITER = 10     

# ================= 1. CORE SOLVER UTILS =================

def apply_step(solver, F_next):
    """
    Tries to solve for F_next starting from the solver's CURRENT state.
    """
    u_checkpoint = solver.u_free.copy()
    solver.update_macro_deformation(F_next)
    
    # Use relaxed tolerance (1.0) and generous iterations (30)
    res, P = solver.solve(max_iter=30, tol=1.0, reset=False)
    
    if res is None:
        solver.set_initial_guess(u_checkpoint)
        return None
    return P

def ramp_to_deformation(solver, F_target, log_tag="", dt_start=0.2):
    """
    Adaptive ramping. Accepts dt_start to be gentle for compression.
    """
    F_start = solver.F_macro.copy()
    t_current = 0.0
    dt = dt_start  # Use the passed start step
    dt_min = 1e-4
    
    while t_current < 1.0:
        if t_current + dt > 1.0: dt = 1.0 - t_current
        
        t_next = t_current + dt
        F_next = F_start + t_next * (F_target - F_start)
        
        P = apply_step(solver, F_next)
        
        if P is not None:
            t_current = t_next
            # Only increase speed if we are stable (and step wasn't tiny)
            if dt < 0.5 and dt > 0.01: dt *= 1.2 
        else:
            print(f"[{log_tag}] Ramp back dt={dt:.1e}", flush=True)
            dt *= 0.5
            if dt < dt_min: return None
            
    return solver.calculate_avg_stress()

# ================= 2. ROBUST STEP-WISE RELAXATION =================

def fast_relaxation_solver(solver, job_type, params, log_tag):
    
    # --- 1. SETUP ---
    F_base = np.eye(3)
    target_val = 1.0
    
    if job_type in ['UT', 'UC']:
        active_ax, val = params
        relax_axes = [0,1,2]; relax_axes.remove(active_ax)
        target_val = val
        F_base[active_ax, active_ax] = val
        
        # Guess Incompressible
        lat_guess = val**(-0.5)
        
    elif job_type in ['ET', 'EC']:
        plane, relax_axis, val = params
        relax_axes = [relax_axis]
        target_val = val
        F_base[plane[0], plane[0]] = val
        F_base[plane[1], plane[1]] = val
        
        # Guess Incompressible
        lat_guess = val**(-2.0)

    # --- 2. DEFINE INCREMENTS ---
    # For compression, we must go slow.
    n_steps = 5  # 5 increments
    
    # Current lateral stretch (starts at 1.0)
    current_lat = [1.0] * len(relax_axes)
    
    # Reset Solver
    solver.u_free = np.zeros(solver.pbc_manager.n_free)
    solver.apply_boundary_conditions(np.eye(3), pbc_tol=1e-5, verbose=False)

    print(f"[{log_tag}] Starting Step-Wise Relaxation ({n_steps} steps)...", flush=True)

    # --- 3. STEPPING LOOP ---
    for step in range(1, n_steps + 1):
        t = step / n_steps
        lam_step = 1.0 + t * (target_val - 1.0)
        
        F_step_base = np.eye(3)
        if job_type in ['UT', 'UC']:
            F_step_base[active_ax, active_ax] = lam_step
        else:
            F_step_base[plane[0], plane[0]] = lam_step
            F_step_base[plane[1], plane[1]] = lam_step

        def make_F(lat_vars):
            F = F_step_base.copy()
            if len(relax_axes) == 1:
                F[relax_axes[0], relax_axes[0]] = lat_vars[0]
            else:
                F[relax_axes[0], relax_axes[0]] = lat_vars[0]
                F[relax_axes[1], relax_axes[1]] = lat_vars[1]
            return F

        def objective(lat_vars):
            F_try = make_F(lat_vars)
            P = apply_step(solver, F_try) 
            if P is None: return [1e9]*len(relax_axes)
            res = []
            for ax in relax_axes: res.append(P[ax, ax])
            return res

        try:
            sol = optimize.root(objective, current_lat, method='hybr', tol=TOL_RELAX)
            if sol.success:
                current_lat = sol.x # Update guess
                # print(f"[{log_tag}] Step {step} OK. Lat={current_lat}", flush=True)
                
                if step == n_steps:
                    F_final = make_F(sol.x)
                    P_final = apply_step(solver, F_final)
                    return F_final, P_final
            else:
                print(f"[{log_tag}] Step {step}/{n_steps} Optim Failed.", flush=True)
                return None, None
                
        except Exception as e:
            print(f"[{log_tag}] Step {step} Error: {e}", flush=True)
            return None, None

    return None, None

# ================= 3. WORKER PROCESS =================

def worker_task(job_data):
    job_type, params = job_data
    
    if job_type in ['UT', 'UC']: ax, val = params; log_tag = f"{job_type}_Ax{ax}_{val:.3f}"
    elif job_type in ['ET', 'EC']: _, _, val = params; log_tag = f"{job_type}_Val{val:.3f}"
    elif job_type == 'SS': idx, val = params; log_tag = f"SS_{idx}_{val:.3f}"
    else: log_tag = "ID"

    try:
        # --- VERBOSE FIX: DO NOT SILENCE STDOUT ---
        # sys.stdout = open(os.devnull, 'w') 
        # ------------------------------------------
        
        solver = RVE_Solver(MESH_FILE)
        F_final, P_final = None, None
        
        if job_type == 'ID':
            solver.apply_boundary_conditions(np.eye(3), verbose=False)
            F_final = np.eye(3)
            P_final = solver.calculate_avg_stress()
            
        elif job_type == 'SS':
            idx, val = params
            F_target = np.eye(3); F_target[idx[0], idx[1]] = val
            solver.u_free = np.zeros(solver.pbc_manager.n_free)
            solver.apply_boundary_conditions(np.eye(3), verbose=False)
            P_final = ramp_to_deformation(solver, F_target, log_tag)
            F_final = F_target
            
        else:
            res = fast_relaxation_solver(solver, job_type, params, log_tag)
            if res: F_final, P_final = res

        # sys.stdout = sys.__stdout__ # Restore not needed
        
        if P_final is not None:
            return np.concatenate([F_final.flatten(), P_final.flatten()])
        return None

    except Exception as e:
        print(f"[{log_tag}] WORKER EXCEPTION: {e}", flush=True)
        # sys.stdout = sys.__stdout__
        return None

# ================= 4. GENERATOR LOGIC =================

def generate_jobs():
    jobs = []
    jobs.append(('ID', None))
    # 1. Uniaxial Tension
    for lam in np.linspace(1.05, 1.6, N_POINTS_PER_PATH):
        for axis in [0, 1, 2]: jobs.append(('UT', (axis, lam)))
    # 2. Uniaxial Compression
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

# ================= 5. SMART FILTERING =================

def filter_jobs(all_jobs, df):
    """
    Returns only the jobs that are NOT in the dataframe.
    Checks the 'Prescribed' components of F.
    """
    if df.empty: return all_jobs
    
    # Pre-compute data structures for fast lookup
    # 1. Shear rows (where any off-diagonal > 1e-4)
    shear_cols = ['F12','F13','F21','F23','F31','F32']
    is_shear = df[shear_cols].abs().sum(axis=1) > 1e-4
    df_shear = df[is_shear]
    df_diag  = df[~is_shear]
    
    jobs_todo = []
    
    for job in all_jobs:
        job_type, params = job
        found = False
        
        # --- IDENTITY ---
        if job_type == 'ID':
            err = (df_diag[['F11','F22','F33']] - 1.0).abs().sum(axis=1)
            if (err < 1e-3).any(): found = True
            
        # --- SIMPLE SHEAR ---
        elif job_type == 'SS':
            idx, val = params
            col_name = f"F{idx[0]+1}{idx[1]+1}"
            matches = (df_shear[col_name] - val).abs() < 1e-3
            if matches.any(): found = True
            
        # --- UNIAXIAL T/C ---
        elif job_type in ['UT', 'UC']:
            ax, val = params
            col_name = f"F{ax+1}{ax+1}"
            matches = (df_diag[col_name] - val).abs() < 1e-3
            if matches.any(): found = True
            
        # --- EQUIBIAXIAL T/C ---
        elif job_type in ['ET', 'EC']:
            plane, relax_axis, val = params
            col1 = f"F{plane[0]+1}{plane[0]+1}"
            col2 = f"F{plane[1]+1}{plane[1]+1}"
            
            match1 = (df_diag[col1] - val).abs() < 1e-3
            match2 = (df_diag[col2] - val).abs() < 1e-3
            if (match1 & match2).any(): found = True
            
        if not found:
            jobs_todo.append(job)
            
    return jobs_todo

# ================= 6. MAIN =================

if __name__ == "__main__":
    if not os.path.exists(MESH_FILE):
        print("Mesh not found."); sys.exit(1)
    
    try: N_WORKERS = int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count()))
    except: N_WORKERS = 1
    
    print(f"--- Starting FAST FEANN Generator ({N_WORKERS} workers) ---")
    
    cols = [f"F{i}{j}" for i in range(1,4) for j in range(1,4)] + \
           [f"P{i}{j}" for i in range(1,4) for j in range(1,4)]
    
    if not os.path.exists(OUTPUT_FILE):
        pd.DataFrame(columns=cols).to_csv(OUTPUT_FILE, index=False)
        df_exist = pd.DataFrame(columns=cols)
    else:
        try:
            df_exist = pd.read_csv(OUTPUT_FILE)
            print(f"Found {len(df_exist)} existing data points.")
        except:
            df_exist = pd.DataFrame(columns=cols)

    # 1. Generate All
    all_jobs = generate_jobs()
    print(f"Total Jobs Defined: {len(all_jobs)}")
    
    # 2. Filter Smartly
    jobs_to_run = filter_jobs(all_jobs, df_exist)
    print(f"Jobs Remaining after Filter: {len(jobs_to_run)}")
    
    if len(jobs_to_run) == 0:
        print("All jobs completed. Exiting.")
        sys.exit(0)

    # 3. Run Missing
    with Pool(processes=N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(worker_task, jobs_to_run, chunksize=1)):
            if res is not None:
                append_locked(OUTPUT_FILE, pd.DataFrame([res], columns=cols))
                print(f"[{i+1}/{len(jobs_to_run)}] Saved.", flush=True)
            else:
                print(f"[{i+1}/{len(jobs_to_run)}] Failed.", flush=True)

    print("--- Done ---")