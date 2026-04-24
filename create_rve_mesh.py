import gmsh
import sys
import os
import numpy as np
import random

# --- CONFIG ---
Lx = 323.6
Ly = 323.6
Lz = 10.0
N_FIBERS = 100
VF = 0.30
R = np.sqrt((VF * Lx * Ly) / (N_FIBERS * np.pi))

print(f"Generating Simple RVE: {N_FIBERS} fibers, R={R:.4f}")

# --- PLACEMENT (Safe Mode: Inside Box) ---
def get_centers():
    centers = []
    margin = 1.1 * R
    r_sq = (2.2 * R)**2 # Ensure gaps
    for i in range(200000):
        x = random.uniform(margin, Lx - margin)
        y = random.uniform(margin, Ly - margin)
        if all((x-cx)**2 + (y-cy)**2 > r_sq for cx,cy in centers):
            centers.append((x, y))
            if len(centers) == N_FIBERS: break
    return centers

centers = get_centers()
if len(centers) < N_FIBERS:
    print("RSA Failed.")
    sys.exit(1)

# --- GMSH ---
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("simple_rve")

# 1. Create Geometry
# Box
box = gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)

# Fibers (Embedded, no cuts needed if we mesh carefully)
# Actually, for multimaterial we need volumes.
# We will use Fragment which is robust.
fiber_tags = []
for (x, y) in centers:
    c = gmsh.model.occ.addCylinder(x, y, 0, 0, 0, Lz, R)
    fiber_tags.append(c)

gmsh.model.occ.synchronize()

# Fragment: intersects box and fibers, creates compatible topology
# Input: Box (object), Fibers (tool)
# Output: Matrix volume (with holes filled by fibers) and Fiber volumes
print("Fragmenting...")
out, map = gmsh.model.occ.fragment([(3, box)], [(3, t) for t in fiber_tags])
gmsh.model.occ.synchronize()

# 2. Identify Volumes (Matrix vs Fiber)
# Fibers are the ones with centers matching our list
final_matrix = []
final_fibers = []

for dim, tag in out:
    if dim != 3: continue
    bbox = gmsh.model.getBoundingBox(dim, tag)
    cx = (bbox[0]+bbox[3])/2
    cy = (bbox[1]+bbox[4])/2
    
    is_fiber = False
    for fx, fy in centers:
        if (cx-fx)**2 + (cy-fy)**2 < (R+0.1)**2:
            is_fiber = True
            break
    
    if is_fiber: final_fibers.append(tag)
    else: final_matrix.append(tag)

# 3. Periodicity
# Get all boundary surfaces
surfs = gmsh.model.getBoundary(out, combined=False, oriented=False)
s_left, s_right, s_bot, s_top, s_back, s_front = [], [], [], [], [], []
eps = 1e-3

for _, s in surfs:
    bb = gmsh.model.getBoundingBox(2, s)
    cx, cy, cz = (bb[0]+bb[3])/2, (bb[1]+bb[4])/2, (bb[2]+bb[5])/2
    
    if abs(cx-0)<eps: s_left.append(s)
    elif abs(cx-Lx)<eps: s_right.append(s)
    elif abs(cy-0)<eps: s_bot.append(s)
    elif abs(cy-Ly)<eps: s_top.append(s)
    elif abs(cz-0)<eps: s_back.append(s)
    elif abs(cz-Lz)<eps: s_front.append(s)

# Apply translation periodicity to mesh
# Note: For this to work perfectly, surfaces must be identical.
# Since we used Safe Mode (no fibers touching boundary), Left and Right surfaces ARE identical (just flat squares).
tx = [1,0,0,Lx, 0,1,0,0, 0,0,1,0, 0,0,0,1]
ty = [1,0,0,0, 0,1,0,Ly, 0,0,1,0, 0,0,0,1]
tz = [1,0,0,0, 0,1,0,0, 0,0,1,Lz, 0,0,0,1]

for sl, sr in zip(s_left, s_right): gmsh.model.mesh.setPeriodic(2, [sr], [sl], tx)
for sb, st in zip(s_bot, s_top):    gmsh.model.mesh.setPeriodic(2, [st], [sb], ty)
for sbk, sf in zip(s_back, s_front): gmsh.model.mesh.setPeriodic(2, [sf], [sbk], tz)

# 4. Physical Groups
gmsh.model.addPhysicalGroup(3, final_matrix, 101)
gmsh.model.setPhysicalName(3, 101, "Matrix")
gmsh.model.addPhysicalGroup(3, final_fibers, 102)
gmsh.model.setPhysicalName(3, 102, "Fibers")

# 5. Mesh
gmsh.option.setNumber("Mesh.ElementOrder", 2)
gmsh.option.setNumber("Mesh.HighOrderOptimize", 1)
gmsh.option.setNumber("Mesh.Optimize", 1)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 9.0)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 9.0)

print("Meshing...")
gmsh.model.mesh.generate(3)

os.makedirs("meshes", exist_ok=True)
gmsh.write("meshes/rve_100fibers_periodic_safe.msh")
print("Done.")
gmsh.finalize()