import os
import sys
import numpy as np
import meshio

def load_3d_mesh(mesh_path):
    """
    Load a Gmsh TET10 RVE mesh and return:
      nodes: (N,3)
      elements: (Ne,10) int, 0-based, in Gmsh TET10 ordering
      material_index_map: (Ne,), 0 for matrix, 1 for fiber
    """
    print(f"\nLoading mesh from: {mesh_path}\n")

    if not os.path.exists(mesh_path):
        raise FileNotFoundError(mesh_path)

    mesh = meshio.read(mesh_path)

    nodes = mesh.points[:, :3].astype(float)
    print(f"  Nodes read: {nodes.shape[0]}")

    # find physical tag name
    phys_name = None
    for k in mesh.cell_data.keys():
        if k in ("gmsh:physical", "physical"):
            phys_name = k
            break
    if phys_name is None:
        raise RuntimeError("No 'gmsh:physical' or 'physical' in cell_data.")

    phys_blocks = mesh.cell_data[phys_name]

    print("  Cell blocks in file:")
    for cb in mesh.cells:
        print(f"    {cb.type:<8} {cb.data.shape}")
    print("")

    all_elems = []
    all_tags = []

    for (cb, tags) in zip(mesh.cells, phys_blocks):
        ctype = cb.type.lower()
        conn = cb.data.astype(int)
        tags = np.asarray(tags, dtype=int)

        # keep only tet10
        if not ((ctype == "tetra10") or (ctype == "tetra" and conn.shape[1] == 10)):
            continue

        if conn.shape[1] != 10 or tags.shape[0] != conn.shape[0]:
            continue

        # positive physical tags only
        mask = tags > 0
        if not np.any(mask):
            continue

        conn = conn[mask]
        tags = tags[mask]

        all_elems.append(conn)
        all_tags.append(tags)

        print(f"  Accepted {conn.shape[0]} TET10 elements from '{cb.type}' block.")

    if not all_elems:
        raise RuntimeError("No valid TET10 elements with physical tags found.")

    elements = np.vstack(all_elems)
    phys_tags = np.concatenate(all_tags)

    # gmsh is 1-based -> convert to 0-based if needed
    if elements.min() == 1:
        elements -= 1

    print(f"  Total TET10 elements loaded: {elements.shape[0]}")

    # Material mapping:
    unique = np.unique(phys_tags[phys_tags > 0])
    print(f"  Unique physical tags: {unique}")

    if unique.size == 0:
        raise RuntimeError("No positive physical tags in TET10 elements.")

    matrix_tag = int(unique.min())
    fiber_tags = [int(t) for t in unique if t != matrix_tag]
    fiber_set = set(fiber_tags)

    print(f"  Material mapping: matrix_tag={matrix_tag}, fiber_tags={fiber_tags}")

    material_index_map = np.zeros(elements.shape[0], dtype=int)
    if fiber_tags:
        material_index_map[np.isin(phys_tags, list(fiber_set))] = 1

    n_matrix = int(np.sum(material_index_map == 0))
    n_fiber = int(np.sum(material_index_map == 1))
    print(f"  Elements as matrix: {n_matrix}, as fibers: {n_fiber}")

    # No orientation modifications here.
    return nodes, elements, material_index_map


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "meshes/rve_gmshModel_TET10_coarse.msh"

    nodes, elems, mat_map = load_3d_mesh(path)
    print(f"\nMesh summary: {nodes.shape[0]} nodes, {elems.shape[0]} elements.\n")
