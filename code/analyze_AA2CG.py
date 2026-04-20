import MDAnalysis as mda
import numpy as np
import json
import itertools
import matplotlib.pyplot as plt
from MDAnalysis.analysis.rdf import InterRDF
from scipy.signal import find_peaks
import os

# ===============================
# Load CG trajectory
# ===============================
u = mda.Universe("AA2CG/cg_trajectory.pdb")

outdir = "AA2CG"
# ===============================
# Atom types
# ===============================
atom_types = sorted(set(u.atoms.names))
print(" Atom types:", atom_types)

results = {
    "atom_types": atom_types,
    "bond_lengths": {},
    "angles": {},
    "rdf_peaks": {}
}

# ===============================
# Bond lengths (ALL combinations)
# ===============================
for a1, a2 in itertools.combinations(atom_types, 2):
    distances = []

    for res in u.residues:
        atoms1 = res.atoms.select_atoms(f"name {a1}")
        atoms2 = res.atoms.select_atoms(f"name {a2}")

        if len(atoms1) == 1 and len(atoms2) == 1:
            d = np.linalg.norm(atoms1.positions[0] - atoms2.positions[0])
            distances.append(d)

    if distances:
        distances = np.array(distances)
        results["bond_lengths"][f"{a1}-{a2}"] = {
            "mean_bond_length(Å)": round(float(distances.mean()), 4),
            "std_bond_length(Å)": round(float(distances.std()), 4)
        }

# ===============================
# Angles (ALL combinations, if >=3 atoms)
# ===============================
if len(atom_types) >= 3:
    for a1, a2, a3 in itertools.permutations(atom_types, 3):
        angles = []

        for res in u.residues:
            A = res.atoms.select_atoms(f"name {a1}")
            B = res.atoms.select_atoms(f"name {a2}")
            C = res.atoms.select_atoms(f"name {a3}")

            if len(A) == len(B) == len(C) == 1:
                v1 = A.positions[0] - B.positions[0]
                v2 = C.positions[0] - B.positions[0]

                cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
                angles.append(ang)

        if angles:
            angles = np.array(angles)
            results["angles"][f"{a1}-{a2}-{a3}"] = {
                "mean_angle_deg": round(float(angles.mean()), 4),
                "std_angle_deg": round(float(angles.std()), 4)
            }

# ===============================
# RDFs + peak detection (ALL pairs)
# ===============================
for a1, a2 in itertools.combinations_with_replacement(atom_types, 2):
    group1 = u.select_atoms(f"name {a1}")
    group2 = u.select_atoms(f"name {a2}")

    if len(group1) == 0 or len(group2) == 0:
        continue

    rdf = InterRDF(
        group1,
        group2,
        exclusion_block=(1, 1)
    )
    rdf.run()

    r = rdf.results.bins
    g = rdf.results.rdf

    mask = r > 0.2
    r_valid = r[mask]
    g_valid = g[mask]

    peaks, props = find_peaks(g_valid, height=1.0)

    if len(peaks) == 0:
        continue

    p = peaks[0]
    r_peak = float(r_valid[p])
    g_peak = float(g_valid[p])

    pair_key = f"{a1}-{a2}"
    results["rdf_peaks"][pair_key] = {
        "Rmin (Å)": round(r_peak, 4),
        "peak_height g(r)": round(g_peak, 4) 
    }

    # ---- Plot RDF ----
    plt.figure(figsize=(6, 4))
    plt.plot(r, g, lw=2)
    plt.scatter(r_peak, g_peak, color="red")

    plt.annotate(
        f"{r_peak:.2f} Å",
        xy=(r_peak, g_peak),
        xytext=(r_peak + 0.8, g_peak),
        arrowprops=dict(arrowstyle="->")
    )

    plt.xlabel("r (Å)")
    plt.ylabel("g(r)")
    plt.title(f"RDF: {a1}–{a2}")
    plt.tight_layout()
    plt.savefig(f"{outdir}/rdf_{a1}_{a2}.png", dpi=300)
    plt.close()

# ===============================
# Save JSON
# ===============================
with open(f"{outdir}/AA2CG_results.json", "w") as f:
    json.dump(results, f, indent=4)

print(" ✓ AA2CG analysis complete")