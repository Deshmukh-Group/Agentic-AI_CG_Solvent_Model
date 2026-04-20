"""
Topology Creator Agent for NAMD Simulations

This agent handles the complete workflow for creating NAMD input files:
1. Coarse-grains molecules based on mapping scheme (center of mass)
2. Creates PSFgen script with bead definitions
3. Uses Packmol to pack molecules at experimental density
4. Generates final NAMD_input.pdb and NAMD_input.psf files
"""

import os
import json
import numpy as np
import requests
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import subprocess
import tempfile
from common import LLMAgent, AgentRole

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class AtomData:
    """Atom information from all-atom structure"""
    index: int
    name: str
    resname: str
    resid: int
    x: float
    y: float
    z: float
    element: str
    mass: float
    charge: float = 0.0

@dataclass
class BeadData:
    """Coarse-grained bead information"""
    index: int
    name: str
    resname: str
    resid: int
    x: float
    y: float
    z: float
    mass: float
    charge: float = 0.0
    atom_indices: List[int] = None  # AA atoms that map to this bead

@dataclass
class CGMolecule:
    """Single coarse-grained molecule"""
    mol_id: int
    beads: List[BeadData]
    bonds: List[Tuple[int, int]]  # Bead index pairs

# ============================================================================
# Topology Creator Agent
# ============================================================================

class TopologyCreatorAgent(LLMAgent):
    """Agent for creating NAMD topology and coordinate files"""

    def __init__(self, api_key: str, url: str, output_dir: str = ".", prompts_dir: str = "prompts"):
        super().__init__(AgentRole.TOPOLOGY, api_key, url)
        self.output_dir = output_dir
        self.prompts_dir = prompts_dir
        self.mapping_scheme = None
        self.aa_structure = None
        self.cg_molecules = []
        self.prev_bead_params = self._load_prev_bead_params()
        self._load_prompts()

    def _load_prompts(self):
        """Load prompts from files"""
        with open(os.path.join(self.prompts_dir, "topology_system_prompt.txt"), 'r') as f:
            self.system_prompt = f.read().strip()
        with open(os.path.join(self.prompts_dir, "topology_prompt_template.txt"), 'r') as f:
            self.prompt_template = f.read().strip()

    def _load_prev_bead_params(self) -> Dict:
        """Load previous bead parameters from cdhm_dict"""
        prev_params_file = "cdhm_dict.json"
        if os.path.exists(prev_params_file):
            try:
                with open(prev_params_file, 'r') as f:
                    prev_params = json.load(f)
                print(f"[topology] Loaded {len(prev_params)} residue types from {prev_params_file}")
                return prev_params
            except Exception as e:
                print(f"[topology] Warning: Could not load {prev_params_file}: {e}")
                return {}
        return {}

    def _get_bead_params_from_prev(self, bead_type: str, dict_resname: str = None) -> Optional[tuple]:
        """Get epsilon and Rmin/2 from cdhm_dict"""
        if not self.prev_bead_params:
            return None
    
        # Try molecule-specific lookup
        if dict_resname and dict_resname in self.prev_bead_params:
            res_params = self.prev_bead_params[dict_resname]
            if bead_type in res_params:
                bead_data = res_params[bead_type]
                if isinstance(bead_data, list) and len(bead_data) == 3:
                    return (bead_data[1], bead_data[2])  # epsilon, rmin_half
        
        # Search all residues
        for residue, beads in self.prev_bead_params.items():
            if bead_type in beads:
                bead_data = beads[bead_type]
                if isinstance(bead_data, list) and len(bead_data) == 3:
                    return (bead_data[1], bead_data[2])
        
        return None
        
    def set_mapping_scheme(self, mapping_scheme):
        """Set the CG mapping scheme"""
        self.mapping_scheme = mapping_scheme
        print(f"✓ Mapping scheme loaded: {len(mapping_scheme.bead_types)} bead types")
    
    # ========================================================================
    # Step 1a: Coarse-Grain Molecules (Center of Mass)
    # ========================================================================
    
    def load_aa_structure(self, pdb_file: str) -> List[AtomData]:
        """Load all-atom structure from PDB file"""
        atoms = []
        
        # Element masses (g/mol)
        mass_table = {
            'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
            'S': 32.065, 'P': 30.974, 'F': 18.998, 'Cl': 35.453,
            'Br': 79.904, 'I': 126.904
        }
        
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    try:
                        index = int(line[6:11].strip())
                        name = line[12:16].strip()
                        resname = line[17:20].strip()
                        resid = int(line[22:26].strip())
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        element = line[76:78].strip() if len(line) > 76 else name[0]
                        
                        mass = mass_table.get(element, 12.011)
                        
                        atoms.append(AtomData(
                            index=index, name=name, resname=resname, resid=resid,
                            x=x, y=y, z=z, element=element, mass=mass
                        ))
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse line: {line.strip()}")
                        continue
        
        self.aa_structure = atoms
        print(f"✓ Loaded {len(atoms)} atoms from {pdb_file}")
        return atoms
    
    def create_mapping_rules(self, molecule_name: str = "DMA") -> Dict:
        """
        Create mapping rules from AA atoms to CG beads using LLM
        Returns a dictionary mapping bead names to atom selections
        """

        atom_info = "\n".join([f"  {a.name} - resid {a.resid}" for a in self.aa_structure])

        prompt = self.prompt_template.format(
            molecule_name=molecule_name,
            atom_info=atom_info,
            bead_types=', '.join(self.mapping_scheme.bead_types),
            bead_descriptions=json.dumps(self.mapping_scheme.bead_descriptions, indent=2),
            connectivity=self.mapping_scheme.connectivity
        )
        
        # Call LLM using LLMAgent
        result = self.call(prompt, self.system_prompt, temperature=0.5)
        print (f"Topology Agent says: {result}")

        if result and "mapping_rules" in result:
            return result["mapping_rules"]

        # Return empty dict if LLM fails - let caller handle
        return {}
    
    def coarse_grain_molecule(self, atoms: List[AtomData],
                              mapping_rules: Dict,
                              mol_id: int = 1) -> CGMolecule:
        """
        Coarse-grain a single molecule using center of mass
        Handle dummy beads with mass splitting
        """
        beads = []
        bead_index = 1

        # Get unique residue ID for this molecule
        resid = atoms[0].resid if atoms else 1

        # First pass: create core beads (with atoms)
        core_beads = {}
        for bead_name, rule in mapping_rules.items():
            if bead_name not in self.mapping_scheme.dummy_beads:
                # Find atoms belonging to this bead
                bead_atoms = [a for a in atoms if a.name in rule["atom_names"]]

                if not bead_atoms:
                    print(f"Warning: No atoms found for bead {bead_name}")
                    continue

                # Calculate center of mass
                total_mass = sum(a.mass for a in bead_atoms)
                com_x = sum(a.x * a.mass for a in bead_atoms) / total_mass
                com_y = sum(a.y * a.mass for a in bead_atoms) / total_mass
                com_z = sum(a.z * a.mass for a in bead_atoms) / total_mass

                # Create core bead
                bead = BeadData(
                    index=bead_index,
                    name=bead_name,
                    resname="CG",
                    resid=resid,
                    x=com_x,
                    y=com_y,
                    z=com_z,
                    mass=total_mass,  # Will be adjusted if dummy beads exist
                    charge=sum(a.charge for a in bead_atoms),
                    atom_indices=[a.index for a in bead_atoms]
                )

                core_beads[bead_name] = bead
                beads.append(bead)
                bead_index += 1

        # Calculate total mass before dummy handling
        total_mass = sum(b.mass for b in core_beads.values())
        print (f"  Total mass before dummy handling: {total_mass:.3f} Da")        

        # Second pass: handle dummy beads and mass splitting
        # First, count how many dummies each core bead has
        core_to_dummies = {}
        for dummy_bead_name in self.mapping_scheme.dummy_beads:
            # Find which core bead this dummy is connected to
            connected_core = None
            for conn in self.mapping_scheme.connectivity:
                if len(conn) == 2:
                    if conn[0] == dummy_bead_name and conn[1] in core_beads:
                        connected_core = conn[1]
                        break
                    elif conn[1] == dummy_bead_name and conn[0] in core_beads:
                        connected_core = conn[0]
                        break

            if connected_core:
                if connected_core not in core_to_dummies:
                    core_to_dummies[connected_core] = []
                core_to_dummies[connected_core].append(dummy_bead_name)

        # Now split masses: each core bead and its dummies share the mass equally
        for core_name, dummy_names in core_to_dummies.items():
            print (f"  Processing core bead {core_name} with dummies {dummy_names}")
            if core_name in core_beads:
                core_bead = core_beads[core_name]
                original_mass = core_bead.mass
                num_parts = 1 + len(dummy_names)  # core + dummies
                split_mass = original_mass / num_parts

                # Set core bead mass
                core_bead.mass = split_mass

                # Create dummy beads with z-offset positioning
                for i, dummy_name in enumerate(dummy_names):
                    # Position dummy beads 0.2 Å away in z-direction from core
                    # Alternate positive and negative offsets for multiple dummies
                    z_offset = 0.2 if i % 2 == 0 else -0.2

                    dummy_bead = BeadData(
                        index=bead_index,
                        name=dummy_name,
                        resname="CG",
                        resid=resid,
                        x=core_bead.x,
                        y=core_bead.y,
                        z=core_bead.z + z_offset,  # 0.2 Å offset in z-direction
                        mass=split_mass,  # Equal share of mass
                        charge=0.0,  # Charge will be set based on mapping scheme description
                        atom_indices=[]  # No atoms for dummy beads
                    )

                    beads.append(dummy_bead)
                    bead_index += 1

                print(f"✓ Created {len(dummy_names)} dummy bead(s) for {core_name}")
                print(f"  Original mass: {original_mass:.3f}, Split mass each: {split_mass:.3f}")
                print(f"  Total mass conserved: {original_mass:.3f} = {(1 + len(dummy_names)) * split_mass:.3f}")

        # Define bonds based on connectivity in mapping scheme
        bonds = []
        bead_name_to_idx = {b.name: i for i, b in enumerate(beads)}

        for conn in self.mapping_scheme.connectivity:
            if len(conn) == 2:
                bead1_name, bead2_name = conn
                if bead1_name in bead_name_to_idx and bead2_name in bead_name_to_idx:
                    idx1 = bead_name_to_idx[bead1_name]
                    idx2 = bead_name_to_idx[bead2_name]
                    bonds.append((idx1, idx2))

        return CGMolecule(mol_id=mol_id, beads=beads, bonds=bonds)
    
    def coarse_grain_system(self, aa_pdb: str, molecule_name: str = "DMA") -> Tuple[str, Dict]:
        """
        Coarse-grain entire system and create namd_inp.pdb
        Returns CG PDB path and mapping rules
        """
        print("\n[Step 1a] Coarse-Graining System...")

        # Load AA structure
        self.load_aa_structure(aa_pdb)

        # Get mapping rules from LLM
        print("  Creating mapping rules...")
        mapping_rules = self.create_mapping_rules(molecule_name)

        # Group atoms by molecule (by residue ID)
        molecules = {}
        for atom in self.aa_structure:
            if atom.resid not in molecules:
                molecules[atom.resid] = []
            molecules[atom.resid].append(atom)

        print(f"  Found {len(molecules)} molecules")

        # Coarse-grain each molecule
        self.cg_molecules = []
        for mol_id, (resid, atoms) in enumerate(molecules.items(), 1):
            cg_mol = self.coarse_grain_molecule(atoms, mapping_rules, mol_id)
            self.cg_molecules.append(cg_mol)

        # Write CG PDB
        output_pdb = os.path.join(self.output_dir, "namd_inp.pdb")
        self.write_cg_pdb(output_pdb)

        print(f"✓ Created {output_pdb} with {len(self.cg_molecules)} CG molecules")
        return output_pdb, mapping_rules
    
    def write_cg_pdb(self, filename: str):
        """Write CG molecules to PDB file"""
        with open(filename, 'w') as f:
            f.write("REMARK   CG Structure\n")
            atom_index = 1
            
            for mol in self.cg_molecules:
                for bead in mol.beads:
                    # PDB format
                    line = f"ATOM  {atom_index:5d}  {bead.name:<4s} {'CG':>3s} " \
                           f"{bead.resid:4d}    {bead.x:8.3f}{bead.y:8.3f}{bead.z:8.3f}" \
                           f"  1.00  0.00           {bead.name[0]:>2s}\n"
                    f.write(line)
                    atom_index += 1
                f.write("TER\n")
            
            f.write("END\n")

    # ========================================================================
    # Step 1b: Convert AA Trajectory to CG
    # ========================================================================
    def convert_aa_to_cg_trajectory(self, aa_psf: str, aa_dcd: str, cg_pdb_output: str, mapping_rules: Dict):
        """
        Convert all-atom trajectory to coarse-grained trajectory
        """
        print("\n[Step 1b: Convert AA to CG Trajectory]")
        
        import MDAnalysis as mda
        import warnings
        warnings.filterwarnings('ignore')
        
        # Load the universe
        u = mda.Universe(aa_psf, aa_dcd)
        
        # Select all atoms (solvent molecules)
        Solvent = u.select_atoms("all")
        
        total_frames = len(u.trajectory)
        print(f"Loaded trajectory: {len(Solvent.residues)} residues, {total_frames} frames")
        print(f"  >> Processing last 100 frames for conversion")
        print(f"Unique resnames: {np.unique(Solvent.residues.resnames)}")
        print(f"Unique atom names: {np.unique(Solvent.names)}")
        
        # Create output directory if needed
        output_dir = os.path.dirname(cg_pdb_output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Determine frame range (last 100 frames)
        start_frame = max(0, total_frames - 100)
        frames_to_process = range(start_frame, total_frames)
        
        # Count total CG beads
        num_residues = len(Solvent.residues)
        cg_beads_per_residue = len([b for b in mapping_rules.keys() 
                                    if b not in self.mapping_scheme.dummy_beads])
        total_cg_atoms = num_residues * cg_beads_per_residue
        
        print(f"  CG system: {num_residues} residues × {cg_beads_per_residue} beads = {total_cg_atoms} total atoms")
        
        with open(cg_pdb_output, "w") as outfile:
            for frame_idx, frame_index in enumerate(frames_to_process):
                u.trajectory[frame_index]
                
                # Extract box dimensions from trajectory
                try:
                    box = u.trajectory.ts.dimensions  # [a, b, c, alpha, beta, gamma]
                    if box is not None:
                        box_x, box_y, box_z = box[0], box[1], box[2]
                        alpha, beta, gamma = box[3], box[4], box[5]
                    else:
                        raise AttributeError
                except (AttributeError, TypeError):
                    # Fallback: estimate from atom positions
                    positions = u.atoms.positions
                    min_coords = np.min(positions, axis=0)
                    max_coords = np.max(positions, axis=0)
                    box_dims = max_coords - min_coords
                    box_x, box_y, box_z = box_dims[0], box_dims[1], box_dims[2]
                    alpha, beta, gamma = 90.0, 90.0, 90.0
                
                # Write CRYST1 record (PDB standard format)
                outfile.write(f"CRYST1{box_x:9.3f}{box_y:9.3f}{box_z:9.3f}"
                            f"{alpha:7.2f}{beta:7.2f}{gamma:7.2f} P 1           1\n")
                
                # Write MODEL record for multi-frame PDB
                outfile.write(f"MODEL     {frame_idx + 1:4d}\n")
                
                atom_index = 1
                
                # Process each residue
                for residue_index, residue in enumerate(Solvent.residues):
                    resname = residue.resname
                    residue_id = residue.resid
                    
                    # Keep residue number within PDB limits (1-9999)
                    pdb_resid = (residue_index % 9999) + 1
                    
                    # Process each CG bead for this residue (in consistent order)
                    for cg_bead in sorted(mapping_rules.keys()):  # Sort for consistency
                        # Skip dummy beads
                        if cg_bead in self.mapping_scheme.dummy_beads:
                            continue
                        
                        atom_names = mapping_rules[cg_bead].get('atom_names', [])
                        if not atom_names:
                            continue
                        
                        # Build selection string
                        atom_names_str = ' '.join(atom_names)
                        selection_str = f"resid {residue_id} and name {atom_names_str}"
                        
                        try:
                            atoms = u.select_atoms(selection_str)
                        except Exception as e:
                            if residue_index == 0 and frame_idx == 0:
                                print(f"  Selection error for {cg_bead}: {e}")
                            continue
                        
                        if len(atoms) == 0:
                            if residue_index == 0 and frame_idx == 0:
                                print(f"  Warning: No atoms found for {cg_bead} (sel: {selection_str})")
                            continue
                        
                        # Compute center of mass using MDAnalysis
                        com = atoms.center_of_mass()
                        
                        # Keep atom index within PDB limits (1-99999)
                        pdb_atomid = (atom_index % 99999)
                        if pdb_atomid == 0:
                            pdb_atomid = 99999
                        
                        outfile.write(
                            f"ATOM  {pdb_atomid:5d} "
                            f"{cg_bead:<4s}"
                            f"{resname:>3s} "
                            f"A"
                            f"{pdb_resid:4d}    "
                            f"{com[0]:8.3f}{com[1]:8.3f}{com[2]:8.3f}"
                            f"{1.00:6.2f}{0.00:6.2f}          "
                            f"{cg_bead[0]:>2s}\n"
                        )

                        atom_index += 1
                
                # End of model
                outfile.write("TER\n")
                outfile.write("ENDMDL\n")
                
                # Progress reporting
                if (frame_idx + 1) % 10 == 0 or frame_idx == 0 or frame_idx == len(frames_to_process) - 1:
                    print(f"  Processed frame {frame_idx + 1}/{len(frames_to_process)} "
                        f"(trajectory frame {frame_index}, {atom_index-1} CG atoms)")
        
        print(f"✓ CG trajectory written to {cg_pdb_output}")
        print(f"  Total frames written: {len(frames_to_process)}")
        print(f"  Atoms per frame: {atom_index-1}")
        
        return cg_pdb_output

    # ========================================================================
    # Step 2: Create PSFgen Script
    # ========================================================================
    
    def create_psfgen_script(self, parameters: Dict, dict_resname: str = "XXX") -> str:

        """
        Create PSFgen script with bead definitions and topology
        """
        print("\n[Step 2] Creating PSFgen Script...")
        
        script_file = os.path.join(self.output_dir, "psf_creation.pgn")
        rtf_file = os.path.join(self.output_dir, "cg_topology.rtf")
        prm_file = os.path.join(self.output_dir, "cg_parameters.prm")
        
        # Create topology file (RTF)
        self._create_rtf_file(rtf_file, parameters)
        
        # Create parameter file (PRM)
        self._create_prm_file(prm_file, parameters, dict_resname)
        
        # Create PSFgen script
        with open(script_file, 'w') as f:
            f.write("# PSFgen script for CG system\n\n")
            f.write("package require psfgen\n")
            f.write("topology cg_topology.rtf\n\n")
            
            f.write("# Create segments\n")
            f.write("segment CG {\n")
            f.write("  pdb namd_inp.pdb\n")
            f.write("}\n\n")
            
            f.write("# Read coordinates\n")
            f.write("coordpdb namd_inp.pdb CG\n\n")
            
            f.write("# Guess missing coordinates and bonds\n")
            f.write("guesscoord\n\n")
            
            f.write("# Write output files\n")
            f.write("writepsf NAMD_input.psf\n")
            f.write("writepdb NAMD_input.pdb\n\n")
            
            f.write("exit\n")
        
        print(f"✓ Created {script_file}")
        print(f"✓ Created {rtf_file}")
        print(f"✓ Created {prm_file}")
        
        return script_file
    
    def _create_rtf_file(self, filename: str, parameters: Dict):
        """Create CHARMM topology file (RTF)"""
        with open(filename, 'w') as f:
            f.write("* CG Topology File\n")
            f.write("*\n\n")

            # Version
            f.write("   36     1\n\n")

            # Define atom types and masses
            f.write("! Atom type definitions\n")
            bead_masses = {}

            for bead_type in self.mapping_scheme.bead_types:
                # Get mass from first molecule
                mass = 15.0  # Default
                for mol in self.cg_molecules:
                    for bead in mol.beads:
                        if bead.name == bead_type:
                            mass = bead.mass
                            break
                    if mass != 15.0:
                        break

                bead_masses[bead_type] = mass
                f.write(f"MASS  -1  {bead_type:<4s}  {mass:7.3f}\n")

            f.write("\n")

            # Define residue topology
            f.write("RESI CG          0.00  ! CG molecule\n")
            f.write("GROUP\n")

            # Add atoms
            for bead_type in self.mapping_scheme.bead_types:
                f.write(f"ATOM {bead_type:<4s} {bead_type:<4s} 0.00\n")

            # Add bonds
            if self.mapping_scheme.connectivity:
                f.write("\n! Bonds\n")
                for conn in self.mapping_scheme.connectivity:
                    if len(conn) == 2:
                        f.write(f"BOND {conn[0]:<4s} {conn[1]:<4s}\n")

            # Generate angles and dihedrals
            angles, dihedrals = self._generate_angles_dihedrals()

            # Add angles
            if angles:
                f.write("\n! Angles\n")
                for angle in angles:
                    f.write(f"ANGLE {angle[0]:<4s} {angle[1]:<4s} {angle[2]:<4s}\n")

            # Add dihedrals
            if dihedrals:
                f.write("\n! Dihedrals\n")
                for dihedral in dihedrals:
                    f.write(f"DIHE {dihedral[0]:<4s} {dihedral[1]:<4s} {dihedral[2]:<4s} {dihedral[3]:<4s}\n")

            # Add patches (if any dummy beads)
            if self.mapping_scheme.dummy_beads:
                f.write("\n! Dummy bead patches\n")
                for dummy in self.mapping_scheme.dummy_beads:
                    f.write(f"! {dummy} is a dummy bead for electrostatics\n")

            f.write("\nEND\n")
    
    def _create_prm_file(self, filename: str, parameters: Dict, dict_resname: str = "XXX"):
        """Create CHARMM parameter file (PRM) template with placeholders"""
        with open(filename, 'w') as f:
            f.write("* CG Parameters Template\n")
            f.write("*\n\n")

            f.write("BONDS\n")
            f.write("!\n")
            f.write("!V(bond) = Kb(b - b0)**2\n")
            f.write("!\n")
            f.write("!Kb: kcal/mole/A**2\n")
            f.write("!b0: A\n")
            f.write("!\n")
            f.write("!atom type   Kb          b0\n")
            f.write("\n")

            # Write bond parameters with placeholders (only for non-dummy connections)
            for conn in self.mapping_scheme.connectivity:
                if len(conn) == 2:
                    bead1, bead2 = conn
                    # Skip dummy-to-dummy bonds (shouldn't exist per constraints)
                    if bead1 in self.mapping_scheme.dummy_beads and bead2 in self.mapping_scheme.dummy_beads:
                        continue
                    
                    kb_placeholder = f"${{{bead1}_{bead2}_kb}}"
                    bl_placeholder = f"${{{bead1}_{bead2}_bl}}"

                    f.write(f"{bead1:<6s} {bead2:<6s} {kb_placeholder} {bl_placeholder}\n")

            f.write("\n")

            # Generate angles and dihedrals
            angles, dihedrals = self._generate_angles_dihedrals()

            # Angles
            if angles:
                f.write("ANGLES\n")
                f.write("!\n")
                f.write("!V(angle) = Ktheta(Theta - Theta0)**2\n")
                f.write("!\n")
                f.write("!Ktheta: kcal/mole/rad**2\n")
                f.write("!Theta0: degrees\n")
                f.write("!\n")
                f.write("!atom types     Ktheta    Theta0\n")
                f.write("\n")

                for angle in angles:
                    ktheta_placeholder = f"${{{angle[0]}_{angle[1]}_{angle[2]}_ktheta}}"
                    theta0_placeholder = f"${{{angle[0]}_{angle[1]}_{angle[2]}_theta0}}"
                    f.write(f"{angle[0]:<6s} {angle[1]:<6s} {angle[2]:<6s} {ktheta_placeholder} {theta0_placeholder}\n")

                f.write("\n")

            # Dihedrals
            if dihedrals:
                f.write("DIHEDRALS\n")
                f.write("!\n")
                f.write("!V(dihedral) = Kchi(1 + cos(n(chi - delta)))\n")
                f.write("!\n")
                f.write("!Kchi: kcal/mole\n")
                f.write("!n: multiplicity (1, 2, 3, 4, 6)\n")
                f.write("!delta: degrees\n")
                f.write("!\n")
                f.write("!atom types             Kchi    n   delta\n")
                f.write("\n")

                for dihedral in dihedrals:
                    kchi_placeholder = f"${{{dihedral[0]}_{dihedral[1]}_{dihedral[2]}_{dihedral[3]}_kchi}}"
                    n_placeholder = f"${{{dihedral[0]}_{dihedral[1]}_{dihedral[2]}_{dihedral[3]}_n}}"
                    delta_placeholder = f"${{{dihedral[0]}_{dihedral[1]}_{dihedral[2]}_{dihedral[3]}_delta}}"
                    f.write(f"{dihedral[0]:<6s} {dihedral[1]:<6s} {dihedral[2]:<6s} {dihedral[3]:<6s} {kchi_placeholder} {n_placeholder} {delta_placeholder}\n")

                f.write("\n")

            # Non-bonded parameters (LJ)
            f.write("NONBONDED nbxmod  5 atom cdiel shift vatom vdistance vswitch -\n")
            f.write("cutnb 14.0 ctofnb 12.0 ctonnb 10.0 eps 1.0 e14fac 1.0 wmin 1.5\n")
            f.write("                !adm jr., 5/08/91, suggested cutoff scheme\n")
            f.write("\n")

            f.write("!\n")
            f.write("!epsilon: kcal/mole, Eps,i,j = sqrt(eps,i * eps,j)\n")
            f.write("!Rmin/2: A, Rmin,i,j = Rmin/2,i + Rmin/2,j\n")
            f.write("!\n")
            f.write("!atom  ignored    epsilon      Rmin/2   ignored   eps,1-4       Rmin/2,1-4\n")
            f.write("\n")

            # Write self-interaction parameters ONLY for core beads
            core_beads = [b for b in self.mapping_scheme.bead_types 
                        if b not in self.mapping_scheme.dummy_beads]

            for bead_type in core_beads:
                # Check if parameters exist in cdhm_dict.json
                prev_params = self._get_bead_params_from_prev(bead_type, "XXX") 
                
                if prev_params:
                    # Use hardcoded values from cdhm_dict.json
                    epsilon, rmin_half = prev_params
                    f.write(f"{bead_type:<6s}  0.0  {epsilon:.4f}  {rmin_half:.4f}\n")
                    print(f"[topology] Using fixed params for {bead_type}: epsilon={epsilon}, Rmin/2={rmin_half}")
                else:
                    # Use placeholders for optimization
                    epsilon_placeholder = f"${{{bead_type}_epsilon}}"
                    rmin_placeholder = f"${{{bead_type}_rminby2}}"
                    f.write(f"{bead_type:<6s}  0.0  {epsilon_placeholder}  {rmin_placeholder}\n")

            # Write dummy beads with zero VdW parameters (no placeholders)
            if self.mapping_scheme.dummy_beads:
                f.write("\n! Dummy beads (no VdW interactions)\n")
                for dummy_bead in self.mapping_scheme.dummy_beads:
                    f.write(f"{dummy_bead:<6s}  0.0  0.0  0.0\n")

            # NBFIX section for cross-interactions based on interaction_matrix
            # Only write cross-interactions that are explicitly in the matrix
            cross_interactions = []
            
            for bead1, interaction_list in self.mapping_scheme.interaction_matrix.items():
                # Skip dummy beads (they have empty lists)
                if bead1 in self.mapping_scheme.dummy_beads:
                    continue
                    
                for bead2 in interaction_list:
                    # Skip self-interactions (already handled above)
                    if bead1 == bead2:
                        continue
                    
                    # Skip dummy beads
                    if bead2 in self.mapping_scheme.dummy_beads:
                        continue
                    
                    # Ensure we only add each pair once (symmetric)
                    pair = tuple(sorted([bead1, bead2]))
                    if pair not in cross_interactions:
                        cross_interactions.append(pair)

            if cross_interactions:
                f.write("\n! Cross-interaction parameters\n")
                f.write("NBFIX\n")
                f.write("!               Emin        Rmin\n")
                f.write("!            (kcal/mol)     (A)\n")
                
                for bead1, bead2 in cross_interactions:
                    epsilon_placeholder = f"${{{bead1}_{bead2}_epsilon}}"
                    rmin_placeholder = f"${{{bead1}_{bead2}_rmin}}"
                    f.write(f"{bead1:<6s} {bead2:<6s} {epsilon_placeholder} {rmin_placeholder}\n")

            f.write("\nEND\n")

    def _generate_angles_dihedrals(self) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str, str]]]:
        """
        Generate angles and dihedrals from connectivity.
        Angles: for every triplet bead1-bead2-bead3 where bead1-bead2 and bead2-bead3 are connected.
                Dummy beads CAN participate in angles.
        Dihedrals: for every quadruplet bead1-bead2-bead3-bead4 where all consecutive pairs are connected.
                Dummy beads do NOT participate in dihedrals.
        """
        if not self.mapping_scheme or not self.mapping_scheme.connectivity:
            return [], []

        dummy_beads = set(self.mapping_scheme.dummy_beads or [])

        # Build adjacency list for angles (includes dummy beads)
        adj_angles = {}
        for bead in self.mapping_scheme.bead_types:
            adj_angles[bead] = set()
        for conn in self.mapping_scheme.connectivity:
            if len(conn) == 2:
                bead1, bead2 = conn
                adj_angles[bead1].add(bead2)
                adj_angles[bead2].add(bead1)

        # Build adjacency list for dihedrals (excludes dummy beads)
        adj_dihedrals = {}
        for bead in self.mapping_scheme.bead_types:
            if bead not in dummy_beads:
                adj_dihedrals[bead] = set()
        for conn in self.mapping_scheme.connectivity:
            if len(conn) == 2:
                bead1, bead2 = conn
                if bead1 not in dummy_beads and bead2 not in dummy_beads:
                    adj_dihedrals[bead1].add(bead2)
                    adj_dihedrals[bead2].add(bead1)

        angles = []
        dihedrals = []

        # Generate angles: find all paths of length 2 (includes dummy beads)
        for bead2 in adj_angles:
            neighbors = sorted(adj_angles[bead2])  # Sort neighbors for consistent ordering
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    bead1 = neighbors[i]
                    bead3 = neighbors[j]
                    # Order as bead1-bead2-bead3 with bead2 central
                    angle = (bead1, bead2, bead3)
                    if angle not in angles:
                        angles.append(angle)

        # Generate dihedrals: find all paths of length 3 (excludes dummy beads)
        for bead2 in adj_dihedrals:
            for bead3 in adj_dihedrals[bead2]:
                if bead3 <= bead2:  # Avoid duplicates
                    continue
                for bead1 in adj_dihedrals[bead2]:
                    if bead1 == bead3:
                        continue
                    for bead4 in adj_dihedrals[bead3]:
                        if bead4 == bead2 or bead4 == bead1:
                            continue
                        # Ensure consistent ordering
                        dihedral = (bead1, bead2, bead3, bead4)
                        if dihedral not in dihedrals:
                            dihedrals.append(dihedral)

        return angles, dihedrals

    # def _generate_angles_dihedrals(self) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str, str]]]:
    #     """
    #     Generate angles and dihedrals from connectivity.
    #     Angles: for every triplet bead1-bead2-bead3 where bead1-bead2 and bead2-bead3 are connected.
    #     Dihedrals: for every quadruplet bead1-bead2-bead3-bead4 where all consecutive pairs are connected.
    #     Dummy beads do not participate in angles or dihedrals.
    #     """
    #     if not self.mapping_scheme or not self.mapping_scheme.connectivity:
    #         return [], []

    #     dummy_beads = set(self.mapping_scheme.dummy_beads or [])

    #     # Build adjacency list, excluding connections to dummies for angle/dihedral generation
    #     adj = {}
    #     for bead in self.mapping_scheme.bead_types:
    #         if bead not in dummy_beads:
    #             adj[bead] = set()
    #     for conn in self.mapping_scheme.connectivity:
    #         if len(conn) == 2:
    #             bead1, bead2 = conn
    #             if bead1 not in dummy_beads and bead2 not in dummy_beads:
    #                 adj[bead1].add(bead2)
    #                 adj[bead2].add(bead1)

    #     angles = []
    #     dihedrals = []

    #     # Generate angles: find all paths of length 2 among core beads
    #     for bead2 in adj:
    #         neighbors = sorted(adj[bead2])  # Sort neighbors for consistent ordering
    #         for i in range(len(neighbors)):
    #             for j in range(i+1, len(neighbors)):
    #                 bead1 = neighbors[i]
    #                 bead3 = neighbors[j]
    #                 # Order as bead1-bead2-bead3 with bead2 central
    #                 angle = (bead1, bead2, bead3)
    #                 if angle not in angles:
    #                     angles.append(angle)

    #     # Generate dihedrals: find all paths of length 3 among core beads
    #     for bead2 in adj:
    #         for bead3 in adj[bead2]:
    #             if bead3 <= bead2:  # Avoid duplicates
    #                 continue
    #             for bead1 in adj[bead2]:
    #                 if bead1 == bead3:
    #                     continue
    #                 for bead4 in adj[bead3]:
    #                     if bead4 == bead2 or bead4 == bead1:
    #                         continue
    #                     # Ensure consistent ordering
    #                     dihedral = (bead1, bead2, bead3, bead4)
    #                     if dihedral not in dihedrals:
    #                         dihedrals.append(dihedral)

    #     return angles, dihedrals

    # ========================================================================
    # Step 3: Packmol System Generation
    # ========================================================================
    
    def create_packmol_input(self, n_molecules: int, box_size: float = 50.0,
                            density: float = 1.095) -> str:
        """
        Create Packmol input file to pack molecules at experimental density
        """
        print("\n[Step 3] Creating Packmol Input...")
        
        # Calculate number of molecules needed for target density
        if len(self.cg_molecules) > 0:
            # Get molecular weight of CG molecule
            mol_mass = sum(bead.mass for bead in self.cg_molecules[0].beads)
        else:
            raise ValueError("No CG molecules available for molecular weight calculation")
        
        # Calculate molecules needed
        # V = L^3 (Angstrom^3), convert to cm^3
        volume_cm3 = (box_size ** 3) * 1e-24
        
        # Mass = density * volume (g)
        mass_g = density * volume_cm3
        
        # Moles = mass / molecular_weight
        moles = mass_g / mol_mass
        
        # Molecules = moles * Avogadro
        n_molecules_calc = int(moles * 6.022e23)
        
        print(f"  Box size: {box_size}³ Å³")
        print(f"  Target density: {density} g/cm³")
        print(f"  Molecular weight: {mol_mass:.2f} g/mol")
        print(f"  Calculated molecules needed: {n_molecules_calc}")
        
        n_molecules = n_molecules_calc if n_molecules is None else n_molecules
        
        # Create single molecule PDB
        single_mol_pdb = os.path.join(self.output_dir, "single_molecule.pdb")
        with open(single_mol_pdb, 'w') as f:
            f.write("REMARK   Single CG Molecule\n")
            for i, bead in enumerate(self.cg_molecules[0].beads, 1):
                line = f"ATOM  {i:5d}  {bead.name:<4s} CG     1    " \
                       f"{bead.x:8.3f}{bead.y:8.3f}{bead.z:8.3f}" \
                       f"  1.00  0.00           {bead.name[0]:>2s}\n"
                f.write(line)
            f.write("END\n")
        
        # Create Packmol input
        packmol_input = os.path.join(self.output_dir, "pack.inp")
        packed_pdb = os.path.join(self.output_dir, "packed_system.pdb")
        
        with open(packmol_input, 'w') as f:
            f.write("# Packmol input for CG system\n\n")
            f.write(f"tolerance 2.0\n")
            f.write(f"filetype pdb\n")
            f.write(f"output {packed_pdb}\n\n")
            
            f.write(f"structure {single_mol_pdb}\n")
            f.write(f"  number {n_molecules}\n")
            f.write(f"  inside box 0. 0. 0. {box_size} {box_size} {box_size}\n")
            f.write(f"end structure\n")
        
        print(f"✓ Created {packmol_input}")
        print(f"  Will pack {n_molecules} molecules in {box_size}³ Å³ box")
        
        return packmol_input
    
    def run_packmol(self, packmol_input: str) -> str:
        """Execute Packmol"""
        print("\n  Executing Packmol...")
        
        try:
            result = subprocess.run(
                ['./executables/packmol'],
                stdin=open(packmol_input, 'r'),
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                packed_pdb = os.path.join(self.output_dir, "packed_system.pdb")
                print(f"✓ Packmol completed successfully")
                print(f"✓ Created {packed_pdb}")
                return packed_pdb
            else:
                print(f"✗ Packmol failed:")
                print(result.stderr)
                return None
                
        except subprocess.TimeoutExpired:
            print("✗ Packmol timeout")
            return None
        except FileNotFoundError:
            print("✗ Packmol not found. Please install packmol.")
            return None
    
    # ========================================================================
    # Step 4: Execute PSFgen
    # ========================================================================
    
    def run_psfgen(self, psfgen_script: str, packed_pdb: str) -> Tuple[str, str]:
        """
        Execute PSFgen to create final PSF and PDB files
        """
        print("\n[Step 4] Executing PSFgen...")
        
        # Copy packed system as namd_inp.pdb
        namd_inp = os.path.join(self.output_dir, "namd_inp.pdb")
        
        if packed_pdb and os.path.exists(packed_pdb):
            import shutil
            shutil.copy(packed_pdb, namd_inp)
            print(f"  Using packed system: {packed_pdb}")
        else:
            print(f"  Using existing namd_inp.pdb")
        
        # Execute PSFgen
        try:
            result = subprocess.run(
                ['../executables/psfgen', os.path.basename(psfgen_script)],
                cwd=self.output_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            psf_file = os.path.join(self.output_dir, "NAMD_input.psf")
            pdb_file = os.path.join(self.output_dir, "NAMD_input.pdb")
            
            if result.returncode == 0 and os.path.exists(psf_file):
                print(f"✓ PSFgen completed successfully")
                print(f"✓ Created {psf_file}")
                print(f"✓ Created {pdb_file}")
                return psf_file, pdb_file
            else:
                print(f"✗ PSFgen failed:")
                print(result.stdout)
                print(result.stderr)
                return None, None
                
        except subprocess.TimeoutExpired:
            print("✗ PSFgen timeout")
            return None, None
        except FileNotFoundError:
            print("✗ PSFgen not found.")
            return None, None
    
    # ========================================================================
    # Complete Workflow
    # ========================================================================
    
    def create_namd_inputs(self, aa_pdb: str, parameters: Dict,
                          molecule_name: str = "DMA",
                          n_molecules: int = None,
                          box_size: float = 50.0,
                          density: float = 1.095) -> Dict:
        """
        Complete workflow to create NAMD input files
        
        Returns dict with paths to created files
        """
        print("\n" + "="*80)
        print("TOPOLOGY CREATOR AGENT - NAMD INPUT GENERATION")
        print("="*80)
        
        results = {}
        
        # Step 1a: Coarse-grain system
        cg_pdb, mapping_rules = self.coarse_grain_system(aa_pdb, molecule_name)
        results['cg_pdb'] = cg_pdb

        # Step 1b: Convert AA Trajectory to CG
        aa_psf_path = os.path.join("../..", "AA_Agent", "Run", "NAMD_input.psf")
        aa_dcd_path = os.path.join("../..", "AA_Agent", "Run", "output.dcd")
        cg_trajectory_pdb = os.path.join("AA2CG", "cg_trajectory.pdb")
        cg_trajectory = self.convert_aa_to_cg_trajectory(aa_psf_path, aa_dcd_path, cg_trajectory_pdb, mapping_rules)
        results['cg_trajectory'] = cg_trajectory

        # Step 2: Create PSFgen script
        psfgen_script = self.create_psfgen_script(parameters)
        results['psfgen_script'] = psfgen_script
        
        # Step 3: Create Packmol input and run
        packmol_input = self.create_packmol_input(n_molecules, box_size, density)
        results['packmol_input'] = packmol_input
        
        packed_pdb = self.run_packmol(packmol_input)
        results['packed_pdb'] = packed_pdb
        
        # Step 4: Run PSFgen
        psf_file, pdb_file = self.run_psfgen(psfgen_script, packed_pdb)
        results['psf_file'] = psf_file
        results['pdb_file'] = pdb_file
        
        print("\n" + "="*80)
        print("NAMD INPUT GENERATION COMPLETE")
        print("="*80)
        print(f"\nFinal files:")
        print(f"  CG Trajectory: {cg_trajectory}")
        print(f"  PSF: {psf_file}")
        print(f"  PDB: {pdb_file}")

        return results
        
    