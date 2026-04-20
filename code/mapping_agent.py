"""
Mapping Agent for CG Force Field Optimization

This agent decides CG mapping schemes and bead types.
"""

import os
import json
from typing import Dict, List, Any, Optional
from common import LLMAgent, AgentRole, MappingScheme
from smiles_parser import SMILESParser


class MappingAgent(LLMAgent):
    """Agent for deciding CG mapping scheme"""

    def __init__(self, api_key: str, url: str, prompts_dir: str = "prompts"):
        super().__init__(AgentRole.MAPPING, api_key, url)
        self.prompts_dir = prompts_dir
        self._load_prompts()

    def _load_prompts(self):
        """Load prompts from files"""
        with open(os.path.join(self.prompts_dir, "mapping_system_prompt.txt"), 'r') as f:
            self.system_prompt = f.read().strip()
        with open(os.path.join(self.prompts_dir, "mapping_prompt_template.txt"), 'r') as f:
            self.prompt_template = f.read().strip()

    def propose_mapping(self, molecule_info: Dict) -> Optional[MappingScheme]:
        """Propose CG mapping scheme for a molecule"""

        # Extract connectivity from SMILES
        smiles_parser = SMILESParser()
        connectivity_data = smiles_parser.parse(molecule_info.get('smiles', ''))

        # Get connected components for validation
        connected_components = []
        if connectivity_data['success']:
            connected_components = smiles_parser.get_connected_components(connectivity_data['connectivity'])

        # Load previous mapping schemes to avoid
        previous_mappings = ""
        prev_file = "prev_mapping_scheme.json"
        if os.path.exists(prev_file):
            try:
                with open(prev_file, 'r') as f:
                    prev_data = json.load(f)
                # Handle both single scheme (dict) and multiple schemes (list)
                if isinstance(prev_data, dict):
                    prev_data = [prev_data]
                previous_mappings = ", ".join([json.dumps(scheme, indent=2) for scheme in prev_data])
            except Exception as e:
                print(f"Warning: Could not load previous mappings: {e}")

        previous_errors = ""
        attempt = 0
        max_attempts = 3

        while attempt < max_attempts:
            prompt = self.prompt_template.format(
                molecule_name=molecule_info.get('name', 'Unknown'),
                smiles=molecule_info.get('smiles', 'N/A'),
                structure=molecule_info.get('structure', 'N/A'),
                polarity=molecule_info.get('polarity', 'N/A'),
                dipole_moment=molecule_info.get('dipole_moment', 'N/A'),
                molecular_weight=molecule_info.get('molecular_weight', 'N/A'),
                targets=json.dumps(molecule_info.get('targets', {}), indent=2),
                connectivity_matrix=json.dumps(connectivity_data.get('connectivity', {}), indent=2),
                connected_components=json.dumps([list(comp) for comp in connected_components], indent=2),
                atom_count=connectivity_data.get('atom_count', 0),
                previous_errors=previous_errors,
                previous_mappings=previous_mappings
            )

            result = self.call(prompt, self.system_prompt, temperature=0.7)

            if result and "mapping" in result:
                mapping_data = result["mapping"]

                # Create mapping scheme
                mapping_scheme = MappingScheme(
                    bead_types=mapping_data.get("bead_types", []),
                    bead_descriptions=mapping_data.get("bead_descriptions", {}),
                    connectivity=mapping_data.get("connectivity", []),
                    dummy_beads=mapping_data.get("dummy_beads", []),
                    interaction_matrix=mapping_data.get("interaction_matrix", {})
                )

                # Basic validation: must have at least one core bead
                core_beads = [b for b in mapping_scheme.bead_types if b not in (mapping_scheme.dummy_beads or [])]
                if not core_beads:
                    error_msg = "No core beads defined in mapping"
                    print(f"ERROR: {error_msg}")
                    if previous_errors:
                        previous_errors += f"\n{error_msg}"
                    else:
                        previous_errors = error_msg
                    attempt += 1
                    continue

                # Validate connectivity constraints
                is_valid, errors = self.validate_mapping_connectivity(mapping_scheme, connectivity_data)
                if is_valid:
                    return mapping_scheme
                else:
                    error_text = "\n".join(errors)
                    print("ERROR: Mapping violates connectivity constraints - rejecting invalid mapping")
                    print("Each core bead must contain at least 2 heavy atoms")
                    if previous_errors:
                        previous_errors += f"\n{error_text}"
                    else:
                        previous_errors = error_text
                    attempt += 1
                    print(f"Validation failed on attempt {attempt}, retrying with feedback...")
            else:
                attempt += 1
                error_msg = "LLM failed to return valid mapping JSON"
                print(f"ERROR: {error_msg}")
                if previous_errors:
                    previous_errors += f"\n{error_msg}"
                else:
                    previous_errors = error_msg

        return None

    def validate_mapping_connectivity(self, mapping_scheme: MappingScheme, connectivity_data: Dict) -> tuple[bool, list[str]]:
        """
        Validate that the mapping respects connectivity constraints.
        Ensures that atoms grouped in the same bead are actually connected.
        Returns (is_valid, list_of_errors)
        """
        errors = []
        if not connectivity_data.get('success', False):
            errors.append("WARNING: No connectivity data available for validation")
            return True, errors  # Can't validate, assume OK

        connectivity = connectivity_data.get('connectivity', {})
        connected_components = SMILESParser().get_connected_components(connectivity)

        # Check each bead
        for bead_type, description in mapping_scheme.bead_descriptions.items():
            # Skip dummy beads (DUP, DUN)
            if bead_type in (mapping_scheme.dummy_beads or []):
                continue

            bead_atoms = self.extract_atoms_from_description(description)

            if not bead_atoms:
                continue

            # Filter atoms to only those in connectivity matrix (ignore hydrogens/implicit atoms)
            connectivity_atoms = set(connectivity.keys())
            valid_bead_atoms = [atom for atom in bead_atoms if atom in connectivity_atoms]

            if valid_bead_atoms:
                # Check chemical connectivity (allows grouping of atoms sharing bonding partners)
                is_connected, conn_errors = self.atoms_are_chemically_connected(valid_bead_atoms, connectivity)
                if not is_connected:
                    errors.append(f"Bead {bead_type} contains chemically disconnected atoms: {valid_bead_atoms}")
                    errors.extend(conn_errors)

            # STRICT CONSTRAINT: No single heavy atoms as CG beads (defeats coarse-graining purpose)
            heavy_atoms = [atom for atom in bead_atoms if self._is_heavy_atom(atom)]

            # EXCEPTION: Allow single O atom for water (standard CG practice)
            is_water_exception = (len(heavy_atoms) == 1 and
                                heavy_atoms[0].startswith('O') and
                                len(connectivity) == 1)  # Only one atom total

            if len(heavy_atoms) < 2 and not is_water_exception:
                errors.append(f"Bead {bead_type} contains only {len(heavy_atoms)} heavy atoms: {heavy_atoms}. Each core bead must contain at least 2 heavy atoms.")

        is_valid = len(errors) == 0
        return is_valid, errors

    def atoms_are_chemically_connected(self, atom_list: List[str], connectivity: Dict[str, List[str]]) -> tuple[bool, list[str]]:
        """
        STRICT CONNECTIVITY RULE: Atoms can only be grouped if they are either:
        1. Directly connected (bonded to each other), OR
        2. Connected through another common heavy atom (1st order neighbors)

        Every pair of atoms in the group must satisfy this rule.
        Returns (is_connected, list_of_errors)
        """
        errors = []
        if not atom_list:
            return True, errors

        # Remove duplicates
        atom_list = list(set(atom_list))

        if len(atom_list) == 1:
            return True, errors  # Single atom is trivially connected

        # Check every pair of atoms in the group
        for i in range(len(atom_list)):
            for j in range(i + 1, len(atom_list)):
                atom1 = atom_list[i]
                atom2 = atom_list[j]

                # Check if atoms are directly connected
                if atom2 in connectivity.get(atom1, []):
                    continue  # Directly connected, OK

                # Check if they share a common heavy atom neighbor (1st order)
                atom1_neighbors = set(connectivity.get(atom1, []))
                atom2_neighbors = set(connectivity.get(atom2, []))

                # Find common neighbors
                common_neighbors = atom1_neighbors & atom2_neighbors

                # Check if any common neighbor is a heavy atom
                has_common_heavy_neighbor = any(self._is_heavy_atom(neighbor) for neighbor in common_neighbors)

                if not has_common_heavy_neighbor:
                    errors.append(f"Atoms {atom1} and {atom2} are not properly connected. Direct bond: {atom2 in connectivity.get(atom1, [])}. Common heavy neighbors: {[n for n in common_neighbors if self._is_heavy_atom(n)]}")

        is_connected = len(errors) == 0
        return is_connected, errors

    def _share_common_bonding_partner(self, atom_list: List[str], connectivity: Dict[str, List[str]]) -> bool:
        """
        Check if all atoms in the list share a common heavy atom bonding partner.
        This ensures chemical coherence in CG groupings.
        """
        if len(atom_list) <= 2:
            return True  # Small groups are OK

        # Find all heavy atom neighbors for each atom
        heavy_neighbors = {}
        for atom in atom_list:
            neighbors = connectivity.get(atom, [])
            heavy_neighbors[atom] = [n for n in neighbors if self._is_heavy_atom(n)]

        # Check if there's a common heavy atom that connects multiple atoms in the list
        all_heavy_neighbors = set()
        for neighbors in heavy_neighbors.values():
            all_heavy_neighbors.update(neighbors)

        # For each potential common neighbor, check how many atoms in our list it connects
        for common_neighbor in all_heavy_neighbors:
            connected_count = 0
            for atom in atom_list:
                if common_neighbor in heavy_neighbors.get(atom, []):
                    connected_count += 1

            # If this common neighbor connects more than one atom in our list,
            # then those atoms are chemically related through it
            if connected_count >= 2:
                return True

        return False

    def _is_heavy_atom(self, atom_symbol: str) -> bool:
        """
        Check if an atom is a heavy atom (atomic mass > hydrogen).
        Heavy atoms include: C, N, O, S, P, F, Cl, Br, I, B, Si, etc.
        """
        # Remove the number suffix (e.g., "C1" -> "C")
        atom_element = ''.join([c for c in atom_symbol if not c.isdigit()])

        heavy_elements = {'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'Si', 'Se', 'Te', 'As', 'Al', 'Mg', 'Na', 'K'}
        return atom_element in heavy_elements

    def extract_atoms_from_description(self, description: str) -> List[str]:
        """
        Extract atom identifiers from bead description.
        Looks for patterns like "C1, S2, O3" or "atoms C1-S2-C3"
        """
        atoms = []

        # Look for atom patterns (e.g., C1, S2, O3) but avoid duplicates
        import re
        atom_pattern = r'([A-Z][a-z]?\d+)'
        matches = re.findall(atom_pattern, description)

        # Remove duplicates while preserving order
        seen = set()
        for match in matches:
            if match not in seen:
                atoms.append(match)
                seen.add(match)

        return atoms

    def atoms_are_connected(self, atom_list: List[str], connected_components: List[set]) -> bool:
        """
        Check if all atoms in the list belong to the same connected component.
        """
        if not atom_list:
            return True

        # Remove duplicates from atom list
        atom_list = list(set(atom_list))

        # Find which component each atom belongs to
        atom_components = {}
        for atom in atom_list:
            found = False
            for i, component in enumerate(connected_components):
                if atom in component:
                    atom_components[atom] = i
                    found = True
                    break
            if not found:
                print(f"WARNING: Atom {atom} not found in any connected component")
                return False

        # Check if all atoms are in the same component
        component_ids = list(atom_components.values())
        return len(set(component_ids)) == 1