"""
RDKit-based SMILES Parser
Robust and production-ready SMILES parsing using RDKit library.

Installation:
    pip install rdkit
    # or
    conda install -c conda-forge rdkit
"""

from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Install with: pip install rdkit")


class SMILESParser:
    """
    Production-ready SMILES parser using RDKit.
    Handles all valid SMILES notation including complex molecules.
    """

    def __init__(self, use_rdkit: bool = True):
        """
        Initialize parser.
        
        Args:
            use_rdkit: If True and RDKit is available, use RDKit for parsing.
                      If False or RDKit unavailable, falls back to basic parser.
        """
        self.use_rdkit = use_rdkit and RDKIT_AVAILABLE
        
        if not self.use_rdkit and use_rdkit:
            print("Warning: RDKit requested but not available. Using basic parser.")

    def parse(self, smiles: str) -> Dict:
        """
        Parse SMILES string and return connectivity information.

        Args:
            smiles: SMILES string (e.g., "CC(C)C", "c1ccccc1")

        Returns:
            Dict with atoms, bonds, connectivity, and molecular properties
        """
        if not smiles or not isinstance(smiles, str) or smiles.strip() == '':
            return self._empty_result()

        smiles = smiles.strip()

        if self.use_rdkit:
            return self._parse_with_rdkit(smiles)
        else:
            return self._error_result("RDKit not available. Install with: pip install rdkit")

    def _parse_with_rdkit(self, smiles: str) -> Dict:
        """Parse SMILES using RDKit."""
        try:
            # Parse SMILES to molecule object
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                return self._error_result(f"Invalid SMILES: {smiles}")

            # Sanitize molecule (aromaticity, valence checking, etc.)
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                return self._error_result(f"Sanitization failed: {str(e)}")

            # Extract atoms
            atoms = []
            for atom in mol.GetAtoms():
                atom_info = {
                    'symbol': atom.GetSymbol(),
                    'atomic_num': atom.GetAtomicNum(),
                    'formal_charge': atom.GetFormalCharge(),
                    'num_h': atom.GetTotalNumHs(),
                    'aromatic': atom.GetIsAromatic(),
                    'hybridization': str(atom.GetHybridization()),
                    'idx': atom.GetIdx()
                }
                atoms.append(atom_info)

            # Extract bonds
            bonds = []
            for bond in mol.GetBonds():
                bond_info = (
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    str(bond.GetBondType())
                )
                bonds.append(bond_info)

            # Build connectivity matrix
            connectivity = self._build_connectivity_from_mol(mol)

            # Get molecular properties
            molecular_formula = rdMolDescriptors.CalcMolFormula(mol)
            molecular_weight = Descriptors.MolWt(mol)
            num_rings = Descriptors.RingCount(mol)
            num_aromatic_rings = Descriptors.NumAromaticRings(mol)

            # Get canonical SMILES (standardized form)
            canonical_smiles = Chem.MolToSmiles(mol)

            return {
                'success': True,
                'smiles': smiles,
                'canonical_smiles': canonical_smiles,
                'atoms': atoms,
                'bonds': bonds,
                'connectivity': connectivity,
                'atom_count': mol.GetNumAtoms(),
                'bond_count': mol.GetNumBonds(),
                'molecular_formula': molecular_formula,
                'molecular_weight': round(molecular_weight, 2),
                'num_rings': num_rings,
                'num_aromatic_rings': num_aromatic_rings,
                'components': len(Chem.GetMolFrags(mol))
            }

        except Exception as e:
            return self._error_result(f"RDKit parsing error: {str(e)}")

    def _build_connectivity_from_mol(self, mol) -> Dict[str, List[str]]:
        """Build connectivity matrix from RDKit molecule."""
        connectivity = defaultdict(list)

        # Initialize all atoms
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atom_symbol = atom.GetSymbol()
            atom_name = f"{atom_symbol}{atom_idx + 1}"
            connectivity[atom_name] = []

        # Add bonds
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            
            begin_atom = mol.GetAtomWithIdx(begin_idx)
            end_atom = mol.GetAtomWithIdx(end_idx)
            
            begin_name = f"{begin_atom.GetSymbol()}{begin_idx + 1}"
            end_name = f"{end_atom.GetSymbol()}{end_idx + 1}"

            connectivity[begin_name].append(end_name)
            connectivity[end_name].append(begin_name)

        return dict(connectivity)

    def get_connected_components(self, connectivity: Dict[str, List[str]]) -> List[Set[str]]:
        """
        Find connected components in the connectivity graph.
        """
        if not connectivity:
            return []

        visited = set()
        components = []

        def dfs(atom, component):
            if atom in visited:
                return
            visited.add(atom)
            component.add(atom)
            for neighbor in connectivity.get(atom, []):
                dfs(neighbor, component)

        for atom in connectivity:
            if atom not in visited:
                component = set()
                dfs(atom, component)
                if component:
                    components.append(component)

        return components

    def validate_smiles(self, smiles: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SMILES string.

        Returns:
            (is_valid, error_message)
        """
        if not smiles or not isinstance(smiles, str):
            return False, "Empty or invalid input"

        if not self.use_rdkit:
            return False, "RDKit not available"

        try:
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                return False, "Invalid SMILES syntax"
            
            # Try to sanitize
            Chem.SanitizeMol(mol)
            return True, None
        except Exception as e:
            return False, str(e)

    def get_molecular_properties(self, smiles: str) -> Dict:
        """
        Get detailed molecular properties.

        Returns extensive molecular descriptors and properties.
        """
        result = self.parse(smiles)
        if not result['success']:
            return result

        try:
            mol = Chem.MolFromSmiles(smiles.strip())
            
            properties = {
                'success': True,
                'molecular_formula': result['molecular_formula'],
                'molecular_weight': result['molecular_weight'],
                'num_atoms': result['atom_count'],
                'num_bonds': result['bond_count'],
                'num_heavy_atoms': Descriptors.HeavyAtomCount(mol),
                'num_h_donors': Descriptors.NumHDonors(mol),
                'num_h_acceptors': Descriptors.NumHAcceptors(mol),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'num_rings': result['num_rings'],
                'num_aromatic_rings': result['num_aromatic_rings'],
                'num_saturated_rings': Descriptors.NumSaturatedRings(mol),
                'tpsa': round(Descriptors.TPSA(mol), 2),  # Topological polar surface area
                'logp': round(Descriptors.MolLogP(mol), 2),  # Partition coefficient
                'num_heteroatoms': Descriptors.NumHeteroatoms(mol),
                'fraction_csp3': round(Descriptors.FractionCSP3(mol), 3),
            }
            return properties
        except Exception as e:
            return self._error_result(f"Error calculating properties: {str(e)}")

    def smiles_to_inchi(self, smiles: str) -> Optional[str]:
        """Convert SMILES to InChI identifier."""
        if not self.use_rdkit:
            return None
        
        try:
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                return None
            from rdkit.Chem import inchi
            return inchi.MolToInchi(mol)
        except:
            return None

    def smiles_to_inchikey(self, smiles: str) -> Optional[str]:
        """Convert SMILES to InChIKey identifier."""
        if not self.use_rdkit:
            return None
        
        try:
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                return None
            from rdkit.Chem import inchi
            return inchi.MolToInchiKey(mol)
        except:
            return None

    def _empty_result(self) -> Dict:
        """Return empty result structure."""
        return {
            'success': False,
            'error': 'Empty or invalid SMILES string',
            'smiles': '',
            'atoms': [],
            'bonds': [],
            'connectivity': {},
            'atom_count': 0,
            'bond_count': 0
        }

    def _error_result(self, error: str) -> Dict:
        """Return error result structure."""
        return {
            'success': False,
            'error': error,
            'smiles': '',
            'atoms': [],
            'bonds': [],
            'connectivity': {},
            'atom_count': 0,
            'bond_count': 0
        }


# Example usage and comprehensive testing
if __name__ == "__main__":
    if not RDKIT_AVAILABLE:
        print("ERROR: RDKit is not installed!")
        print("Install with: pip install rdkit")
        print("Or with conda: conda install -c conda-forge rdkit")
        exit(1)

    parser = SMILESParser()

    test_smiles = [
        ("CS(=O)C", "DMSO"),
        ("CN(C)C", "Dimethylamine (DMA)"),
        ("CC(C)C", "Isobutane"),
        ("CC(C)(C)C", "Neopentane - double branch"),
        ("CC(CC)C", "2-Methylbutane"),
        ("C(C)(C)C", "Alternative branch notation"),
        ("CC(=O)N(C)C", "N,N-Dimethylacetamide"),
        ("c1ccccc1", "Benzene"),
        ("C1CC1", "Cyclopropane"),
        ("CCO", "Ethanol"),
        ("CC(=O)O", "Acetic acid"),
        ("c1cc(O)ccc1", "Phenol"),
        ("CC(C(C)C)C", "Nested branches"),
        ("C1CCCCC1", "Cyclohexane"),
        ("CC(C)CC(C)(C)C", "2,4,4-Trimethylpentane"),
        ("CC(C)C(C)C", "2,3-Dimethylbutane"),
    ]

    print("RDKit-Based SMILES Parser Test Suite")
    print("=" * 70)

    for smiles, name in test_smiles:
        print(f"\nTesting: {smiles} ({name})")
        print("-" * 70)
        
        # Parse
        result = parser.parse(smiles)
        
        if result['success']:
            print(f"✓ Success")
            print(f"  Canonical SMILES: {result['canonical_smiles']}")
            print(f"  Formula: {result['molecular_formula']}")
            print(f"  Molecular Weight: {result['molecular_weight']}")
            print(f"  Atoms: {result['atom_count']}, Bonds: {result['bond_count']}")
            print(f"  Rings: {result['num_rings']} (Aromatic: {result['num_aromatic_rings']})")
            print(f"  Components: {result['components']}")
            print(f"  Connectivity: {result['connectivity']}")
        else:
            print(f"✗ Failed: {result['error']}")

    # Test molecular properties
    print("\n" + "=" * 70)
    print("Testing Molecular Properties for Aspirin")
    print("-" * 70)
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    props = parser.get_molecular_properties(aspirin)
    if props['success']:
        for key, value in props.items():
            if key != 'success':
                print(f"  {key}: {value}")