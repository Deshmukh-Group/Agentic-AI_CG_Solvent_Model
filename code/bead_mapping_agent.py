"""
Bead Mapping Agent for CG Force Field Optimization

This agent handles bead name matching between mapping schemes and chemistry databases:
1. Analyzes bead descriptions from mapping_scheme.json
2. Matches them to beads in cdhm_dict.json using LLM
3. Replaces bead names in mapping_scheme with matched names
"""

import os
import json
import copy
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from common import LLMAgent, AgentRole


# ============================================================================
# Bead Mapping Agent
# ============================================================================

class BeadMappingAgent(LLMAgent):
    """Agent for matching and replacing bead names in mapping schemes"""

    def __init__(self, api_key: str, url: str, prompts_dir: str = "prompts"):
        super().__init__(AgentRole.BOUNDARY, api_key, url)
        self.prompts_dir = prompts_dir
        self.cdhm_dict = None
        self.mapping_scheme = None
        self._load_prompts()

    def _load_prompts(self):
        """Load prompts from files"""
        system_prompt_file = os.path.join(self.prompts_dir, "bead_matching_system_prompt.txt")
        prompt_template_file = os.path.join(self.prompts_dir, "bead_matching_prompt_template.txt")
        
        # Check if custom prompts exist, otherwise use defaults
        if os.path.exists(system_prompt_file):
            with open(system_prompt_file, 'r') as f:
                self.system_prompt = f.read().strip()
        else:
            self.system_prompt = self._get_default_system_prompt()
        
        if os.path.exists(prompt_template_file):
            with open(prompt_template_file, 'r') as f:
                self.prompt_template = f.read().strip()
        else:
            self.prompt_template = self._get_default_prompt_template()

    def _get_default_system_prompt(self) -> str:
        """Default system prompt for bead matching"""
        return """You are a chemistry expert specializing in molecular structure analysis.

Your task is to match bead descriptions from a coarse-grained mapping scheme to actual bead types 
in a chemistry database by analyzing their chemical composition and functional groups.

Focus on:
1. Atom types and counts (C, H, S, O, N, etc.)
2. Functional groups (methyl, sulfoxide, sulfone, amine, etc.)
3. Chemical structure (number of methyls, presence of sulfur-oxygen bonds, aromatic rings, etc.)

Important rules:
- The replacement name MUST be a BEAD NAME from the chemistry database
- DO NOT replace beads with residue names
- Residue names are context only

Return your answer as a JSON object with exact format:
{
  "matches": {
    "BEAD_NAME": {
      "matched_bead": "BEAD_NAME",
      "confidence": "high|medium|low",
      "reasoning": "explanation of the match"
    }
  },
  "name_mapping": {
    "OLD_BEAD": "BEAD_NAME"
    },
    "matches": {
    "OLD_BEAD": {
        "matched_bead": "BEAD_NAME",
        "confidence": "high",
        "reasoning": "..."
    }
    }
}

Important rules:
- Only match core beads (non-dummy beads)
- Match based on chemical composition, not just names
- If no good match exists, keep the original name
- Dummy beads (like DUP, DUN) should not be matched - exclude them from name_mapping
"""

    def _get_default_prompt_template(self) -> str:
        """Default prompt template for bead matching"""
        return """Match the bead descriptions from the mapping scheme to beads in the chemistry database.

## Mapping Scheme Bead Descriptions:
{bead_descriptions}

## Available Beads in Chemistry Database (cdhm_dict):
{cdhm_beads}

## Current Bead Types:
{bead_types}

## Dummy Beads (DO NOT MATCH):
{dummy_beads}

## Task:
For each NON-DUMMY bead in the mapping scheme, find the best matching bead from the chemistry database.

Key matching criteria:
- Match based on CHEMICAL COMPOSITION and FUNCTIONAL GROUPS
- Consider atom counts, functional groups, and molecular structure
- Ignore dummy beads - they don't represent real chemical groups

Return the name_mapping showing which current bead names should be replaced with which bead names.
Only include non-dummy beads in the name_mapping.
"""

    # ========================================================================
    # Step 1: Load Chemistry Database
    # ========================================================================

    def load_cdhm_dict(self, cdhm_dict_file: str = "cdhm_dict.json") -> Dict:
        """Load chemistry database (cdhm_dict) from file"""
        try:
            with open(cdhm_dict_file, 'r') as f:
                self.cdhm_dict = json.load(f)
            print(f"✓ Loaded {len(self.cdhm_dict)} bead types from {cdhm_dict_file}")
            return self.cdhm_dict
        except FileNotFoundError:
            print(f"✗ File not found: {cdhm_dict_file}")
            return {}
        except json.JSONDecodeError as e:
            print(f"✗ JSON decode error in {cdhm_dict_file}: {e}")
            return {}

    # ========================================================================
    # Step 2: Load Mapping Scheme
    # ========================================================================

    def load_mapping_scheme(self, mapping_scheme_file: str = "mapping_scheme.json") -> Dict:
        """Load mapping scheme from file"""
        try:
            with open(mapping_scheme_file, 'r') as f:
                self.mapping_scheme = json.load(f)
            print(f"✓ Loaded mapping scheme from {mapping_scheme_file}")
            print(f"  Bead types: {self.mapping_scheme.get('bead_types', [])}")
            print(f"  Dummy beads: {self.mapping_scheme.get('dummy_beads', [])}")
            return self.mapping_scheme
        except FileNotFoundError:
            print(f"✗ File not found: {mapping_scheme_file}")
            return {}
        except json.JSONDecodeError as e:
            print(f"✗ JSON decode error in {mapping_scheme_file}: {e}")
            return {}

    # ========================================================================
    # Step 3: Format Chemistry Database for LLM
    # ========================================================================

    def _format_cdhm_beads(self) -> str:
        """Format cdhm_dict beads for LLM prompt"""
        if not self.cdhm_dict:
            return "No beads available"

        formatted = []
        for bead_name, beads in self.cdhm_dict.items():
            formatted.append(f"### {bead_name}:")
            for bead_name, bead_data in beads.items():
                if isinstance(bead_data, list) and len(bead_data) >= 1:
                    atoms = bead_data[0]
                    epsilon = bead_data[1] if len(bead_data) > 1 else "N/A"
                    rmin = bead_data[2] if len(bead_data) > 2 else "N/A"
                    formatted.append(f"  - {bead_name}: atoms {atoms}")

        return "\n".join(formatted)

    # ========================================================================
    # Step 4: Match Beads Using LLM
    # ========================================================================

    def match_beads(self) -> Optional[Dict]:
        """
        Use LLM to match bead descriptions to beads in cdhm_dict
        
        Returns:
            Dictionary with 'matches' and 'name_mapping' keys
        """
        print("\n[Step 1] Matching Beads Based on Chemical Descriptions...")

        if not self.mapping_scheme or not self.cdhm_dict:
            print("✗ Missing mapping_scheme or cdhm_dict. Load them first.")
            return None

        # Extract info from mapping scheme
        bead_descriptions = self.mapping_scheme.get("bead_descriptions", {})
        bead_types = self.mapping_scheme.get("bead_types", [])
        dummy_beads = self.mapping_scheme.get("dummy_beads", [])

        # Format cdhm beads
        cdhm_beads = self._format_cdhm_beads()

        # Build prompt
        prompt = self.prompt_template.format(
            bead_descriptions=json.dumps(bead_descriptions, indent=2),
            cdhm_beads=cdhm_beads,
            bead_types=json.dumps(bead_types),
            dummy_beads=json.dumps(dummy_beads) if dummy_beads else "None"
        )

        # Call LLM
        result = self.call(prompt, self.system_prompt, temperature=0.3)

        if result and "name_mapping" in result:
            matches = result.get("matches", {})
            name_mapping = result["name_mapping"]

            print("\n[Matching Results]")
            for old_name, new_name in name_mapping.items():
                if old_name in matches:
                    match_info = matches[old_name]
                    confidence = match_info.get('confidence', 'unknown')
                    reasoning = match_info.get('reasoning', 'N/A')
                    print(f"  {old_name} → {new_name}")
                    print(f"    Confidence: {confidence}")
                    print(f"    Reasoning: {reasoning}")

            return result
        else:
            print("✗ LLM call failed or returned invalid format")
            return None

    # ========================================================================
    # Step 5: Apply Name Mapping to Mapping Scheme
    # ========================================================================

    def apply_name_mapping(self, name_mapping: Dict, matches: Dict, confidence_threshold: str = "high") -> Dict:
        """
        Apply name mapping to all occurrences in mapping_scheme
        
        Args:
            name_mapping: Dict of {old_name: new_name}
            matches: Dict with confidence information for each mapping
            confidence_threshold: Minimum confidence level to apply (high/medium/low)
            
        Returns:
            Updated mapping scheme with replaced names
        """
        print("\n[Step 2] Applying Name Mapping to Mapping Scheme...")
        
        # Filter name_mapping based on confidence threshold
        filtered_mapping = {}
        for old_name, new_name in name_mapping.items():
            if old_name in matches:
                confidence = matches[old_name].get('confidence', 'low')
                if confidence == confidence_threshold:
                    filtered_mapping[old_name] = new_name
                else:
                    print(f"  ⚠ Skipping {old_name} → {new_name} (confidence: {confidence})")
            else:
                print(f"  ⚠ Skipping {old_name} → {new_name} (no match info)")
        
        print(f"  Applied {len(filtered_mapping)} of {len(name_mapping)} mappings (threshold: {confidence_threshold})")
        name_mapping = filtered_mapping

        if not self.mapping_scheme:
            print("✗ No mapping scheme loaded")
            return {}

        updated = copy.deepcopy(self.mapping_scheme)

        # 1. Update bead_types list
        if "bead_types" in updated:
            old_types = updated["bead_types"]
            updated["bead_types"] = [
                name_mapping.get(bead, bead) 
                for bead in updated["bead_types"]
            ]
            print(f"  Bead types: {old_types} → {updated['bead_types']}")

        # 2. Update bead_descriptions keys
        if "bead_descriptions" in updated:
            new_descriptions = {}
            for old_key, description in updated["bead_descriptions"].items():
                new_key = name_mapping.get(old_key, old_key)
                new_descriptions[new_key] = description
                if new_key != old_key:
                    print(f"  Description key: {old_key} → {new_key}")
            updated["bead_descriptions"] = new_descriptions

        # 3. Update connectivity
        if "connectivity" in updated:
            updated["connectivity"] = [
                [name_mapping.get(bead, bead) for bead in connection]
                for connection in updated["connectivity"]
            ]
            print(f"  Updated connectivity: {updated['connectivity']}")

        # 4. Update dummy_beads
        if "dummy_beads" in updated:
            updated["dummy_beads"] = [
                name_mapping.get(bead, bead)
                for bead in updated["dummy_beads"]
            ]
            print(f"  Dummy beads: {updated['dummy_beads']}")

        # 5. Update interaction_matrix
        if "interaction_matrix" in updated:
            new_matrix = {}
            for old_key, interaction_list in updated["interaction_matrix"].items():
                new_key = name_mapping.get(old_key, old_key)
                new_matrix[new_key] = [
                    name_mapping.get(bead, bead)
                    for bead in interaction_list
                ]
            updated["interaction_matrix"] = new_matrix
            print(f"  Updated interaction matrix keys: {list(new_matrix.keys())}")

        return updated