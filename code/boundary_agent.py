"""
Boundary Agent for CG Force Field Optimization

This agent sets parameter ranges based on chemical intuition and adaptively
adjusts boundaries (both expansion and contraction) based on optimization progress.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from common import LLMAgent, AgentRole, MappingScheme, ParameterBoundary


class BoundaryAgent(LLMAgent):
    """Agent for setting parameter boundaries"""

    def __init__(self, api_key: str, url: str, prompts_dir: str = "prompts"):
        super().__init__(AgentRole.BOUNDARY, api_key, url)
        self.prompts_dir = prompts_dir
        self._load_prompts()

    def _load_prompts(self):
        """Load prompts from files"""
        # Note: system_prompt is loaded conditionally in set_boundaries
        with open(os.path.join(self.prompts_dir, "boundary_prompt_template.txt"), 'r') as f:
            self.prompt_template = f.read().strip()
        with open(os.path.join(self.prompts_dir, "boundary_adjustment_prompt.txt"), 'r') as f:
            self.adjustment_prompt_template = f.read().strip()

    def _extract_parameters_from_file(self, param_file: str = "namd_setup/cg_parameters.prm") -> List[str]:
        """Extract parameter names from the CG parameter file"""
        import re

        parameters = []
        try:
            with open(param_file, 'r') as f:
                content = f.read()

            # Find all placeholders like ${PARAM_NAME}
            placeholders = re.findall(r'\$\{([^}]+)\}', content)
            parameters = list(set(placeholders))  # Remove duplicates

        except FileNotFoundError:
            print(f"Warning: Parameter file {param_file} not found. Using empty parameter list.")
        except Exception as e:
            print(f"Error reading parameter file: {e}")

        return sorted(parameters)

    def set_boundaries(self, mapping_scheme: MappingScheme,
                          molecule_info: Dict,
                          aa_reference: Optional[Dict] = None) -> Optional[ParameterBoundary]:
        """Set initial parameter boundaries based on chemical intuition"""

        # Extract actual parameters from the parameter file
        param_file = "namd_setup/cg_parameters.prm"
        required_params = self._extract_parameters_from_file(param_file)

        # Add charge parameters for dummy beads
        if mapping_scheme.dummy_beads:
            for dummy in mapping_scheme.dummy_beads:
                required_params.append(f"{dummy}_charge")

        if not required_params:
            print("Warning: No parameters found in parameter file. Cannot set boundaries.")
            return None

        aa_info = ""
        if aa_reference:
            aa_info = f"\nAll-atom reference data:\n{json.dumps(aa_reference, separators=(',', ':'))}"

        # Load appropriate system prompt based on AA2CG availability
        aa2cg_path = os.path.join("AA2CG", "AA2CG_results.json")
        if os.path.exists(aa2cg_path):
            with open(os.path.join(self.prompts_dir, "boundary_system_prompt_minimal.txt"), 'r') as f:
                self.system_prompt = f.read().strip()
        else:
            with open(os.path.join(self.prompts_dir, "boundary_system_prompt.txt"), 'r') as f:
                self.system_prompt = f.read().strip()

        print("  Analyzing AA2CG trajectory...")

        # Analyze AA2CG trajectory
        os.system(f"python3 analyze_AA2CG.py")

        try:
            with open(os.path.join("AA2CG", "AA2CG_results.json"), 'r') as f:
                aa2cg_data = json.load(f)
            aa2cg_data = f"\n\n## IMPORTANT:\nAll-atom mapped to CG (AA2CG) reference values:\n{json.dumps(aa2cg_data, separators=(',', ':'))}"
            print (aa2cg_data)

            # Add to system prompt
            self.system_prompt += aa2cg_data 

        except Exception as e:
            print(f"Error reading AA2CG_results.json: {e}")
            
        prompt = self.prompt_template.format(
            molecule_name=molecule_info.get('name', 'Unknown'),
            bead_types=', '.join(mapping_scheme.bead_types),
            connectivity=mapping_scheme.connectivity,
            dummy_beads=', '.join(mapping_scheme.dummy_beads) if mapping_scheme.dummy_beads else 'None',
            bead_descriptions=json.dumps(mapping_scheme.bead_descriptions, separators=(',', ':')),
            interaction_matrix=json.dumps(mapping_scheme.interaction_matrix, separators=(',', ':')),
            required_params=', '.join(required_params),
            aa_info=aa_info
        )

        # Add instruction for conservative initial boundaries
        prompt += "\n\nIMPORTANT: Set CONSERVATIVE initial boundaries (not too wide) to allow for both expansion and contraction during optimization. "
        prompt += "Start with physically reasonable ranges (guess from the standard deviation) that can be adjusted in both directions using the AA2CG mean data."

        result = self.call(prompt, self.system_prompt, temperature=0.5)

        if result and "boundaries" in result:
            bounds = result["boundaries"]
            var_names = bounds.get("var_names", [])
            min_var = bounds.get("min_var", [])
            max_var = bounds.get("max_var", [])
            recommended_start = bounds.get("recommended_start", [])
            physical_constraints = bounds.get("physical_constraints", {})

            # Validate that all arrays have the same length
            n_params = len(var_names)
            if len(min_var) != n_params or len(max_var) != n_params or len(recommended_start) != n_params:
                print(f"[boundary] Array length mismatch: var_names={n_params}, min_var={len(min_var)}, max_var={len(max_var)}, recommended_start={len(recommended_start)}")
                return None

            return ParameterBoundary(
                var_names=var_names,
                min_var=min_var,
                max_var=max_var,
                recommended_start=recommended_start,
                physical_constraints=physical_constraints
            )
        return None

    def characterize_boundary_features(self, current_boundaries: ParameterBoundary,
                                     best_params_list: List[Dict],
                                     boundary_hit_counts: Dict[str, int]) -> Dict[str, Any]:
        """Compute quantitative features for boundary adjustment decisions"""

        features = {}

        # 1. Boundary proximity analysis
        features['boundary_proximity'] = {}
        for param_name in current_boundaries.var_names:
            param_idx = current_boundaries.var_names.index(param_name)
            min_val = current_boundaries.min_var[param_idx]
            max_val = current_boundaries.max_var[param_idx]

            # Get all values for this parameter from best params
            param_values = [params[param_name] for params in best_params_list if param_name in params]

            if param_values:
                # Distance to boundaries (normalized)
                min_distance = min(abs(v - min_val) for v in param_values)
                max_distance = min(abs(v - max_val) for v in param_values)

                # Clustering near boundaries (within 10% of range)
                range_width = max_val - min_val
                near_min = sum(1 for v in param_values if abs(v - min_val) / range_width < 0.1)
                near_max = sum(1 for v in param_values if abs(v - max_val) / range_width < 0.1)
                
                # Clustering in interior (between 20% and 80% of range)
                in_interior = sum(1 for v in param_values if 0.2 < (v - min_val) / range_width < 0.8)

                features['boundary_proximity'][param_name] = {
                    'min_distance': min_distance,
                    'max_distance': max_distance,
                    'near_min_count': near_min,
                    'near_max_count': near_max,
                    'in_interior_count': in_interior,
                    'range_width': range_width,
                    'normalized_min_dist': min_distance / range_width if range_width > 0 else 0,
                    'normalized_max_dist': max_distance / range_width if range_width > 0 else 0
                }

        # 2. Parameter distribution analysis
        features['distribution_stats'] = {}
        for param_name in current_boundaries.var_names:
            param_values = [params[param_name] for params in best_params_list if param_name in params]
            if len(param_values) > 1:
                param_idx = current_boundaries.var_names.index(param_name)
                min_val = current_boundaries.min_var[param_idx]
                max_val = current_boundaries.max_var[param_idx]
                range_width = max_val - min_val
                
                features['distribution_stats'][param_name] = {
                    'mean': float(np.mean(param_values)),
                    'std': float(np.std(param_values)),
                    'min': min(param_values),
                    'max': max(param_values),
                    'range_used': max(param_values) - min(param_values),
                    'range_usage_ratio': (max(param_values) - min(param_values)) / range_width if range_width > 0 else 0,
                    'centroid_position': (np.mean(param_values) - min_val) / range_width if range_width > 0 else 0.5
                }

        # 3. Boundary hit analysis
        features['boundary_hit_analysis'] = {
            'total_hits': sum(boundary_hit_counts.values()),
            'frequent_hitters': [p for p, c in boundary_hit_counts.items() if c > 2],
            'hit_distribution': boundary_hit_counts
        }

        # 4. Expansion and contraction signals
        features['adjustment_signals'] = {
            'expand': [],
            'contract': [],
            'shift': []
        }

        # Signal: Parameters clustering near boundaries (EXPAND)
        for param_name, proximity in features['boundary_proximity'].items():
            if proximity['near_min_count'] >= 2:
                features['adjustment_signals']['expand'].append(
                    f"{param_name}_lower: {proximity['near_min_count']} samples cluster near lower bound"
                )
            if proximity['near_max_count'] >= 2:
                features['adjustment_signals']['expand'].append(
                    f"{param_name}_upper: {proximity['near_max_count']} samples cluster near upper bound"
                )

        # Signal: Parameters frequently hitting boundaries (EXPAND)
        for param_name, count in boundary_hit_counts.items():
            if count > 2:
                features['adjustment_signals']['expand'].append(
                    f"{param_name}: hit boundary {count} times"
                )

        # Signal: Parameters very close to boundaries (EXPAND)
        for param_name, proximity in features['boundary_proximity'].items():
            if proximity['normalized_min_dist'] < 0.05:
                features['adjustment_signals']['expand'].append(
                    f"{param_name}_lower: samples within 5% of lower bound"
                )
            if proximity['normalized_max_dist'] < 0.05:
                features['adjustment_signals']['expand'].append(
                    f"{param_name}_upper: samples within 5% of upper bound"
                )

        # Signal: Parameters using narrow range with interior clustering (CONTRACT)
        for param_name, stats in features['distribution_stats'].items():
            proximity = features['boundary_proximity'].get(param_name, {})
            # Contract if using <25% of range AND most samples are in interior
            if stats['range_usage_ratio'] < 0.25 and proximity.get('in_interior_count', 0) >= 3:
                features['adjustment_signals']['contract'].append(
                    f"{param_name}: using only {stats['range_usage_ratio']*100:.1f}% of available range, samples concentrated in interior"
                )

        # Signal: Parameters with consistent offset from boundary (SHIFT)
        for param_name, stats in features['distribution_stats'].items():
            centroid = stats.get('centroid_position', 0.5)
            range_usage = stats.get('range_usage_ratio', 0)
            
            # If centroid is significantly offset and range usage is moderate
            if range_usage < 0.5:  # Not using full range
                if centroid < 0.3:  # Clustered near lower bound
                    features['adjustment_signals']['shift'].append(
                        f"{param_name}: centroid at {centroid*100:.1f}% (near lower bound), consider shifting down"
                    )
                elif centroid > 0.7:  # Clustered near upper bound
                    features['adjustment_signals']['shift'].append(
                        f"{param_name}: centroid at {centroid*100:.1f}% (near upper bound), consider shifting up"
                    )

        return features

    def adjust_boundaries(self, current_boundaries: ParameterBoundary,
                           best_params_list: List[Dict],
                           boundary_hit_counts: Dict[str, int],
                           molecule_info: Dict,
                           boundary_recommendations: Dict = None) -> Optional[ParameterBoundary]:
        """Intelligently adjust boundaries - expand, contract, or shift per parameter"""

        if not best_params_list:
            return None

        # Compute quantitative features for decision making
        features = self.characterize_boundary_features(current_boundaries, best_params_list, boundary_hit_counts)

        # Prepare data for the prompt
        best_params_summary = []
        for i, params in enumerate(best_params_list[:5]):  # Top 5
            best_params_summary.append(f"Rank {i+1}: {json.dumps(params, indent=2)}")

        boundary_hits = []
        for param, count in boundary_hit_counts.items():
            if count > 2:
                boundary_hits.append(f"{param}: hit {count} times")

        prompt = self.adjustment_prompt_template.format(
            current_boundaries=json.dumps(asdict(current_boundaries), indent=2),
            features=json.dumps(features, indent=2),
            best_params_summary=chr(10).join(best_params_summary),
            boundary_hits=chr(10).join(boundary_hits) if boundary_hits else "None",
            boundary_recommendations=json.dumps(boundary_recommendations or {}, indent=2),
            molecule_name=molecule_info.get('name', 'Unknown')
        )

        result = self.call(prompt, self.system_prompt, temperature=0.4)

        if result and "boundaries" in result:
            bounds = result["boundaries"]
            if bounds is not None:
                new_boundary = ParameterBoundary(
                    var_names=bounds.get("var_names", current_boundaries.var_names),
                    min_var=bounds.get("min_var", current_boundaries.min_var),
                    max_var=bounds.get("max_var", current_boundaries.max_var),
                    recommended_start=bounds.get("recommended_start", current_boundaries.recommended_start),
                    physical_constraints=bounds.get("physical_constraints", current_boundaries.physical_constraints)
                )
                
                # Log the adjustments made
                self._log_boundary_adjustments(current_boundaries, new_boundary, features)
                
                return new_boundary
        return None

    def _log_boundary_adjustments(self, old_bounds: ParameterBoundary, 
                                  new_bounds: ParameterBoundary,
                                  features: Dict[str, Any]):
        """Log boundary adjustments for transparency"""
        print("\n" + "="*80)
        print("BOUNDARY ADJUSTMENT SUMMARY")
        print("="*80)
        
        for i, param_name in enumerate(old_bounds.var_names):
            old_min = old_bounds.min_var[i]
            old_max = old_bounds.max_var[i]
            new_min = new_bounds.min_var[i]
            new_max = new_bounds.max_var[i]
            
            if old_min != new_min or old_max != new_max:
                old_range = old_max - old_min
                new_range = new_max - new_min
                
                # Determine adjustment type
                adjustment_type = []
                if new_min < old_min and new_max > old_max:
                    adjustment_type.append("EXPAND BOTH")
                elif new_min < old_min:
                    adjustment_type.append("EXPAND LOWER")
                elif new_max > old_max:
                    adjustment_type.append("EXPAND UPPER")
                elif new_min > old_min and new_max < old_max:
                    adjustment_type.append("CONTRACT")
                elif new_min > old_min or new_max < old_max:
                    adjustment_type.append("SHIFT")
                
                print(f"\n{param_name}: {' & '.join(adjustment_type)}")
                print(f"  Old: [{old_min:.4f}, {old_max:.4f}] (range: {old_range:.4f})")
                print(f"  New: [{new_min:.4f}, {new_max:.4f}] (range: {new_range:.4f})")
                
                # Show relevant metrics
                if param_name in features.get('distribution_stats', {}):
                    stats = features['distribution_stats'][param_name]
                    print(f"  Usage: {stats['range_usage_ratio']*100:.1f}% of range")
                    print(f"  Centroid: {stats['centroid_position']*100:.1f}% position")
                
                if param_name in features.get('boundary_proximity', {}):
                    prox = features['boundary_proximity'][param_name]
                    print(f"  Near bounds: {prox['near_min_count']} lower, {prox['near_max_count']} upper")
        
        print("\n" + "="*80 + "\n")

    def get_genetic_boundaries(self, parents: List[Dict],
                          global_boundaries: ParameterBoundary,
                          buffer_fraction: float = 0.30) -> ParameterBoundary:
        """
        Create GA-focused boundaries around parent locations

        Args:
            parents: List of parent dicts with 'params' and 'score'
            global_boundaries: Current global boundaries (for validation)
            buffer_fraction: Buffer size (0.30 = 30% expansion around parents)

        Returns:
            New ParameterBoundary focused around parents with buffer
        """
        parent1_params = parents[0]['params']
        parent2_params = parents[1]['params']

        genetic_min = []
        genetic_max = []
        genetic_start = []

        for i, param_name in enumerate(global_boundaries.var_names):
            p1_val = parent1_params[param_name]
            p2_val = parent2_params[param_name]

            global_min = global_boundaries.min_var[i]
            global_max = global_boundaries.max_var[i]
            global_range = global_max - global_min

            # Parent range
            parent_min = min(p1_val, p2_val)
            parent_max = max(p1_val, p2_val)
            parent_range = parent_max - parent_min

            # Buffer calculation
            if parent_range < 1e-6:  # Parents very similar
                # Use buffer based on global range
                buffer = buffer_fraction * global_range
            else:
                # Use buffer based on parent separation
                buffer = buffer_fraction * parent_range

            # Set genetic bounds (clipped to global physical limits)
            gen_min = max(global_min, parent_min - buffer)
            gen_max = min(global_max, parent_max + buffer)
            gen_start = (p1_val + p2_val) / 2  # Midpoint of parents

            genetic_min.append(gen_min)
            genetic_max.append(gen_max)
            genetic_start.append(gen_start)

            # Log the adjustment
            print(f"    {param_name}: [{parent_min:.3f}, {parent_max:.3f}] "
                  f"→ [{gen_min:.3f}, {gen_max:.3f}] (buffer={buffer:.3f})")

        return ParameterBoundary(
            var_names=global_boundaries.var_names,
            min_var=genetic_min,
            max_var=genetic_max,
            recommended_start=genetic_start,
            physical_constraints=global_boundaries.physical_constraints
        )