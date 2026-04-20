"""
Optimization Agent for CG Force Field Optimization

This agent proposes parameter updates based on diagnostic reports.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional
from common import LLMAgent, AgentRole, OptimizationState, DiagnosticReport, ParameterBoundary
from memory import OptimizationMemory


class PheromoneTrails:
    """Pheromone trail system for single agent optimization using ant colony optimization principles"""

    def __init__(self, var_names, min_var, max_var, grid_size=20):
        self.var_names = var_names
        self.min_var = np.array(min_var)
        self.max_var = np.array(max_var)
        self.dimensions = len(var_names)
        self.grid_size = grid_size

        # Heatmap: Success signals (higher = better region)
        self.success_heatmap = np.zeros([grid_size] * self.dimensions)

        # Warning map: Wasteland signals (higher = avoid)
        self.warning_heatmap = np.zeros([grid_size] * self.dimensions)

        # Treasure locations: Best finds with metadata
        self.treasures = []

    def _position_to_grid(self, position):
        """Convert continuous position to grid indices"""
        indices = []
        for i, val in enumerate(position):
            # Normalize to [0, 1]
            normalized = (val - self.min_var[i]) / (self.max_var[i] - self.min_var[i])
            # Map to grid
            idx = int(normalized * (self.grid_size - 1))
            idx = np.clip(idx, 0, self.grid_size - 1)
            indices.append(idx)
        return tuple(indices)

    def add_success_signal(self, position, score):
        """Agent leaves success pheromone"""
        grid_idx = self._position_to_grid(position)

        # Stronger signal for better scores (inverse scoring)
        signal_strength = 1.0 / (1.0 + abs(score))

        # Add to heatmap with decay
        self.success_heatmap[grid_idx] += signal_strength

        # Add to treasures if exceptional
        if score < 10.0:  # Threshold for "treasure" (good performance across all properties)
            self.treasures.append({
                'position': position.copy(),
                'score': score,
                'timestamp': len(self.treasures)
            })

    def add_warning_signal(self, position, score):
        """Agent warns others about poor regions"""
        if score > 50.0:  # Only warn about really bad areas (poor performance across properties)
            grid_idx = self._position_to_grid(position)
            self.warning_heatmap[grid_idx] += 1.0

    def get_region_quality(self, position):
        """Query signal quality at a position"""
        grid_idx = self._position_to_grid(position)
        success = self.success_heatmap[grid_idx]
        warning = self.warning_heatmap[grid_idx]
        return {
            'success_signal': float(success),
            'warning_signal': float(warning),
            'quality_score': float(success - warning)
        }

    def get_top_treasures(self, top_k=3):
        """Get best treasure locations"""
        if not self.treasures:
            return []
        sorted_treasures = sorted(self.treasures, key=lambda x: x['score'])
        return sorted_treasures[:top_k]

    def get_exploration_suggestion(self):
        """Suggest unexplored region (low signal areas)"""
        total_signal = self.success_heatmap + self.warning_heatmap
        flat_signal = total_signal.flatten()

        # Find areas with lowest exploration
        unexplored_idx = np.argmin(flat_signal)

        # Convert back to position
        grid_coords = np.unravel_index(unexplored_idx, total_signal.shape)
        position = []
        for i, coord in enumerate(grid_coords):
            normalized = coord / (self.grid_size - 1)
            val = self.min_var[i] + normalized * (self.max_var[i] - self.min_var[i])
            position.append(val)

        return np.array(position)


class OptimizationAgent(LLMAgent):
    """Agent for parameter optimization"""

    def __init__(self, api_key: str, url: str, prompts_dir: str = "prompts", nforks: int = 1):
        super().__init__(AgentRole.OPTIMIZATION, api_key, url)
        self.prompts_dir = prompts_dir
        self.nforks = nforks
        self._load_prompts()

        # Initialize memory system
        self.memory = OptimizationMemory(self.system_prompt)

        # Initialize pheromone trail system (will be set up when boundaries are known)
        self.pheromone_trails = None

    def _initialize_pheromone_trails(self, boundaries: ParameterBoundary):
        """Initialize pheromone trail system with parameter boundaries"""
        if self.pheromone_trails is None:
            # Adaptive grid size based on dimensionality to prevent memory issues
            n_dims = len(boundaries.var_names)
            if n_dims <= 3:
                grid_size = 20
            elif n_dims <= 6:
                grid_size = 10
            elif n_dims <= 10:
                grid_size = 5
            elif n_dims <= 15:
                grid_size = 3
            else:
                grid_size = 2  # For very high-dimensional spaces, use minimal grid

            self.pheromone_trails = PheromoneTrails(
                var_names=boundaries.var_names,
                min_var=boundaries.min_var,
                max_var=boundaries.max_var,
                grid_size=grid_size
            )

    def _build_pheromone_context(self, state: OptimizationState) -> str:
        """Build pheromone trail context for the prompt"""
        if not self.pheromone_trails or not self.pheromone_trails.treasures:
            return ""

        context = "\n**Pheromone Trails (Ant Colony Optimization):**\n"

        # Share treasure locations
        treasures = self.pheromone_trails.get_top_treasures(top_k=3)
        if treasures:
            context += "- **🏆 Treasure Map (Best Parameter Sets):**\n"
            for i, treasure in enumerate(treasures):
                param_str = ", ".join([f"{name}={val:.3f}" for name, val in zip(self.pheromone_trails.var_names, treasure['position'])])
                context += f"  {i+1}. Score {treasure['score']:.3f} at [{param_str}]\n"

        # Suggest exploration if current position has low quality
        if state.best_params:
            current_pos = np.array([state.best_params[name] for name in self.pheromone_trails.var_names])
            quality = self.pheromone_trails.get_region_quality(current_pos)
            if quality['quality_score'] < 0.5:  # Low quality region
                suggestion = self.pheromone_trails.get_exploration_suggestion()
                param_str = ", ".join([f"{name}={val:.3f}" for name, val in zip(self.pheromone_trails.var_names, suggestion)])
                context += f"- **🗺️  Exploration Suggestion:** Consider [{param_str}] (unexplored region)\n"

        return context

    def _load_prompts(self):
        """Load prompts from files"""
        with open(os.path.join(self.prompts_dir, "optimization_system_prompt.txt"), 'r') as f:
            self.system_prompt = f.read().strip()
        with open(os.path.join(self.prompts_dir, "optimization_prompt_template.txt"), 'r') as f:
            self.prompt_template = f.read().strip()

        # NEW: Load genetic crossover prompts
        try:
            with open(os.path.join(self.prompts_dir, "genetic_crossover_system_prompt.txt"), 'r') as f:
                self.genetic_system_prompt = f.read().strip()
            with open(os.path.join(self.prompts_dir, "genetic_crossover_prompt_template.txt"), 'r') as f:
                self.genetic_prompt_template = f.read().strip()
        except FileNotFoundError:
            print("Warning: Genetic crossover prompts not found")
            self.genetic_system_prompt = self.system_prompt
            self.genetic_prompt_template = self.prompt_template

    def propose_parameters(self, state: OptimizationState,
                             diagnostic: DiagnosticReport,
                             boundaries: ParameterBoundary,
                             targets: Dict,
                             hypothesis: Optional[Dict] = None) -> Optional[Dict]:
        """Propose next parameter set based on hypothesis, diagnostics, memory, and pheromone signals"""

        # Initialize pheromone trail system if not done
        self._initialize_pheromone_trails(boundaries)

        # Reset conversation history to prevent context window overflow
        self.reset_history()

        # Get memory context
        memory_context = self.memory.get_context_message(state.iteration)
        # No truncation for full memory visibility

        # Include all warnings and recommendations
        truncated_warnings = diagnostic.warnings
        truncated_recommendations = diagnostic.recommendations

        # Add pheromone trail context
        pheromone_context = self._build_pheromone_context(state)

        # Format hypothesis from Hypothesis Agent (or placeholder if unavailable)
        hypothesis_str = json.dumps(hypothesis, indent=2) if hypothesis else "No hypothesis available — use diagnostic recommendations to guide parameter selection."

        prompt = self.prompt_template.format(
            nforks=self.nforks,
            iteration=state.iteration,
            phase=state.phase,
            best_score=f"{state.best_score:.3f}",
            recent_scores=[f"{s:.3f}" for s in state.recent_scores[-5:]],
            stuck_counter=state.stuck_counter,
            best_params=json.dumps(state.best_params),
            phase_state=diagnostic.phase_state,
            density_assessment=diagnostic.density_assessment,
            hvap_assessment=diagnostic.hvap_assessment,
            surface_tension_assessment=diagnostic.surface_tension_assessment,
            warnings=json.dumps(truncated_warnings),
            recommendations=json.dumps(truncated_recommendations),
            boundaries=json.dumps({name: {"min": mn, "max": mx} for name, mn, mx in
                                      zip(boundaries.var_names, boundaries.min_var, boundaries.max_var)}),
            targets=json.dumps(targets),
            crash_regions=json.dumps(state.crash_regions[-3:] if state.crash_regions else []),
            memory_context=memory_context + pheromone_context,
            hypothesis=hypothesis_str,
        )

        # Add nforks info to the system prompt
        self.system_prompt = self.system_prompt.replace("{nforks}", str(self.nforks))

        # Log the prompt for debugging
        with open(f"llm_prompts/llm_prompt_iter_{state.iteration}.txt", "w") as f:
            f.write(f"System Prompt:\n{self.system_prompt}\n\nUser Prompt:\n{prompt}\n")

        print(f"[optimization] Prompt length: {len(prompt)} chars, estimated tokens: {len(prompt) // 4}")
        result = self.call(prompt, self.system_prompt, temperature=0.7)

        if result and "action" in result:
            return result
        return None

    def update_memory(self, iteration: int, proposal: Dict, scores: List[float], params_list: List[Dict]):
        """Update memory with best fork result"""
        # Handle both single value and list inputs
        if not isinstance(scores, list):
            scores = [scores]
        if not isinstance(params_list, list):
            params_list = [params_list]
        
        best_idx = np.argmin(scores)
        best_score = scores[best_idx]
        best_params = params_list[best_idx]
        
        self.memory.update(iteration, proposal, best_score, best_params)
        
        # Update pheromone trails for all forks
        if self.pheromone_trails:
            for params, score in zip(params_list, scores):
                position = np.array([params[name] for name in self.pheromone_trails.var_names])
                if score < 20.0:
                    self.pheromone_trails.add_success_signal(position, score)
                else:
                    self.pheromone_trails.add_warning_signal(position, score)

        # Save memory to memories folder
        memory_file = f"memories/optimization_memory_iter_{iteration}.json"
        self.memory.save_to_json(memory_file)

    def propose_genetic_offspring(self,
                                  generation: int,
                                  iteration: int,
                                  parents: List[Dict],
                                  genetic_boundaries: ParameterBoundary,
                                  global_boundaries: ParameterBoundary,
                                  targets: Dict,
                                  state: OptimizationState,
                                  hypothesis: Optional[Dict] = None) -> Optional[Dict]:
        """
        Perform genetic crossover between two parents

        Args:
            generation: Current generation number
            iteration: Current iteration number
            parents: List of 2 parent dicts with 'params', 'score', 'iteration'
            genetic_boundaries: Focused boundaries around parents
            global_boundaries: Global physical boundaries
            targets: Target property values
            state: Current optimization state
            hypothesis: Pre-generated hypothesis from HypothesisAgent (optional)

        Returns:
            Dict with 'reasoning', 'crossover_map', 'mutation_applied', 'action'
        """
        if len(parents) < 2:
            print("Error: Need at least 2 parents for crossover")
            return None

        parent1 = parents[0]
        parent2 = parents[1]

        # Build parent comparison analysis
        parent_comparison = self._compare_parents(parent1, parent2, targets)

        # Extract parameter blocks for display
        p1_params = parent1['params']
        p2_params = parent2['params']

        # Calculate buffer percentage
        buffer_percent = self._calculate_buffer_percent(genetic_boundaries, parent1, parent2)

        # Get memory context
        memory_context = self.memory.get_context_message(iteration) if hasattr(self, 'memory') else ""

        # Generate parameter blocks dynamically
        parameter_blocks = self._generate_parameter_blocks(p1_params, p2_params)

        # Format hypothesis from Hypothesis Agent (or placeholder if unavailable)
        hypothesis_str = json.dumps(hypothesis, indent=2) if hypothesis else "No hypothesis available — use parent analysis to guide block selection."

        # Format the genetic crossover prompt
        prompt = self.genetic_prompt_template.format(
            generation=generation,
            iteration=iteration,
            parent1=json.dumps(p1_params, indent=2),
            parent1_score=parent1['score'],
            parent1_iter=parent1['iteration'],
            parent2=json.dumps(p2_params, indent=2),
            parent2_score=parent2['score'],
            parent2_iter=parent2['iteration'],
            parent_comparison=parent_comparison,
            buffer_percent=buffer_percent,
            genetic_boundaries=self._format_boundaries(genetic_boundaries, "GENETIC"),
            global_boundaries=self._format_boundaries(global_boundaries, "GLOBAL"),
            parameter_blocks=parameter_blocks,
            targets=json.dumps(targets, indent=2),
            best_score=state.best_score,
            stuck_counter=state.stuck_counter,
            memory_context=memory_context,
            hypothesis=hypothesis_str,
        )

        # Log the genetic prompt
        with open(f"llm_prompts/genetic_prompt_gen_{generation}_iter_{iteration}.txt", "w") as f:
            f.write(f"System Prompt:\n{self.genetic_system_prompt}\n\n")
            f.write(f"User Prompt:\n{prompt}\n")

        print(f"[genetic] Prompt length: {len(prompt)} chars")

        # Call LLM with genetic system prompt
        result = self.call(prompt, self.genetic_system_prompt, temperature=0.8)  # Higher temp for diversity

        if result and "action" in result:
            return result
        return None

    def _compare_parents(self, parent1: Dict, parent2: Dict, targets: Dict) -> str:
        """Compare parents and identify strengths"""
        # This would ideally use actual property breakdowns
        lines = []
        lines.append(f"Score Difference: {abs(parent1['score'] - parent2['score']):.4f}")
        lines.append(f"Iteration Gap: {abs(parent1['iteration'] - parent2['iteration'])} iterations apart")

        if parent1['score'] < parent2['score']:
            lines.append("Parent 1 is overall better performer")
        else:
            lines.append("Parent 2 is overall better performer")

        # Compare parameter values
        param_diffs = []
        for key in parent1['params'].keys():
            diff = abs(parent1['params'][key] - parent2['params'][key])
            if diff > 0.01:  # Significant difference
                param_diffs.append(f"{key}: Δ={diff:.3f}")

        if param_diffs:
            lines.append(f"Major parameter differences: {', '.join(param_diffs[:5])}")

        return "\n".join(lines)

    def _calculate_buffer_percent(self, genetic_bounds: ParameterBoundary,
                                 parent1: Dict, parent2: Dict) -> float:
        """Calculate average buffer percentage"""
        buffers = []
        for i, param_name in enumerate(genetic_bounds.var_names):
            p1_val = parent1['params'][param_name]
            p2_val = parent2['params'][param_name]
            gen_min = genetic_bounds.min_var[i]
            gen_max = genetic_bounds.max_var[i]

            parent_min = min(p1_val, p2_val)
            parent_max = max(p1_val, p2_val)
            parent_range = parent_max - parent_min

            if parent_range > 1e-6:
                lower_buffer = (parent_min - gen_min)
                upper_buffer = (gen_max - parent_max)
                avg_buffer = (lower_buffer + upper_buffer) / 2
                buffer_pct = (avg_buffer / parent_range) * 100
                buffers.append(buffer_pct)

        return np.mean(buffers) if buffers else 30.0

    def _format_boundaries(self, boundaries: ParameterBoundary, label: str) -> str:
        """Format boundaries for display"""
        lines = [f"{label} Boundaries:"]
        for i, param_name in enumerate(boundaries.var_names):
            min_val = boundaries.min_var[i]
            max_val = boundaries.max_var[i]
            range_val = max_val - min_val
            lines.append(f"  {param_name}: [{min_val:.4f}, {max_val:.4f}] (range: {range_val:.4f})")
        return "\n".join(lines)

    def _generate_parameter_blocks(self, p1_params: Dict, p2_params: Dict) -> str:
        """Generate parameter blocks description for crossover"""
        blocks = []
        block_num = 1

        # Group parameters by type
        bond_params = {}
        lj_params = {}
        cross_lj_params = {}
        charge_params = {}

        for param_name in p1_params.keys():
            if '_bl' in param_name and param_name.replace('_bl', '_kb') in p1_params:
                # Bond parameters
                bond_base = param_name.replace('_bl', '')
                if bond_base not in bond_params:
                    bond_params[bond_base] = []
                bond_params[bond_base].append(param_name)
                bond_params[bond_base].append(param_name.replace('_bl', '_kb'))

            elif '_epsilon' in param_name and not '_' in param_name.split('_epsilon')[1]:
                # Self LJ parameters (like CM_epsilon)
                bead_type = param_name.replace('_epsilon', '')
                if bead_type not in lj_params:
                    lj_params[bead_type] = []
                lj_params[bead_type].append(param_name)
                rmin_param = f"{bead_type}_rminby2"
                if rmin_param in p1_params:
                    lj_params[bead_type].append(rmin_param)
                else:
                    rmin_param = f"{bead_type}_rmin"
                    if rmin_param in p1_params:
                        lj_params[bead_type].append(rmin_param)

            elif '_epsilon' in param_name and '_' in param_name.split('_epsilon')[1]:
                # Cross LJ parameters
                parts = param_name.split('_epsilon')[0].split('_')
                if len(parts) == 2:
                    bead1, bead2 = parts
                    key = f"{bead1}_{bead2}"
                    if key not in cross_lj_params:
                        cross_lj_params[key] = []
                    cross_lj_params[key].append(param_name)
                    rmin_param = f"{bead1}_{bead2}_rmin"
                    if rmin_param in p1_params:
                        cross_lj_params[key].append(rmin_param)

            elif '_charge' in param_name:
                # Charge parameters
                charge_params[param_name] = True

        # Generate block descriptions
        for bond_base, params in bond_params.items():
            params = list(set(params))  # Remove duplicates
            if len(params) >= 2:
                blocks.append(f"Block {block_num} - {bond_base.replace('_', '-')} Bond:")
                blocks.append(f"  Parameters: {params}")
                p1_vals = [f"{p}={p1_params.get(p, 'N/A')}" for p in params]
                p2_vals = [f"{p}={p2_params.get(p, 'N/A')}" for p in params]
                blocks.append(f"  Parent 1: [{', '.join(p1_vals)}]")
                blocks.append(f"  Parent 2: [{', '.join(p2_vals)}]")
                blocks.append("  → Inherit BOTH parameters from same parent")
                blocks.append("")
                block_num += 1

        for bead_type, params in lj_params.items():
            params = list(set(params))
            if len(params) >= 2:
                blocks.append(f"Block {block_num} - {bead_type}-{bead_type} LJ Interaction:")
                blocks.append(f"  Parameters: {params}")
                p1_vals = [f"{p}={p1_params.get(p, 'N/A')}" for p in params]
                p2_vals = [f"{p}={p2_params.get(p, 'N/A')}" for p in params]
                blocks.append(f"  Parent 1: [{', '.join(p1_vals)}]")
                blocks.append(f"  Parent 2: [{', '.join(p2_vals)}]")
                blocks.append("  → Inherit BOTH parameters from same parent")
                blocks.append("")
                block_num += 1

        for cross_key, params in cross_lj_params.items():
            params = list(set(params))
            if len(params) >= 2:
                bead1, bead2 = cross_key.split('_')
                blocks.append(f"Block {block_num} - {bead1}-{bead2} Cross LJ Interaction:")
                blocks.append(f"  Parameters: {params}")
                p1_vals = [f"{p}={p1_params.get(p, 'N/A')}" for p in params]
                p2_vals = [f"{p}={p2_params.get(p, 'N/A')}" for p in params]
                blocks.append(f"  Parent 1: [{', '.join(p1_vals)}]")
                blocks.append(f"  Parent 2: [{', '.join(p2_vals)}]")
                blocks.append("  → Inherit BOTH parameters from same parent")
                blocks.append("")
                block_num += 1

        if charge_params:
            charge_list = list(charge_params.keys())
            blocks.append(f"Block {block_num} - Electrostatic Charges:")
            blocks.append(f"  Parameters: {charge_list}")
            p1_vals = [f"{p}={p1_params.get(p, 'N/A')}" for p in charge_list]
            p2_vals = [f"{p}={p2_params.get(p, 'N/A')}" for p in charge_list]
            blocks.append(f"  Parent 1: [{', '.join(p1_vals)}]")
            blocks.append(f"  Parent 2: [{', '.join(p2_vals)}]")
            blocks.append("  → Inherit ALL charge parameters from same parent (must maintain charge neutrality!)")
            blocks.append("")
            block_num += 1

        return "\n".join(blocks)