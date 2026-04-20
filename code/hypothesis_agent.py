"""
Hypothesis Agent for CG Force Field Optimization

This agent generates scientific hypotheses about which parameters to change
and why, based on diagnostic reports and the current optimization state.
The generated hypothesis is then fed directly into the Optimization Agent.
"""

import os
import json
from typing import Dict, List, Optional
from common import LLMAgent, AgentRole, OptimizationState, DiagnosticReport, ParameterBoundary


class HypothesisAgent(LLMAgent):
    """Agent for generating scientific hypotheses to guide parameter optimization"""

    def __init__(self, api_key: str, url: str, prompts_dir: str = "prompts"):
        super().__init__(AgentRole.HYPOTHESIS, api_key, url)
        self.prompts_dir = prompts_dir
        self._load_prompts()

    def _load_prompts(self):
        """Load prompts from files"""
        with open(os.path.join(self.prompts_dir, "hypothesis_system_prompt.txt"), 'r') as f:
            self.system_prompt = f.read().strip()
        with open(os.path.join(self.prompts_dir, "hypothesis_prompt_template.txt"), 'r') as f:
            self.prompt_template = f.read().strip()
        try:
            with open(os.path.join(self.prompts_dir, "hypothesis_genetic_prompt_template.txt"), 'r') as f:
                self.genetic_prompt_template = f.read().strip()
        except FileNotFoundError:
            print("Warning: hypothesis_genetic_prompt_template.txt not found; genetic hypothesis disabled")
            self.genetic_prompt_template = None

    def generate_hypothesis(self,
                            state: OptimizationState,
                            diagnostic: DiagnosticReport,
                            boundaries: ParameterBoundary,
                            targets: Dict,
                            memory_context: str = "") -> Optional[Dict]:
        """
        Generate a scientific hypothesis for the next optimization step.

        Returns:
            hypothesis dict with keys: parameters_being_changed, scientific_rationale,
            expected_benefit, test_method, suggested_direction
        """
        self.reset_history()

        prompt = self.prompt_template.format(
            iteration=state.iteration,
            phase=state.phase,
            best_score=f"{state.best_score:.3f}",
            best_params=json.dumps(state.best_params),
            stuck_counter=state.stuck_counter,
            phase_state=diagnostic.phase_state,
            density_assessment=diagnostic.density_assessment,
            hvap_assessment=diagnostic.hvap_assessment,
            surface_tension_assessment=diagnostic.surface_tension_assessment,
            warnings=json.dumps(diagnostic.warnings),
            recommendations=json.dumps(diagnostic.recommendations),
            boundaries=json.dumps({
                name: {"min": mn, "max": mx}
                for name, mn, mx in zip(boundaries.var_names, boundaries.min_var, boundaries.max_var)
            }),
            targets=json.dumps(targets),
            memory_context=memory_context,
        )

        os.makedirs("llm_prompts", exist_ok=True)
        with open(f"llm_prompts/hypothesis_prompt_iter_{state.iteration}.txt", "w") as f:
            f.write(f"System Prompt:\n{self.system_prompt}\n\nUser Prompt:\n{prompt}\n")

        print(f"[hypothesis] Prompt length: {len(prompt)} chars, estimated tokens: {len(prompt) // 4}")
        result = self.call(prompt, self.system_prompt, temperature=0.7)

        if result and "hypothesis" in result:
            return result["hypothesis"]
        return None

    def generate_genetic_hypothesis(self,
                                    generation: int,
                                    iteration: int,
                                    parents: List[Dict],
                                    state: OptimizationState,
                                    boundaries: ParameterBoundary,
                                    targets: Dict) -> Optional[Dict]:
        """
        Generate a hypothesis about the expected outcome of a genetic crossover.

        Returns:
            hypothesis dict describing what the offspring should achieve
        """
        if not self.genetic_prompt_template or len(parents) < 2:
            return None

        self.reset_history()

        parent1, parent2 = parents[0], parents[1]
        prompt = self.genetic_prompt_template.format(
            generation=generation,
            iteration=iteration,
            parent1=json.dumps(parent1["params"], indent=2),
            parent1_score=parent1["score"],
            parent1_iter=parent1["iteration"],
            parent2=json.dumps(parent2["params"], indent=2),
            parent2_score=parent2["score"],
            parent2_iter=parent2["iteration"],
            boundaries=json.dumps({
                name: {"min": mn, "max": mx}
                for name, mn, mx in zip(boundaries.var_names, boundaries.min_var, boundaries.max_var)
            }),
            targets=json.dumps(targets),
            best_score=state.best_score,
            stuck_counter=state.stuck_counter,
        )

        os.makedirs("llm_prompts", exist_ok=True)
        with open(f"llm_prompts/hypothesis_genetic_gen_{generation}_iter_{iteration}.txt", "w") as f:
            f.write(f"System Prompt:\n{self.system_prompt}\n\nUser Prompt:\n{prompt}\n")

        print(f"[hypothesis-genetic] Prompt length: {len(prompt)} chars, estimated tokens: {len(prompt) // 4}")
        result = self.call(prompt, self.system_prompt, temperature=0.7)

        if result and "hypothesis" in result:
            return result["hypothesis"]
        return None
