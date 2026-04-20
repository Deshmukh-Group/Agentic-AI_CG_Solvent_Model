"""
Diagnostic Agent for CG Force Field Optimization
This agent analyzes simulation results and provides diagnostic reports.
Supports multi-temperature optimization.
"""
import os
import json
import numpy as np
from typing import Dict, List, Any, Optional
from common import LLMAgent, AgentRole, DiagnosticReport


class DiagnosticAgent(LLMAgent):
    """Agent for system diagnostics and analysis - multi-temperature aware"""

    def __init__(self, api_key: str, url: str, prompts_dir: str = "prompts", boundary_agent=None):
        super().__init__(AgentRole.DIAGNOSTIC, api_key, url)
        self.prompts_dir = prompts_dir
        self.boundary_agent = boundary_agent
        self._load_prompts()
        self._load_molecule_info()

    def _load_prompts(self):
        """Load prompts from files"""
        with open(os.path.join(self.prompts_dir, "diagnostic_system_prompt.txt"), 'r') as f:
            self.system_prompt = f.read().strip()
        with open(os.path.join(self.prompts_dir, "diagnostic_prompt_template.txt"), 'r') as f:
            self.prompt_template = f.read().strip()

    def _load_molecule_info(self):
        """Load molecule info from file"""
        try:
            with open("molecular_info/molecule_info.json", 'r') as f:
                self.molecule_info = json.load(f)
        except FileNotFoundError:
            self.molecule_info = {}

    def _load_phase_analysis(self, sim_dir="Simulation_Runs") -> Optional[Dict]:
        """Load phase analysis data from file if available"""
        phase_file = os.path.join(sim_dir, "phase_analysis.dat")
        if not os.path.exists(phase_file):
            return None
        phase_data = {}
        try:
            with open(phase_file, 'r') as f:
                for line in f:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        phase_data[key.strip()] = value.strip()
        except Exception:
            return None
        return phase_data

    def _guess_phase_state(self, phase_analysis: Optional[Dict]) -> str:
        if not phase_analysis:
            return "unknown"
        consensus = phase_analysis.get("Consensus Phase Classification", "").lower()
        if "liquid" in consensus:
            return "liquid"
        if any(x in consensus for x in ["crystal", "crystalline", "glassy", "solid"]):
            return "solid"
        return "unstable / intermediate"

    def _load_memory_context(self, iteration: int, memories_dir: str = "memories") -> str:
        """Load the most recent memory document for context"""
        try:
            memory_files = [
                f for f in os.listdir(memories_dir)
                if f.startswith("optimization_memory_iter_") and f.endswith(".json")
            ]
            if not memory_files:
                return "No previous optimization memory available yet."
            
            memory_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]), reverse=True)
            latest_file = os.path.join(memories_dir, memory_files[0])
            
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            summary = []
            if data.get('best_score', float('inf')) != float('inf'):
                summary.append(f"Best composite score so far: {data['best_score']:.6f}")
            if data.get('milestones'):
                latest = data['milestones'][-1]
                summary.append(f"Recent trend: {latest.get('improvement_rate', 0):.6f} improvement/iter")
                if latest.get('stagnation_count', 0) > 0:
                    summary.append(f"Stagnation warning: {latest['stagnation_count']} iter without progress")
            return "Recent optimization memory:\n" + "\n".join(f"  • {line}" for line in summary)
        except Exception as e:
            return f"(Memory context unavailable: {str(e)})"

    def _format_multi_temp_fork_analysis(
        self,
        all_fork_results_by_temp: Dict[str, List[Dict]],
        all_fork_params: List[Dict],
        all_fork_composite_scores: List[float],
        targets_by_temp: Dict[str, Dict]
    ) -> str:
        """Create structured multi-temperature fork comparison"""
        if not all_fork_results_by_temp or not all_fork_composite_scores:
            return ""

        lines = ["\n" + "═" * 80]
        lines.append("MULTI-TEMPERATURE • MULTI-FORK ANALYSIS")
        lines.append("═" * 80)

        # Sort forks by composite (averaged) score
        ranked = sorted(
            enumerate(all_fork_composite_scores),
            key=lambda x: x[1]  # lower score = better
        )
        best_idx = ranked[0][0]
        worst_idx = ranked[-1][0]

        lines.append(f"🏆 Best composite fork  : #{best_idx}   (avg score = {all_fork_composite_scores[best_idx]:.4f})")
        lines.append(f"❌ Worst composite fork : #{worst_idx}  (avg score = {all_fork_composite_scores[worst_idx]:.4f})")

        # Best fork — per temperature
        lines.append("\nBest fork performance per temperature:")
        for temp_key in sorted(targets_by_temp):
            temp_key_int = int(temp_key.rstrip("K"))
            res = all_fork_results_by_temp[temp_key_int][best_idx]
            tgt = targets_by_temp[temp_key]
            lines.append(f"  {temp_key}")
            for prop in sorted(tgt):
                val = res.get(prop, np.nan)
                target = tgt[prop]
                if target != 0 and not np.isnan(val):
                    dev_pct = abs(val - target) / target * 100
                    lines.append(f"    {prop:20} {val:8.4f}   target {target:8.4f}   dev {dev_pct:5.2f}%")
                else:
                    lines.append(f"    {prop:20} {val:8.4f if not np.isnan(val) else 'N/A'}")

        # Parameter contrast best vs worst
        lines.append("\nParameter differences — best vs worst fork:")
        p_best = all_fork_params[best_idx]
        p_worst = all_fork_params[worst_idx]
        param_diffs = [
            (name, abs(p_best[name] - p_worst[name]), p_best[name], p_worst[name])
            for name in p_best if name in p_worst
        ]
        param_diffs.sort(key=lambda x: x[1], reverse=True)

        for name, delta, vb, vw in param_diffs[:10]:
            lines.append(f"  {name:24} Δ = {delta:.6f}   best = {vb:.6f}   worst = {vw:.6f}")

        # Property sensitivity across forks
        lines.append("\nProperty variance across forks (averaged over temperatures):")
        all_props = set()
        for temp_res in all_fork_results_by_temp.values():
            if temp_res:
                all_props.update(temp_res[0].keys())

        for prop in sorted(all_props):
            fork_means = []
            for fork_i in range(len(all_fork_composite_scores)):
                vals = [all_fork_results_by_temp[t][fork_i].get(prop, np.nan)
                        for t in all_fork_results_by_temp]
                if any(not np.isnan(v) for v in vals):
                    fork_means.append(np.nanmean(vals))
            if len(fork_means) > 1:
                std = np.std(fork_means)
                rmin, rmax = min(fork_means), max(fork_means)
                lines.append(f"  {prop:20} std = {std:.4f}   range [{rmin:.4f} – {rmax:.4f}]")

        lines.append("═" * 80)
        return "\n".join(lines)

    def diagnose_system(
        self,
        iteration: int,
        results_by_temp: Dict[str, Dict],                   # best fork results per temperature
        targets_by_temp: Dict[str, Dict],                   # temperature → property targets
        params: Dict,                                       # best fork parameters
        trajectory_stats: Optional[Dict] = None,
        current_boundaries: Optional[Any] = None,
        best_params_list: Optional[List[Dict]] = None,
        boundary_hit_counts: Optional[Dict[str, int]] = None,
        recent_scores: Optional[List[float]] = None,
        stuck_counter: int = 0,
        all_fork_results_by_temp: Optional[Dict[str, List[Dict]]] = None,
        all_fork_params: Optional[List[Dict]] = None,
        all_fork_scores: Optional[List[float]] = None       # composite scores per fork
    ) -> Optional[DiagnosticReport]:
        """
        Main diagnostic method — now multi-temperature aware
        """
        memory_context = self._load_memory_context(iteration)

        # Trajectory / crash info
        traj_str = json.dumps(trajectory_stats, indent=2) if trajectory_stats else ""
        crash_info_str = ""
        if trajectory_stats and ("error_lines" in trajectory_stats or "last_lines" in trajectory_stats):
            crash_info_str = "\nCrash / stability information:\n" + traj_str

        # Phase info
        phase_analysis = self._load_phase_analysis()
        phase_info = json.dumps(phase_analysis, indent=2) if phase_analysis else ""
        phase_state = self._guess_phase_state(phase_analysis)

        # Multi-fork + multi-temperature analysis
        fork_analysis_str = ""
        if all_fork_results_by_temp and all_fork_params and all_fork_scores:
            fork_analysis_str = self._format_multi_temp_fork_analysis(
                all_fork_results_by_temp,
                all_fork_params,
                all_fork_scores,
                targets_by_temp
            )

        # Format best results clearly per temperature
        best_results_formatted = "Best parameters applied — results per temperature:\n"
        for temp_key in sorted(results_by_temp.keys()):
            res = results_by_temp[temp_key]
            tgt = targets_by_temp.get(temp_key, {})
            best_results_formatted += f"\n{temp_key}:\n"
            for prop in sorted(res):
                val = res.get(prop, np.nan)
                target = tgt.get(prop, 0.0)
                if target != 0 and not np.isnan(val):
                    dev = abs(val - target) / target * 100
                    best_results_formatted += f"  {prop:20} {val:9.4f}   target {target:9.4f}   dev {dev:5.2f}%\n"
                else:
                    best_results_formatted += f"  {prop:20} {val if not np.isnan(val) else 'N/A':9}\n"

        # Structured deviation dictionary
        deviations = {}
        for temp_key, tgt_dict in targets_by_temp.items():
            res = results_by_temp.get(temp_key, {})
            for prop, target in tgt_dict.items():
                val = res.get(prop, 0.0)
                if target != 0:
                    deviations[f"{prop} @ {temp_key}"] = f"{abs(val - target)/target*100:.2f}%"
                else:
                    deviations[f"{prop} @ {temp_key}"] = f"diff {abs(val - target):.4f}"

        # Build final prompt
        prompt = self.prompt_template.format(
            iteration=iteration,
            best_results_by_temperature=best_results_formatted,
            targets_by_temperature=json.dumps(targets_by_temp, indent=2),
            current_parameters=json.dumps(params, indent=2),
            stuck_counter=stuck_counter,
            memory_context=memory_context,
            traj_info=traj_str,
            crash_info=crash_info_str,
            phase_info=phase_info,
            deviations_by_property_and_temp=json.dumps(deviations, indent=2),
            fork_analysis=fork_analysis_str
        )

        # Call LLM
        response = self.call(prompt, self.system_prompt, temperature=0.6)

        if not response or "diagnosis" not in response:
            return None

        diag = response["diagnosis"]

        # Optional boundary adjustment
        boundary_adjustment = None
        if self.boundary_agent and current_boundaries and best_params_list and boundary_hit_counts:
            rec_text = " ".join(diag.get("recommendations", [])).lower()
            boundary_rec = diag.get("boundary_recommendations", {})
            should_adjust = (
                any(count > 2 for count in boundary_hit_counts.values()) or
                stuck_counter >= 20 or
                "boundary" in rec_text or "expand" in rec_text or "narrow" in rec_text or
                bool(boundary_rec)
            )
            if should_adjust:
                boundary_adjustment = self.boundary_agent.adjust_boundaries(
                    current_boundaries,
                    best_params_list,
                    boundary_hit_counts,
                    self.molecule_info,
                    boundary_rec
                )

        return DiagnosticReport(
            iteration=iteration,
            phase_state=phase_state,
            density_assessment=diag.get("density_assessment", "Not evaluated"),
            hvap_assessment=diag.get("hvap_assessment", "Not evaluated"),
            surface_tension_assessment=diag.get("surface_tension_assessment", "Not evaluated"),
            warnings=diag.get("warnings", []),
            recommendations=diag.get("recommendations", []),
            confidence_score=diag.get("confidence_score", 0.5),
            boundary_adjustment=boundary_adjustment
        )