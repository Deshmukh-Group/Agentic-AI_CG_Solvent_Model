"""
Multi-Agent Coarse-Grained Force Field Optimization Framework

This framework uses specialized LLM agents for:
1. Mapping Agent: Decides CG mapping scheme and bead types
2. Topology Agent: Constructs CG topology based on mapping scheme
3. Boundary Agent: Sets parameter ranges based on chemical intuition
4. Diagnostic Agent: Analyzes simulation results and system behavior
5. Optimization Agent: Proposes parameter updates based on results
"""

import os
import json
import numpy as np
import requests
import random
import subprocess
import re
import shutil
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
from mapping_agent import MappingAgent
from bead_mapping_agent import BeadMappingAgent
from topology_creator_agent import TopologyCreatorAgent
from boundary_agent import BoundaryAgent
from diagnostic_agent import DiagnosticAgent
from hypothesis_agent import HypothesisAgent
from optimization_agent import OptimizationAgent
from common import (
    AgentRole, MappingScheme, ParameterBoundary, DiagnosticReport,
    OptimizationState, LLMAgent
)

# ============================================================================
# Multi-Agent Orchestrator
# ============================================================================

class MultiAgentOrchestrator:
    """Coordinates multiple agents for CG force field optimization"""

    def __init__(self, api_key: str, url: str, output_dir: str = ".", prompts_dir: str = "prompts", nforks: int = 1, temperatures: Optional[List[int]] = None):
        self.api_key = api_key
        self.url = url
        self.output_dir = output_dir
        self.prompts_dir = prompts_dir
        self.nforks = nforks
        self.temperatures = temperatures or [298]
        self.use_completions = False  # Use chat API

        # Initialize agents
        self.mapping_agent = MappingAgent(api_key, url, prompts_dir)

        # Create namd_setup directory for topology files
        namd_setup_dir = "./namd_setup"
        os.makedirs(namd_setup_dir, exist_ok=True)

        self.bead_mapping_agent = BeadMappingAgent(api_key, url, prompts_dir)
        self.topology_agent = TopologyCreatorAgent(api_key, url, namd_setup_dir, self.prompts_dir)
        self.boundary_agent = BoundaryAgent(api_key, url, prompts_dir)
        self.diagnostic_agent = DiagnosticAgent(api_key, url, prompts_dir, boundary_agent=self.boundary_agent)
        self.hypothesis_agent = HypothesisAgent(api_key, url, prompts_dir)
        self.optimization_agent = OptimizationAgent(api_key, url, prompts_dir, nforks=nforks)
        
        # State tracking
        self.optimization_state = None
        self.mapping_scheme = None
        self.boundaries = None
        self.actions_log = {"iterations": []}
        self.diagnostic_reports = []

    def _create_simulation_dirs(self):
        base = os.path.join(self.output_dir, "Simulation_Runs")
        os.makedirs(base, exist_ok=True)

        for T in self.temperatures:
            tdir = os.path.join(base, f"{T}K")
            os.makedirs(tdir, exist_ok=True)

            for fork in range(self.nforks):
                os.makedirs(os.path.join(tdir, f"fork_{fork}"), exist_ok=True)
                
    def _extract_json_from_response(self, text: str) -> Optional[Dict]:
        """Extract JSON from response text, handling markdown code blocks"""
        # Remove markdown code blocks if present
        text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
        text = text.strip()
        
        # Try to find JSON object in text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Try parsing entire text as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
        
    def initialize_system(self, molecule_info: Dict, 
                         aa_reference: Optional[Dict] = None,
                         use_existing_mapping: Optional[MappingScheme] = None) -> bool:
        """Initialize the optimization system"""
        
        print("\n" + "="*80)
        print("INITIALIZING MULTI-AGENT SYSTEM")
        print("Order: Mapping → Topology → Boundary → Initial Parameters")
        print(f"LLM Mode: {'Completions API' if self.use_completions else 'Chat API'}")
        print("="*80)

        print("\n[INIT] Creating Simulation_Runs directory structure...")
        self._create_simulation_dirs()

        raw_mapping = None
        # Step 1: Decide mapping scheme
        if use_existing_mapping:
            print("\n[1/4] Using provided mapping scheme...")
            self.mapping_scheme = use_existing_mapping
        else:
            # Check if mapping scheme file already exists
            mapping_file = os.path.join(self.output_dir, "mapping_scheme.json")
            if os.path.exists(mapping_file):
                print("\n[1/4] Using existing mapping scheme from file")
                try:
                    with open(mapping_file, 'r') as f:
                        mapping_data = json.load(f)
                    raw_mapping = MappingScheme(**mapping_data)
                    print(f"  ✓ Loaded mapping with {len(raw_mapping.bead_types)} bead types")
                    print(f"  ✓ Dummy beads: {', '.join(raw_mapping.dummy_beads) if raw_mapping.dummy_beads else 'None'}")
                except Exception as e:
                    print(f"  ✗ Error loading existing mapping: {e}")
                    self.mapping_scheme = None
            else:
                self.mapping_scheme = None

            # If no existing mapping, try to get from agent
            if not raw_mapping:
                print("\n[1/4] Mapping Agent: Proposing CG mapping scheme...")
                raw_mapping = self.mapping_agent.propose_mapping(molecule_info)

                print(f"  ✓ Proposed {len(raw_mapping.bead_types)} bead types")
                print(f"  ✓ Bead types: {', '.join(raw_mapping.bead_types)}")
                print(f"  ✓ Dummy beads: {', '.join(raw_mapping.dummy_beads) if raw_mapping.dummy_beads else 'None'}")

                # Save mapping scheme
                with open(os.path.join(self.output_dir, "mapping_scheme_raw.json"), 'w') as f:
                    json.dump(asdict(raw_mapping), f, indent=2)

            # --------------------------------------------------
            # Bead Mapping Agent (POST-PROCESSING STEP)
            # --------------------------------------------------
            print("  → Bead Mapping Agent: Assigning chemical bead names from existing library...")

            self.bead_mapping_agent.load_cdhm_dict("cdhm_dict.json")
            self.bead_mapping_agent.mapping_scheme = asdict(raw_mapping)

            match_result = self.bead_mapping_agent.match_beads()

            if match_result and "name_mapping" in match_result:
                updated_mapping_dict = self.bead_mapping_agent.apply_name_mapping(
                    match_result["name_mapping"],
                    match_result["matches"]
                )
                self.mapping_scheme = MappingScheme(**updated_mapping_dict)
                print("  ✓ Bead-aware bead naming applied")

                # Save updated mapping scheme
                with open(os.path.join(self.output_dir, "mapping_scheme.json"), 'w') as f:
                    json.dump(asdict(self.mapping_scheme), f, indent=2)
            else:
                print("  ⚠ Bead mapping failed — using generic bead names")
                self.mapping_scheme = raw_mapping
        
        if len(set(self.mapping_scheme.bead_types)) != len(self.mapping_scheme.bead_types):
            raise ValueError("Bead mapping produced duplicate bead names — aborting")
        
        # Step 2: Create initial topology
        print("\n[2/4] Topology Agent: Creating NAMD input files...")
        self.topology_agent.set_mapping_scheme(self.mapping_scheme)

        # Use default parameters for topology creation (will be replaced with template placeholders)
        initial_params = {}

        # Load molecule info for topology creation
        with open("molecular_info/molecule_info.json", 'r') as f:
            molecule_info = json.load(f)

        # Check for AA PDB file
        aa_pdb_candidates = ["molecular_info/AA.pdb", "molecular_info/aa.pdb"]
        aa_pdb_path = None
        for candidate in aa_pdb_candidates:
            if os.path.exists(candidate):
                aa_pdb_path = candidate
                break

        if not aa_pdb_path:
            print("  WARNING: No all-atom PDB file found. Looking for AA structure...")
            # Try to find any PDB that might contain AA structure
            pdb_files = [f for f in os.listdir('.') if f.endswith('.pdb')]
            for pdb_file in pdb_files:
                try:
                    with open(pdb_file, 'r') as f:
                        lines = f.readlines()[:50]  # Check first 50 lines
                        atom_count = sum(1 for line in lines if line.startswith('ATOM'))
                        if atom_count > 10:  # Assume AA if more than 10 atoms
                            aa_pdb_path = pdb_file
                            print(f"  Found potential AA PDB: {pdb_file} ({atom_count} atoms)")
                            break
                except:
                    continue

        if not aa_pdb_path:
            print("  ERROR: No suitable all-atom PDB file found for topology creation")
            print("  Please provide an AA PDB file (e.g., molecular_info/AA.pdb) with all-atom coordinates")
            print("  Skipping topology creation - assuming NAMD_input.psf and NAMD_input.pdb already exist")
            return True  # Continue without topology creation

        print(f"  Using AA PDB: {aa_pdb_path}")

        # Create NAMD inputs
        try:
            topology_results = self.topology_agent.create_namd_inputs(
                aa_pdb=aa_pdb_path,
                parameters=initial_params,
                molecule_name=molecule_info.get('name', 'DMA'),
                n_molecules=None,  # Auto-calculate
                box_size=50.0,
                density=molecule_info.get('targets', {}).get('Density', 1.095)
            )

            if not topology_results.get('psf_file') or not topology_results.get('pdb_file'):
                print("  ERROR: Topology agent failed to create NAMD input files")
                print("  Topology results:", topology_results)
                return False

            print(f"  ✓ Created PSF: {topology_results['psf_file']}")
            print(f"  ✓ Created PDB: {topology_results['pdb_file']}")

            # Save topology results
            with open(os.path.join(self.output_dir, "topology_results.json"), 'w') as f:
                json.dump(topology_results, f, indent=2)

        except Exception as e:
            print(f"  ERROR: Exception during topology creation: {e}")
            print("  Skipping topology creation - assuming existing NAMD files")
            # Continue with initialization assuming existing files
        
        # Step 3: Set parameter boundaries
        print("\n[3/4] Boundary Agent: Setting parameter boundaries...")

        # Check if boundaries file already exists
        boundaries_file = os.path.join(self.output_dir, "parameter_boundaries.json")
        if os.path.exists(boundaries_file):
            print("  Using existing parameter boundaries from file")
            try:
                with open(boundaries_file, 'r') as f:
                    boundaries_data = json.load(f)
                self.boundaries = ParameterBoundary(**boundaries_data)
                print(f"  ✓ Loaded {len(self.boundaries.var_names)} parameters from existing file")
            except Exception as e:
                print(f"  ✗ Error loading existing boundaries: {e}")
                self.boundaries = None
        else:
            self.boundaries = None

        # If no existing boundaries, try to get from agent
        if not self.boundaries:
            print("  Setting parameter boundaries based on chemical intuition...")
            self.boundaries = self.boundary_agent.set_boundaries(
                self.mapping_scheme, molecule_info, aa_reference
            )

        if not self.boundaries:
            print("ERROR: Boundary agent failed to set boundaries")
            return False

        print(f"  ✓ Defined {len(self.boundaries.var_names)} parameters")
        print(f"  ✓ Parameters: {', '.join(self.boundaries.var_names)}")

        # Save boundaries
        with open(os.path.join(self.output_dir, "parameter_boundaries.json"), 'w') as f:
            json.dump(asdict(self.boundaries), f, indent=2)
        
        # Step 4: Create initial simulation.prm with boundary values
        print("\n[4/4] Creating initial simulation.prm with boundary values...")
        initial_params = {name: val for name, val in zip(
            self.boundaries.var_names, self.boundaries.recommended_start
        )}

        # Save initial parameters
        with open(os.path.join(self.output_dir, "params.json"), 'w') as f:
            json.dump({"iteration": 0, "parameters": initial_params}, f, indent=2)
        
        # Initialize optimization state
        self.optimization_state = OptimizationState(
            iteration=0,
            phase="exploration",
            best_score=float('inf'),
            best_params={},
            recent_scores=[],
            stuck_counter=0,
            crash_regions=[]
        )
        self.boundary_hit_counts = {}

        print("\n✓ System initialization complete!")
        return True
    
    def run_optimization(self, targets_by_temp: Dict[str, Dict], num_iterations: int = 1000,
                        simulation_func=None, skip_completed: bool = True) -> Dict:
        """
        Run optimization with temperature-specific targets
        
        Args:
            targets_by_temp: Dict mapping temperature strings (e.g., "298K") to target dictionaries
            num_iterations: Number of optimization iterations
            simulation_func: Optional custom simulation function
            skip_completed: If True, skip iterations already in actions_log
        """

        if not self.optimization_state:
            raise RuntimeError("System not initialized. Call initialize_system() first.")

        assert self.optimization_state is not None
        assert self.boundaries is not None

        # Track completed iterations for restart
        completed_iters = set()
        if skip_completed and self.actions_log.get('iterations'):
            completed_iters = {it['iteration'] for it in self.actions_log['iterations']}
            if completed_iters:
                print(f"\n⚠️  Restart mode: Skipping {len(completed_iters)} already completed iterations")
                print(f"    Completed: {sorted(completed_iters)}")

        print("\n" + "="*80)
        print("STARTING OPTIMIZATION")
        print(f"LLM Mode: {'Completions API' if self.use_completions else 'Chat API'}")
        print(f"Temperatures: {self.temperatures}")
        print(f"Forks per temperature: {self.nforks}")
        print("\nTemperature-Specific Targets:")
        for temp_str, targets in targets_by_temp.items():
            print(f"  {temp_str}: {targets}")
        print("="*80)

        i = -1
        for i in range(num_iterations):
            current_iter = i + 1
            
            # Skip already completed iterations in restart mode
            if skip_completed and current_iter in completed_iters:
                continue
            
            self.optimization_state.iteration = current_iter

            # Check if this is a genetic generation boundary
            is_genetic_iteration = (i + 1) > 0 and (i + 1) % 20 == 0

            if is_genetic_iteration:
                generation_num = (i + 1) // 20
                print(f"\n{'='*80}")
                print(f"🧬 GENETIC GENERATION {generation_num} - ITERATION {i+1}")
                print(f"{'='*80}")

                # Get best 2 parents from memory
                parents = self._get_best_parents_from_memory(n=2)

                if len(parents) >= 2:
                    # Create new GA-focused boundaries with generous buffer
                    buffer_fraction = 0.30

                    new_boundaries = self.boundary_agent.get_genetic_boundaries(
                        parents=parents,
                        global_boundaries=self.boundaries,
                        buffer_fraction=buffer_fraction
                    )

                    # Update boundaries for next 20 iterations
                    self.boundaries = new_boundaries

                    print(f"  ✅ Created GA boundaries with {buffer_fraction*100:.0f}% buffer")
                    print(f"  📍 Parent 1: score={parents[0]['score']:.4f} (iter {parents[0]['iteration']})")
                    print(f"  📍 Parent 2: score={parents[1]['score']:.4f} (iter {parents[1]['iteration']})")

                    # Save new boundaries
                    with open(os.path.join(self.output_dir, "parameter_boundaries.json"), 'w') as f:
                        json.dump(asdict(self.boundaries), f, indent=2)
                else:
                    print(f"  ⚠️ Not enough parents for GA, using current boundaries")

            print(f"\n{'='*80}")
            print(f"ITERATION {i+1}/{num_iterations} - Phase: {self.optimization_state.phase.upper()}")
            print(f"{'='*80}")

            # Update optimization phase
            self._update_phase(i, num_iterations)
            
            # Get diagnostic from previous iteration (if not first)
            diagnostic = None
            if i > 0 and hasattr(self, 'last_results'):
                print("\n[1/5] Diagnostic Agent: Analyzing previous results...")
                diagnostic = self.diagnostic_agent.diagnose_system(
                    iteration=i,
                    results_by_temp=self.last_results,               # Dict[temp → properties]
                    targets_by_temp=targets_by_temp,
                    params=self.last_params,
                    trajectory_stats=crash_info,
                    current_boundaries=self.boundaries,
                    best_params_list=self._get_best_params_list(),
                    boundary_hit_counts=self.boundary_hit_counts,
                    recent_scores=self.optimization_state.recent_scores,
                    stuck_counter=self.optimization_state.stuck_counter,
                    all_fork_results_by_temp=self.last_all_results,  # Dict[temp → List[fork results]]
                    all_fork_params=self.last_all_params,
                    all_fork_scores=self.last_all_scores             # List of composite scores per fork
                )
                if diagnostic:
                    print(f"  Phase State: {diagnostic.phase_state}")
                    print(f"  Confidence: {diagnostic.confidence_score:.2f}")
                    if diagnostic.warnings:
                        print(f"  ⚠ Warnings: {len(diagnostic.warnings)}")
                    print(f"  Recommendations: {len(diagnostic.recommendations)}")

                    # Save diagnostic
                    self.actions_log["iterations"][-1]["diagnostic"] = asdict(diagnostic)
                    self.diagnostic_reports.append(asdict(diagnostic))

                    # Save diagnostic reports to separate JSON file
                    with open(os.path.join(self.output_dir, "diagnostic_reports.json"), 'w') as f:
                        json.dump(self.diagnostic_reports, f, indent=2)

                    # Check for boundary adjustment (NO 15-EPOCH RESTRICTION!)
                    if diagnostic.boundary_adjustment:
                        print(f"  🔄 Diagnostic agent adjusting boundaries")
                        self.boundaries = diagnostic.boundary_adjustment
                        self.boundary_hit_counts = {}  # Reset counters

                        # Save updated boundaries
                        with open(os.path.join(self.output_dir, "parameter_boundaries.json"), 'w') as f:
                            json.dump(asdict(self.boundaries), f, indent=2)
            else:
                # Create dummy diagnostic for first iteration
                diagnostic = DiagnosticReport(
                    iteration=0,
                    phase_state="unknown",
                    density_assessment="Not yet evaluated",
                    hvap_assessment="Not yet evaluated",
                    surface_tension_assessment="Not yet evaluated",
                    warnings=[],
                    recommendations=["Start with recommended initial parameters"],
                    confidence_score=0.5
                )
            
            # Generate hypothesis, then propose parameters
            print(f"\n[2/5] Hypothesis Agent: Generating scientific hypothesis...")
            hypothesis = None
            if diagnostic is not None and self.boundaries is not None:
                if is_genetic_iteration and len(self._get_best_parents_from_memory(n=2)) >= 2:
                    parents_for_hypothesis = self._get_best_parents_from_memory(n=2)
                    generation_num = (i + 1) // 20
                    hypothesis = self.hypothesis_agent.generate_genetic_hypothesis(
                        generation=generation_num,
                        iteration=i + 1,
                        parents=parents_for_hypothesis,
                        state=self.optimization_state,
                        boundaries=self.boundaries,
                        targets=targets_by_temp,
                    )
                else:
                    memory_context = self.optimization_agent.memory.get_context_message(self.optimization_state.iteration)
                    hypothesis = self.hypothesis_agent.generate_hypothesis(
                        state=self.optimization_state,
                        diagnostic=diagnostic,
                        boundaries=self.boundaries,
                        targets=targets_by_temp,
                        memory_context=memory_context,
                    )

            if hypothesis:
                print(f"  ✓ Hypothesis: {hypothesis.get('scientific_rationale', '')[:100]}...")
            else:
                print("  ⚠ Hypothesis generation failed — optimization agent will proceed without it")

            print(f"\n[3/5] Optimization Agent: Proposing parameters...")
            proposal = None
            params_list = None
            max_regeneration_attempts = 3

            if diagnostic is None or self.boundaries is None:
                print("  ✗ Missing diagnostic or boundaries, using best known parameters")
                params_list = [self.optimization_state.best_params.copy()] * self.nforks
            else:
                original_proposal = None

                for attempt in range(max_regeneration_attempts + 1):
                    if is_genetic_iteration and len(self._get_best_parents_from_memory(n=2)) >= 2:
                        # Use genetic crossover
                        parents = self._get_best_parents_from_memory(n=2)
                        generation_num = (i + 1) // 20
                        proposal = self.optimization_agent.propose_genetic_offspring(
                            generation=generation_num,
                            iteration=i+1,
                            parents=parents,
                            genetic_boundaries=self.boundaries,
                            global_boundaries=self.boundaries,
                            targets=targets_by_temp,  # Pass all temperature-specific targets
                            state=self.optimization_state,
                            hypothesis=hypothesis,
                        )
                    else:
                        # Normal gradient optimization
                        proposal = self.optimization_agent.propose_parameters(
                            state=self.optimization_state,
                            diagnostic=diagnostic,
                            boundaries=self.boundaries,
                            targets=targets_by_temp,  # Pass all temperature-specific targets
                            hypothesis=hypothesis,
                        )
                    # print(f"  → Proposal attempt {attempt + 1}: {proposal}")

                    if not proposal or "action" not in proposal:
                        print(f"  ✗ Proposal attempt {attempt + 1} failed (no action), trying again...")
                        continue

                    # Extract fork parameters from proposal
                    candidate_action = proposal["action"]
                    
                    if self.nforks > 1:
                        # Multi-fork mode: extract lists
                        fork_params_list = []
                        for fork_idx in range(self.nforks):
                            fork_params = {}
                            all_valid = True
                            for param_name, values in candidate_action.items():
                                if isinstance(values, list) and len(values) == self.nforks:
                                    fork_params[param_name] = values[fork_idx]
                                else:
                                    print(f"  ERROR: Parameter {param_name} doesn't have {self.nforks} values")
                                    all_valid = False
                                    break
                            if not all_valid:
                                fork_params_list = None
                                break
                            fork_params_list.append(fork_params)
                        
                        if fork_params_list is None or len(fork_params_list) != self.nforks:
                            print(f"  ✗ Proposal attempt {attempt + 1} failed (invalid fork structure)")
                            if attempt == 0:
                                original_proposal = proposal.copy()
                            if attempt >= max_regeneration_attempts:
                                print(f"  ✗ All attempts failed, using best known for all forks")
                                params_list = [self.optimization_state.best_params.copy()] * self.nforks
                                proposal = original_proposal
                                break
                            continue
                        else:
                            params_list = fork_params_list
                    else:
                        # Single fork mode: use params directly
                        params_list = [candidate_action]
                    
                    # Validate all forks
                    all_valid = True
                    all_violations = []
                    for fork_idx, fork_params in enumerate(params_list):
                        is_valid, violations = self._validate_parameters(fork_params)
                        if not is_valid:
                            all_valid = False
                            all_violations.extend([f"Fork {fork_idx}: {v}" for v in violations])
                    
                    if all_valid:
                        print(f"  ✓ Valid parameters proposed for all {len(params_list)} forks (attempt {attempt + 1})")
                        break
                    else:
                        violation_msg = "; ".join(all_violations[:5])  # Show first 5
                        print(f"  ⚠ Proposal attempt {attempt + 1} failed validation: {violation_msg}")

                        if attempt == 0:
                            original_proposal = proposal.copy()
                            proposal["validation_failures"] = all_violations

                        if attempt >= max_regeneration_attempts:
                            print(f"  ✗ All {max_regeneration_attempts + 1} attempts failed, using best known")
                            params_list = [self.optimization_state.best_params.copy()] * self.nforks
                            proposal = original_proposal
                            break

                if params_list is None:
                    params_list = [self.optimization_state.best_params.copy()] * self.nforks

            # Log action (log all forks' params for compatibility)
            self.actions_log["iterations"].append({
                "iteration": i + 1,
                "phase": self.optimization_state.phase if self.optimization_state else "unknown",
                "hypothesis": hypothesis,
                "proposal": proposal,
                "parameters": params_list,  # Log all forks
                "all_fork_parameters": params_list if self.nforks > 1 else None
            })

            # Create fork directories and save parameters for all temperatures
            for T in self.temperatures:
                for fork_idx, fork_params in enumerate(params_list):
                    if self.nforks > 1:
                        fork_dir = os.path.join(self.output_dir, f"Simulation_Runs/{T}K/fork_{fork_idx}")
                        os.makedirs(fork_dir, exist_ok=True)
                        params_file = os.path.join(fork_dir, "params.json")
                    else:
                        # Single fork - save in temperature directory
                        fork_dir = os.path.join(self.output_dir, f"Simulation_Runs/{T}K")
                        params_file = os.path.join(fork_dir, "params.json")
                    
                    with open(params_file, 'w') as f:
                        json.dump({"iteration": i+1, "fork": fork_idx, "temperature": T, "parameters": fork_params}, f, indent=2)

            # Update params.json in root for backward compatibility
            with open(os.path.join(self.output_dir, "params.json"), 'w') as f:
                json.dump({"iteration": i+1, "parameters": params_list}, f, indent=2)

            # Create simulation.prm from template for each temperature and fork
            try:
                for T in self.temperatures:
                    result = subprocess.run(
                        ["python3", "update_params.py", f"{self.nforks}", f"{T}"],
                        capture_output=True,
                        text=True,
                        cwd=self.output_dir if os.path.exists(os.path.join(self.output_dir, "update_params.py")) else "."
                    )
                    if result.returncode == 0:
                        print(f"  ✓ Created simulation.prm for {T}K")
                    else:
                        print(f"  ✗ Failed to create simulation.prm for {T}K: {result.stderr}")
            except Exception as e:
                print(f"  ✗ Error creating simulation.prm: {e}")
            
            # Run simulation
            print(f"\n[4/5] Running {'parallel ' if self.nforks > 1 else ''}simulation at {len(self.temperatures)} temperature(s)...")
            
            if simulation_func:
                # Custom simulation function - need to adapt for multi-temp
                results_list = [simulation_func(p) for p in params_list]
            else:
                # Run simulations for all temperatures
                all_temp_results = {T: [] for T in self.temperatures}
                all_temp_scores = {T: [] for T in self.temperatures}

                # Run parallel simulations
                os.system(f"bash run_parallel.sh")
                
                for T in self.temperatures:
                    temp_str = f"{T}K"
                    temp_targets = targets_by_temp.get(temp_str, {})
                    
                    print(f"  🌡 Running simulations at {T} K (targets: {temp_targets})")
                    
                    if self.nforks > 1:
                        for fork_idx in range(self.nforks):
                            result = self._obtain_score(
                                f"Simulation_Runs/{T}K/fork_{fork_idx}/result.dat"
                            )
                            all_temp_results[T].append(result)

                            # Compute scalar score per temperature using temperature-specific targets
                            if result.get("Density", 0) == 10000:
                                score_T = 10000
                            else:
                                score_T = sum(
                                    abs(result.get(name, 0) - target) / target * 100
                                    for name, target in temp_targets.items() if target != 0
                                )
                            all_temp_scores[T].append(score_T)
                    else:
                        # Single fork - run for this temperature
                        os.system(f"bash run.sh {T}")
                        result = self._obtain_score(f"Simulation_Runs/{T}K/result.dat")
                        all_temp_results[T].append(result)
                        
                        if result.get("Density", 0) == 10000:
                            score_T = 10000
                        else:
                            score_T = sum(
                                abs(result.get(name, 0) - target) / target * 100
                                for name, target in temp_targets.items() if target != 0
                            )
                        all_temp_scores[T].append(score_T)
                
                # Aggregate scores across temperatures (average all temperature scores)
                scores = []
                for fork_idx in range(self.nforks):
                    # Average scores across all temperatures
                    fork_scores = [all_temp_scores[T][fork_idx] for T in self.temperatures]
                    scores.append(np.mean(fork_scores))
                
                # Store temperature-specific results (don't average properties)
                results_list = all_temp_results

            # Find best fork (based on averaged scores across temperatures)
            best_fork_idx = int(np.argmin(scores))
            best_score = scores[best_fork_idx]
            best_params = params_list[best_fork_idx]
            
            # Get temperature-specific results for best fork
            best_results_by_temp = {T: all_temp_results[T][best_fork_idx] for T in self.temperatures}

            if self.nforks > 1 or len(self.temperatures) > 1:
                print(f"\n  🏆 Best fork: {best_fork_idx} with average score {best_score:.4f}")
                for fork_idx, s in enumerate(scores):
                    status = "✓" if fork_idx == best_fork_idx else " "
                    print(f"     {status} Fork {fork_idx}: {s:.4f}")

            # Store for diagnostic agent - use dictionary with temperature-specific results
            self.last_results = best_results_by_temp
            self.last_params = best_params
            self.last_all_results = all_temp_results  # Store all temperature-fork results
            self.last_all_params = params_list        # Store all fork params
            self.last_all_scores = scores             # Store all averaged scores
            self.last_temp_scores = all_temp_scores   # Store temperature-specific scores
            
            # Handle crashes
            crash_info = None
            if best_score >= 10000:
                print(f"  ✗ CRASH detected in best fork")
                self.optimization_state.crash_regions.append(best_params.copy())
                # Bound crash regions to last 10
                if len(self.optimization_state.crash_regions) > 10:
                    self.optimization_state.crash_regions = self.optimization_state.crash_regions[-10:]
                
                # Collect crash information from the first temperature
                T_first = self.temperatures[0]
                if self.nforks > 1:
                    crash_info = self._get_crash_info(f"Simulation_Runs/{T_first}K/fork_{best_fork_idx}/output.log")
                else:
                    crash_info = self._get_crash_info(f"Simulation_Runs/{T_first}K/output.log")
                
                # Re-run diagnostic with crash info and ALL fork results
                if diagnostic:
                    diagnostic = self.diagnostic_agent.diagnose_system(
                        iteration=i,
                        results_by_temp=best_results_by_temp,
                        targets_by_temp=targets_by_temp,
                        params=best_params,
                        trajectory_stats=crash_info,
                        current_boundaries=self.boundaries,
                        best_params_list=self._get_best_params_list(),
                        boundary_hit_counts=self.boundary_hit_counts,
                        recent_scores=self.optimization_state.recent_scores,
                        # NEW: Pass all fork results for diagnostic analysis
                        all_fork_results_by_temp=all_temp_results if self.nforks > 1 else None,
                        all_fork_params=params_list if self.nforks > 1 else None,
                        all_fork_scores=scores if self.nforks > 1 else None
                    )
                    # Update diagnostic in logs and save
                    self.actions_log["iterations"][-1]["diagnostic"] = asdict(diagnostic)
                    try:
                        self.diagnostic_reports[-1] = asdict(diagnostic)
                    except IndexError:
                        self.diagnostic_reports.append(asdict(diagnostic))
                    
                    with open(os.path.join(self.output_dir, "diagnostic_reports.json"), 'w') as f:
                        json.dump(self.diagnostic_reports, f, indent=2)
            else:
                # Success - display results for each temperature
                print(f"\n[5/5] Results (Best Fork {best_fork_idx}):")
                
                for T in self.temperatures:
                    temp_str = f"{T}K"
                    temp_targets = targets_by_temp.get(temp_str, {})
                    temp_results = best_results_by_temp[T]
                    
                    print(f"\n  Temperature: {temp_str}")
                    for name, value in temp_results.items():
                        target_val = temp_targets.get(name, 0)
                        if target_val != 0:
                            pct_dev = abs(value - target_val) / target_val * 100
                            print(f"    {name}: {value:.4f} (target: {target_val:.4f}, dev: {pct_dev:.2f}%)")
                        else:
                            dev = abs(value - target_val)
                            print(f"    {name}: {value:.4f} (target: {target_val:.4f}, dev: {dev:.4f})")
                    
                    # Show temperature-specific score
                    temp_score = all_temp_scores[T][best_fork_idx]
                    print(f"    Score at {temp_str}: {temp_score:.4f}")
                
                print(f"\n  Average Composite Score: {best_score:.4f} (averaged across temperatures)")
    
                # Check for boundary hits (using any temperature's results)
                T_first = self.temperatures[0]
                for name in self.boundaries.var_names:
                    idx = self.boundaries.var_names.index(name)
                    if best_params.get(name) == self.boundaries.min_var[idx] or \
                       best_params.get(name) == self.boundaries.max_var[idx]:
                        self.boundary_hit_counts[name] = self.boundary_hit_counts.get(name, 0) + 1
    
                # Update state
                self.optimization_state.recent_scores.append(best_score)
                if len(self.optimization_state.recent_scores) > 10:
                    self.optimization_state.recent_scores.pop(0)
                
                # Check for improvement
                if best_score < self.optimization_state.best_score:
                    self.optimization_state.best_score = best_score
                    self.optimization_state.best_params = best_params.copy()
                    self.optimization_state.stuck_counter = 0
                    print(f"\n  ★ NEW BEST SCORE: {best_score:.4f}")
                else:
                    self.optimization_state.stuck_counter += 1
                
                # Add scores to log
                self.actions_log["iterations"][-1]["results_by_temperature"] = {
                    f"{T}K": best_results_by_temp[T] for T in self.temperatures
                }
                self.actions_log["iterations"][-1]["composite_score"] = best_score
                self.actions_log["iterations"][-1]["best_fork_idx"] = best_fork_idx
                self.actions_log["iterations"][-1]["temperature_scores"] = {
                    f"{T}K": all_temp_scores[T] for T in self.temperatures
                }
                if self.nforks > 1:
                    self.actions_log["iterations"][-1]["all_fork_results_by_temp"] = all_temp_results
                    self.actions_log["iterations"][-1]["all_fork_scores"] = scores
    
                # Update optimization agent memory with ALL fork results
                if proposal:
                    self.optimization_agent.update_memory(i + 1, proposal, scores, params_list)
    
                # Save actions log
                with open(os.path.join(self.output_dir, "actions.json"), 'w') as f:
                    json.dump(self.actions_log, f, indent=2)
                
                # Check convergence across all temperatures
                converged = True
                for T in self.temperatures:
                    temp_str = f"{T}K"
                    temp_targets = targets_by_temp.get(temp_str, {})
                    temp_results = best_results_by_temp[T]
                    if not self._check_convergence(temp_results, temp_targets):
                        converged = False
                        break
                
                if converged:
                    print(f"\n🎉 CONVERGENCE ACHIEVED at iteration {i+1}!")
                    break
        
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"Best Score: {self.optimization_state.best_score:.4f}")
        print(f"Best Parameters:")
        print(json.dumps(self.optimization_state.best_params, indent=2))
        
        return {
            "best_score": self.optimization_state.best_score,
            "best_params": self.optimization_state.best_params,
            "best_results": self.last_results if self.last_results else {},
            "iterations": i + 1
        }
    
    def _update_phase(self, iteration: int, total_iterations: int):
        """Update optimization phase based on progress"""
        if self.optimization_state is None:
            return
        if iteration < total_iterations * 0.3:
            self.optimization_state.phase = "exploration"
        elif iteration < total_iterations * 0.7:
            self.optimization_state.phase = "refinement"
        else:
            self.optimization_state.phase = "convergence"
    
    def _validate_parameters(self, params: Dict) -> Tuple[bool, List[str]]:
        """Validate parameter values against boundaries

        Returns:
            (is_valid: bool, violation_messages: List[str])
        """
        if self.boundaries is None:
            return True, []  # No boundaries to check

        violations = []
        for name in self.boundaries.var_names:
            if name not in params:
                violations.append(f"Missing parameter: {name}")
                continue

            value = params[name]
            idx = self.boundaries.var_names.index(name)
            min_val = self.boundaries.min_var[idx]
            max_val = self.boundaries.max_var[idx]

            if not (min_val <= value <= max_val):
                direction = "above" if value > max_val else "below"
                bound = max_val if value > max_val else min_val
                diff = abs(value - bound)
                violations.append(
                    f"{name}: {value:.4f} is {direction} bound {bound:.4f} by {diff:.4f}"
                )

        return len(violations) == 0, violations
    
    def _check_convergence(self, results: Dict, targets: Dict) -> bool:
        """Check if optimization has converged"""
        # Check if all properties within 1 std (assuming std = 5% of target)
        for name, target in targets.items():
            std = target * 0.05  # 5% tolerance
            if abs(results.get(name, 0) - target) > std:
                return False
        return True

    def _get_best_parents_from_memory(self, n=2) -> List[Dict]:
        """Get top N parents from optimization memory/population"""

        # Option A: From optimization agent memory (if using population tracking)
        if hasattr(self.optimization_agent.memory, 'population'):
            population = self.optimization_agent.memory.population
            if len(population) >= n:
                sorted_pop = sorted(population, key=lambda x: x['score'])
                return sorted_pop[:n]

        # Option B: From actions log (fallback)
        valid_iterations = [
            {
                'params': iter_data['parameters'][0] if isinstance(iter_data['parameters'], list) else iter_data['parameters'],
                'score': iter_data.get('composite_score', float('inf')),
                'iteration': iter_data['iteration']
            }
            for iter_data in self.actions_log['iterations']
            if 'composite_score' in iter_data and iter_data['composite_score'] < 10000
        ]

        if len(valid_iterations) >= n:
            sorted_iters = sorted(valid_iterations, key=lambda x: x['score'])
            return sorted_iters[:n]

        return []

    def _get_best_params_list(self) -> List[Dict]:
        """Get list of best parameter sets from memory"""
        # For now, return the current best
        if self.optimization_state and self.optimization_state.best_params:
            return [self.optimization_state.best_params]
        return []
    
    def _obtain_score(self, file_path: str = "Simulation_Runs/result.dat") -> Dict:
        """Read simulation results from file"""
        result = {}
        try:
            with open(os.path.join(self.output_dir, file_path), 'r') as file:
                for line in file:
                    if line.strip():
                        key, value = line.strip().split(': ')
                        result[key] = float(value)
        except Exception as e:
            print(f"Error reading results from {file_path}: {e}")
            result = {"Density": 10000, "Heat_of_Vaporization": 10000, "Surface_Tension": 10000, "Dipole_Moment": 0}
        return result

    def _get_crash_info(self, log_file: str = "Simulation_Runs/output.log") -> Dict:
        """Extract crash information from log file"""
        crash_info = {"error_lines": [], "last_lines": []}
        log_path = os.path.join(self.output_dir, log_file)

        try:
            # Grep for Error lines
            import subprocess
            result = subprocess.run(['grep', 'Error', log_path],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                crash_info["error_lines"] = result.stdout.strip().split('\n')

            # Get last 10 lines
            result = subprocess.run(['tail', '-10', log_path],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                crash_info["last_lines"] = result.stdout.strip().split('\n')

        except Exception as e:
            print(f"Error reading crash info from {log_file}: {e}")

        return crash_info


# ============================================================================
# Usage
# ============================================================================

def main():
    """Usage of the multi-agent framework"""

    # Delete and recreate memories and llm_prompts directories for fresh run
    memories_dir = "memories"
    llm_prompts_dir = "llm_prompts"

    if os.path.exists(memories_dir):
        shutil.rmtree(memories_dir)
    if os.path.exists(llm_prompts_dir):
        shutil.rmtree(llm_prompts_dir)

    os.makedirs(memories_dir, exist_ok=True)
    os.makedirs(llm_prompts_dir, exist_ok=True)

    # API configuration for chat API
    API_KEY = 'YOUR_API_KEY_HERE'
    API_URL = "https://llm-api.arc.vt.edu/api/v1/chat/completions"

    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator(
        api_key=API_KEY,
        url=API_URL,
        output_dir=".",
        prompts_dir="prompts",
        nforks=8,
        temperatures=[298, 313]
    )

    # Load molecule information from files
    with open("molecular_info/molecule_info.json", 'r') as f:
        molecule_info = json.load(f)

    with open("molecular_info/aa_reference.json", 'r') as f:
        aa_reference = json.load(f)

    # Initialize system
    success = orchestrator.initialize_system(molecule_info, aa_reference)
    if not success:
        print("Failed to initialize system")
        return

    # Define temperature-specific targets from molecule properties
    targets_by_temp = {}
    for temp_str in ["298K", "313K"]:
        if temp_str in molecule_info["properties"]:
            targets_by_temp[temp_str] = molecule_info["properties"][temp_str]
    
    print(f"\nOptimization targets by temperature:")
    for temp_str, targets in targets_by_temp.items():
        print(f"  {temp_str}: {targets}")

    # Run optimization with temperature-specific targets
    results = orchestrator.run_optimization(targets_by_temp, num_iterations=1000)

    print("\nFinal Results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()