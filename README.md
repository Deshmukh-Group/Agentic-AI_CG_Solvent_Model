# CG Solvent - Multi-Agent Coarse-Grained Force Field Optimization Framework

A multi-agent AI-driven framework for optimizing coarse-grained (CG) force field parameters for molecular simulations using Large Language Model (LLM) agents.

## Overview

This framework leverages specialized LLM agents to automate the complex process of CG force field parameter optimization for molecular solvents. The system iteratively runs molecular dynamics simulations, analyzes results, and proposes parameter updates to match experimental properties.

### Key Features

- **Multi-Agent Architecture**: Coordinated agents for mapping, topology creation, boundary setting, diagnostics, and optimization
- **LLM-Powered Decision Making**: Uses Large Language Models for chemical intuition and parameter decisions
- **Automated MD Simulations**: Runs NAMD simulations with automated setup and execution
- **Property Matching**: Optimizes to match experimental density, heat of vaporization, and surface tension

## Architecture

### Agent Types

| Agent | Purpose |
|-------|---------|
| **Mapping Agent** | Decides CG mapping scheme and bead types from molecular structure |
| **Bead Mapping Agent** | Assigns atoms to CG beads based on chemical groups |
| **Topology Agent** | Constructs CG topology files (PSF) from mapping scheme |
| **Boundary Agent** | Sets parameter ranges based on chemical intuition |
| **Diagnostic Agent** | Analyzes simulation results and system behavior |
| **Hypothesis Agent** | Generates hypotheses for parameter adjustments using genetic algorithms |
| **Optimization Agent** | Proposes parameter updates based on diagnostic results |

### Data Flow

```
SMILES → Mapping Agent → Topology Agent → Boundary Agent
                                    ↓
                            NAMD Simulation
                                    ↓
                            Diagnostic Agent
                                    ↓
                            Hypothesis/Optimization Agent
                                    ↓
                            Parameter Updates (loop)
```

## Project Structure

```
CG_Solvent/
├── master_agent.py           # Main orchestrator coordinating all agents
├── common.py                 # Shared data structures and base classes
├── mapping_agent.py          # CG mapping scheme generation
├── bead_mapping_agent.py     # Atom-to-bead assignment
├── topology_creator_agent.py # PSF topology file generation
├── boundary_agent.py         # Parameter boundary/range setting
├── diagnostic_agent.py       # Simulation result analysis
├── hypothesis_agent.py       # Hypothesis generation with genetic algorithms
├── optimization_agent.py     # Parameter optimization logic
├── smiles_parser.py          # SMILES molecular structure parsing
├── analyze_AA2CG.py          # Analysis tools
├── extract_boundary_table.py # Boundary extraction utilities
├── update_params.py          # Parameter update utilities
├── mapping_scheme.json       # Current mapping scheme configuration
│
├── executables/
│   ├── packmol               # Molecular packing tool
│   └── psfgen                # PSF topology generator (NAMD)
│
├── molecular_info/
│   ├── molecule_info.json    # Target molecule properties
│   ├── aa_reference.json     # All-atom reference data
│   ├── AA.pdb                # All-atom structure
│   └── calc_313.dat          # Calculated properties at 313K
│
└── prompts/
    ├── *_system_prompt.txt   # System prompts for each agent
    └── *_prompt_template.txt # User prompt templates
```

## Installation

### Prerequisites

- Python 3.8+
- NAMD (for molecular dynamics simulations)
- OpenMM or access to NAMD executable
- LLM API key (OpenAI or compatible API)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/CG_Solvent.git
   cd CG_Solvent
   ```

2. Install dependencies:
   ```bash
   pip install numpy requests json-repair
   ```

3. Configure your LLM API key:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

4. Ensure NAMD and required executables are available in `executables/`

## Usage

### Basic Usage

Run the optimization with default settings:

```python
from master_agent import MultiAgentOrchestrator

orchestrator = MultiAgentOrchestrator(
    api_key="your-api-key",
    url="https://api.openai.com/v1/chat/completions",
    output_dir="./results",
    nforks=4,
    temperatures=[298, 313]
)

orchestrator.run_full_optimization(max_iterations=50)
```

### Configuration

Key parameters:
- `api_key`: Your LLM API key
- `url`: API endpoint (supports OpenAI-compatible APIs)
- `output_dir`: Directory for simulation results
- `nforks`: Number of parallel simulation forks
- `temperatures`: Simulation temperatures (K)

### Input Molecule

Configure your target molecule in `molecular_info/molecule_info.json`:

```json
{
    "name": "N,N-Dimethylacetamide (DMA)",
    "smiles": "CC(=O)N(C)C",
    "properties": {
        "298K": {
            "Density": 0.9361,
            "Heat_of_Vaporization": 10.951,
            "Surface_Tension": 32.43
        }
    }
}
```

## Output

The framework generates:
- `Simulation_Runs/` - MD simulation trajectories
- `namd_setup/` - Topology and parameter files
- `prev_mapping_scheme.json` - Current mapping scheme
- Optimization logs and diagnostic reports

### Diagnostic Reports

Each iteration produces a diagnostic report including:
- Phase state assessment (liquid/gas/solid/unstable)
- Density, heat of vaporization, and surface tension evaluations
- Warnings and recommendations
- Confidence score

## Extending the Framework

### Adding New Agents

1. Create a new agent class inheriting from `LLMAgent`
2. Define agent-specific prompts in `prompts/`
3. Register the agent in `MultiAgentOrchestrator`

### Custom Property Targets

Modify `molecular_info/molecule_info.json` to add experimental properties:
- Density (g/cm³)
- Heat of Vaporization (kcal/mol)
- Surface Tension (dyn/cm)
- Dipole Moment (D)

## Technical Details

### CG Mapping

The framework supports various mapping schemes:
- Single-site (united atom)
- Multiple-site (polarizable)
- Dummy bead representations

### Parameter Optimization

Uses a combination of:
- Bayesian optimization hints from LLM
- Genetic algorithm crossover for hypothesis generation
- Boundary-based constrained optimization

### Simulation Pipeline

1. Parse molecular structure from SMILES
2. Generate CG mapping scheme
3. Create topology (PSF) and coordinates
4. Set up NAMD simulation with initial parameters
5. Run MD simulation
6. Analyze trajectory for properties
7. Diagnose system behavior
8. Propose parameter updates
9. Repeat until convergence

## Requirements

- Python 3.8+
- numpy
- requests
- json-repair
- NAMD 2.x or 3.x
- OpenAI API key (or compatible LLM)

## License

MIT License

## Contributing

Contributions welcome! Please submit pull requests or open issues for:
- Bug fixes
- New agent types
- Additional property targets
- Performance improvements

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{cg_solvent,
  title = {CG Solvent: Multi-Agent CG Force Field Optimization},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/CG_Solvent}
}
```

## Support

- Issues: https://github.com/yourusername/CG_Solvent/issues
- Documentation: See inline code documentation