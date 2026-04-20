import json
import re
import os
import sys

def update_params(params_file="params.json", output_dir=".", temperature=None):
    """Read cg_parameters.prm template, replace placeholders with values from params.json, 
    save as simulation.prm in the appropriate directory"""

    # Read parameters from params.json
    if not os.path.exists(params_file):
        print(f"Error: {params_file} not found")
        return False

    with open(params_file, 'r') as f:
        params_data = json.load(f)

    params = params_data.get("parameters", {})
    iteration = params_data.get("iteration", "unknown")
    fork_id = params_data.get("fork", None)
    temp_in_file = params_data.get("temperature", None)

    # Use temperature from file if not provided as argument
    if temperature is None:
        temperature = temp_in_file

    if fork_id is not None:
        if temperature is not None:
            print(f"Processing Fork {fork_id}, Temperature {temperature}K, Iteration {iteration}")
        else:
            print(f"Processing Fork {fork_id}, Iteration {iteration}")
    else:
        if temperature is not None:
            print(f"Processing Temperature {temperature}K, Iteration {iteration}")
        else:
            print(f"Processing Single Fork, Iteration {iteration}")
    
    print(f"Parameters read from {params_file}:")
    print(json.dumps(params, indent=2))

    # Read template file
    template_path = "namd_setup/cg_parameters.prm"
    if not os.path.exists(template_path):
        print(f"Error: Template file {template_path} not found")
        return False

    with open(template_path, 'r') as f:
        template_content = f.read()

    # Find all placeholders ${PARAM_NAME}
    placeholders = re.findall(r'\$\{([^}]+)\}', template_content)
    unique_placeholders = set(placeholders)

    print(f"Found {len(unique_placeholders)} unique placeholders in template")

    # Replace each placeholder with parameter value
    updated_content = template_content
    missing_params = []
    
    for placeholder in unique_placeholders:
        if placeholder in params:
            value = params[placeholder]
            updated_content = updated_content.replace(f"${{{placeholder}}}", str(value))
            print(f"  ${{{placeholder}}} -> {value}")
        else:
            missing_params.append(placeholder)
            print(f"  WARNING: No value found for parameter '${{{placeholder}}}'")

    if missing_params:
        print(f"\nWARNING: {len(missing_params)} parameters missing:")
        for param in sorted(missing_params):
            print(f"  - {param}")

    # Determine output path
    output_file = os.path.join(output_dir, "simulation.prm")
    
    # Write output file
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(updated_content)

    print(f"✓ Created {output_file} with substituted parameters")
    return True


def update_all_forks_and_temps(nforks=1, temperatures=None):
    """Update parameters for all forks and temperatures
    
    Args:
        nforks: Number of forks (parallel runs with different parameters)
        temperatures: List of temperatures (e.g., [298, 313]) or None for single temperature
    """
    
    if temperatures is None:
        # Try to auto-detect temperatures from directory structure
        if os.path.exists("Simulation_Runs"):
            temp_dirs = [d for d in os.listdir("Simulation_Runs") 
                        if d.endswith("K") and os.path.isdir(os.path.join("Simulation_Runs", d))]
            if temp_dirs:
                temperatures = [int(d[:-1]) for d in temp_dirs]
                print(f"Auto-detected temperatures: {temperatures}")
            else:
                temperatures = [298]  # Default
        else:
            temperatures = [298]
    
    print("="*80)
    print(f"UPDATING PARAMETERS FOR {nforks} FORK(S) AT {len(temperatures)} TEMPERATURE(S)")
    print(f"Temperatures: {temperatures}")
    print("="*80)
    
    success_count = 0
    total_updates = nforks * len(temperatures)
    
    for temp in temperatures:
        print(f"\n{'='*80}")
        print(f"TEMPERATURE: {temp}K")
        print(f"{'='*80}")
        
        for fork_idx in range(nforks):
            if nforks > 1:
                # Multi-fork mode: separate directories per temperature and fork
                fork_dir = f"Simulation_Runs/{temp}K/fork_{fork_idx}"
                params_file = os.path.join(fork_dir, "params.json")
                output_dir = fork_dir
                
                print(f"\n--- Temperature {temp}K, Fork {fork_idx} ---")
            else:
                # Single fork mode: separate directories per temperature only
                temp_dir = f"Simulation_Runs/{temp}K"
                params_file = os.path.join(temp_dir, "params.json")
                output_dir = temp_dir
                
                print(f"\n--- Temperature {temp}K, Single Fork ---")
            
            if update_params(params_file, output_dir, temperature=temp):
                success_count += 1
            else:
                print(f"✗ Failed to update fork {fork_idx} at {temp}K")
    
    print("\n" + "="*80)
    print(f"COMPLETED: {success_count}/{total_updates} updates successful")
    print("="*80)
    
    return success_count == total_updates


if __name__ == "__main__":
    # Parse arguments: python update_params.py [nforks] [temperature]
    # Examples:
    #   python update_params.py 8 298    # 8 forks at 298K only
    #   python update_params.py 8        # 8 forks, auto-detect temperatures
    #   python update_params.py          # auto-detect both
    
    nforks = 1
    temperatures = None
    
    if len(sys.argv) > 1:
        try:
            nforks = int(sys.argv[1])
        except ValueError:
            print(f"Error: Invalid nforks argument '{sys.argv[1]}', using default of 1")
            nforks = 1
    else:
        # Try to detect nforks from directory structure
        if os.path.exists("Simulation_Runs"):
            # Look for temperature directories first
            temp_dirs = [d for d in os.listdir("Simulation_Runs") 
                        if d.endswith("K") and os.path.isdir(os.path.join("Simulation_Runs", d))]
            
            if temp_dirs:
                # Count forks in first temperature directory
                first_temp_dir = os.path.join("Simulation_Runs", temp_dirs[0])
                fork_dirs = [d for d in os.listdir(first_temp_dir) 
                           if d.startswith("fork_") and os.path.isdir(os.path.join(first_temp_dir, d))]
                nforks = len(fork_dirs) if fork_dirs else 1
                
                if nforks > 1:
                    print(f"Auto-detected {nforks} forks from {temp_dirs[0]} directory")
    
    if len(sys.argv) > 2:
        # Single temperature provided
        try:
            temp = int(sys.argv[2])
            temperatures = [temp]
            print(f"Using single temperature: {temp}K")
        except ValueError:
            print(f"Error: Invalid temperature argument '{sys.argv[2]}', will auto-detect")
            temperatures = None
    
    success = update_all_forks_and_temps(nforks, temperatures)
    sys.exit(0 if success else 1)