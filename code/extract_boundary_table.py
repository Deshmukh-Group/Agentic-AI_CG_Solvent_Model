import json
import csv

def extract_boundary_table(actions_file, output_csv):
    """Extract boundary adjustments and genetic operations into a CSV table"""
    
    with open(actions_file, 'r') as f:
        data = json.load(f)

    # Initial boundaries
    initial_bounds = {
        'var_names': ['CSM_DUP_bl', 'CSM_DUP_kb', 'CSM_OCM_bl', 'CSM_OCM_epsilon', 'CSM_OCM_kb', 'CSM_OCM_rmin', 'CSM_epsilon', 'CSM_rminby2', 'OCM_DUN_bl', 'OCM_DUN_kb', 'OCM_epsilon', 'OCM_rminby2', 'DUP_charge', 'DUN_charge'],
        'min_var': [0.25, 76.4, 1.38, -0.99, 4.55, 2.92, -0.99, 1.41, 0.25, 76.4, -0.77, 1.78, 0.01, -0.51],
        'max_var': [0.87, 173.6, 1.83, -0.02, 30.45, 4.12, -0.02, 2.09, 0.87, 173.6, -0.02, 2.52, 0.51, -0.01]
    }

    rows = []
    prev_bounds = initial_bounds

    for iteration in data['iterations']:
        iter_num = iteration['iteration']
        
        # Check for genetic operations
        proposal = iteration.get('proposal', {})
        reasoning = proposal.get('reasoning', '')
        is_genetic = 'crossover' in reasoning.lower() or 'mutation' in reasoning.lower() or 'genetic' in reasoning.lower()
        
        # Check for boundary adjustments
        bounds = iteration.get('diagnostic', {}).get('boundary_adjustment')
        if bounds and bounds is not None:
            for i, param in enumerate(bounds['var_names']):
                old_min = prev_bounds['min_var'][i] if i < len(prev_bounds['min_var']) else 'N/A'
                old_max = prev_bounds['max_var'][i] if i < len(prev_bounds['max_var']) else 'N/A'
                new_min = bounds['min_var'][i]
                new_max = bounds['max_var'][i]
                
                if old_min != new_min or old_max != new_max:
                    change_type = []
                    if isinstance(old_min, (int, float)) and isinstance(old_max, (int, float)):
                        if new_min < old_min and new_max > old_max:
                            change_type.append('EXPANDED')
                        elif new_min < old_min:
                            change_type.append('EXPANDED_LOWER')
                        elif new_max > old_max:
                            change_type.append('EXPANDED_UPPER')
                        elif new_min > old_min and new_max < old_max:
                            change_type.append('CONTRACTED')
                        elif new_min > old_min or new_max < old_max:
                            change_type.append('SHIFTED')
                        else:
                            change_type.append('CHANGED')
                    else:
                        change_type.append('INITIAL')
                    
                    genetic_op = 'Yes' if is_genetic else 'No'
                    rows.append({
                        'Epoch': iter_num,
                        'Parameter': param,
                        'Adjustment Type': ' & '.join(change_type),
                        'Old Min': old_min,
                        'Old Max': old_max,
                        'New Min': new_min,
                        'New Max': new_max,
                        'Genetic Operation': genetic_op
                    })
            
            prev_bounds = bounds
        elif is_genetic:
            # Genetic operation without boundary adjustment
            rows.append({
                'Epoch': iter_num,
                'Parameter': 'All Parameters',
                'Adjustment Type': 'GENETIC_OPERATION',
                'Old Min': '-',
                'Old Max': '-',
                'New Min': '-',
                'New Max': '-',
                'Genetic Operation': 'Yes'
            })

    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Parameter', 'Adjustment Type', 'Old Min', 'Old Max', 'New Min', 'New Max', 'Genetic Operation']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    print(f"Extracted {len(rows)} boundary adjustment records to {output_csv}")

if __name__ == "__main__":
    extract_boundary_table('/home/swarnadeep/Scratch_Falcon/CG-Solvent/DMSO/Agentic_AI_2.9_V2/actions.json', 'boundary_adjustments.csv')