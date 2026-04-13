
import json
import sys

notebook_path = 'notebooks/unsupervised_part_discovery_analysis.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

found = False

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Look for the problematic block
        if "cluster_metadata = {" in source and "'n_clusters': n_clusters," in source:
            print("Found problematic metadata overwriting cell.")
            
            lines = cell['source']
            new_lines = []
            
            # We want to replace the re-definition with an update
            # But we need to be careful. 'cluster_metadata' might be a variable from previous scope
            # In the previous cell (or same cell), cluster_parts_per_class returns cluster_metadata.
            
            # Let's check if we can find where cluster_metadata comes from.
            # If this is a standalone cell, it might rely on global state.
            
            # Strategy:
            # Replace:
            # cluster_metadata = { ... }
            # with:
            # if 'cluster_metadata' not in locals(): cluster_metadata = {}
            # cluster_metadata.update({ ... })
            
            skip_block = False
            for line in lines:
                if "cluster_metadata = {" in line:
                    new_lines.append("# Update existing metadata instead of overwriting\n")
                    new_lines.append("if 'cluster_metadata' not in locals(): cluster_metadata = {}\n")
                    new_lines.append("cluster_metadata.update({\n")
                    skip_block = True
                elif skip_block and "}" in line:
                    new_lines.append("})\n")
                    skip_block = False
                else:
                    new_lines.append(line)
            
            cell['source'] = new_lines
            found = True
            break

if found:
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print("Successfully patched notebook metadata saving.")
else:
    print("Could not find cell to patch.")
