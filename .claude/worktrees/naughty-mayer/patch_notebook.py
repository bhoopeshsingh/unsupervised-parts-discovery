
import json
import sys

notebook_path = 'notebooks/unsupervised_part_discovery_analysis.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "cluster_parts(" in source and "cluster_parts_per_class" not in source:
            print("Found global clustering cell. Replacing...")
            
            new_source = [
                "# Cluster the parts per class (to ensure purity)\n",
                "# We use per-class clustering to guarantee that each cluster contains only one object class\n",
                "cluster_labels, cluster_metadata, metrics = cluster_parts_per_class(\n",
                "    features=features,\n",
                "    class_labels=labels,\n",
                "    class_names=data_config['dataset']['classes'],\n",
                "    k_range=(5, 20),\n",
                "    n_init=3,\n",
                "    random_state=42\n",
                ")\n"
            ]
            cell['source'] = new_source
            found = True
            break

if found:
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print("Successfully patched notebook.")
else:
    print("Could not find the global clustering cell (or it's already patched).")
