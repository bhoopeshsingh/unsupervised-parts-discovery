import json

notebook_path = 'notebooks/unsupervised_part_discovery_analysis.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Create a training cell
training_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Train Part Discovery Model\n",
        "# We re-train the model with the new stable configuration (frozen backbone, 6 slots)\n",
        "print(\"Starting Model Training... (This may take a while)\")\n",
        "!python experiments/train_part_discovery.py\n",
        "print(\"Training Completed!\")"
    ]
}

# Insert after imports (assuming cell 0 or 1 is imports)
# Let's verify where to insert. Usually after the first code cell.
insert_idx = 1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        insert_idx = i + 1
        break

nb['cells'].insert(insert_idx, training_cell)

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Added training cell to notebook at index {insert_idx}")
