from semantic_utils import semantic_grounding

semantic_path = semantic_grounding(
    clusters_json="./runs/exp1/clusters/clusters.json",
    out_path="./runs/exp1/clusters/semantic_labels.json",
    candidate_labels=["head", "wing", "leg", "beak", "tail", "body"],
    top_k=5,
    human_confirm=False   # set True if you want manual confirmation
)

print("Semantic labels saved at:", semantic_path)