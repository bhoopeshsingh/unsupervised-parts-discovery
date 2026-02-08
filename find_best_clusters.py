#!/usr/bin/env python
"""Quick analysis to find best clusters"""
import numpy as np
import json

print('Loading data...')
masks = np.load('parts/extracted/masks.npy')
cluster_labels = np.load('parts/clusters/cluster_labels.npy')

with open('parts/clusters/cluster_metadata.json', 'r') as f:
    cluster_meta = json.load(f)

print(f'Total masks: {len(masks)}')
print(f'Total clusters: {len(cluster_meta)}')

results = []
for cid_str, meta in cluster_meta.items():
    cid = int(cid_str)
    size = meta['size']

    indices = np.where(cluster_labels == cid)[0][:30]
    if len(indices) < 5:
        continue

    cluster_masks = masks[indices]
    # Compute contrast (max - min) / mean as quality metric
    contrasts = [(m.max() - m.min()) / (m.mean() + 1e-8) for m in cluster_masks]
    contrast_mean = np.mean(contrasts)
    # How consistent are peak locations across masks in this cluster
    peak_positions = [np.unravel_index(np.argmax(m), m.shape) for m in cluster_masks]
    peak_y_std = np.std([p[0] for p in peak_positions])
    peak_x_std = np.std([p[1] for p in peak_positions])
    position_consistency = 1.0 / (1.0 + peak_y_std + peak_x_std)  # Higher = more consistent

    peak = np.mean([m.max() for m in cluster_masks])
    variance = np.mean([m.var() for m in cluster_masks])

    # Score: prioritize high contrast + consistent peak positions
    score = contrast_mean * 100 + position_consistency * 50 + variance * 10000

    results.append((cid, score, size, contrast_mean, peak, position_consistency))

results.sort(key=lambda x: x[1], reverse=True)

print()
print('='*70)
print('TOP 15 BEST CLUSTERS')
print('='*70)
print('Rank   Cluster   Score     Size      Contrast    Peak      PosConsist')
print('-'*75)
for i, (cid, score, size, contrast, peak, pos_cons) in enumerate(results[:15], 1):
    print(f'{i:<7}{cid:<10}{score:<10.2f}{size:<10}{contrast:<12.4f}{peak:<10.4f}{pos_cons:<10.4f}')

print()
print('TOP 10 cluster IDs to label:')
top_10 = [r[0] for r in results[:10]]
print(top_10)

# Save to file
with open('parts/clusters/best_clusters.json', 'w') as f:
    json.dump({'top_10': top_10, 'all_ranked': [(cid, float(score)) for cid, score, _, _, _, _ in results]}, f)
print('\nSaved to parts/clusters/best_clusters.json')

