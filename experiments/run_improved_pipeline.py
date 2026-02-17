#!/usr/bin/env python
"""
Run Improved Part Discovery Pipeline

This script runs the full improved pipeline with:
1. Enhanced spatial coherence loss (bigger impact on part quality)
2. Part filtering (removes bad parts before clustering)
3. Visual features weighted higher than slot features
4. More clusters (100-200) for better granularity
5. Cluster refinement (merges small clusters)

Usage:
    python experiments/run_improved_pipeline.py [--skip-training] [--n-clusters 100]
"""

import sys
sys.path.append('.')

import argparse
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60 + '\n')

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n❌ FAILED: {description}")
        sys.exit(1)
    print(f"\n✓ COMPLETED: {description}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Run improved part discovery pipeline')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training step (use existing checkpoint)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path for extraction (default: read from config)')
    parser.add_argument('--n-clusters', type=int, default=None,
                        help='Number of clusters (default: read from config)')
    parser.add_argument('--visual-weight', type=float, default=None,
                        help='Weight for visual features (default: read from config)')
    parser.add_argument('--slot-weight', type=float, default=None,
                        help='Weight for slot features (default: read from config)')
    parser.add_argument('--min-cluster-size', type=int, default=None,
                        help='Minimum cluster size for refinement (default: read from config)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of images for testing')
    parser.add_argument('--refine', action='store_true', help='Apply post-processing refinement')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("IMPROVED PART DISCOVERY PIPELINE")
    print("="*60)
    print("\nConfiguration:")
    print(f"  - Skip training: {args.skip_training}")
    print(f"  - N clusters: {args.n_clusters}")
    print(f"  - Visual weight: {args.visual_weight}")
    print(f"  - Slot weight: {args.slot_weight}")
    print(f"  - Min cluster size: {args.min_cluster_size}")

    # Step 1: Train (optional)
    if not args.skip_training:
        run_command(
            ['python', 'experiments/train_part_discovery.py'],
            "Training Part Discovery Model with Enhanced Spatial Coherence Loss"
        )
    else:
        print("\n⏭ Skipping training step (using existing checkpoint)")
        if not Path(args.checkpoint).exists():
            print(f"❌ ERROR: Checkpoint not found at {args.checkpoint}")
            sys.exit(1)

    # Step 2: Extract parts with filtering
    # Rely on defaults in extract_parts.py which now read from unified_config.yaml
    extract_cmd = ['python', 'experiments/extract_parts.py']
    
    # Only pass checkpoint if explicitly provided to this script, otherwise let extract_parts find it in config
    if args.checkpoint:
         extract_cmd.extend(['--checkpoint', args.checkpoint])
         
    if args.limit:
        extract_cmd.extend(['--limit', str(args.limit)])

    run_command(
        extract_cmd,
        "Extracting Parts with Quality Filtering"
    )

    # Step 3: Cluster with improved settings
    # Rely on defaults in cluster_parts.py which now read from unified_config.yaml
    cluster_cmd = ['python', 'experiments/cluster_parts.py']
    
    # We only pass args if they were explicitly provided (not None)
    # But for now, since we want to force config file usage, we just run it without args
    # except maybe refine which is a flag
    if args.refine:
        cluster_cmd.append('--refine')
        
    run_command(
        cluster_cmd,
        "Clustering Parts with Visual Features + Refinement"
    )

    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nOutputs:")
    print("  - Checkpoint: checkpoints/part_discovery/best_model.pt")
    print("  - Extracted parts: parts/extracted/")
    print("  - Cluster results: parts/clusters/")
    print("\nNext steps:")
    print("  1. Review parts/clusters/clustering_metrics.json for quality metrics")
    print("  2. Review parts/clusters/cluster_metadata.json for per-cluster stats")
    print("  3. Check parts/clusters/tsne.png for cluster visualization")
    print("  4. Run 'python -m src.clustering.streamlit_labeler' to label clusters")


if __name__ == '__main__':
    main()

