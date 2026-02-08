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
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/part_discovery/best_model.pt',
                        help='Checkpoint path for extraction')
    parser.add_argument('--n-clusters', type=int, default=100,
                        help='Number of clusters (recommended: 100-200)')
    parser.add_argument('--visual-weight', type=float, default=1.5,
                        help='Weight for visual features')
    parser.add_argument('--slot-weight', type=float, default=0.3,
                        help='Weight for slot features')
    parser.add_argument('--min-cluster-size', type=int, default=10,
                        help='Minimum cluster size for refinement')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of images for testing')
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
    extract_cmd = [
        'python', 'experiments/extract_parts.py',
        '--checkpoint', args.checkpoint,
        '--output-dir', './parts/extracted'
    ]
    if args.limit:
        extract_cmd.extend(['--limit', str(args.limit)])

    run_command(
        extract_cmd,
        "Extracting Parts with Quality Filtering"
    )

    # Step 3: Cluster with improved settings
    run_command(
        [
            'python', 'experiments/cluster_parts.py',
            '--parts-dir', './parts/extracted',
            '--output-dir', './parts/clusters',
            '--n-clusters', str(args.n_clusters),
            '--visual-weight', str(args.visual_weight),
            '--slot-weight', str(args.slot_weight),
            '--min-cluster-size', str(args.min_cluster_size),
            '--refine'
        ],
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

