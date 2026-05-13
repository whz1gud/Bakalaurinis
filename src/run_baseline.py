"""
Baseline Evaluation Script

Evaluates DINOv2 WITHOUT any fine-tuning to establish baseline performance.
This gives us the "before" results to compare against fine-tuned models.
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
import numpy as np

from src.data.dataset import DISC21EvalDataset, load_groundtruth
from src.models.similarity_model import create_model
from src.evaluation.metrics import evaluate_retrieval, extract_embeddings


def run_baseline_evaluation(
    queries_dir: str = "data/queries_dev",
    refs_dir: str = "data/refs",
    groundtruth_path: str = "data/dev_queries_groundtruth.csv",
    output_dir: str = "results",
    batch_size: int = 64,
    num_workers: int = 4,
    backbone: str = "dinov2_vits14",
):
    """
    Run baseline evaluation with pretrained DINOv2 (no fine-tuning).
    
    Args:
        queries_dir: Path to query images
        refs_dir: Path to reference images
        groundtruth_path: Path to ground truth CSV
        output_dir: Where to save results
        batch_size: Batch size for inference
        num_workers: Number of data loading workers
    """
    print("\n" + "=" * 70)
    print(f"BASELINE EVALUATION - DINOv2 {backbone} (No Fine-tuning)")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # Step 1: Load Model
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 1: Loading DINOv2 model...")
    print("-" * 70)
    
    model = create_model(
        backbone=backbone,
        embedding_dim=128,
        freeze_backbone=True,
        device=device,
    )
    model.eval()
    
    # =========================================================================
    # Step 2: Load Datasets
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 2: Loading datasets...")
    print("-" * 70)
    
    # Load ground truth
    groundtruth = load_groundtruth(groundtruth_path)
    
    # Create datasets
    query_dataset = DISC21EvalDataset(queries_dir)
    ref_dataset = DISC21EvalDataset(refs_dir)
    
    # Create dataloaders
    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    ref_loader = DataLoader(
        ref_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # =========================================================================
    # Step 3: Extract Embeddings
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 3: Extracting embeddings...")
    print("-" * 70)
    
    start_time = time.time()
    
    print("\nExtracting query embeddings...")
    query_embeddings, query_ids = extract_embeddings(model, query_loader, device)
    print(f"  Shape: {query_embeddings.shape}")
    
    print("\nExtracting reference embeddings...")
    ref_embeddings, ref_ids = extract_embeddings(model, ref_loader, device)
    print(f"  Shape: {ref_embeddings.shape}")
    
    extraction_time = time.time() - start_time
    print(f"\nTotal extraction time: {extraction_time:.1f}s")
    
    # =========================================================================
    # Step 4: Evaluate
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 4: Computing metrics...")
    print("-" * 70)
    
    results = evaluate_retrieval(
        query_embeddings=query_embeddings,
        ref_embeddings=ref_embeddings,
        query_ids=query_ids,
        ref_ids=ref_ids,
        groundtruth=groundtruth,
        k_values=[1, 5, 10, 50, 100],
    )
    
    # =========================================================================
    # Step 5: Save Results
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 5: Saving results...")
    print("-" * 70)
    
    # Add metadata
    results_with_meta = {
        "experiment": "baseline",
        "model": backbone,
        "embedding_dim": 128,
        "fine_tuned": False,
        "queries_dir": queries_dir,
        "refs_dir": refs_dir,
        "num_queries": len(query_ids),
        "num_refs": len(ref_ids),
        "num_groundtruth": len(groundtruth),
        "extraction_time_seconds": extraction_time,
        "timestamp": datetime.now().isoformat(),
        "metrics": results,
    }
    
    # Save to JSON
    backbone_short = backbone.replace('dinov2_', '')
    output_path = Path(output_dir) / f"baseline_{backbone_short}_results.json"
    with open(output_path, 'w') as f:
        json.dump(results_with_meta, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Metric':<20} {'Score':>10}")
    print("-" * 30)
    for metric, score in results.items():
        print(f"{metric:<20} {score:>10.4f}")
    print("-" * 30)
    
    print("\n[OK] Baseline evaluation complete!")
    print("     These results represent DINOv2 performance WITHOUT fine-tuning.")
    print("     After training with different loss functions, we'll compare against these.")
    
    return results


def run_quick_test(num_samples: int = 100, backbone: str = 'dinov2_vits14'):
    """
    Run a quick test with a small subset to verify everything works.
    """
    print("\n" + "=" * 70)
    print(f"QUICK TEST MODE (100 samples, {backbone})")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\nLoading model...")
    model = create_model(
        backbone=backbone,
        embedding_dim=128,
        freeze_backbone=True,
        device=device,
    )
    model.eval()
    
    # Load small subset
    print("Loading data...")
    query_dataset = DISC21EvalDataset("data/queries_dev")
    ref_dataset = DISC21EvalDataset("data/refs")
    groundtruth = load_groundtruth("data/dev_queries_groundtruth.csv")
    
    # Limit to first N samples
    query_dataset.image_paths = query_dataset.image_paths[:num_samples]
    query_dataset.image_ids = query_dataset.image_ids[:num_samples]
    ref_dataset.image_paths = ref_dataset.image_paths[:num_samples * 5]
    ref_dataset.image_ids = ref_dataset.image_ids[:num_samples * 5]
    
    query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False, num_workers=0)
    ref_loader = DataLoader(ref_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Extract and evaluate
    print("Extracting embeddings...")
    query_emb, query_ids = extract_embeddings(model, query_loader, device)
    ref_emb, ref_ids = extract_embeddings(model, ref_loader, device)
    
    print("Evaluating...")
    results = evaluate_retrieval(
        query_emb, ref_emb, query_ids, ref_ids, groundtruth, k_values=[1, 5, 10]
    )
    
    print("\n[OK] Quick test passed! Full evaluation should work.")
    return results


BACKBONE_CHOICES = {
    'vits': 'dinov2_vits14',
    'vitb': 'dinov2_vitb14',
}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run baseline evaluation")
    parser.add_argument("--quick", action="store_true", help="Run quick test with 100 samples")
    parser.add_argument("--backbone", type=str, default="vits",
                        choices=list(BACKBONE_CHOICES.keys()),
                        help="DINOv2 backbone size (vits=ViT-S/14, vitb=ViT-B/14)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loading workers")
    args = parser.parse_args()
    
    backbone_name = BACKBONE_CHOICES[args.backbone]
    
    if args.quick:
        run_quick_test(backbone=backbone_name)
    else:
        run_baseline_evaluation(
            batch_size=args.batch_size,
            num_workers=args.workers,
            backbone=backbone_name,
        )


