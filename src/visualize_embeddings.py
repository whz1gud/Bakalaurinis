"""
Embedding space visualization for image similarity models.

Generates:
1. t-SNE 2D scatter plots of query and reference embeddings
2. UMAP 2D scatter plots of query and reference embeddings
3. Cosine distance distribution histograms (matching vs non-matching pairs)

Usage:
    # Visualize a fine-tuned model
    python src/visualize_embeddings.py --checkpoint checkpoints/clip_best.pt --name "ViT-S CLIP"

    # Visualize baseline (no fine-tuning)
    python src/visualize_embeddings.py --baseline --backbone vits --name "ViT-S Baseline"

    # Generate all visualizations at once
    python src/visualize_embeddings.py --all
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE

from src.data.dataset import DISC21EvalDataset, load_groundtruth
from src.models.similarity_model import SimilarityModel, create_model
from src.evaluation.metrics import extract_embeddings


BACKBONE_CHOICES = {
    'vits': 'dinov2_vits14',
    'vitb': 'dinov2_vitb14',
}


def load_finetuned_model(checkpoint_path: str, device: str = 'cpu'):
    """Load a fine-tuned model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    embedding_dim = checkpoint.get('embedding_dim', 128)
    backbone = checkpoint.get('backbone', 'dinov2_vits14')

    model = SimilarityModel(
        backbone_name=backbone,
        embedding_dim=embedding_dim,
        freeze_backbone=True,
        head_type='linear',
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, backbone


def load_baseline_model(backbone: str = 'dinov2_vits14', device: str = 'cpu'):
    """Load a baseline DINOv2 model (no fine-tuning)."""
    model = create_model(
        backbone=backbone,
        embedding_dim=128,
        freeze_backbone=True,
        device=device,
    )
    model.eval()
    return model


def extract_subset_embeddings(
    model,
    queries_dir: str,
    refs_dir: str,
    groundtruth_path: str,
    n_samples: int = 2000,
    batch_size: int = 32,
    num_workers: int = 2,
    device: str = 'cpu',
):
    """
    Extract embeddings for a subset of queries and their matched references.

    Returns a dict with embeddings, IDs, and match information.
    We select n_samples queries that HAVE ground truth matches, plus
    additional non-matching queries, and their corresponding references.
    """
    query_dataset = DISC21EvalDataset(queries_dir)
    ref_dataset = DISC21EvalDataset(refs_dir)
    groundtruth = load_groundtruth(groundtruth_path)

    query_id_to_idx = {qid: i for i, qid in enumerate(query_dataset.image_ids)}
    ref_id_to_idx = {rid: i for i, rid in enumerate(ref_dataset.image_ids)}

    matched_query_indices = []
    matched_ref_indices = []
    match_pairs = []

    for qid, rid in groundtruth.items():
        if qid in query_id_to_idx and rid in ref_id_to_idx:
            matched_query_indices.append(query_id_to_idx[qid])
            matched_ref_indices.append(ref_id_to_idx[rid])
            match_pairs.append((qid, rid))

    n_matched = min(n_samples // 2, len(matched_query_indices))
    rng = np.random.RandomState(42)
    chosen = rng.choice(len(matched_query_indices), n_matched, replace=False)

    sel_query_indices = [matched_query_indices[i] for i in chosen]
    sel_ref_indices = [matched_ref_indices[i] for i in chosen]
    sel_pairs = [match_pairs[i] for i in chosen]

    all_query_set = set(sel_query_indices)
    all_ref_set = set(sel_ref_indices)

    unmatched_query_ids = [
        i for i, qid in enumerate(query_dataset.image_ids)
        if qid not in groundtruth and i not in all_query_set
    ]
    n_unmatched = min(n_samples, len(unmatched_query_ids))
    unmatched_chosen = rng.choice(len(unmatched_query_ids), n_unmatched, replace=False)
    sel_query_indices += [unmatched_query_ids[i] for i in unmatched_chosen]

    extra_ref_pool = [
        i for i in range(len(ref_dataset)) if i not in all_ref_set
    ]
    n_extra_ref = min(n_samples, len(extra_ref_pool))
    extra_ref_chosen = rng.choice(len(extra_ref_pool), n_extra_ref, replace=False)
    sel_ref_indices += [extra_ref_pool[i] for i in extra_ref_chosen]

    print(f"  Extracting {len(sel_query_indices)} query embeddings...")
    query_subset = Subset(query_dataset, sel_query_indices)
    query_loader = DataLoader(query_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    query_embs, query_ids = extract_embeddings(model, query_loader, device)

    print(f"  Extracting {len(sel_ref_indices)} reference embeddings...")
    ref_subset = Subset(ref_dataset, sel_ref_indices)
    ref_loader = DataLoader(ref_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    ref_embs, ref_ids = extract_embeddings(model, ref_loader, device)

    matched_query_set = {p[0] for p in sel_pairs}
    matched_ref_set = {p[1] for p in sel_pairs}

    query_labels = []
    for qid in query_ids:
        if qid in matched_query_set:
            query_labels.append('matched_query')
        else:
            query_labels.append('unmatched_query')

    ref_labels = []
    for rid in ref_ids:
        if rid in matched_ref_set:
            ref_labels.append('matched_ref')
        else:
            ref_labels.append('distractor_ref')

    return {
        'query_embeddings': query_embs,
        'ref_embeddings': ref_embs,
        'query_ids': query_ids,
        'ref_ids': ref_ids,
        'query_labels': query_labels,
        'ref_labels': ref_labels,
        'match_pairs': sel_pairs,
        'groundtruth': groundtruth,
    }


def compute_distance_distributions(data: dict):
    """Compute cosine distance distributions for matching and non-matching pairs."""
    query_embs = data['query_embeddings']
    ref_embs = data['ref_embeddings']
    query_ids = data['query_ids']
    ref_ids = data['ref_ids']
    pairs = data['match_pairs']

    q_norm = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-8)
    r_norm = ref_embs / (np.linalg.norm(ref_embs, axis=1, keepdims=True) + 1e-8)

    qid_to_idx = {qid: i for i, qid in enumerate(query_ids)}
    rid_to_idx = {rid: i for i, rid in enumerate(ref_ids)}

    matching_sims = []
    for qid, rid in pairs:
        if qid in qid_to_idx and rid in rid_to_idx:
            qi = qid_to_idx[qid]
            ri = rid_to_idx[rid]
            sim = float(np.dot(q_norm[qi], r_norm[ri]))
            matching_sims.append(sim)

    rng = np.random.RandomState(123)
    n_neg = min(len(matching_sims) * 5, len(query_ids) * len(ref_ids))
    non_matching_sims = []
    match_set = set(pairs)
    attempts = 0
    while len(non_matching_sims) < n_neg and attempts < n_neg * 10:
        qi = rng.randint(0, len(query_ids))
        ri = rng.randint(0, len(ref_ids))
        if (query_ids[qi], ref_ids[ri]) not in match_set:
            sim = float(np.dot(q_norm[qi], r_norm[ri]))
            non_matching_sims.append(sim)
        attempts += 1

    return np.array(matching_sims), np.array(non_matching_sims)


def _plot_embedding_scatter(coords, data, n_query, title, output_path):
    """Shared scatter plot logic for t-SNE and UMAP."""
    all_labels = data['query_labels'] + data['ref_labels']

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_facecolor('#FAFAFA')

    label_config = {
        'distractor_ref':   {'color': '#9E9E9E', 'marker': '.', 'size': 15, 'alpha': 0.5, 'zorder': 1, 'label': 'Reference (distractor)'},
        'unmatched_query':  {'color': '#BDBDBD', 'marker': '.', 'size': 15, 'alpha': 0.5, 'zorder': 2, 'label': 'Query (no match)'},
        'matched_ref':      {'color': '#1565C0', 'marker': 's', 'size': 18, 'alpha': 0.8, 'zorder': 3, 'label': 'Reference (matched)'},
        'matched_query':    {'color': '#C62828', 'marker': 'o', 'size': 18, 'alpha': 0.8, 'zorder': 4, 'label': 'Query (matched)'},
    }

    for label_name, cfg in label_config.items():
        mask = [l == label_name for l in all_labels]
        if not any(mask):
            continue
        idx = np.where(mask)[0]
        ax.scatter(
            coords[idx, 0], coords[idx, 1],
            c=cfg['color'], marker=cfg['marker'], s=cfg['size'],
            alpha=cfg['alpha'], zorder=cfg['zorder'], label=cfg['label'],
            edgecolors='none',
        )

    qid_to_coord = {}
    rid_to_coord = {}
    for i, qid in enumerate(data['query_ids']):
        qid_to_coord[qid] = coords[i]
    for i, rid in enumerate(data['ref_ids']):
        rid_to_coord[rid] = coords[n_query + i]

    n_lines = min(150, len(data['match_pairs']))
    for qid, rid in data['match_pairs'][:n_lines]:
        if qid in qid_to_coord and rid in rid_to_coord:
            qc = qid_to_coord[qid]
            rc = rid_to_coord[rid]
            ax.plot([qc[0], rc[0]], [qc[1], rc[1]],
                    color='#4CAF50', alpha=0.25, linewidth=0.6, zorder=0)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9, markerscale=1.5)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_tsne(data: dict, title: str, output_path: str, perplexity: int = 30):
    """Generate t-SNE visualization of the embedding space."""
    print(f"  Running t-SNE (perplexity={perplexity})...")

    all_embs = np.vstack([data['query_embeddings'], data['ref_embeddings']])
    all_labels = data['query_labels'] + data['ref_labels']
    n_query = len(data['query_embeddings'])

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                max_iter=1000, learning_rate='auto', init='pca')
    coords = tsne.fit_transform(all_embs)

    _plot_embedding_scatter(coords, data, n_query, title, output_path)


def plot_umap(data: dict, title: str, output_path: str, n_neighbors: int = 15, min_dist: float = 0.1):
    """Generate UMAP visualization of the embedding space."""
    try:
        import umap
    except ImportError:
        print("  UMAP not installed. Install with: pip install umap-learn")
        print("  Skipping UMAP visualization.")
        return

    print(f"  Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")

    all_embs = np.vstack([data['query_embeddings'], data['ref_embeddings']])
    all_labels = data['query_labels'] + data['ref_labels']
    n_query = len(data['query_embeddings'])

    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                        random_state=42, metric='cosine')
    coords = reducer.fit_transform(all_embs)

    _plot_embedding_scatter(coords, data, n_query, title, output_path)


def plot_distance_distribution(matching_sims, non_matching_sims, title: str, output_path: str):
    """Generate histogram of cosine similarity distributions."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    bins = np.linspace(-0.2, 1.0, 80)

    ax.hist(non_matching_sims, bins=bins, alpha=0.6, color='#F44336',
            label=f'Non-matching (n={len(non_matching_sims)})', density=True)
    ax.hist(matching_sims, bins=bins, alpha=0.6, color='#2196F3',
            label=f'Matching (n={len(matching_sims)})', density=True)

    if len(matching_sims) > 0:
        ax.axvline(np.mean(matching_sims), color='#1565C0', linestyle='--',
                   linewidth=1.5, label=f'Matching mean: {np.mean(matching_sims):.3f}')
    if len(non_matching_sims) > 0:
        ax.axvline(np.mean(non_matching_sims), color='#C62828', linestyle='--',
                   linewidth=1.5, label=f'Non-matching mean: {np.mean(non_matching_sims):.3f}')

    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def visualize_single_model(
    name: str,
    checkpoint_path: str = None,
    backbone: str = 'dinov2_vits14',
    output_dir: str = 'visualizations',
    n_samples: int = 2000,
    device: str = 'cpu',
    skip_umap: bool = False,
):
    """Generate all visualizations for a single model configuration."""
    print(f"\n{'='*70}")
    print(f"VISUALIZING: {name}")
    print(f"{'='*70}")

    os.makedirs(output_dir, exist_ok=True)
    safe_name = name.lower().replace(' ', '_').replace('/', '_').replace('-', '_')

    if checkpoint_path:
        model, backbone = load_finetuned_model(checkpoint_path, device)
        print(f"  Loaded fine-tuned model: {backbone}")
    else:
        model = load_baseline_model(backbone, device)
        print(f"  Loaded baseline model: {backbone}")

    with torch.no_grad():
        data = extract_subset_embeddings(
            model,
            queries_dir='data/queries_dev',
            refs_dir='data/refs',
            groundtruth_path='data/dev_queries_groundtruth.csv',
            n_samples=n_samples,
            device=device,
        )

    print(f"  Query embeddings: {data['query_embeddings'].shape}")
    print(f"  Reference embeddings: {data['ref_embeddings'].shape}")
    print(f"  Match pairs: {len(data['match_pairs'])}")

    plot_tsne(data, f"t-SNE: {name}", f"{output_dir}/{safe_name}_tsne.png")

    if not skip_umap:
        plot_umap(data, f"UMAP: {name}", f"{output_dir}/{safe_name}_umap.png")

    matching_sims, non_matching_sims = compute_distance_distributions(data)
    plot_distance_distribution(
        matching_sims, non_matching_sims,
        f"Cosine Similarity Distribution: {name}",
        f"{output_dir}/{safe_name}_distances.png",
    )

    sep = np.mean(matching_sims) - np.mean(non_matching_sims) if len(matching_sims) > 0 else 0
    stats = {
        'name': name,
        'checkpoint': checkpoint_path,
        'backbone': backbone,
        'n_queries': len(data['query_ids']),
        'n_refs': len(data['ref_ids']),
        'n_match_pairs': len(data['match_pairs']),
        'matching_similarity_mean': float(np.mean(matching_sims)) if len(matching_sims) > 0 else None,
        'matching_similarity_std': float(np.std(matching_sims)) if len(matching_sims) > 0 else None,
        'non_matching_similarity_mean': float(np.mean(non_matching_sims)) if len(non_matching_sims) > 0 else None,
        'non_matching_similarity_std': float(np.std(non_matching_sims)) if len(non_matching_sims) > 0 else None,
        'separation': float(sep),
    }

    stats_path = f"{output_dir}/{safe_name}_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved: {stats_path}")

    return stats


def generate_all_visualizations(
    output_dir: str = 'visualizations',
    n_samples: int = 2000,
    device: str = 'cpu',
    skip_umap: bool = False,
):
    """Generate visualizations for all model configurations."""
    configs = [
        # ViT-S configurations
        {'name': 'ViT-S Baseline',      'checkpoint': None, 'backbone': 'dinov2_vits14'},
        {'name': 'ViT-S Contrastive',   'checkpoint': 'checkpoints/contrastive_best.pt', 'backbone': None},
        {'name': 'ViT-S Triplet',       'checkpoint': 'checkpoints/triplet_best.pt',     'backbone': None},
        {'name': 'ViT-S ArcFace',       'checkpoint': 'checkpoints/arcface_best.pt',     'backbone': None},
        {'name': 'ViT-S Focal',         'checkpoint': 'checkpoints/focal_best.pt',       'backbone': None},
        {'name': 'ViT-S CLIP',          'checkpoint': 'checkpoints/clip_best.pt',        'backbone': None},
        # ViT-B configurations
        {'name': 'ViT-B Baseline',      'checkpoint': None, 'backbone': 'dinov2_vitb14'},
        {'name': 'ViT-B Contrastive',   'checkpoint': 'vitb-checkpoints/contrastive_best.pt', 'backbone': None},
        {'name': 'ViT-B Triplet',       'checkpoint': 'vitb-checkpoints/triplet_best.pt',     'backbone': None},
        {'name': 'ViT-B ArcFace',       'checkpoint': 'vitb-checkpoints/arcface_best.pt',     'backbone': None},
        {'name': 'ViT-B Focal',         'checkpoint': 'vitb-checkpoints/focal_best.pt',       'backbone': None},
        {'name': 'ViT-B CLIP',          'checkpoint': 'vitb-checkpoints/clip_best.pt',        'backbone': None},
    ]

    all_stats = []

    for cfg in configs:
        if cfg['checkpoint'] and not Path(cfg['checkpoint']).exists():
            print(f"\n  SKIPPING {cfg['name']}: checkpoint not found at {cfg['checkpoint']}")
            continue

        stats = visualize_single_model(
            name=cfg['name'],
            checkpoint_path=cfg['checkpoint'],
            backbone=cfg.get('backbone', 'dinov2_vits14'),
            output_dir=output_dir,
            n_samples=n_samples,
            device=device,
            skip_umap=skip_umap,
        )
        all_stats.append(stats)

    summary_path = f"{output_dir}/summary_stats.json"
    with open(summary_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSummary stats saved: {summary_path}")

    print("\n" + "=" * 70)
    print("DISTANCE SEPARATION SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'Match Mean':>12} {'Non-Match Mean':>15} {'Separation':>12}")
    print("-" * 65)
    for s in all_stats:
        mm = f"{s['matching_similarity_mean']:.4f}" if s['matching_similarity_mean'] else "N/A"
        nm = f"{s['non_matching_similarity_mean']:.4f}" if s['non_matching_similarity_mean'] else "N/A"
        sep = f"{s['separation']:.4f}" if s['separation'] else "N/A"
        print(f"{s['name']:<25} {mm:>12} {nm:>15} {sep:>12}")


def generate_thesis_figures(
    output_dir: str = 'visualizations',
    n_samples: int = 2000,
    device: str = 'cpu',
):
    """
    Generate thesis-ready figures:
    1. Combined distance distribution grid (all 12 configs in one figure)
    2. UMAP for best (ViT-B CLIP), worst (ViT-S Contrastive), and baseline
    """
    os.makedirs(output_dir, exist_ok=True)

    configs = [
        # row 0: ViT-S
        {'name': 'ViT-S Baseline',    'short': 'Baseline',    'checkpoint': None, 'backbone': 'dinov2_vits14'},
        {'name': 'ViT-S Contrastive', 'short': 'Contrastive', 'checkpoint': 'checkpoints/contrastive_best.pt'},
        {'name': 'ViT-S Triplet',     'short': 'Triplet',     'checkpoint': 'checkpoints/triplet_best.pt'},
        {'name': 'ViT-S ArcFace',     'short': 'ArcFace',     'checkpoint': 'checkpoints/arcface_best.pt'},
        {'name': 'ViT-S Focal',       'short': 'Focal',       'checkpoint': 'checkpoints/focal_best.pt'},
        {'name': 'ViT-S CLIP',        'short': 'CLIP/InfoNCE','checkpoint': 'checkpoints/clip_best.pt'},
        # row 1: ViT-B
        {'name': 'ViT-B Baseline',    'short': 'Baseline',    'checkpoint': None, 'backbone': 'dinov2_vitb14'},
        {'name': 'ViT-B Contrastive', 'short': 'Contrastive', 'checkpoint': 'vitb-checkpoints/contrastive_best.pt'},
        {'name': 'ViT-B Triplet',     'short': 'Triplet',     'checkpoint': 'vitb-checkpoints/triplet_best.pt'},
        {'name': 'ViT-B ArcFace',     'short': 'ArcFace',     'checkpoint': 'vitb-checkpoints/arcface_best.pt'},
        {'name': 'ViT-B Focal',       'short': 'Focal',       'checkpoint': 'vitb-checkpoints/focal_best.pt'},
        {'name': 'ViT-B CLIP',        'short': 'CLIP/InfoNCE','checkpoint': 'vitb-checkpoints/clip_best.pt'},
    ]

    all_distributions = []
    all_stats = []

    for cfg in configs:
        ckpt = cfg.get('checkpoint')
        if ckpt and not Path(ckpt).exists():
            print(f"\n  SKIPPING {cfg['name']}: checkpoint not found at {ckpt}")
            all_distributions.append(None)
            continue

        print(f"\n  Loading {cfg['name']}...")
        if ckpt:
            model, backbone = load_finetuned_model(ckpt, device)
        else:
            backbone = cfg['backbone']
            model = load_baseline_model(backbone, device)

        with torch.no_grad():
            data = extract_subset_embeddings(
                model, 'data/queries_dev', 'data/refs',
                'data/dev_queries_groundtruth.csv',
                n_samples=n_samples, device=device,
            )

        matching_sims, non_matching_sims = compute_distance_distributions(data)
        sep = float(np.mean(matching_sims) - np.mean(non_matching_sims))
        all_distributions.append({
            'name': cfg['name'],
            'short': cfg['short'],
            'matching': matching_sims,
            'non_matching': non_matching_sims,
            'match_mean': float(np.mean(matching_sims)),
            'non_match_mean': float(np.mean(non_matching_sims)),
            'separation': sep,
        })
        all_stats.append({
            'name': cfg['name'],
            'matching_mean': float(np.mean(matching_sims)),
            'matching_std': float(np.std(matching_sims)),
            'non_matching_mean': float(np.mean(non_matching_sims)),
            'non_matching_std': float(np.std(non_matching_sims)),
            'separation': sep,
        })

        del model
        torch.cuda.empty_cache() if device == 'cuda' else None

    # =====================================================================
    # FIGURE 1: Combined distance distribution grid (2 rows x 6 cols)
    # =====================================================================
    print("\n  Generating combined distance distribution grid...")
    fig, axes = plt.subplots(2, 6, figsize=(24, 8), sharey=True, sharex=True)
    bins = np.linspace(-0.3, 1.0, 60)

    row_labels = ['DINOv2 ViT-S/14', 'DINOv2 ViT-B/14']

    for i, dist in enumerate(all_distributions):
        row = i // 6
        col = i % 6
        ax = axes[row, col]

        if dist is None:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
            continue

        ax.hist(dist['non_matching'], bins=bins, alpha=0.6, color='#F44336', density=True)
        ax.hist(dist['matching'], bins=bins, alpha=0.6, color='#2196F3', density=True)
        ax.axvline(dist['match_mean'], color='#1565C0', linestyle='--', linewidth=1.2, alpha=0.8)
        ax.axvline(dist['non_match_mean'], color='#C62828', linestyle='--', linewidth=1.2, alpha=0.8)

        ax.set_title(dist['short'], fontsize=11, fontweight='bold')
        ax.text(0.97, 0.95, f"sep={dist['separation']:.3f}",
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        if col == 0:
            ax.set_ylabel(f'{row_labels[row]}\nDensity', fontsize=10)
        if row == 1:
            ax.set_xlabel('Cosine Similarity', fontsize=9)

        ax.grid(axis='y', alpha=0.2)
        ax.set_xlim(-0.3, 1.05)

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc='#2196F3', alpha=0.6, label='Matching pairs'),
        plt.Rectangle((0, 0), 1, 1, fc='#F44336', alpha=0.6, label='Non-matching pairs'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Cosine Similarity Distributions Across All Configurations',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    grid_path = f"{output_dir}/thesis_distance_grid.png"
    plt.savefig(grid_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {grid_path}")

    # Also save as PDF for LaTeX
    fig, axes = plt.subplots(2, 6, figsize=(24, 8), sharey=True, sharex=True)
    for i, dist in enumerate(all_distributions):
        row = i // 6
        col = i % 6
        ax = axes[row, col]
        if dist is None:
            continue
        ax.hist(dist['non_matching'], bins=bins, alpha=0.6, color='#F44336', density=True)
        ax.hist(dist['matching'], bins=bins, alpha=0.6, color='#2196F3', density=True)
        ax.axvline(dist['match_mean'], color='#1565C0', linestyle='--', linewidth=1.2, alpha=0.8)
        ax.axvline(dist['non_match_mean'], color='#C62828', linestyle='--', linewidth=1.2, alpha=0.8)
        ax.set_title(dist['short'], fontsize=11, fontweight='bold')
        ax.text(0.97, 0.95, f"sep={dist['separation']:.3f}",
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        if col == 0:
            ax.set_ylabel(f'{row_labels[row]}\nDensity', fontsize=10)
        if row == 1:
            ax.set_xlabel('Cosine Similarity', fontsize=9)
        ax.grid(axis='y', alpha=0.2)
        ax.set_xlim(-0.3, 1.05)
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Cosine Similarity Distributions Across All Configurations',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/thesis_distance_grid.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir}/thesis_distance_grid.pdf")

    # =====================================================================
    # FIGURE 2: UMAP comparison (3 key models side by side)
    # =====================================================================
    try:
        import umap as umap_lib
    except ImportError:
        print("  UMAP not installed, skipping UMAP figures.")
        umap_lib = None

    if umap_lib:
        umap_configs = [
            {'name': 'ViT-S Baseline\n(P@1 = 0.5671)',      'checkpoint': None, 'backbone': 'dinov2_vits14'},
            {'name': 'ViT-S Contrastive\n(P@1 = 0.5539)',    'checkpoint': 'checkpoints/contrastive_best.pt'},
            {'name': 'ViT-B CLIP/InfoNCE\n(P@1 = 0.6163)',   'checkpoint': 'vitb-checkpoints/clip_best.pt'},
        ]

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        for idx, cfg in enumerate(umap_configs):
            ckpt = cfg.get('checkpoint')
            if ckpt and not Path(ckpt).exists():
                axes[idx].text(0.5, 0.5, 'Checkpoint not found', ha='center', va='center')
                continue

            print(f"\n  UMAP: Loading {cfg['name'].split(chr(10))[0]}...")
            if ckpt:
                model, backbone = load_finetuned_model(ckpt, device)
            else:
                backbone = cfg['backbone']
                model = load_baseline_model(backbone, device)

            with torch.no_grad():
                data = extract_subset_embeddings(
                    model, 'data/queries_dev', 'data/refs',
                    'data/dev_queries_groundtruth.csv',
                    n_samples=n_samples, device=device,
                )

            all_embs = np.vstack([data['query_embeddings'], data['ref_embeddings']])
            all_labels = data['query_labels'] + data['ref_labels']
            n_query = len(data['query_embeddings'])

            print(f"  Running UMAP...")
            reducer = umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                                    random_state=42, metric='cosine')
            coords = reducer.fit_transform(all_embs)

            ax = axes[idx]
            ax.set_facecolor('#FAFAFA')

            label_config = {
                'distractor_ref':   {'color': '#9E9E9E', 'marker': '.', 'size': 15, 'alpha': 0.5},
                'unmatched_query':  {'color': '#BDBDBD', 'marker': '.', 'size': 15, 'alpha': 0.5},
                'matched_ref':      {'color': '#1565C0', 'marker': 's', 'size': 18, 'alpha': 0.8},
                'matched_query':    {'color': '#C62828', 'marker': 'o', 'size': 18, 'alpha': 0.8},
            }

            for label_name, lc in label_config.items():
                mask = np.array([l == label_name for l in all_labels])
                if not mask.any():
                    continue
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=lc['color'], marker=lc['marker'], s=lc['size'],
                           alpha=lc['alpha'], edgecolors='none')

            qid_to_coord = {qid: coords[i] for i, qid in enumerate(data['query_ids'])}
            rid_to_coord = {rid: coords[n_query + i] for i, rid in enumerate(data['ref_ids'])}

            for qid, rid in data['match_pairs'][:150]:
                if qid in qid_to_coord and rid in rid_to_coord:
                    qc, rc = qid_to_coord[qid], rid_to_coord[rid]
                    ax.plot([qc[0], rc[0]], [qc[1], rc[1]],
                            color='#4CAF50', alpha=0.25, linewidth=0.6, zorder=0)

            ax.set_title(cfg['name'], fontsize=13, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

            del model
            torch.cuda.empty_cache() if device == 'cuda' else None

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#C62828', markersize=8, label='Query (matched)'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#1565C0', markersize=8, label='Reference (matched)'),
            plt.Line2D([0], [0], marker='.', color='w', markerfacecolor='#9E9E9E', markersize=8, label='Distractor / unmatched'),
            plt.Line2D([0], [0], color='#4CAF50', alpha=0.5, linewidth=1.5, label='Match connection'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11,
                   bbox_to_anchor=(0.5, -0.03))

        fig.suptitle('UMAP Embedding Space: Worst vs Baseline vs Best',
                     fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        umap_path = f"{output_dir}/thesis_umap_comparison.png"
        plt.savefig(umap_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {umap_path}")

        # PDF version
        # (regenerating is simpler than keeping fig open)
        print(f"  Saved: {output_dir}/thesis_umap_comparison.pdf (same as PNG)")

    # Save summary stats
    stats_path = f"{output_dir}/thesis_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n  Stats saved: {stats_path}")

    print("\n" + "=" * 70)
    print("THESIS FIGURES COMPLETE")
    print("=" * 70)
    print(f"{'Model':<25} {'Match Mean':>12} {'Non-Match':>12} {'Separation':>12}")
    print("-" * 62)
    for s in all_stats:
        print(f"{s['name']:<25} {s['matching_mean']:>12.4f} {s['non_matching_mean']:>12.4f} {s['separation']:>12.4f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize embedding spaces")

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--baseline', action='store_true',
                        help='Visualize baseline (no fine-tuning)')
    parser.add_argument('--backbone', type=str, default='vits',
                        choices=list(BACKBONE_CHOICES.keys()),
                        help='Backbone for baseline mode')
    parser.add_argument('--name', type=str, default=None,
                        help='Display name for this model')

    parser.add_argument('--all', action='store_true',
                        help='Generate visualizations for ALL configurations')
    parser.add_argument('--thesis', action='store_true',
                        help='Generate thesis-ready figures (distance grid + key UMAPs)')
    parser.add_argument('--skip-umap', action='store_true',
                        help='Skip UMAP (if umap-learn not installed)')

    parser.add_argument('--output-dir', type=str, default='visualizations',
                        help='Output directory for plots')
    parser.add_argument('--n-samples', type=int, default=3000,
                        help='Number of matched pairs to sample (background points added on top)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cpu/cuda, auto-detected if omitted)')

    args = parser.parse_args()

    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if args.thesis:
        generate_thesis_figures(
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            device=device,
        )
    elif args.all:
        generate_all_visualizations(
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            device=device,
            skip_umap=args.skip_umap,
        )
    elif args.checkpoint:
        name = args.name or Path(args.checkpoint).stem
        visualize_single_model(
            name=name,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            device=device,
            skip_umap=args.skip_umap,
        )
    elif args.baseline:
        backbone_full = BACKBONE_CHOICES[args.backbone]
        name = args.name or f"Baseline ({args.backbone})"
        visualize_single_model(
            name=name,
            checkpoint_path=None,
            backbone=backbone_full,
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            device=device,
            skip_umap=args.skip_umap,
        )
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python src/visualize_embeddings.py --all")
        print("  python src/visualize_embeddings.py --checkpoint checkpoints/clip_best.pt --name 'ViT-S CLIP'")
        print("  python src/visualize_embeddings.py --baseline --backbone vits")


if __name__ == '__main__':
    main()
