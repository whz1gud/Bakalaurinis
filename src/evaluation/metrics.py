"""
Evaluation metrics for image similarity/retrieval.

Implements standard retrieval metrics:
- Precision@K
- Recall@K
- Mean Average Precision (mAP)
- Micro Average Precision (μAP) - DISC21 official metric
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
from tqdm import tqdm


def compute_similarity_matrix(
    query_embeddings: np.ndarray,
    ref_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity matrix between queries and references.
    
    Args:
        query_embeddings: Query embeddings [N_q, D]
        ref_embeddings: Reference embeddings [N_r, D]
        
    Returns:
        Similarity matrix [N_q, N_r]
    """
    # Embeddings should already be L2-normalized, but ensure it
    query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
    ref_norm = ref_embeddings / (np.linalg.norm(ref_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Cosine similarity via dot product
    similarity = query_norm @ ref_norm.T
    
    return similarity


def precision_at_k(
    similarity_matrix: np.ndarray,
    query_ids: List[str],
    ref_ids: List[str],
    groundtruth: Dict[str, str],
    k: int = 1,
) -> float:
    """
    Compute Precision@K.
    
    For each query, check if the correct reference is in top-K results.
    
    Args:
        similarity_matrix: Similarity scores [N_q, N_r]
        query_ids: List of query IDs
        ref_ids: List of reference IDs
        groundtruth: Dict mapping query_id -> correct_ref_id
        k: Number of top results to consider
        
    Returns:
        Precision@K score (0-1)
    """
    correct = 0
    total = 0
    
    # Create ref_id to index mapping
    ref_id_to_idx = {ref_id: idx for idx, ref_id in enumerate(ref_ids)}
    
    for q_idx, query_id in enumerate(query_ids):
        if query_id not in groundtruth:
            continue
        
        correct_ref_id = groundtruth[query_id]
        if correct_ref_id not in ref_id_to_idx:
            continue  # Skip if correct reference not in our ref set
        
        # Get top-K reference indices
        top_k_indices = np.argsort(similarity_matrix[q_idx])[-k:][::-1]
        top_k_ref_ids = [ref_ids[i] for i in top_k_indices]
        
        if correct_ref_id in top_k_ref_ids:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0


def recall_at_k(
    similarity_matrix: np.ndarray,
    query_ids: List[str],
    ref_ids: List[str],
    groundtruth: Dict[str, str],
    k: int = 1,
) -> float:
    """
    Compute Recall@K.
    
    For image similarity with single correct answer, Recall@K = Precision@K.
    Included for completeness and consistency with literature.
    """
    return precision_at_k(similarity_matrix, query_ids, ref_ids, groundtruth, k)


def average_precision(
    similarities: np.ndarray,
    relevant_indices: List[int],
) -> float:
    """
    Compute Average Precision for a single query.
    
    Args:
        similarities: Similarity scores for all references
        relevant_indices: Indices of relevant (correct) references
        
    Returns:
        Average Precision score
    """
    if len(relevant_indices) == 0:
        return 0.0
    
    # Sort by similarity (descending)
    sorted_indices = np.argsort(similarities)[::-1]
    
    # Compute precision at each relevant position
    precisions = []
    num_relevant_found = 0
    
    for rank, idx in enumerate(sorted_indices, start=1):
        if idx in relevant_indices:
            num_relevant_found += 1
            precision = num_relevant_found / rank
            precisions.append(precision)
    
    return np.mean(precisions) if precisions else 0.0


def mean_average_precision(
    similarity_matrix: np.ndarray,
    query_ids: List[str],
    ref_ids: List[str],
    groundtruth: Dict[str, str],
) -> float:
    """
    Compute Mean Average Precision (mAP).
    
    Args:
        similarity_matrix: Similarity scores [N_q, N_r]
        query_ids: List of query IDs
        ref_ids: List of reference IDs
        groundtruth: Dict mapping query_id -> correct_ref_id
        
    Returns:
        mAP score (0-1)
    """
    ref_id_to_idx = {ref_id: idx for idx, ref_id in enumerate(ref_ids)}
    
    aps = []
    for q_idx, query_id in enumerate(query_ids):
        if query_id not in groundtruth:
            continue
        
        correct_ref_id = groundtruth[query_id]
        if correct_ref_id not in ref_id_to_idx:
            continue
        
        relevant_idx = ref_id_to_idx[correct_ref_id]
        ap = average_precision(similarity_matrix[q_idx], [relevant_idx])
        aps.append(ap)
    
    return np.mean(aps) if aps else 0.0


def micro_average_precision(
    similarity_matrix: np.ndarray,
    query_ids: List[str],
    ref_ids: List[str],
    groundtruth: Dict[str, str],
) -> float:
    """
    Compute Micro Average Precision (μAP) - DISC21 official metric.
    
    Treats each query-reference pair equally (unlike mAP which averages per query).
    """
    ref_id_to_idx = {ref_id: idx for idx, ref_id in enumerate(ref_ids)}
    
    all_scores = []
    all_labels = []
    
    for q_idx, query_id in enumerate(query_ids):
        if query_id not in groundtruth:
            continue
        
        correct_ref_id = groundtruth[query_id]
        if correct_ref_id not in ref_id_to_idx:
            continue
        
        correct_ref_idx = ref_id_to_idx[correct_ref_id]
        
        for r_idx, ref_id in enumerate(ref_ids):
            all_scores.append(similarity_matrix[q_idx, r_idx])
            all_labels.append(1 if r_idx == correct_ref_idx else 0)
    
    if not all_scores:
        return 0.0
    
    # Sort by score (descending)
    sorted_indices = np.argsort(all_scores)[::-1]
    sorted_labels = np.array(all_labels)[sorted_indices]
    
    # Compute AP
    num_positive = sum(all_labels)
    if num_positive == 0:
        return 0.0
    
    precisions = []
    num_positive_found = 0
    
    for rank, label in enumerate(sorted_labels, start=1):
        if label == 1:
            num_positive_found += 1
            precision = num_positive_found / rank
            precisions.append(precision)
    
    return np.mean(precisions) if precisions else 0.0


def evaluate_retrieval(
    query_embeddings: np.ndarray,
    ref_embeddings: np.ndarray,
    query_ids: List[str],
    ref_ids: List[str],
    groundtruth: Dict[str, str],
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Run full retrieval evaluation.
    
    Args:
        query_embeddings: Query embeddings [N_q, D]
        ref_embeddings: Reference embeddings [N_r, D]
        query_ids: List of query IDs
        ref_ids: List of reference IDs
        groundtruth: Dict mapping query_id -> correct_ref_id
        k_values: List of K values for Precision@K
        
    Returns:
        Dictionary of metric names to scores
    """
    print("Computing similarity matrix...")
    sim_matrix = compute_similarity_matrix(query_embeddings, ref_embeddings)
    
    results = {}
    
    # Precision@K
    for k in k_values:
        p_at_k = precision_at_k(sim_matrix, query_ids, ref_ids, groundtruth, k)
        results[f'P@{k}'] = p_at_k
        print(f"  Precision@{k}: {p_at_k:.4f}")
    
    # Mean Average Precision
    map_score = mean_average_precision(sim_matrix, query_ids, ref_ids, groundtruth)
    results['mAP'] = map_score
    print(f"  mAP: {map_score:.4f}")
    
    # Micro AP (DISC21 official)
    uap = micro_average_precision(sim_matrix, query_ids, ref_ids, groundtruth)
    results['uAP'] = uap
    print(f"  uAP: {uap:.4f}")
    
    return results


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract embeddings from a dataset using the model.
    
    Args:
        model: The similarity model
        dataloader: DataLoader yielding (images, ids)
        device: Device to use
        
    Returns:
        Tuple of (embeddings array, list of IDs)
    """
    model.eval()
    
    all_embeddings = []
    all_ids = []
    
    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        images, ids = batch
        images = images.to(device)
        
        embeddings = model(images)
        all_embeddings.append(embeddings.cpu().numpy())
        all_ids.extend(ids)
    
    embeddings = np.vstack(all_embeddings)
    return embeddings, all_ids


