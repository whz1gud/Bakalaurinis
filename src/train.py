"""
Training script for image similarity with different loss functions.

Supports:
- Contrastive Loss
- Triplet Loss  
- ArcFace Loss
- Focal Loss (from object detection domain)
- CLIP/InfoNCE Loss (from multimodal learning domain)

Features:
- Checkpoint saving every epoch and every N minutes
- Resume from checkpoint
- Validation during training
- Logging to file
- Selectable backbone (ViT-S, ViT-B)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# pytorch-metric-learning for loss functions
from pytorch_metric_learning import losses, miners

from src.data.dataset import DISC21TrainDataset, DISC21PairDataset, DISC21EvalDataset, load_groundtruth
from src.models.similarity_model import SimilarityModel, ArcFaceHead
from src.evaluation.metrics import evaluate_retrieval, extract_embeddings


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) from object detection.
    
    Down-weights easy examples and focuses training on hard ones.
    L = -(1-p)^gamma * log(p)
    """
    
    def __init__(self, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def info_nce_loss(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE / NT-Xent loss (CLIP-style) from multimodal learning.
    
    Treats all other items in the batch as negatives.
    Symmetric: image-to-image and image-to-image (both directions).
    """
    logits = embeddings_a @ embeddings_b.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = (nn.functional.cross_entropy(logits, labels) +
            nn.functional.cross_entropy(logits.T, labels)) / 2
    return loss


class Trainer:
    """
    Trainer class for image similarity models.
    
    Handles training loop, checkpointing, logging, and evaluation.
    """
    
    def __init__(
        self,
        loss_name: str = 'contrastive',
        backbone: str = 'dinov2_vits14',
        embedding_dim: int = 128,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs',
        checkpoint_every_minutes: int = 10,
        eval_every_epochs: int = 2,
        device: str = 'cuda',
        num_workers: int = 4,
    ):
        self.loss_name = loss_name
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_every_minutes = checkpoint_every_minutes
        self.eval_every_epochs = eval_every_epochs
        self.device = device
        self.num_workers = num_workers
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.last_checkpoint_time = time.time()
        self.training_history = []
        
        # Initialize model, loss, optimizer
        self._setup_model()
        self._setup_loss()
        self._setup_optimizer()
        self._setup_data()
        
    def _setup_model(self):
        """Initialize the similarity model."""
        print(f"\n[Setup] Loading DINOv2 model ({self.backbone})...")
        self.model = SimilarityModel(
            backbone_name=self.backbone,
            embedding_dim=self.embedding_dim,
            freeze_backbone=True,
            head_type='linear',
        ).to(self.device)
        
        self.classification_head = None
        
        print(f"[Setup] Model ready on {self.device}")
        
    def _setup_loss(self):
        """Initialize the loss function."""
        print(f"[Setup] Initializing {self.loss_name} loss...")
        
        if self.loss_name == 'contrastive':
            self.loss_fn = losses.ContrastiveLoss(pos_margin=0.0, neg_margin=1.0)
            self.miner = miners.PairMarginMiner(pos_margin=0.0, neg_margin=1.0)
            
        elif self.loss_name == 'triplet':
            margin = float(os.getenv('TRIPLET_MARGIN', '1.0'))
            self.loss_fn = losses.TripletMarginLoss(margin=margin)
            self.miner = miners.TripletMarginMiner(margin=margin, type_of_triplets="hard")
            
        elif self.loss_name == 'arcface':
            self.loss_fn = None
            self.miner = None
            
        elif self.loss_name == 'focal':
            self.loss_fn = None
            self.miner = None
            
        elif self.loss_name == 'clip':
            self.temperature = float(os.getenv('CLIP_TEMPERATURE', '0.07'))
            self.loss_fn = 'clip'
            self.miner = None
            
        else:
            raise ValueError(f"Unknown loss: {self.loss_name}")
            
        print(f"[Setup] Loss function ready: {self.loss_name}")
        
    def _setup_optimizer(self):
        """Initialize the optimizer."""
        # Different learning rates for backbone and head
        backbone_params = list(self.model.backbone.parameters())
        head_params = list(self.model.head.parameters())
        
        param_groups = [
            {'params': backbone_params, 'lr': self.learning_rate * 0.1},  # Lower LR for backbone
            {'params': head_params, 'lr': self.learning_rate},
        ]
        
        self.optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs, eta_min=1e-6
        )
        
        print(f"[Setup] Optimizer ready: AdamW with lr={self.learning_rate}")
        
    def _setup_data(self):
        """Initialize data loaders."""
        print(f"[Setup] Loading datasets...")
        
        # Training data -- CLIP/InfoNCE needs pair dataset, others use standard
        if self.loss_name == 'clip':
            self.train_dataset = DISC21PairDataset('data/train')
        else:
            self.train_dataset = DISC21TrainDataset('data/train')
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        # Validation data (dev queries + refs)
        self.val_query_dataset = DISC21EvalDataset('data/queries_dev')
        self.val_ref_dataset = DISC21EvalDataset('data/refs')
        self.val_groundtruth = load_groundtruth('data/dev_queries_groundtruth.csv')
        
        self.val_query_loader = DataLoader(
            self.val_query_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.val_ref_loader = DataLoader(
            self.val_ref_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        # Classification-based losses need a classification head
        if self.loss_name == 'arcface':
            num_classes = len(self.train_dataset)
            self.classification_head = ArcFaceHead(
                embedding_dim=self.embedding_dim,
                num_classes=num_classes,
                scale=30.0,
                margin=0.5,
            ).to(self.device)
            self.optimizer.add_param_group({
                'params': self.classification_head.parameters(),
                'lr': self.learning_rate,
            })
            self.loss_fn = nn.CrossEntropyLoss()
            print(f"[Setup] ArcFace head ready with {num_classes} classes")
            
        elif self.loss_name == 'focal':
            num_classes = len(self.train_dataset)
            self.classification_head = ArcFaceHead(
                embedding_dim=self.embedding_dim,
                num_classes=num_classes,
                scale=30.0,
                margin=0.0,  # No angular margin -- Focal Loss handles difficulty weighting
            ).to(self.device)
            self.optimizer.add_param_group({
                'params': self.classification_head.parameters(),
                'lr': self.learning_rate,
            })
            self.loss_fn = FocalLoss(gamma=2.0)
            print(f"[Setup] Focal Loss head ready with {num_classes} classes (gamma=2.0)")
        
        print(f"[Setup] Training samples: {len(self.train_dataset)}")
        print(f"[Setup] Validation queries: {len(self.val_query_dataset)}")
        print(f"[Setup] Validation refs: {len(self.val_ref_dataset)}")
        
    def save_checkpoint(self, epoch: int, is_best: bool = False, reason: str = "epoch"):
        """Save a checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'loss_name': self.loss_name,
            'backbone': self.backbone,
            'embedding_dim': self.embedding_dim,
            'training_history': self.training_history,
        }
        
        if self.classification_head is not None:
            checkpoint['classification_head_state_dict'] = self.classification_head.state_dict()
        
        # Include backbone short name in filename to avoid cross-backbone overwrites
        backbone_short = self.backbone.replace('dinov2_', '')  # e.g. "vits14" or "vitb14"
        prefix = f"{backbone_short}_{self.loss_name}"
        
        filename = f"{prefix}_epoch_{epoch:03d}_{reason}.pt"
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"[Checkpoint] Saved: {filename}")
        
        # Save as best if applicable
        if is_best:
            best_path = self.checkpoint_dir / f"{prefix}_best.pt"
            torch.save(checkpoint, best_path)
            print(f"[Checkpoint] New best model saved!")
        
        # Keep only last 3 regular checkpoints (not best)
        self._cleanup_old_checkpoints()
        
        self.last_checkpoint_time = time.time()
        
    def _cleanup_old_checkpoints(self):
        """Keep only the last 3 checkpoints (excluding best)."""
        backbone_short = self.backbone.replace('dinov2_', '')
        prefix = f"{backbone_short}_{self.loss_name}"
        pattern = f"{prefix}_epoch_*.pt"
        checkpoints = sorted(self.checkpoint_dir.glob(pattern))
        
        # Don't delete best checkpoint
        checkpoints = [c for c in checkpoints if 'best' not in c.name]
        
        # Keep last 3
        if len(checkpoints) > 3:
            for old_ckpt in checkpoints[:-3]:
                old_ckpt.unlink()
                print(f"[Checkpoint] Removed old: {old_ckpt.name}")
                
    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint and resume training."""
        print(f"\n[Resume] Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        self.training_history = checkpoint.get('training_history', [])
        
        if self.classification_head is not None and 'classification_head_state_dict' in checkpoint:
            self.classification_head.load_state_dict(checkpoint['classification_head_state_dict'])
        elif self.classification_head is not None and 'arcface_head_state_dict' in checkpoint:
            self.classification_head.load_state_dict(checkpoint['arcface_head_state_dict'])
        
        print(f"[Resume] Resuming from epoch {self.start_epoch}")
        print(f"[Resume] Best metric so far: {self.best_metric:.4f}")
        
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        if self.classification_head is not None:
            self.classification_head.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            
            if self.loss_name == 'clip':
                view1, view2, labels = batch
                view1 = view1.to(self.device)
                view2 = view2.to(self.device)
                
                emb_a = self.model(view1)
                emb_b = self.model(view2)
                loss = info_nce_loss(emb_a, emb_b, temperature=self.temperature)
            else:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                embeddings = self.model(images)
                
                if self.loss_name in ('arcface', 'focal'):
                    logits = self.classification_head(embeddings, labels)
                    loss = self.loss_fn(logits, labels)
                else:
                    hard_pairs = self.miner(embeddings, labels)
                    loss = self.loss_fn(embeddings, labels, hard_pairs)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}",
            })
            
            elapsed = time.time() - self.last_checkpoint_time
            if elapsed > self.checkpoint_every_minutes * 60:
                self.save_checkpoint(epoch, reason="timed")
        
        self.scheduler.step()
        
        avg_loss = total_loss / num_batches
        current_lr = self.optimizer.param_groups[0]['lr']
        
        return {
            'epoch': epoch,
            'avg_loss': avg_loss,
            'learning_rate': current_lr,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and return metrics."""
        print("\n[Validation] Extracting embeddings...")
        
        self.model.eval()
        
        # Extract query embeddings
        query_embeddings, query_ids = extract_embeddings(
            self.model, self.val_query_loader, self.device
        )
        
        # Extract reference embeddings  
        ref_embeddings, ref_ids = extract_embeddings(
            self.model, self.val_ref_loader, self.device
        )
        
        # Evaluate
        print("[Validation] Computing metrics...")
        results = evaluate_retrieval(
            query_embeddings, ref_embeddings,
            query_ids, ref_ids,
            self.val_groundtruth,
            k_values=[1, 5, 10],
        )
        
        return results
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop."""
        
        # Resume if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        print("\n" + "=" * 70)
        print(f"TRAINING: {self.loss_name.upper()} LOSS | {self.backbone}")
        print("=" * 70)
        print(f"Backbone: {self.backbone}")
        print(f"Epochs: {self.start_epoch} -> {self.num_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Device: {self.device}")
        print(f"Checkpoint every: {self.checkpoint_every_minutes} minutes")
        print("=" * 70 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.num_epochs):
            epoch_start = time.time()
            
            # Train one epoch
            train_stats = self.train_one_epoch(epoch)
            
            epoch_time = time.time() - epoch_start
            print(f"\n[Epoch {epoch}] Loss: {train_stats['avg_loss']:.4f} | "
                  f"LR: {train_stats['learning_rate']:.2e} | "
                  f"Time: {epoch_time/60:.1f}min")
            
            # Save checkpoint after each epoch
            self.save_checkpoint(epoch, reason="epoch")
            
            # Validate periodically
            if (epoch + 1) % self.eval_every_epochs == 0 or epoch == self.num_epochs - 1:
                val_results = self.validate()
                
                # Check if this is the best model
                current_metric = val_results['P@1']
                is_best = current_metric > self.best_metric
                
                if is_best:
                    self.best_metric = current_metric
                    self.save_checkpoint(epoch, is_best=True, reason="best")
                
                # Log results
                train_stats['val_results'] = val_results
                print(f"[Epoch {epoch}] Validation P@1: {current_metric:.4f} "
                      f"(Best: {self.best_metric:.4f})")
            
            # Save training history
            self.training_history.append(train_stats)
            
        # Final summary
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best P@1: {self.best_metric:.4f}")
        backbone_short = self.backbone.replace('dinov2_', '')
        print(f"Best checkpoint: {self.checkpoint_dir}/{backbone_short}_{self.loss_name}_best.pt")
        print("=" * 70)
        
        # Save training history
        history_path = self.log_dir / f"{backbone_short}_{self.loss_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"Training history saved to: {history_path}")
        
        return self.best_metric


BACKBONE_CHOICES = {
    'vits': 'dinov2_vits14',
    'vitb': 'dinov2_vitb14',
}


def main():
    parser = argparse.ArgumentParser(description="Train image similarity model")
    
    parser.add_argument('--loss', type=str, default='contrastive',
                        choices=['contrastive', 'triplet', 'arcface', 'focal', 'clip'],
                        help='Loss function to use')
    parser.add_argument('--backbone', type=str, default='vits',
                        choices=list(BACKBONE_CHOICES.keys()),
                        help='DINOv2 backbone size (vits=ViT-S/14, vitb=ViT-B/14)')
    
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    
    parser.add_argument('--checkpoint-minutes', type=int, default=10,
                        help='Save checkpoint every N minutes')
    parser.add_argument('--eval-epochs', type=int, default=2,
                        help='Validate every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    backbone_name = BACKBONE_CHOICES[args.backbone]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: Running on CPU - this will be slow!")
    
    trainer = Trainer(
        loss_name=args.loss,
        backbone=backbone_name,
        embedding_dim=128,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        checkpoint_every_minutes=args.checkpoint_minutes,
        eval_every_epochs=args.eval_epochs,
        device=device,
        num_workers=args.workers,
    )
    
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()

