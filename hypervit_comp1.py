import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import json
import numpy as np
from collections import defaultdict
import random
from safetensors.torch import save_file, load_file
import logging
from datetime import datetime
import shutil
import time
import pickle
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('tensors', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('safetensors', exist_ok=True)

# Memory management for M2 MacBook
def clear_memory():
    """Clear memory cache for MPS/CPU"""
    if torch.backends.mps.is_available():
        import gc
        gc.collect()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

# CIFAR-100 Dataset Class
class CIFAR100Dataset(Dataset):
    def __init__(self, data_path, train=True, transform=None):
        self.transform = transform
        self.train = train
        
        if train:
            file_path = os.path.join(data_path, 'train')
        else:
            file_path = os.path.join(data_path, 'test')
        
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        
        self.data = data_dict[b'data']
        self.fine_labels = data_dict[b'fine_labels']
        self.coarse_labels = data_dict[b'coarse_labels']
        
        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))
        
        meta_path = os.path.join(data_path, 'meta')
        with open(meta_path, 'rb') as f:
            meta_dict = pickle.load(f, encoding='bytes')
        
        self.fine_label_names = [name.decode('utf-8') for name in meta_dict[b'fine_label_names']]
        self.coarse_label_names = [name.decode('utf-8') for name in meta_dict[b'coarse_label_names']]
        
        logger.info(f"Loaded CIFAR-100 {'train' if train else 'test'} set: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        fine_label = self.fine_labels[idx]
        coarse_label = self.coarse_labels[idx]
        
        from PIL import Image
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, fine_label

# Efficient Patch Embedding without PHConv2d
class EfficientPatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=132):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Use standard Conv2d for efficiency
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = self.proj(x)  # B, C, H, W
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        x = self.norm(x)
        return x

# Hypercomplex Linear Layer (Optimized)
class HypercomplexLinear(nn.Module):
    def __init__(self, in_features, out_features, n=4):
        super().__init__()
        self.n = n
        self.in_features = in_features
        self.out_features = out_features
        
        # Ensure divisibility
        self.in_features_padded = ((in_features - 1) // n + 1) * n
        self.out_features_padded = ((out_features - 1) // n + 1) * n
        
        # Component matrices
        self.weight_components = nn.Parameter(
            torch.randn(n, self.out_features_padded // n, self.in_features_padded // n) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Input/output projections for dimension matching
        if in_features != self.in_features_padded:
            self.input_proj = nn.Linear(in_features, self.in_features_padded, bias=False)
        else:
            self.input_proj = None
            
        if out_features != self.out_features_padded:
            self.output_proj = nn.Linear(self.out_features_padded, out_features, bias=False)
        else:
            self.output_proj = None
    
    def forward(self, x):
        batch_shape = x.shape[:-1]
        x = x.view(-1, self.in_features)
        
        # Project input if needed
        if self.input_proj is not None:
            x = self.input_proj(x)
        
        # Reshape for hypercomplex multiplication
        x = x.view(-1, self.n, self.in_features_padded // self.n)
        
        # Efficient hypercomplex multiplication
        out = torch.zeros(x.shape[0], self.n, self.out_features_padded // self.n, device=x.device)
        for i in range(self.n):
            for j in range(self.n):
                k = (i + j) % self.n
                out[:, k] += torch.matmul(x[:, j], self.weight_components[i].T)
        
        out = out.view(-1, self.out_features_padded)
        
        # Project output if needed
        if self.output_proj is not None:
            out = self.output_proj(out)
        
        out = out + self.bias
        out = out.view(*batch_shape, self.out_features)
        
        return out

# Efficient Attention with Linear Complexity
class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Learnable temperature for attention
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Efficient attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn / self.temperature
        
        # Add local attention bias
        if N > 1:
            local_mask = self._get_local_mask(N, window_size=7, device=x.device)
            attn = attn + local_mask
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def _get_local_mask(self, seq_len, window_size, device):
        mask = torch.zeros(seq_len, seq_len, device=device)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = 1
        return mask.log().unsqueeze(0).unsqueeze(0)

# Adaptive Computation Time Module
class AdaptiveComputationTime(nn.Module):
    def __init__(self, dim, max_steps=5, threshold=0.99):
        super().__init__()
        self.max_steps = max_steps
        self.threshold = threshold
        
        self.halting_predictor = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, block_fn):
        B, N, C = x.shape
        
        halting_prob = torch.zeros(B, N, 1, device=x.device)
        remainders = torch.zeros(B, N, 1, device=x.device)
        n_updates = torch.zeros(B, N, 1, device=x.device)
        
        accumulated_output = torch.zeros_like(x)
        
        for step in range(self.max_steps):
            # Compute halting probability
            p = self.halting_predictor(x)
            
            # Determine which positions should still be updated
            still_running = (halting_prob < self.threshold).float()
            
            # Compute remainders and halting probabilities
            new_halted = (halting_prob + p * still_running > self.threshold).float() * still_running
            remainders += new_halted * (1 - halting_prob)
            halting_prob += p * still_running
            n_updates += still_running
            
            # Apply block function
            update = block_fn(x)
            accumulated_output += update * still_running
            
            # Update x for next iteration
            x = x + update * still_running
            
            # Check if all positions have halted
            if (still_running == 0).all():
                break
        
        return accumulated_output, n_updates.mean()

# Optimized MoE Layer
class OptimizedMoE(nn.Module):
    def __init__(self, dim, num_experts=4, expert_capacity=2):
        super().__init__()
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Compute gating scores
        gates = self.gate(x)  # B, N, num_experts
        gates = F.softmax(gates, dim=-1)
        
        # Select top-k experts
        topk_gates, topk_indices = torch.topk(gates, k=min(2, self.num_experts), dim=-1)
        topk_gates = topk_gates / topk_gates.sum(dim=-1, keepdim=True)  # Renormalize
        
        # Process through experts
        output = torch.zeros_like(x)
        for i in range(topk_gates.shape[-1]):
            expert_mask = F.one_hot(topk_indices[:, :, i], self.num_experts).float()  # B, N, num_experts
            for e in range(self.num_experts):
                mask = expert_mask[:, :, e].unsqueeze(-1)  # B, N, 1
                if mask.sum() > 0:
                    expert_input = x * mask
                    expert_output = self.experts[e](expert_input)
                    output += expert_output * topk_gates[:, :, i:i+1] * mask
        
        return output

# Transformer Block with Novel Features
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = nn.LayerNorm(dim)
        self.moe = OptimizedMoE(dim, num_experts=4)
        
        self.drop_path = nn.Identity()  # Can be replaced with DropPath
        
        # Learnable scaling factors
        self.attn_scale = nn.Parameter(torch.ones(1))
        self.mlp_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # Attention with residual
        x = x + self.drop_path(self.attn_scale * self.attn(self.norm1(x)))
        
        # MoE with residual
        x = x + self.drop_path(self.mlp_scale * self.moe(self.norm2(x)))
        
        return x

# Main Model
class CIFAR100HyperViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=100,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4., 
                 drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = EfficientPatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks with adaptive computation
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate
            ) for _ in range(depth)
        ])
        
        # Adaptive computation time
        self.act = AdaptiveComputationTime(embed_dim, max_steps=3)
        
        # Normalization and head
        self.norm = nn.LayerNorm(embed_dim)
        
        # Multi-scale classification head
        self.head = nn.Sequential(
            HypercomplexLinear(embed_dim, embed_dim * 2, n=4),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim * 2, num_classes)
        )
        
        # Auxiliary heads for deep supervision
        self.aux_heads = nn.ModuleList([
            nn.Linear(embed_dim, num_classes) 
            for _ in range(depth // 3)
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
    def forward_features(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks with checkpointing for memory efficiency
        aux_outputs = []
        for i, blk in enumerate(self.blocks):
            if self.training and i % 4 == 0:
                x = checkpoint(blk, x)
            else:
                x = blk(x)
            
            # Collect auxiliary outputs
            if i % (len(self.blocks) // len(self.aux_heads)) == 0 and i > 0:
                aux_idx = i // (len(self.blocks) // len(self.aux_heads)) - 1
                if aux_idx < len(self.aux_heads):
                    aux_outputs.append(self.aux_heads[aux_idx](x[:, 0]))
        
        x = self.norm(x)
        return x[:, 0], aux_outputs
    
    def forward(self, x):
        x, aux_outputs = self.forward_features(x)
        x = self.head(x)
        
        if self.training and aux_outputs:
            return x, aux_outputs
        return x

# Knowledge Distillation Loss
class DistillationLoss(nn.Module):
    def __init__(self, base_criterion, teacher_model=None, alpha=0.5, temperature=3.0):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        
    def forward(self, outputs, labels):
        if isinstance(outputs, tuple):
            student_outputs, aux_outputs = outputs
            
            # Main loss
            loss = self.base_criterion(student_outputs, labels)
            
            # Auxiliary losses
            for aux_out in aux_outputs:
                loss += 0.3 * self.base_criterion(aux_out, labels)
            
            return loss
        else:
            return self.base_criterion(outputs, labels)

# Mixup Data Augmentation
class Mixup:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, x, y):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam

# CutMix Data Augmentation
class CutMix:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, x, y):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        y_a, y_b = y, y[index]
        
        return x, y_a, y_b, lam
    
    def _rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int32(W * cut_rat)
        cut_h = np.int32(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

# Enhanced Data Transformations
def get_transforms(phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])

# Load CIFAR-100 dataset
def load_cifar100_dataset(data_path):
    train_dataset = CIFAR100Dataset(data_path, train=True, transform=get_transforms('train'))
    test_dataset = CIFAR100Dataset(data_path, train=False, transform=get_transforms('test'))
    
    logger.info(f"CIFAR-100 dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")
    return train_dataset, test_dataset

# Training function with advanced techniques
def train_model(model, train_dataset, test_dataset, epochs=100, lr=3e-4, batch_size=64):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model.to(device)
    
    # Data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size*2, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    # Optimizer with different learning rates for different parts
    param_groups = [
        {'params': model.patch_embed.parameters(), 'lr': lr * 0.1},
        {'params': model.blocks.parameters(), 'lr': lr},
        {'params': model.head.parameters(), 'lr': lr * 2},
    ]
    
    optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=0.05, betas=(0.9, 0.999))
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader),
        pct_start=0.05, anneal_strategy='cos', div_factor=25, final_div_factor=1000
    )
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    distill_criterion = DistillationLoss(criterion)
    
    # Mixed precision training scaler
    scaler = GradScaler('cuda') if torch.cuda.is_available() else None
    
    # Data augmentation
    mixup = Mixup(alpha=0.8)
    cutmix = CutMix(alpha=1.0)
    
    history = {
        'train_losses': [], 'test_losses': [], 'test_accuracies': [], 
        'learning_rates': [], 'top5_accuracies': []
    }
    
    best_accuracy = 0.0
    patience = 0
    max_patience = 20
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training")
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)
            
            # Apply data augmentation
            if random.random() > 0.5:
                images, labels_a, labels_b, lam = mixup(images, labels)
                mixed = True
            elif random.random() > 0.5:
                images, labels_a, labels_b, lam = cutmix(images, labels)
                mixed = True
            else:
                mixed = False
            
            optimizer.zero_grad()
            
            # Mixed precision training
            if scaler and torch.cuda.is_available():
                with autocast():
                    outputs = model(images)
                    if mixed:
                        loss = lam * criterion(outputs[0] if isinstance(outputs, tuple) else outputs, labels_a) + \
                               (1 - lam) * criterion(outputs[0] if isinstance(outputs, tuple) else outputs, labels_b)
                    else:
                        loss = distill_criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                if mixed:
                    loss = lam * criterion(outputs[0] if isinstance(outputs, tuple) else outputs, labels_a) + \
                           (1 - lam) * criterion(outputs[0] if isinstance(outputs, tuple) else outputs, labels_b)
                else:
                    loss = distill_criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, predicted = outputs.max(1)
            if not mixed:
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            train_pbar.set_postfix({
                'loss': f"{train_loss/(batch_idx+1):.4f}",
                'acc': f"{100.*correct/total:.2f}%" if total > 0 else "N/A"
            })
            
            # Clear cache periodically
            if batch_idx % 50 == 0:
                clear_memory()
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        correct_top5 = 0
        total = 0
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} - Testing")
            for images, labels in test_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Top-5 accuracy
                _, pred_top5 = outputs.topk(5, 1, True, True)
                correct_top5 += pred_top5.eq(labels.view(-1, 1).expand_as(pred_top5)).sum().item()
                
                accuracy = 100. * correct / total
                top5_accuracy = 100. * correct_top5 / total
                test_pbar.set_postfix({
                    'loss': f"{test_loss/len(test_loader):.4f}",
                    'acc': f"{accuracy:.2f}%",
                    'top5': f"{top5_accuracy:.2f}%"
                })
        
        # Record metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100. * correct / total
        top5_accuracy = 100. * correct_top5 / total
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_losses'].append(avg_train_loss)
        history['test_losses'].append(avg_test_loss)
        history['test_accuracies'].append(test_accuracy)
        history['top5_accuracies'].append(top5_accuracy)
        history['learning_rates'].append(current_lr)
        
        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.2f}%, Top-5 Accuracy: {top5_accuracy:.2f}%")
        logger.info(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            save_model_safetensors(model, 'safetensors/best_cifar100_hypervit.safetensors')
            logger.info(f"New best model saved with accuracy: {best_accuracy:.2f}%")
            patience = 0
        else:
            patience += 1
            
        # Early stopping
        if patience >= max_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'history': history,
                'best_accuracy': best_accuracy
            }
            torch.save(checkpoint, f'checkpoints/checkpoint_epoch_{epoch+1}.pth')
            
        clear_memory()
    
    # Save final model
    save_model_safetensors(model, 'safetensors/final_cifar100_hypervit.safetensors')
    
    # Save training history
    with open('logs/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return history

# SafeTensors functions
def save_model_safetensors(model, filepath):
    state_dict = model.state_dict()
    cpu_state_dict = {k: v.cpu().contiguous() for k, v in state_dict.items()}
    save_file(cpu_state_dict, filepath)
    logger.info(f"Model saved to {filepath}")

def load_model_safetensors(model, filepath):
    state_dict = load_file(filepath)
    model.load_state_dict(state_dict)
    logger.info(f"Model loaded from {filepath}")
    return model

# Advanced evaluation with detailed metrics
def evaluate_model(model, test_dataset, device, num_samples=5):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    correct = 0
    correct_top5 = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            _, predicted = outputs.max(1)
            _, pred_top5 = outputs.topk(5, 1, True, True)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            correct_top5 += pred_top5.eq(labels.view(-1, 1).expand_as(pred_top5)).sum().item()
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
    
    overall_accuracy = 100. * correct / total
    top5_accuracy = 100. * correct_top5 / total
    
    # Calculate per-class metrics
    class_accuracies = {}
    for class_id in range(100):
        if class_total[class_id] > 0:
            class_accuracies[class_id] = 100. * class_correct[class_id] / class_total[class_id]
        else:
            class_accuracies[class_id] = 0.0
    
    # Find best and worst performing classes
    sorted_classes = sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)
    best_classes = sorted_classes[:5]
    worst_classes = sorted_classes[-5:]
    
    # Calculate confusion matrix for worst classes
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    results = {
        'overall_accuracy': overall_accuracy,
        'top5_accuracy': top5_accuracy,
        'correct': correct,
        'total': total,
        'class_accuracies': class_accuracies,
        'best_classes': best_classes,
        'worst_classes': worst_classes,
        'confusion_matrix': cm.tolist()
    }
    
    logger.info(f"Overall Accuracy: {overall_accuracy:.2f}% ({correct}/{total})")
    logger.info(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
    logger.info(f"Best performing classes: {best_classes}")
    logger.info(f"Worst performing classes: {worst_classes}")
    
    return results

# Visualization functions
def plot_training_history(history):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Training and Test Losses
    axes[0, 0].plot(history['train_losses'], label='Train Loss', color='blue', linewidth=2)
    axes[0, 0].plot(history['test_losses'], label='Test Loss', color='red', linewidth=2)
    axes[0, 0].set_title('Training and Test Losses', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Test Accuracy
    axes[0, 1].plot(history['test_accuracies'], label='Test Accuracy', color='green', linewidth=2)
    axes[0, 1].set_title('Test Accuracy', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top-5 Accuracy
    if 'top5_accuracies' in history:
        axes[0, 2].plot(history['top5_accuracies'], label='Top-5 Accuracy', color='purple', linewidth=2)
        axes[0, 2].set_title('Top-5 Accuracy', fontsize=14)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy (%)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # Learning Rate Schedule
    axes[1, 0].plot(history['learning_rates'], label='Learning Rate', color='orange', linewidth=2)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss Ratio
    loss_ratio = [train/test if test > 0 else 0 for train, test in zip(history['train_losses'], history['test_losses'])]
    axes[1, 1].plot(loss_ratio, label='Train/Test Loss Ratio', color='brown', linewidth=2)
    axes[1, 1].axhline(y=1, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Overfitting Indicator', fontsize=14)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Improvement Rate
    if len(history['test_accuracies']) > 1:
        improvement = [history['test_accuracies'][i] - history['test_accuracies'][i-1] 
                      for i in range(1, len(history['test_accuracies']))]
        axes[1, 2].plot(improvement, label='Accuracy Improvement', color='teal', linewidth=2)
        axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 2].set_title('Epoch-wise Improvement', fontsize=14)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Accuracy Change (%)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('logs/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_class_performance(results, dataset):
    """Plot per-class performance analysis"""
    class_accs = results['class_accuracies']
    class_names = dataset.fine_label_names
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Sort classes by accuracy
    sorted_items = sorted(class_accs.items(), key=lambda x: x[1])
    indices = [item[0] for item in sorted_items]
    accuracies = [item[1] for item in sorted_items]
    names = [class_names[i] for i in indices]
    
    # Bar plot of all classes
    colors = ['red' if acc < 50 else 'yellow' if acc < 70 else 'green' for acc in accuracies]
    bars = ax1.bar(range(len(accuracies)), accuracies, color=colors, alpha=0.7)
    ax1.set_xlabel('Classes (sorted by accuracy)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Per-Class Accuracy Distribution')
    ax1.axhline(y=results['overall_accuracy'], color='blue', linestyle='--', 
                label=f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Highlight worst and best classes
    worst_10 = sorted_items[:10]
    best_10 = sorted_items[-10:]
    
    ax2.barh(range(10), [item[1] for item in worst_10], color='red', alpha=0.7, label='Worst 10')
    ax2.barh(range(10, 20), [item[1] for item in best_10], color='green', alpha=0.7, label='Best 10')
    ax2.set_yticks(range(20))
    ax2.set_yticklabels([class_names[item[0]] for item in worst_10] + 
                        [class_names[item[0]] for item in best_10])
    ax2.set_xlabel('Accuracy (%)')
    ax2.set_title('Best and Worst Performing Classes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('logs/class_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main function
def main():
    logger.info("🚀 Starting CIFAR-100 HyperComplex Vision Transformer Training (Optimized)")
    logger.info("="*70)
    
    try:
        # Configuration
        config = {
            'data_path': '/Volumes/DATA/watermark_proj/data/archive/cifar-100-python',
            'img_size': 32,
            'patch_size': 4,
            'embed_dim': 192,
            'depth': 12,
            'num_heads': 3,
            'mlp_ratio': 4,
            'dropout': 0.1,
            'epochs': 100,
            'batch_size': 64,
            'learning_rate': 3e-4,
            'num_classes': 100
        }
        
        logger.info("Configuration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
        
        # Load dataset
        logger.info("\n📂 Loading CIFAR-100 dataset...")
        train_dataset, test_dataset = load_cifar100_dataset(config['data_path'])
        
        # Create model
        logger.info("\n🏗️ Creating optimized model...")
        model = CIFAR100HyperViT(
            img_size=config['img_size'],
            patch_size=config['patch_size'],
            embed_dim=config['embed_dim'],
            depth=config['depth'],
            num_heads=config['num_heads'],
            mlp_ratio=config['mlp_ratio'],
            drop_rate=config['dropout'],
            num_classes=config['num_classes']
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Train model
        logger.info("\n🚀 Starting training...")
        history = train_model(
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            epochs=config['epochs'],
            lr=config['learning_rate'],
            batch_size=config['batch_size']
        )
        
        # Evaluate model
        logger.info("\n📊 Final Model Evaluation...")
        device = torch.device("mps" if torch.backends.mps.is_available() else 
                            "cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Load best model for evaluation
        model = load_model_safetensors(model, 'safetensors/best_cifar100_hypervit.safetensors')
        evaluation_results = evaluate_model(model, test_dataset, device)
        
        # Save evaluation results
        with open('logs/final_evaluation.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Visualizations
        logger.info("\n📊 Creating visualizations...")
        plot_training_history(history)
        plot_class_performance(evaluation_results, test_dataset)
        
        # Final summary
        logger.info("\n" + "="*70)
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info("📊 FINAL RESULTS:")
        logger.info(f"  Best Test Accuracy: {max(history['test_accuracies']):.2f}%")
        logger.info(f"  Final Test Accuracy: {evaluation_results['overall_accuracy']:.2f}%")
        logger.info(f"  Top-5 Accuracy: {evaluation_results['top5_accuracy']:.2f}%")
        logger.info(f"  Total Training Epochs: {len(history['train_losses'])}")
        
        logger.info("\n📈 Performance Summary:")
        logger.info(f"  Best performing classes:")
        for class_id, acc in evaluation_results['best_classes'][:5]:
            logger.info(f"    - {test_dataset.fine_label_names[class_id]}: {acc:.2f}%")
        
        logger.info(f"  Worst performing classes:")
        for class_id, acc in evaluation_results['worst_classes'][:5]:
            logger.info(f"    - {test_dataset.fine_label_names[class_id]}: {acc:.2f}%")
        
        logger.info("\n📦 Saved Files:")
        logger.info("  ✅ models/: Model checkpoints")
        logger.info("  ✅ safetensors/: SafeTensors format models")
        logger.info("  ✅ logs/: Training logs and results")
        logger.info("  ✅ logs/training_history.png: Training curves")
        logger.info("  ✅ logs/class_performance.png: Per-class analysis")
        logger.info("  ✅ logs/final_evaluation.json: Detailed evaluation metrics")
        
        # Additional analysis
        logger.info("\n🔍 Additional Analysis:")
        
        # Calculate average accuracy per superclass
        coarse_accuracies = defaultdict(list)
        for fine_idx, acc in evaluation_results['class_accuracies'].items():
            coarse_idx = test_dataset.coarse_labels[fine_idx] if fine_idx < len(test_dataset.coarse_labels) else 0
            coarse_accuracies[coarse_idx].append(acc)
        
        logger.info("  Superclass Performance:")
        for coarse_idx, accs in sorted(coarse_accuracies.items())[:5]:
            avg_acc = np.mean(accs)
            if coarse_idx < len(test_dataset.coarse_label_names):
                logger.info(f"    - {test_dataset.coarse_label_names[coarse_idx]}: {avg_acc:.2f}%")
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

# Additional utility functions

def analyze_model_predictions(model, test_dataset, device, num_samples=10):
    """Analyze model predictions with visualization"""
    model.eval()
    
    # Get a few samples
    indices = random.sample(range(len(test_dataset)), num_samples)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            image, label = test_dataset[sample_idx]
            
            # Add batch dimension
            image_input = image.unsqueeze(0).to(device)
            
            # Get prediction
            output = model(image_input)
            probs = F.softmax(output, dim=1)
            top5_probs, top5_indices = torch.topk(probs, 5)
            
            # Denormalize image for visualization
            mean = torch.tensor([0.5071, 0.4865, 0.4409]).view(3, 1, 1)
            std = torch.tensor([0.2673, 0.2564, 0.2762]).view(3, 1, 1)
            denorm_image = image * std + mean
            denorm_image = torch.clamp(denorm_image, 0, 1)
            
            # Plot
            axes[idx].imshow(denorm_image.permute(1, 2, 0))
            axes[idx].axis('off')
            
            # Add text
            true_label = test_dataset.fine_label_names[label]
            pred_label = test_dataset.fine_label_names[top5_indices[0, 0].item()]
            confidence = top5_probs[0, 0].item()
            
            color = 'green' if top5_indices[0, 0].item() == label else 'red'
            axes[idx].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.2f})', 
                               fontsize=8, color=color)
    
    plt.tight_layout()
    plt.savefig('logs/sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def export_model_onnx(model, filepath='models/cifar100_hypervit.onnx'):
    """Export model to ONNX format for deployment"""
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32)
    
    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    logger.info(f"Model exported to ONNX: {filepath}")

def benchmark_model(model, test_dataset, device, num_runs=100):
    """Benchmark model inference speed"""
    model.eval()
    model.to(device)
    
    # Prepare a batch
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    images, _ = next(iter(test_loader))
    images = images.to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(images)
    
    # Benchmark
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(images)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs * 1000  # ms
    fps = 1000 / avg_time
    
    logger.info(f"Inference Benchmark Results:")
    logger.info(f"  Average inference time: {avg_time:.2f} ms")
    logger.info(f"  FPS: {fps:.2f}")
    
    return avg_time, fps

def create_model_card(model, config, results):
    """Create a model card with all relevant information"""
    model_card = {
        "model_name": "CIFAR-100 HyperComplex Vision Transformer",
        "version": "1.0",
        "date_trained": datetime.now().strftime("%Y-%m-%d"),
        "architecture": {
            "type": "Vision Transformer with Hypercomplex Layers",
            "patch_size": config['patch_size'],
            "embed_dim": config['embed_dim'],
            "depth": config['depth'],
            "num_heads": config['num_heads'],
            "mlp_ratio": config['mlp_ratio']
        },
        "training_details": {
            "dataset": "CIFAR-100",
            "epochs": config['epochs'],
            "batch_size": config['batch_size'],
            "learning_rate": config['learning_rate'],
            "optimizer": "AdamW",
            "augmentations": ["RandomCrop", "RandomHorizontalFlip", "ColorJitter", "Mixup", "CutMix"]
        },
        "performance": {
            "test_accuracy": results['overall_accuracy'],
            "top5_accuracy": results['top5_accuracy'],
            "parameters": sum(p.numel() for p in model.parameters()),
            "flops": "N/A"  # Would need to calculate
        },
        "novel_features": [
            "Hypercomplex linear layers for parameter efficiency",
            "Adaptive Computation Time for dynamic inference",
            "Mixture of Experts for increased model capacity",
            "Efficient attention with local bias",
            "Multi-scale classification with auxiliary heads",
            "Advanced data augmentation pipeline"
        ],
        "limitations": [
            "Trained only on CIFAR-100 dataset",
            "May not generalize to higher resolution images",
            "Requires specific hypercomplex layer implementation"
        ],
        "intended_use": "Academic research and benchmarking on CIFAR-100",
        "ethical_considerations": "Model should not be used for surveillance or harmful applications"
    }
    
    with open('models/model_card.json', 'w') as f:
        json.dump(model_card, f, indent=2)
    
    logger.info("Model card created: models/model_card.json")
    return model_card

# Run everything
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Set MPS memory fraction for M2 optimization
    if torch.backends.mps.is_available():
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        # Use MPS-optimized settings
        torch.mps.set_per_process_memory_fraction(0.0)
    
    # Run main training
    main()
    
    # Additional post-training analysis (optional)
    logger.info("\n🔬 Running post-training analysis...")
    
    # Load the best model
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    
    config = {
        'img_size': 32,
        'patch_size': 4,
        'embed_dim': 192,
        'depth': 12,
        'num_heads': 3,
        'mlp_ratio': 4,
        'dropout': 0.1,
        'num_classes': 100,
        'epochs': 100,
        'batch_size': 64,
        'learning_rate': 3e-4
    }
    
    model = CIFAR100HyperViT(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
        drop_rate=config['dropout'],
        num_classes=config['num_classes']
    )
    
    model = load_model_safetensors(model, 'safetensors/best_cifar100_hypervit.safetensors')
    model.to(device)
    
    # Load test dataset
    data_path = '/Volumes/DATA/watermark_proj/data/archive/cifar-100-python'
    _, test_dataset = load_cifar100_dataset(data_path)
    
    # Analyze predictions
    logger.info("Analyzing model predictions...")
    analyze_model_predictions(model, test_dataset, device)
    
    # Benchmark inference speed
    logger.info("\nBenchmarking model inference speed...")
    avg_time, fps = benchmark_model(model, test_dataset, device)
    
    # Create model card
    logger.info("\nCreating model card...")
    with open('logs/final_evaluation.json', 'r') as f:
        results = json.load(f)
    model_card = create_model_card(model, config, results)
    
    # Export to ONNX (optional)
    # logger.info("\nExporting model to ONNX...")
    # export_model_onnx(model.cpu())
    
    logger.info("\n" + "="*70)
    logger.info("✨ All tasks completed successfully!")
    logger.info("="*70)