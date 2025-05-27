import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
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

# Append HyperNets path
sys.path.append('/Volumes/DATA/watermark_proj/HyperNets')
try:
    from layers.ph_layers import PHConv2d, PHMLinear
    logger.info("Successfully imported PHM layers")
except ImportError as e:
    logger.error(f"Failed to import PHM layers: {e}")
    sys.exit(1)

from ndlinear import NdLinear

# Create directories for saving artifacts
os.makedirs('models', exist_ok=True)
os.makedirs('tensors', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('safetensors', exist_ok=True)

# Dataset configurations
DATASET_CONFIGS = {
    'flowers': {
        'path': '/Volumes/DATA/watermark_proj/data/archive/flowers',
        'num_classes': 102,
        'domain_id': 0
    },
    'food101': {
        'path': '/Volumes/DATA/watermark_proj/data/archive/food-101',
        'num_classes': 101,
        'domain_id': 1
    },
    'cifar100': {
        'path': '/Volumes/DATA/watermark_proj/data/cifar-100',
        'num_classes': 100,
        'domain_id': 2
    },
    'caltech256': {
        'path': '/Volumes/DATA/watermark_proj/data/caltech-256',
        'num_classes': 256,
        'domain_id': 3
    },
    'places365': {
        'path': '/Volumes/DATA/watermark_proj/data/places365',
        'num_classes': 365,
        'domain_id': 4
    },
    'original': {
        'path': '/Volumes/DATA/watermark_proj/data/archive/intel',
        'num_classes': 6,
        'domain_id': 5
    }
}

# Hypercomplex Linear Layer using PHM
class PHLinear(nn.Module):
    def __init__(self, n, in_features, out_features):
        super(PHLinear, self).__init__()
        self.phm_linear = PHMLinear(n=n, in_features=in_features, out_features=out_features)
    
    def forward(self, x):
        return self.phm_linear(x)

# Hypercomplex Positional Encoding
class HypercomplexPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000, n=3):
        super(HypercomplexPositionalEncoding, self).__init__()
        self.n = n
        self.dim = dim
        
        # Create hypercomplex positional encodings
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        # Apply different phases for hypercomplex components
        for i in range(n):
            phase_shift = (2 * math.pi * i) / n
            pe[:, i::n] = torch.sin(position * div_term + phase_shift)
            if i * 2 + 1 < dim:
                pe[:, i * 2 + 1::n] = torch.cos(position * div_term + phase_shift)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Quantum-Inspired Multi-Head Attention
class QuantumMultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, n=3):
        super(QuantumMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.n = n
        
        # Hypercomplex Q, K, V projections using PHM
        self.qkv = PHLinear(n=n, in_features=dim, out_features=dim * 3)
        self.proj = PHLinear(n=n, in_features=dim, out_features=dim)
        
        # Quantum superposition parameters
        self.superposition_weights = nn.Parameter(torch.randn(num_heads, num_heads))
        self.entanglement_matrix = nn.Parameter(torch.randn(num_heads, num_heads))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Standard attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Quantum superposition of attention patterns
        superposition_attn = torch.zeros_like(attn)
        for i in range(self.num_heads):
            for j in range(self.num_heads):
                superposition_attn[:, i] += self.superposition_weights[i, j] * attn[:, j]
        
        # Apply entanglement
        entangled_attn = torch.zeros_like(attn)
        for i in range(self.num_heads):
            entangled_attn[:, i] = torch.tanh(
                superposition_attn[:, i] + 
                torch.sum(self.entanglement_matrix[i] * superposition_attn, dim=1)
            )
        
        attn = entangled_attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

# Hypercomplex Mixture of Experts
class HypercomplexMoE(nn.Module):
    def __init__(self, dim, num_experts=8, expert_dim=None, n=3, top_k=2):
        super(HypercomplexMoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.n = n
        expert_dim = expert_dim or dim * 4
        
        # Hypercomplex router using PHM
        self.router = PHLinear(n=n, in_features=dim, out_features=num_experts)
        
        # Hypercomplex experts using PHM and NdLinear
        self.experts = nn.ModuleList([
            nn.Sequential(
                PHLinear(n=n, in_features=dim, out_features=expert_dim),
                nn.GELU(),
                NdLinear([expert_dim], [dim])
            ) for _ in range(num_experts)
        ])
        
        # Domain-specific routing weights
        self.domain_routing = nn.Parameter(torch.randn(6, num_experts))  # 6 domains
        
    def forward(self, x, domain_id=None):
        B, N, C = x.shape
        
        # Get routing weights
        router_logits = self.router(x)  # (B, N, num_experts)
        
        # Apply domain-specific routing if available
        if domain_id is not None:
            domain_bias = self.domain_routing[domain_id].unsqueeze(0).unsqueeze(0)
            router_logits = router_logits + domain_bias
        
        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        # Apply experts
        expert_outputs = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, :, i]
            expert_weight = top_k_weights[:, :, i].unsqueeze(-1)
            
            # Apply each expert to corresponding tokens
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_input = x[mask]
                    if expert_input.numel() > 0:
                        expert_output = self.experts[expert_id](expert_input)
                        expert_outputs[mask] += expert_weight[mask] * expert_output
        
        return expert_outputs

# Meta-Learning MAML wrapper
class MAMLAdapter(nn.Module):
    def __init__(self, model, meta_lr=0.01):
        super(MAMLAdapter, self).__init__()
        self.model = model
        self.meta_lr = meta_lr
        
        # Meta parameters for quick adaptation using NdLinear
        self.adaptation_layers = nn.ModuleList([
            NdLinear([192], [192]) for _ in range(6)  # One per domain
        ])
        
    def forward(self, x, domain_id=None, adaptation_steps=5):
        if domain_id is not None and self.training:
            # Quick adaptation for new domain
            adapted_params = {}
            for name, param in self.model.named_parameters():
                if 'classifier' in name:
                    adapted_params[name] = param - self.meta_lr * param.grad if param.grad is not None else param
            
            # Apply domain-specific adaptation using NdLinear
            x = self.adaptation_layers[domain_id](x)
        
        return self.model(x)

# Fractal Memory Bank
class FractalMemoryBank(nn.Module):
    def __init__(self, dim, depth=5, capacity_base=64):
        super(FractalMemoryBank, self).__init__()
        self.depth = depth
        self.dim = dim
        
        # Multi-scale memory banks
        self.memory_banks = nn.ModuleList([
            nn.Parameter(torch.randn(capacity_base * (2 ** i), dim))
            for i in range(depth)
        ])
        
        # Attention mechanisms for memory retrieval using NdLinear
        self.memory_attention = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
            for _ in range(depth)
        ])
        
        # Memory processing layers using NdLinear
        self.memory_processors = nn.ModuleList([
            NdLinear([dim], [dim]) for _ in range(depth)
        ])
        
    def forward(self, x, scale_level=0):
        if scale_level >= self.depth:
            return x
        
        # Retrieve from memory at current scale
        memory = self.memory_banks[scale_level]
        attended_memory, _ = self.memory_attention[scale_level](x, memory, memory)
        
        # Process with NdLinear
        attended_memory = self.memory_processors[scale_level](attended_memory)
        
        # Recursive call to next scale
        if scale_level < self.depth - 1:
            attended_memory = self.forward(attended_memory, scale_level + 1)
        
        return x + attended_memory

# Gradient Reversal Layer
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

# Domain Adversarial Network
class DomainAdversarialNetwork(nn.Module):
    def __init__(self, dim, num_domains=6):
        super(DomainAdversarialNetwork, self).__init__()
        self.domain_classifier = nn.Sequential(
            NdLinear([dim], [dim // 2]),
            nn.ReLU(),
            NdLinear([dim // 2], [num_domains])
        )
        
    def forward(self, features, alpha=1.0):
        # Gradient reversal layer
        reversed_features = GradientReversalLayer.apply(features, alpha)
        domain_logits = self.domain_classifier(reversed_features)
        return domain_logits

# Advanced Transformer Block
class AdvancedTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1, n=3):
        super(AdvancedTransformerBlock, self).__init__()
        
        # Quantum-inspired attention
        self.attn = QuantumMultiHeadAttention(dim, num_heads, n)
        
        # Hypercomplex MoE
        self.moe = HypercomplexMoE(dim, num_experts=8, n=n)
        
        # Fractal memory
        self.memory = FractalMemoryBank(dim, depth=3)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        # Biological plasticity simulation using NdLinear
        self.plasticity_layer = NdLinear([dim], [dim])
        self.hebbian_learning = nn.Parameter(torch.zeros(dim, dim))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, domain_id=None):
        # Quantum attention with residual connection
        attended = x + self.dropout(self.attn(self.norm1(x)))
        
        # MoE with domain awareness
        moe_output = self.moe(self.norm2(attended), domain_id)
        expert_output = attended + self.dropout(moe_output)
        
        # Fractal memory integration
        memory_output = self.memory(expert_output)
        
        # Biological plasticity update
        if self.training:
            # Hebbian learning rule: neurons that fire together, wire together
            activity = torch.mean(memory_output, dim=0)
            self.hebbian_learning.data += 0.001 * torch.outer(activity, activity)
            plasticity_update = self.plasticity_layer(memory_output)
            memory_output = memory_output + 0.1 * plasticity_update
        
        return self.norm3(memory_output)

# Revolutionary Multi-Domain HyperComplex Vision Transformer
class MultiDomainHyperViT(nn.Module):
    def __init__(self, image_size=96, patch_size=16, dim=192, depth=12, heads=3, 
                 mlp_dim=768, dropout=0.1, n=3, num_domains=6):
        super(MultiDomainHyperViT, self).__init__()
        self.num_domains = num_domains
        self.dim = dim
        num_patches = (image_size // patch_size) ** 2

        # Adaptive Hypercomplex patch embedding using PHConv2d
        self.patch_embed = PHConv2d(n=n, in_features=3, out_features=dim, 
                                   kernel_size=patch_size, stride=patch_size, padding=0)
        
        # Hypercomplex positional encoding
        self.pos_encoding = HypercomplexPositionalEncoding(dim, max_len=num_patches + 1, n=n)
        
        # Domain-specific CLS tokens
        self.cls_tokens = nn.Parameter(torch.randn(num_domains, 1, dim))
        
        # Advanced transformer blocks
        self.transformer_blocks = nn.ModuleList([
            AdvancedTransformerBlock(dim, heads, mlp_dim, dropout, n) 
            for _ in range(depth)
        ])
        
        # Domain-specific classification heads using NdLinear
        self.domain_classifiers = nn.ModuleDict({
            'flowers': nn.Sequential(
                NdLinear([dim], [mlp_dim]),
                nn.GELU(),
                NdLinear([mlp_dim], [102])
            ),
            'food101': nn.Sequential(
                NdLinear([dim], [mlp_dim]),
                nn.GELU(),
                NdLinear([mlp_dim], [101])
            ),
            'cifar100': nn.Sequential(
                NdLinear([dim], [mlp_dim]),
                nn.GELU(),
                NdLinear([mlp_dim], [100])
            ),
            'caltech256': nn.Sequential(
                NdLinear([dim], [mlp_dim]),
                nn.GELU(),
                NdLinear([mlp_dim], [256])
            ),
            'places365': nn.Sequential(
                NdLinear([dim], [mlp_dim]),
                nn.GELU(),
                NdLinear([mlp_dim], [365])
            ),
            'original': nn.Sequential(
                NdLinear([dim], [mlp_dim]),
                nn.GELU(),
                NdLinear([mlp_dim], [6])
            )
        })
        
        # Domain adversarial network
        self.domain_adversarial = DomainAdversarialNetwork(dim, num_domains)
        
        # Universal feature extractor using NdLinear
        self.universal_features = nn.Sequential(
            NdLinear([dim], [mlp_dim]),
            nn.GELU(),
            NdLinear([mlp_dim], [dim])
        )

    def forward(self, x, domain_name=None, domain_id=None, return_features=False):
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, dim)
        
        # Add domain-specific CLS token
        if domain_id is not None:
            cls_tokens = self.cls_tokens[domain_id].expand(x.shape[0], -1, -1)
        else:
            cls_tokens = self.cls_tokens[0].expand(x.shape[0], -1, -1)  # Default to first domain
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer blocks with domain awareness
        for block in self.transformer_blocks:
            x = block(x, domain_id)
        
        # Extract CLS token features
        cls_features = x[:, 0]  # (B, dim)
        
        # Universal feature processing
        universal_features = self.universal_features(cls_features)
        
        if return_features:
            return universal_features
        
        # Domain-specific classification
        if domain_name and domain_name in self.domain_classifiers:
            output = self.domain_classifiers[domain_name](universal_features)
        else:
            # Default to original domain
            output = self.domain_classifiers['original'](universal_features)
        
        return output

    def get_domain_features(self, x, domain_id):
        """Extract domain-invariant features for adversarial training"""
        features = self.forward(x, domain_id=domain_id, return_features=True)
        return features

# Data transformations with advanced augmentation
def get_transforms(dataset_name, phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((110, 110)),
            transforms.RandomCrop(96),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# Dataset loading function
def load_datasets():
    datasets_dict = {}
    for name, config in DATASET_CONFIGS.items():
        if os.path.exists(config['path']):
            try:
                train_dataset = datasets.ImageFolder(
                    root=config['path'], 
                    transform=get_transforms(name, 'train')
                )
                val_dataset = datasets.ImageFolder(
                    root=config['path'].replace('train', 'test') if 'train' in config['path'] else config['path'], 
                    transform=get_transforms(name, 'val')
                )
                datasets_dict[name] = {
                    'train': train_dataset,
                    'val': val_dataset,
                    'config': config
                }
                logger.info(f"Loaded {name} dataset: {len(train_dataset)} train, {len(val_dataset)} val")
            except Exception as e:
                logger.warning(f"Failed to load {name} dataset: {e}")
        else:
            logger.warning(f"Dataset path not found: {config['path']}")
    
    return datasets_dict

# Advanced training function with multi-domain support
def train_multi_domain_model(model, datasets_dict, epochs=50, lr=1e-4, batch_size=8):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model.to(device)
    
    # Create data loaders for all domains
    train_loaders = {}
    val_loaders = {}
    
    for domain_name, domain_data in datasets_dict.items():
        train_loaders[domain_name] = DataLoader(
            domain_data['train'], 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2, 
            pin_memory=False
        )
        val_loaders[domain_name] = DataLoader(
            domain_data['val'], 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=False
        )
    
    # Optimizers and criteria
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    classification_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'val_accuracies': {},
        'domain_losses': []
    }
    
    for domain_name in datasets_dict.keys():
        history['val_accuracies'][domain_name] = []
    
    best_avg_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        domain_loss = 0.0
        num_batches = 0
        
        # Progressive domain mixing - start with single domains, gradually mix
        domain_mixing_prob = min(0.8, epoch / (epochs * 0.5))
        
        # Training loop with domain rotation
        domain_names = list(datasets_dict.keys())
        for batch_idx in range(max(len(loader) for loader in train_loaders.values())):
            # Rotate through domains or mix them
            if random.random() < domain_mixing_prob and len(domain_names) > 1:
                # Multi-domain batch
                selected_domains = random.sample(domain_names, min(3, len(domain_names)))
            else:
                # Single domain batch
                selected_domains = [random.choice(domain_names)]
            
            for domain_name in selected_domains:
                if batch_idx >= len(train_loaders[domain_name]):
                    continue
                
                try:
                    images, labels = next(iter(train_loaders[domain_name]))
                    images, labels = images.to(device), labels.to(device)
                    
                    domain_id = datasets_dict[domain_name]['config']['domain_id']
                    
                    optimizer.zero_grad()
                    
                    # Classification loss
                    outputs = model(images, domain_name=domain_name, domain_id=domain_id)
                    cls_loss = classification_criterion(outputs, labels)
                    
                    # Domain adversarial loss
                    features = model.get_domain_features(images, domain_id)
                    domain_labels = torch.full((images.size(0),), domain_id, device=device)
                    domain_logits = model.domain_adversarial(features, alpha=0.5)
                    adv_loss = domain_criterion(domain_logits, domain_labels)
                    
                    # Total loss
                    total_loss = cls_loss + 0.1 * adv_loss
                    total_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    train_loss += cls_loss.item()
                    domain_loss += adv_loss.item()
                    num_batches += 1
                    
                except StopIteration:
                    continue
                except Exception as e:
                    logger.error(f"Training error: {e}")
                    continue
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_results = {}
        total_val_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for domain_name, val_loader in val_loaders.items():
                domain_correct = 0
                domain_total = 0
                domain_val_loss = 0.0
                
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    domain_id = datasets_dict[domain_name]['config']['domain_id']
                    
                    outputs = model(images, domain_name=domain_name, domain_id=domain_id)
                    loss = classification_criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    domain_correct += (preds == labels).sum().item()
                    domain_total += labels.size(0)
                    domain_val_loss += loss.item()
                
                domain_accuracy = domain_correct / domain_total if domain_total > 0 else 0
                val_results[domain_name] = {
                    'accuracy': domain_accuracy,
                    'loss': domain_val_loss / len(val_loader)
                }
                
                total_correct += domain_correct
                total_samples += domain_total
                total_val_loss += domain_val_loss
        
        # Calculate averages
        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0
        avg_val_loss = total_val_loss / sum(len(loader) for loader in val_loaders.values())
        avg_val_accuracy = total_correct / total_samples if total_samples > 0 else 0
        avg_domain_loss = domain_loss / num_batches if num_batches > 0 else 0
        
        # Update history
        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(avg_val_loss)
        history['domain_losses'].append(avg_domain_loss)
        
        for domain_name in val_results:
            history['val_accuracies'][domain_name].append(val_results[domain_name]['accuracy'])
        
        # Logging
        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        logger.info(f"Avg Val Accuracy: {avg_val_accuracy:.4f}, Domain Loss: {avg_domain_loss:.4f}")
        
        for domain_name, results in val_results.items():
            logger.info(f"{domain_name}: Acc={results['accuracy']:.4f}")
        
        # Save best model
        if avg_val_accuracy > best_avg_accuracy:
            best_avg_accuracy = avg_val_accuracy
            save_model_safetensors(model, 'safetensors/best_hypervit_model.safetensors')
            logger.info(f"New best model saved with accuracy: {best_avg_accuracy:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'history': history,
                'best_accuracy': best_avg_accuracy
            }
            torch.save(checkpoint, f'checkpoints/checkpoint_epoch_{epoch+1}.pth')
    
    # Save final model
    save_model_safetensors(model, 'safetensors/final_hypervit_model.safetensors')
    
    # Save training history
    with open('logs/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return history

# SafeTensors saving function
# SafeTensors saving function (continued from where it stopped)
def save_model_safetensors(model, filepath):
    """Save model state dict using safetensors format"""
    state_dict = model.state_dict()
    # Convert all tensors to CPU and ensure they're contiguous
    cpu_state_dict = {k: v.cpu().contiguous() for k, v in state_dict.items()}
    save_file(cpu_state_dict, filepath)
    logger.info(f"Model saved to {filepath}")

def load_model_safetensors(model, filepath):
    """Load model state dict from safetensors format"""
    state_dict = load_file(filepath)
    model.load_state_dict(state_dict)
    logger.info(f"Model loaded from {filepath}")
    return model

# Advanced evaluation function
def evaluate_model(model, datasets_dict, device):
    """Comprehensive evaluation across all domains"""
    model.eval()
    evaluation_results = {}
    
    with torch.no_grad():
        for domain_name, domain_data in datasets_dict.items():
            val_loader = DataLoader(
                domain_data['val'], 
                batch_size=16, 
                shuffle=False, 
                num_workers=2
            )
            
            correct = 0
            total = 0
            domain_id = domain_data['config']['domain_id']
            
            for images, labels in tqdm(val_loader, desc=f"Evaluating {domain_name}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, domain_name=domain_name, domain_id=domain_id)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
            accuracy = correct / total if total > 0 else 0
            evaluation_results[domain_name] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }
            
            logger.info(f"{domain_name}: {accuracy:.4f} ({correct}/{total})")
    
    # Calculate overall metrics
    total_correct = sum(r['correct'] for r in evaluation_results.values())
    total_samples = sum(r['total'] for r in evaluation_results.values())
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    evaluation_results['overall'] = {
        'accuracy': overall_accuracy,
        'correct': total_correct,
        'total': total_samples
    }
    
    return evaluation_results

# Visualization functions
def plot_training_history(history):
    """Plot comprehensive training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training and validation losses
    axes[0, 0].plot(history['train_losses'], label='Train Loss', color='blue')
    axes[0, 0].plot(history['val_losses'], label='Val Loss', color='red')
    axes[0, 0].set_title('Training and Validation Losses')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Domain adversarial losses
    axes[0, 1].plot(history['domain_losses'], label='Domain Loss', color='green')
    axes[0, 1].set_title('Domain Adversarial Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Validation accuracies per domain
    for domain_name, accuracies in history['val_accuracies'].items():
        axes[1, 0].plot(accuracies, label=domain_name)
    axes[1, 0].set_title('Validation Accuracies by Domain')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate schedule (if available)
    if 'learning_rates' in history:
        axes[1, 1].plot(history['learning_rates'], label='Learning Rate', color='orange')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'Learning Rate\nData Not Available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Learning Rate Schedule')
    
    plt.tight_layout()
    plt.savefig('logs/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_attention_maps(model, sample_images, device):
    """Visualize attention patterns across domains"""
    model.eval()
    attention_maps = {}
    
    with torch.no_grad():
        for domain_name, images in sample_images.items():
            images = images.to(device)
            
            # Hook to capture attention weights
            attention_weights = []
            
            def attention_hook(module, input, output):
                if hasattr(output, 'shape') and len(output.shape) == 4:
                    attention_weights.append(output.cpu())
            
            # Register hooks
            hooks = []
            for name, module in model.named_modules():
                if 'attn' in name:
                    hooks.append(module.register_forward_hook(attention_hook))
            
            # Forward pass
            _ = model(images, domain_name=domain_name)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            attention_maps[domain_name] = attention_weights
    
    return attention_maps

# Model analysis functions
def analyze_expert_utilization(model, datasets_dict, device):
    """Analyze which experts are used for different domains"""
    model.eval()
    expert_usage = defaultdict(lambda: defaultdict(int))
    
    # Hook to capture expert selections
    expert_selections = []
    
    def expert_hook(module, input, output):
        if hasattr(module, 'top_k_indices'):
            expert_selections.append(module.top_k_indices.cpu())
    
    # Register hooks on MoE layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, HypercomplexMoE):
            hooks.append(module.register_forward_hook(expert_hook))
    
    with torch.no_grad():
        for domain_name, domain_data in datasets_dict.items():
            val_loader = DataLoader(domain_data['val'], batch_size=8, shuffle=False)
            domain_id = domain_data['config']['domain_id']
            
            for batch_idx, (images, labels) in enumerate(val_loader):
                if batch_idx >= 10:  # Analyze first 10 batches
                    break
                
                images = images.to(device)
                _ = model(images, domain_name=domain_name, domain_id=domain_id)
                
                # Process expert selections
                for selections in expert_selections:
                    for batch_item in selections:
                        for token_experts in batch_item:
                            for expert_id in token_experts:
                                expert_usage[domain_name][expert_id.item()] += 1
                
                expert_selections.clear()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return dict(expert_usage)

# Cross-domain transfer analysis
def analyze_cross_domain_transfer(model, source_domain, target_domain, datasets_dict, device):
    """Analyze knowledge transfer between domains"""
    model.eval()
    
    # Get source domain features
    source_loader = DataLoader(
        datasets_dict[source_domain]['val'], 
        batch_size=32, 
        shuffle=False
    )
    
    source_features = []
    source_labels = []
    
    with torch.no_grad():
        for images, labels in source_loader:
            images = images.to(device)
            features = model.get_domain_features(
                images, 
                datasets_dict[source_domain]['config']['domain_id']
            )
            source_features.append(features.cpu())
            source_labels.append(labels)
    
    source_features = torch.cat(source_features, dim=0)
    source_labels = torch.cat(source_labels, dim=0)
    
    # Get target domain features
    target_loader = DataLoader(
        datasets_dict[target_domain]['val'], 
        batch_size=32, 
        shuffle=False
    )
    
    target_features = []
    target_labels = []
    
    with torch.no_grad():
        for images, labels in target_loader:
            images = images.to(device)
            features = model.get_domain_features(
                images, 
                datasets_dict[target_domain]['config']['domain_id']
            )
            target_features.append(features.cpu())
            target_labels.append(labels)
    
    target_features = torch.cat(target_features, dim=0)
    target_labels = torch.cat(target_labels, dim=0)
    
    # Compute feature similarity matrix
    similarity_matrix = torch.cosine_similarity(
        source_features.unsqueeze(1), 
        target_features.unsqueeze(0), 
        dim=2
    )
    
    # Analysis results
    transfer_analysis = {
        'avg_similarity': similarity_matrix.mean().item(),
        'max_similarity': similarity_matrix.max().item(),
        'min_similarity': similarity_matrix.min().item(),
        'similarity_std': similarity_matrix.std().item(),
        'source_domain': source_domain,
        'target_domain': target_domain
    }
    
    return transfer_analysis

# Progressive training strategy
def progressive_training(datasets_dict, start_size=64, end_size=96, phases=3):
    """Implement progressive training with increasing image sizes"""
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Starting progressive training on {device}")
    
    # Phase-specific configurations
    phase_configs = []
    sizes = np.linspace(start_size, end_size, phases).astype(int)
    epochs_per_phase = [15, 15, 20]  # Total 50 epochs
    
    for i, (size, epochs) in enumerate(zip(sizes, epochs_per_phase)):
        phase_configs.append({
            'phase': i + 1,
            'image_size': size,
            'epochs': epochs,
            'lr': 1e-4 / (2 ** i),  # Decreasing learning rate
            'batch_size': max(4, 16 // (2 ** i))  # Decreasing batch size
        })
    
    # Initialize model with final size
    model = MultiDomainHyperViT(
        image_size=end_size,
        patch_size=16,
        dim=192,
        depth=12,
        heads=3,
        mlp_dim=768,
        dropout=0.1,
        n=3,
        num_domains=6
    )
    
    total_history = {
        'phase_histories': [],
        'transfer_analyses': [],
        'expert_utilizations': []
    }
    
    for phase_config in phase_configs:
        logger.info(f"\n{'='*50}")
        logger.info(f"PHASE {phase_config['phase']}: Image Size {phase_config['image_size']}")
        logger.info(f"{'='*50}")
        
        # Update transforms for current phase
        current_datasets = {}
        for name, config in DATASET_CONFIGS.items():
            if os.path.exists(config['path']):
                train_transform = transforms.Compose([
                    transforms.Resize((int(phase_config['image_size'] * 1.15), 
                                     int(phase_config['image_size'] * 1.15))),
                    transforms.RandomCrop(phase_config['image_size']),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                         saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                
                val_transform = transforms.Compose([
                    transforms.Resize((phase_config['image_size'], phase_config['image_size'])),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                
                try:
                    train_dataset = datasets.ImageFolder(root=config['path'], transform=train_transform)
                    val_dataset = datasets.ImageFolder(
                        root=config['path'].replace('train', 'test') if 'train' in config['path'] else config['path'],
                        transform=val_transform
                    )
                    current_datasets[name] = {
                        'train': train_dataset,
                        'val': val_dataset,
                        'config': config
                    }
                except Exception as e:
                    logger.warning(f"Failed to load {name} for phase {phase_config['phase']}: {e}")
        
        # Train for this phase
        phase_history = train_multi_domain_model(
            model=model,
            datasets_dict=current_datasets,
            epochs=phase_config['epochs'],
            lr=phase_config['lr'],
            batch_size=phase_config['batch_size']
        )
        
        total_history['phase_histories'].append({
            'phase': phase_config['phase'],
            'config': phase_config,
            'history': phase_history
        })
        
        # Analyze expert utilization after each phase
        expert_util = analyze_expert_utilization(model, current_datasets, device)
        total_history['expert_utilizations'].append({
            'phase': phase_config['phase'],
            'utilization': expert_util
        })
        
        # Cross-domain transfer analysis
        if len(current_datasets) >= 2:
            domain_names = list(current_datasets.keys())
            transfer_analysis = analyze_cross_domain_transfer(
                model, domain_names[0], domain_names[1], current_datasets, device
            )
            total_history['transfer_analyses'].append({
                'phase': phase_config['phase'],
                'analysis': transfer_analysis
            })
        
        # Save phase checkpoint
        checkpoint = {
            'phase': phase_config['phase'],
            'model_state_dict': model.state_dict(),
            'phase_config': phase_config,
            'phase_history': phase_history
        }
        torch.save(checkpoint, f'checkpoints/phase_{phase_config["phase"]}_checkpoint.pth')
        logger.info(f"Phase {phase_config['phase']} completed and saved")
    
    # Save final comprehensive results
    with open('logs/progressive_training_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = json.loads(json.dumps(total_history, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
        json.dump(serializable_history, f, indent=2)
    
    # Final model save
    save_model_safetensors(model, 'safetensors/progressive_final_model.safetensors')
    
    return model, total_history

# Main execution function
def main():
    """Main execution pipeline"""
    logger.info("🚀 Starting Multi-Domain HyperComplex Vision Transformer Training")
    logger.info("="*70)
    
    try:
        # Load datasets
        logger.info("📂 Loading datasets...")
        datasets_dict = load_datasets()
        
        if not datasets_dict:
            logger.error("No datasets found! Please check dataset paths.")
            return
        
        logger.info(f"✅ Loaded {len(datasets_dict)} datasets: {list(datasets_dict.keys())}")
        
        # Choose training strategy
        training_mode = input("\nSelect training mode:\n1. Standard Training\n2. Progressive Training\nEnter choice (1 or 2): ").strip()
        
        if training_mode == "2":
            # Progressive training
            logger.info("🔄 Starting Progressive Training...")
            model, history = progressive_training(datasets_dict)
            
        else:
            # Standard training
            logger.info("🎯 Starting Standard Training...")
            
            # Initialize model
            model = MultiDomainHyperViT(
                image_size=96,
                patch_size=16,
                dim=192,
                depth=12,
                heads=3,
                mlp_dim=768,
                dropout=0.1,
                n=3,
                num_domains=len(datasets_dict)
            )
            
            # Train model
            history = train_multi_domain_model(
                model=model,
                datasets_dict=datasets_dict,
                epochs=50,
                lr=1e-4,
                batch_size=8
            )
        
        # Final evaluation
        logger.info("🔍 Final Model Evaluation...")
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)
        
        evaluation_results = evaluate_model(model, datasets_dict, device)
        
        # Save evaluation results
        with open('logs/final_evaluation.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Plot results
        if training_mode != "2":  # Standard training
            plot_training_history(history)
        
        # Expert utilization analysis
        logger.info("🧠 Analyzing Expert Utilization...")
        expert_util = analyze_expert_utilization(model, datasets_dict, device)
        
        with open('logs/expert_utilization.json', 'w') as f:
            json.dump(expert_util, f, indent=2)
        
        # Print final summary
        logger.info("\n" + "="*70)
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info("📊 FINAL RESULTS:")
        
        for domain_name, results in evaluation_results.items():
            if domain_name != 'overall':
                logger.info(f"  {domain_name}: {results['accuracy']:.4f}")
        
        logger.info(f"  Overall Accuracy: {evaluation_results['overall']['accuracy']:.4f}")
        logger.info("\n📁 Saved Files:")
        logger.info("  • models/: Model checkpoints")
        logger.info("  • safetensors/: SafeTensors format models")
        logger.info("  • logs/: Training logs and results")
        logger.info("  • checkpoints/: Training checkpoints")
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        raise

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Run main pipeline
    main()