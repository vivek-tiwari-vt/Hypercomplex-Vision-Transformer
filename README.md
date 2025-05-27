# Hypercomplex Vision Transformer

This repository contains the implementation of a Hypercomplex Vision Transformer (HyperViT) model, which leverages hypercomplex algebra for efficient and effective image classification.

## Project Overview

The HyperViT model combines the strengths of Vision Transformers with hypercomplex neural networks to create a powerful and parameter-efficient architecture for image classification tasks. This implementation focuses on the CIFAR-100 dataset but can be adapted for other image classification tasks.

## Features

- Hypercomplex neural network layers for parameter efficiency
- Vision Transformer architecture for capturing global dependencies
- Comprehensive evaluation pipeline with detailed metrics
- Training and inference benchmarking
- Model visualization and analysis tools

## Directory Structure

```
├── model_test.py          # Comprehensive model evaluation script
├── hypervit_comp.py       # HyperViT model implementation
├── hypervit_comp1.py      # Enhanced HyperViT implementation
├── ndlinear.py            # N-dimensional linear layer implementation
├── train_hypervit.py      # Training script for HyperViT
├── images/                # Generated visualizations
│   ├── confusion_matrix.png
│   └── training_history.png
├── models/                # Saved model weights
└── logs/                  # Training logs and history
```

## Installation

```bash
# Clone the repository
git clone https://github.com/vivek-tiwari-vt/Hypercomplex-Vision-Transformer.git
cd Hypercomplex-Vision-Transformer

# Install dependencies
pip install torch torchvision numpy matplotlib seaborn scikit-learn tqdm
```

## Usage

### Training

```bash
python train_hypervit.py
```

### Evaluation

```bash
python model_test.py
```

## Results

The HyperViT model achieves competitive performance on the CIFAR-100 dataset while using fewer parameters compared to traditional Vision Transformer architectures.

## License

MIT

## Citation

If you use this code in your research, please cite:

```
@misc{tiwari2023hypervit,
  author = {Tiwari, Vivek},
  title = {Hypercomplex Vision Transformer},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/vivek-tiwari-vt/Hypercomplex-Vision-Transformer}}
}
```

## Acknowledgements

This implementation builds upon the work from the following repositories and papers:

- [HyperNets](https://github.com/eleGAN23/HyperNets)
- [Vision Transformer](https://github.com/google-research/vision_transformer)