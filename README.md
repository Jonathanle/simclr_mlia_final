# SimCLR MLIA Final Project


Melika Morsali Toshmanloui qfc2zn@virginia.edu

Jonathan Le pqq2hu@virginia.edu

Tseganesh Beyene Kebede ykq8wj@virginia.edu




# Running baseline

1. create a data directory ```./data/``` containing all the original brain dataset files
2. run ```python baseline.py```

# SimCLR Implementation for Brain Image Classification

## Overview

This repository contains an implementation of the **SimCLR framework** for contrastive learning, specifically applied to brain image classification tasks. The project explores self-supervised learning (SSL) techniques to learn meaningful representations from unlabeled data and demonstrates how these representations can be fine-tuned for downstream supervised learning tasks. 

Key contributions include:
1. Implementation of the SimCLR framework with custom data augmentation pipelines.
2. Use of ImageNet-pretrained ResNet encoders (ResNet18 and ResNet34) for representation learning.
3. Evaluation of different architectures, batch sizes, and epochs to optimize model performance.
4. Visualization of augmented views, accuracy trends, and detailed experiments.

## Repository Structure

- `Best_Model/`: Contains the notebook and saved model files for the best-performing architecture (ResNet18, batch size 512, and 100 epochs).
- `Experiments/`: Contains notebooks for all other configurations, including ResNet34 and variations in batch sizes and epochs.
- `README.md`: This document, providing an overview of the project.

## Key Features

1. **Self-Supervised Learning**: 
   - Pretraining is conducted using the SimCLR framework, employing a contrastive loss function to maximize agreement between positive pairs of augmented views.
   - Augmentation techniques include random cropping, color jitter, Gaussian blur, and horizontal flipping.

2. **Transfer Learning**:
   - The pretrained encoder is fine-tuned on a labeled dataset for brain image classification tasks, achieving significant improvements in accuracy.

3. **Experimentation**:
   - Comparison of different ResNet architectures (ResNet18, ResNet34) and configurations such as batch size and epochs.
   - Training results, including accuracy trends and performance metrics, are visualized for better insights.

## Performance

The best-performing model achieved:
- **Accuracy**: 90% on the test dataset.
- **Configuration**: ResNet18, batch size 512, and 100 epochs.

Other experiments include:
- ResNet34 with batch size 512 (Accuracy: 89%)
- ResNet18 with batch size 128 (Accuracy: 85%)

## Instructions

### Pretrained Model Training
1. Pretrain the SimCLR model using the script in the `Best_Model/` directory.
2. Ensure you have the dataset files and paths updated as per the instructions in the notebook.

### Fine-Tuning and Evaluation
1. Fine-tune the pretrained encoder on the labeled dataset.
2. Evaluate the performance using the classification script.

### Visualization
1. Generate and visualize augmented views used during pretraining.
2. Plot accuracy trends over epochs for both training and test sets.

### Save and Download Model
Use the included scripts to save and download the trained models for reuse.

## References
1. Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations," ICML 2020. ([Paper](https://arxiv.org/abs/2002.05709))
2. Official SimCLR GitHub repository: [SimCLR](https://github.com/google-research/simclr).

---

Feel free to reach out via the GitHub Issues tab for any questions or clarifications regarding the implementation.
