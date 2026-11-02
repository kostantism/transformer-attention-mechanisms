# Attention Mechanisms and Transformer Networks

This project explores **Attention Mechanisms** and **Transformer Networks**, which form the foundation of state-of-the-art models in natural language processing and computer vision. Developed as part of the Machine Learning and Applications course at Harokopio University of Athens, the project is divided into three distinct parts: implementing Scaled Dot-Product Attention from scratch in NumPy, analyzing the parameter structure of a Transformer Encoder in PyTorch, and performing fine-tuning/linear probing comparisons between a Convolutional Neural Network (ResNet50) and a Vision Transformer (ViT-B/16).

## Description

The project covers three core modules:

* **Part 1 (NumPy Attention):** Custom implementation of the Scaled Dot-Product Attention mechanism using pure NumPy, validated against PyTorch, and visualized with attention heatmaps for self-attention (English) and cross-attention (Greek-to-English).

* **Part 2 (Transformer Parameter Analysis):** Detailed mathematical and programmatic breakdown of a custom `SmallTransformerEncoder` in PyTorch, analyzing layer shapes, tensor unification, and architectural differences from the classic Transformer design.

* **Part 3 (Vision Models Comparison):** Comparative transfer learning evaluation between **ResNet50** and **ViT-B/16** on the real-world **Oxford-IIIT Pet Dataset**, investigating both Linear Probing and Full Fine-Tuning strategies.

## Features

* **Custom Attention Class:** Modular `scaled_dot_product_attention` implementation in NumPy with complete tensor shape verification.

* **Framework Benchmarking:** Direct numerical comparison of custom NumPy outputs with PyTorch's native `F.scaled_dot_product_attention` using `np.allclose`.

* **Attention Heatmaps:** Visualization tools using Matplotlib to analyze cross-lingual token alignments.

* **Transfer Learning Pipeline:** Complete training, validation, and testing pipelines supporting hyperparameter tuning, differential learning rates, and automated performance tracking for computer vision models.

## Prerequisites & Libraries

* Python 3.x

* NumPy

* PyTorch & Torchvision

* Matplotlib (for visualization and heatmaps)

* Scikit-learn

## File Structure

* `part_1.py`: Implementation of Scaled Dot-Product Attention in NumPy, PyTorch validation, and attention heatmap generation.

* `part_2.py`: Definition and parameter breakdown of the `SmallTransformerEncoder` architecture.

* `part_3.py`: Transfer learning experiments (Linear Probing and Full Fine-Tuning) comparing ResNet50 and ViT-B/16 on the Oxford-IIIT Pet Dataset.

* `matrices/`: Directory containing pre-computed embedding and projection weight matrices (`E.npy`, `W_Q_self.npy`, `W_K_self.npy`, `W_V_self.npy`, `W_Q_cross.npy`, `W_K_cross.npy`, `W_V_cross.npy`) along with the vocabulary file (`vocab.json`).

## Usage

To run the respective parts of the project, make sure to update your file paths to load matrices from the `matrices/` folder if necessary, and execute the following commands in your terminal:

```bash
# Part 1: Scaled Dot-Product Attention (NumPy & PyTorch Validation)
python part_1.py

# Part 2: Transformer Encoder Parameter Analysis
python part_2.py

# Part 3: ResNet50 and ViT-B/16 Transfer Learning Experiments
python part_3.py
```