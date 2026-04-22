# pruning-neural-network
#  Self-Pruning Neural Network

##  Overview
This project implements a neural network that **learns to prune itself during training** using learnable gating mechanisms and L1 regularization.

Instead of performing pruning after training, the network dynamically suppresses less important weights during training itself.


##  Core Idea

Each weight has an associated **learnable gate parameter**:

- Gate values ∈ [0,1] (via sigmoid)
- Effective weight = weight × gate
- Gate → 0 ⇒ connection is pruned



##  Loss Function

Total Loss = CrossEntropyLoss + λ × Sparsity Loss

- CrossEntropyLoss → classification performance  
- L1 Sparsity Loss → encourages gates to become zero  


##  Model Architecture

- Custom `PrunableLinear` layer
- Fully connected network:
  - Input: CIFAR-10 images (32×32×3)
  - Hidden layers: 512 → 256
  - Output: 10 classes


##  Results

| Lambda (λ) | Accuracy (%) | Sparsity (%) |
|------------|-------------|--------------|
| 0.0001     | 39.03       | 1.71         |
| 0.001      | 38.71       | 1.70         |
| 0.01       | 37.60       | 1.72         |


##  Observations

- As λ increases, **accuracy decreases slightly**
- Sparsity remains low (~1.7%) across all runs
- The pruning effect is limited in current setup


##  Interpretation

The results suggest that:

- The L1 regularization strength is not strong enough to enforce aggressive pruning  
- The model prioritizes classification accuracy over sparsity  
- The number of training epochs (5) is insufficient for gate values to converge  


##  Trade-off Insight

- Higher λ → Slight drop in accuracy  
- Lower λ → Slightly better accuracy  
- Sparsity remains nearly constant in this experiment  
