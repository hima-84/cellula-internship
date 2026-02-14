# Teeth Classification Using Traditional CNN

A deep learning project for automated classification of dental images into 7 different categories using a custom CNN architecture built with PyTorch.

##  Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Usage](#usage)
- [Results](#results)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)

##  Overview

This project implements a custom Convolutional Neural Network (CNN) called **ClinicalCNN_900K** for classifying dental images into 7 distinct categories. The model is optimized for clinical dental images and uses modern deep learning techniques including batch normalization, dropout, and data augmentation to achieve high accuracy.

The project includes:
- Custom lightweight CNN architecture (~900K parameters)
- Class imbalance handling using weighted loss
- Comprehensive data visualization and analysis
- Confusion matrix and performance metrics
- GPU acceleration support (tested on RTX 3050)

##  Features

- **7-class teeth classification** with clinical-grade accuracy
- **Custom CNN architecture** - ClinicalCNN_900K with ~900K parameters
- **Class imbalance handling** - Weighted cross-entropy loss
- **Data augmentation** - Proper preprocessing for clinical images
- **GPU optimized** - CUDA support for faster training
- **Comprehensive visualizations**:
  - Class distribution analysis
  - Augmentation examples
  - Training/validation curves
  - Confusion matrix
  - Per-class performance metrics
- **Model checkpointing** - Saves best performing model automatically
- **Memory efficient** - Optimized for RTX 3050 (4GB VRAM)

##  Installation

### Prerequisites

- Python 3.7 or higher
- NVIDIA GPU (recommended, tested on RTX 3050)
- CUDA toolkit (for GPU acceleration)
- pytorch 2.7.1+cu118

### Required Dependencies

Install all required packages using pip:

```bash
pip install torch torchvision
pip install matplotlib
pip install numpy
pip install Pillow
pip install tqdm
pip install scikit-learn
pip install seaborn
```

**Standard Library Modules** (no installation needed):
- `os` - File and directory operations
- `shutil` - High-level file operations
- `collections` - Counter for class distribution analysis
- `gc` - Garbage collection for memory management

### Quick Install (requirements.txt)

Create a `requirements.txt` file:

```txt
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.5.0
numpy>=1.21.0
Pillow>=9.0.0
tqdm>=4.65.0
scikit-learn>=1.0.0
seaborn>=0.11.0
```

Then install:

```bash
pip install -r requirements.txt
```

### Verify Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Dataset Structure

The dataset should be organized in the following structure with **7 classes**:

```
Teeth_Dataset/
├── Training/
│   ├── CaS/           # Caries (Cavity on Surface)
│   ├── CoS/           # Crown on Surface  
│   ├── Gum/           # Gum condition
│   ├── MC/            # Missing Crown
│   ├── OC/            # Original Crown
│   ├── OLP/           # Oral Lichen Planus
│   └── OT/            # Oral Thrush
├── Validation/
│   ├── CaS/
│   ├── CoS/
│   ├── Gum/
│   ├── MC/
│   ├── OC/
│   ├── OLP/
│   └── OT/
└── Testing/
    ├── CaS/
    ├── CoS/
    ├── Gum/
    ├── MC/
    ├── OC/
    ├── OLP/
    └── OT/
```

### Dataset Statistics

The project handles **class imbalance** using weighted cross-entropy loss. The training script automatically:
- Analyzes class distribution
- Calculates appropriate class weights
- Applies weighted loss during training

### Image Requirements

- **Format:** JPG, PNG, or other common image formats
- **Size:** Images are automatically resized to 224×224 pixels
- **Channels:** RGB (3 channels)
- **Preprocessing:** Normalized with mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5]

## Training Configuration

### Hyperparameters

```python
IMG_SIZE = 224           # Input image size (224×224)
BATCH_SIZE = 8           # Small batch size for RTX 3050 (4GB VRAM)
NUM_WORKERS = 0          # Single-threaded data loading for stability
NUM_CLASSES = 7          # Number of teeth categories
EPOCHS = 35              # Training epochs
LEARNING_RATE = 0.0005   # Initial learning rate
WEIGHT_DECAY = 0.001
```

### Training Strategy

**Loss Function:**
- Weighted CrossEntropyLoss (handles class imbalance)
- Class weights calculated as: `total_samples / (num_classes × class_samples)`

**Optimizer:**
- Adam optimizer with weight decay (L2 regularization)

**Learning Rate Scheduler:**
- CosineAnnealingLR (T_max=20)
- Smooth learning rate decay for better convergence

**Data Augmentation:**
```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
])
```

**Best Model Selection:**
- Automatically saves model with highest validation accuracy
- Saved as `best_model_improved.pth`

##  Usage

### 1. Prepare Your Dataset

Organize your data following the [Dataset Structure](#dataset-structure) above.

### 2. Training the Model

Open and run the Jupyter notebook:

```bash
jupyter notebook Teeth_classification_Imporved_Net.ipynb
```

**Training Process:**

1. **Setup and Import Libraries** (Cell 0)
   - Imports all dependencies
   - Checks GPU availability
   - Clears GPU cache

2. **Configure Paths** (Cell 1)
   - Update `TRAIN_PATH`, `VAL_PATH`, `TEST_PATH` to your dataset locations

3. **Data Visualization** (Cells 4-5)
   - View class distribution
   - Examine augmentation examples

4. **Train Model** (Cell 10)
   - Runs complete training loop with progress bars
   - Automatically saves best model
   - Displays training/validation metrics each epoch

### 3. Evaluating the Model

After training, evaluate on test set:

```python
# Load model
model = ClinicalCNN_900K(num_classes=7).to(device)
model.load_state_dict(torch.load("best_model_improved.pth"))

# Run evaluation (Cell 16)
test_loss, test_acc, y_true, y_pred = test_model(
    model, test_loader, criterion, device
)
print(f"Test Accuracy: {test_acc:.2f}%")
```

### 4. Making Predictions

Visualize predictions on test images (Cell 19):

```python
# The notebook includes code to:
# - Load random test images
# - Display predictions with confidence scores
# - Show true labels for comparison
```

### 5. Analyzing Results

Generate confusion matrix (Cell 21):

```python
cm = confusion_matrix(all_labels, all_preds)
# Visualizes model performance per class
```

##  Model Architecture

### ClinicalCNN_900K

A custom lightweight CNN architecture specifically designed for clinical dental image classification with approximately **900,000 parameters**.

#### Architecture Details

**Block 1:** (Input → 32 channels)
- Conv2D (3→32, kernel=3×3, padding=1)
- BatchNorm2D + ReLU
- Conv2D (32→32, kernel=3×3, padding=1)
- BatchNorm2D + ReLU
- MaxPool2D (2×2)

**Block 2:** (32 → 64 channels)
- Conv2D (32→64, kernel=3×3, padding=1)
- BatchNorm2D + ReLU
- Conv2D (64→64, kernel=3×3, padding=1)
- BatchNorm2D + ReLU
- MaxPool2D (2×2)

**Block 3:** (64 → 128 channels)
- Conv2D (64→128, kernel=3×3, padding=1)
- BatchNorm2D + ReLU
- Conv2D (128→128, kernel=3×3, padding=1)
- BatchNorm2D + ReLU
- MaxPool2D (2×2)

**Block 4:** (128 → 192 channels)
- Conv2D (128→192, kernel=3×3, padding=1)
- BatchNorm2D + ReLU
- MaxPool2D (2×2)

**Global Average Pooling:**
- AdaptiveAvgPool2D (output size: 2×2)

**Classifier:**
- Flatten
- Linear (192×2×2 → 256)
- ReLU
- Dropout (p=0.5)
- Linear (256 → 7 classes)

#### Key Features

- **Batch Normalization:** Stabilizes training and improves convergence
- **Dropout (0.5):** Prevents overfitting
- **Global Average Pooling:** Reduces parameters while maintaining spatial information
- **Progressive Channel Expansion:** 3 → 32 → 64 → 128 → 192
- **Memory Efficient:** Optimized for GPUs with limited VRAM (4GB+)

## Results

### Model Performance

*Update these values after training:*

| Metric | Value |
|--------|-------|
| **Training Accuracy** | 68.25% |
| **Validation Accuracy** | 72.37% |
| **Test Accuracy** |73.63% |
| **Total Parameters** | ~900,000 |
| **Model Size** | ~3.5 MB |

### Per-Class Performance

The model provides detailed metrics for each of the 7 teeth categories:

----------------------------------------------------------------------
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| CaS   | 0.6154    | 1.0000 | 0.7619   | 8       |
| CoS   | 0.9444    | 1.0000 | 0.9714   | 17      |
| Gum   | 0.7000    | 0.6364 | 0.6667   | 11      |
| MC    | 0.8889    | 0.7619 | 0.8205   | 21      |
| OC    | 0.8182    | 0.7500 | 0.7826   | 12      |
| OLP   | 0.7692    | 0.6667 | 0.7143   | 15      |
| OT    | 0.8824    | 0.9375 | 0.9091   | 16      |
| **Weighted Avg** | **0.8282** | **0.8200** | **0.8182** | **100** |


##  Visualizations

The project generates several visualizations to understand model performance:

### 1. Class Distribution
- **File:** `class_distribution.png`
- Shows training data balance across 7 classes
- Helps identify class imbalance issues

### 2. Data Augmentation Examples
- **File:** `augmentation_examples.png`
- Compares original vs augmented images
- Verifies preprocessing pipeline

### 3. Training Curves
- Training loss and accuracy over epochs
- Validation loss and accuracy over epochs
- Helps identify overfitting/underfitting

### 4. Confusion Matrix
- Heatmap showing prediction accuracy per class
- Identifies which classes are confused with each other
- Generated using scikit-learn and seaborn

### 5. Sample Predictions
- Visual display of model predictions
- Shows predicted class, confidence score, and true label
- Useful for qualitative assessment

##  Saved Models

The training process saves two model files:

1. **best_model_improved900.pth** - Final model after training

To load a saved model:

```python
model = ClinicalCNN_900K(num_classes=7)
model.load_state_dict(torch.load('best_model_improved900.pth'))
model.eval()
```

##  Troubleshooting

### Common Issues

**CUDA Out of Memory:**
- Reduce `BATCH_SIZE` (try 4 or 2)
- Set `NUM_WORKERS = 0`
- Clear GPU cache: `torch.cuda.empty_cache()`

**Slow Training:**
- Ensure GPU is being used: check "Using device: cuda"
- Consider using mixed precision training for RTX GPUs
- Verify CUDA toolkit is properly installed

**Import Errors:**
- Verify all dependencies are installed
- Check Python version (3.7+)
- Try reinstalling PyTorch: `pip install torch torchvision --upgrade`

