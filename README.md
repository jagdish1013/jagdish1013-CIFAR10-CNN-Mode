# jagdish1013-CIFAR10-CNN-Mode
# CIFAR-10 Image Classification Using CNN (PyTorch)

A Convolutional Neural Network (CNN) built with **PyTorch** and **Torchvision** to classify images from the CIFAR-10 dataset into 10 object categories. The model is developed and trained in Google Colab.

## Project Overview

This project demonstrates an end-to-end image-classification workflow:
- Download and load the CIFAR-10 dataset using `torchvision.datasets`.
- Convert images to tensors and normalize them.
- Load batches using PyTorch `DataLoader`.
- Build a CNN with three convolutional blocks and two fully connected layers.
- Train the network for 10 epochs using Cross Entropy Loss and the Adam optimizer.
- Evaluate the trained model on the CIFAR-10 test set and print accuracy.

## Dataset

The project uses **CIFAR-10**, which contains 60,000 colour images of size 32 × 32 across 10 classes:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

- Training set: 50,000 images
- Test set: 10,000 images
- Image channels: 3 (RGB)

The notebook downloads the dataset automatically with `download=True`.

## Model Architecture

Input image: `3 × 32 × 32`

| Layer | Configuration | Output shape (channels × height × width) |
|---|---|---|
| Convolution 1 | 3 → 32 filters, 3×3 kernel, padding 1 + ReLU | 32 × 32 × 32 |
| Max Pooling 1 | 2×2, stride 2 | 32 × 16 × 16 |
| Convolution 2 | 32 → 64 filters, 3×3 kernel, padding 1 + ReLU | 64 × 16 × 16 |
| Max Pooling 2 | 2×2, stride 2 | 64 × 8 × 8 |
| Convolution 3 | 64 → 128 filters, 3×3 kernel, padding 1 + ReLU | 128 × 8 × 8 |
| Max Pooling 3 | 2×2, stride 2 | 128 × 4 × 4 |
| Flatten | 128 × 4 × 4 | 2,048 features |
| Fully connected 1 | 2,048 → 256 + ReLU | 256 |
| Output | 256 → 10 | 10 class scores |

## Training Configuration

- Framework: PyTorch
- Loss function: `nn.CrossEntropyLoss()`
- Optimizer: `optim.Adam(model.parameters())` (default learning rate)
- Batch size: 64
- Epochs: 10
- Image preprocessing: `ToTensor()` and normalization with mean `(0.5, 0.5, 0.5)` and standard deviation `(0.5, 0.5, 0.5)`

## Repository Contents

```text
.
├── CIFAR10_CNN_model.ipynb   # Google Colab/Jupyter notebook
├── README.md                 # Project documentation
└── data/                     # CIFAR-10 data (downloaded when notebook runs; normally not committed)
```

## Run in Google Colab

1. Open `CIFAR10_CNN_model.ipynb` in Google Colab.
2. Select **Runtime → Run all**.
3. The notebook downloads CIFAR-10, trains the CNN, and prints the test accuracy.

## Run Locally

Python 3.9+ is recommended.

```bash
pip install torch torchvision
jupyter notebook CIFAR10_CNN_model.ipynb
```

Run the notebook cells from top to bottom. A GPU can reduce training time; in Colab, select **Runtime → Change runtime type → T4 GPU** if available. The notebook as currently written does not explicitly move the model and batches to CUDA, so GPU execution requires adding device handling to the code.

## Evaluation

The notebook calculates test accuracy as:

```python
accuracy = correct_labels / total_labels * 100
```

The notebook file does not contain a saved execution output for the final accuracy, so no numerical result is claimed here. Run the evaluation cell to obtain the result for your training run.

## Future Improvements

- Add training and validation accuracy/loss plots.
- Add data augmentation and learning-rate scheduling.
- Tune hyperparameters and compare CNN architectures.
- Add a confusion matrix and per-class precision/recall.
- Save and reload the trained model checkpoint.

## Author

**Jagdish Meghwal**  
GitHub: [@jagdish1013](https://github.com/jagdish1013)

---

If you use or extend this project, feel free to fork the repository.

