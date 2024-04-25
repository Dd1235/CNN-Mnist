# MNIST Digit Classifier in PyTorch

## Project Description

This repository contains a simple PyTorch implementation of a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset. The project includes a single Python script, `main.py`, which integrates model training, evaluation, and metric reporting.

## Model Architecture

The CNN architecture defined in the project includes two convolutional layers followed by max-pooling layers, and a fully connected layer to output the class probabilities for the digits 0-9. Here's a brief overview:

- **Convolutional Layer 1**: 8 filters, 3x3 kernel, stride of 1, padding of 1.
- **Max-Pooling Layer 1**: 2x2 pool size, stride of 2.
- **Convolutional Layer 2**: 16 filters, 3x3 kernel, stride of 1, padding of 1.
- **Max-Pooling Layer 2**: 2x2 pool size, stride of 2.
- **Fully Connected Layer**: Maps to 10 output classes.

## Setup and Usage

### Prerequisites

- Python 3.8+
- PyTorch 1.7+
- torchvision
- matplotlib (for plotting metrics and results)
- scikit-learn (for classification metrics)

### Running the Notebook

To run the model on the MNIST dataset, open and execute the cells in `main.ipynb` using Jupyter Notebook or JupyterLab:

```bash
jupyter notebook main.ipynb
```

### Performance Metrics

The MNIST Digit Classifier demonstrates strong performance on both the training and test datasets. Below are the summarized accuracy metrics:

| Metric            | Value  |
| ----------------- | ------ |
| Training Accuracy | 97.15% |
| Test Accuracy     | 97.48% |

### Detailed Classification Report (Test)

The classification report provides a deeper insight into the model's performance across individual classes. Key metrics such as precision, recall, and F1-score for each class are highlighted below:

| Class | Precision | Recall | F1-Score |
| ----- | --------- | ------ | -------- |
| 0     | 0.98      | 0.99   | 0.98     |
| 1     | 0.99      | 0.99   | 0.99     |
| 2     | 0.95      | 0.98   | 0.97     |
| 3     | 0.98      | 0.97   | 0.97     |
| 4     | 0.99      | 0.97   | 0.98     |
| 5     | 0.99      | 0.98   | 0.98     |
| 6     | 0.98      | 0.98   | 0.98     |
| 7     | 0.97      | 0.97   | 0.97     |
| 8     | 0.96      | 0.95   | 0.96     |
| 9     | 0.97      | 0.97   | 0.97     |

For the Confusion Matrix, refer to `main.ipynb` notebook.
