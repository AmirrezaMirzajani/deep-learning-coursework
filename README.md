# Deep Learning Coursework

This repository contains the exercises, Python implementations, detailed solutions, and technical reports prepared for the **Deep Learning course taught by Dr. Maryam Shoaran** during **Summer 2025**.

The exercises follow a progressive path from classical machine-learning classifiers to fully connected neural networks and convolutional neural networks. Most experiments are conducted on the **CIFAR-10 image-classification dataset**.

## Repository Contents

Each exercise is organized in a separate folder and includes:

- Python source code or Jupyter notebooks
- Detailed written solutions
- PDF technical reports
- Experimental results and visualizations
- A separate README file describing the corresponding exercise

## Exercises

### Exercise 1 — k-Nearest Neighbor Classification

Implementation and evaluation of the **k-Nearest Neighbor (kNN)** classifier for CIFAR-10 image classification.

Main topics include:

- Distance-based image classification
- Euclidean-distance computation
- Five-fold cross-validation
- Hyperparameter selection for `k`
- Analysis of training-set size
- Visualization of the distance matrix
- Evaluation on training and test subsets

### Exercise 2 — Linear SVM and Softmax Classifiers

Implementation and comparative evaluation of two fundamental linear classifiers:

- Multiclass Support Vector Machine
- Softmax classifier

Main topics include:

- SVM hinge loss
- Softmax cross-entropy loss
- Naive and vectorized implementations
- Analytical gradient computation
- Numerical gradient checking
- Stochastic Gradient Descent
- L2 regularization
- Learning-rate and regularization tuning
- Visualization of learned class weights

### Exercise 3 — Two-Layer Neural Network

Implementation of a fully connected two-layer neural network from scratch for CIFAR-10 image classification.

Main topics include:

- Forward propagation
- Backpropagation
- ReLU activation
- Softmax loss
- Mini-batch Stochastic Gradient Descent
- L2 regularization
- Learning-rate decay
- Hidden-layer size selection
- Random hyperparameter search
- Training, validation, and test evaluation

### Exercise 4 — Modular Fully Connected Neural-Network Framework

Development of a modular deep-learning framework for building and training fully connected neural networks.

Main topics include:

- Modular affine and activation layers
- Forward and backward passes
- Numerical gradient checking
- Two-layer and multilayer neural networks
- Training Solver implementation
- Mini-batch training
- SGD
- SGD with Momentum
- RMSProp
- Adam
- L2 regularization
- Dropout
- Batch Normalization
- Comparison of optimization algorithms

### Exercise 5 — Convolutional Neural Networks with TensorFlow

Implementation and comparison of fully connected and convolutional neural-network architectures using TensorFlow.

Main topics include:

- TensorFlow low-level operations
- Keras Model API
- Keras Sequential API
- Fully connected neural networks
- Convolutional neural networks
- Batch Normalization
- Max Pooling
- Dropout
- Adam optimization
- CIFAR-10 image classification
- Comparison of different neural-network architectures

## Repository Structure

```text
deep-learning-coursework/
├── Exercise-01-kNN/
│   ├── code/
│   ├── report/
│   └── README.md
│
├── Exercise-02-SVM-Softmax/
│   ├── code/
│   ├── report/
│   └── README.md
│
├── Exercise-03-Two-Layer-Network/
│   ├── code/
│   ├── report/
│   └── README.md
│
├── Exercise-04-Modular-Neural-Network/
│   ├── code/
│   ├── report/
│   └── README.md
│
├── Exercise-05-TensorFlow-CNN/
│   ├── code/
│   ├── report/
│   └── README.md
│
├── requirements.txt
└── README.md
