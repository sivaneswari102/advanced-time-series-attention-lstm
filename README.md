# Variational Autoencoder (VAE) for Anomaly Detection

## Project Description
This project implements a Variational Autoencoder (VAE) using deep learning to detect anomalies in images. The model is trained on normal MNIST digits and tested on artificially corrupted images. Reconstruction error is used to identify anomalies.

## Features
- VAE Encoder–Decoder architecture
- KL Divergence + Reconstruction Loss
- Synthetic anomaly generation
- Reconstruction error based detection
- ROC-AUC evaluation metric
- Visualization of error distribution

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

## How to Run
1. Install dependencies:
   pip install tensorflow numpy matplotlib scikit-learn

2. Run:
   python vae_anomaly_detection.py

## Output
- Trained VAE model
- Reconstruction error histogram
- AUC score for anomaly detection

## Author
Sivaneswari R
