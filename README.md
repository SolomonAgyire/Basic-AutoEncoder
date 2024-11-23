#Autoencoder for Pulse Signal Reconstruction
Overview
This repository contains the implementation of an autoencoder model designed to reconstruct pulse signals. The project utilizes PyTorch Lightning. 

Repository Structure
dataloader.py: Handles the loading and preprocessing of pulse signal data, including normalization and train/validation split.
model.py: Defines the Autoencoder architecture using PyTorch Lightning, including both encoder and decoder components with ReLU activation functions and a final sigmoid activation to constrain output values between 0 and 1.
training.py: Script for training the model, logging metrics, and visualizing both the training process and the output reconstructions.

Data Handling
Data is normalized by dividing each pulse by its maximum value, preserving the physical characteristics inherent to each sample. This normalization step ensures that the model inputs are scaled appropriately without losing significant information about the original signals.

Model Architecture
The model comprises:
An encoder with three linear layers, reducing the dimensionality from 250 to 5, with ReLU activations to introduce non-linearity.
A decoder that mirrors the encoder's architecture, utilizing ReLU activations and ending in a sigmoid activation to reconstruct the original input signal within the range [0, 1].
Mean Squared Error (MSE) loss is employed to quantify the reconstruction error, facilitating regression-oriented learning.

To run the training process:
Ensure you have Python 3.x, PyTorch, PyTorch Lightning, and tensorboard installed.
Place your pulse data in the appropriate directory or update the file_path in dataloader.py.
Execute python training.py to start the training session, which will automatically handle validation and logging.
Visualizations
The training script includes functionality to visualize reconstructed signals against the original data, helping to qualitatively evaluate the model's performance. 
