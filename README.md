# Horseshoe-Measurement-Prediction
Overview

This project develops a neural network model to predict the external curve length of horseshoes based on input attributes: internal curve length, width, and cord length. Implemented in MATLAB, the model uses backpropagation with a [5, 10] layer configuration, achieving a mean squared error (MSE) of 0.047 and 100% accuracy, surpassing a support vector machine (SVM) baseline (78.46% accuracy).

Technologies Used:
Programming Language: MATLAB
Libraries: MATLAB Neural Network Toolbox
Tools: MATLAB Editor, Git

Methodology:
Data Preprocessing: Normalized input attributes to ensure consistent scaling.
Model Architecture: Designed a feedforward neural network with two hidden layers ([5, 10] neurons).
Training: Applied backpropagation with a learning rate optimized for convergence.
Evaluation: Compared MSE and accuracy against an SVM model to validate performance.

Results
Mean Squared Error (MSE): 0.047
Accuracy: 100% on test data
Comparison: Outperformed SVM (78.46% accuracy), demonstrating superior predictive capability.
