Overview

This project implements a simplified Agro-Deep Learning Framework (ADLF) for crop yield prediction using multi-source agricultural data.
The model integrates:
Weather features
Soil parameters
Satellite vegetation index (NDVI)
Crop type information
Environmental/tabular features
The project is inspired by the research paper:
“Improving Crop Production Using an Agro-Deep Learning Framework in Precision Agriculture”
but implemented using publicly available Kaggle datasets and a custom multi-branch neural architecture.
Objective
To design a multi-input deep learning model capable of learning from diverse agricultural data sources and improving yield prediction accuracy compared to traditional ML approaches.

Key Features
Multi-Branch Deep Learning Architecture

Separate branches for:

Image / NDVI input

Weather input

Soil input

Satellite indices

Tabular environmental features

Crop type embeddings

 Advanced Model Components

Dense-Inception Blocks

Spatial Pyramid Pooling (SPP)

Channel & Spatial Attention (CnSAU)

Dual Output Heads (Regression + Classification)

 Baseline Machine Learning Models

Random Forest

Support Vector Machine (SVM)

Logistic Regression

 Performance Metrics

Accuracy

Precision

Recall
Dataset

The project uses the Crop Yield Prediction dataset from Kaggle:

🔗 Dataset Link:
https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset

Dataset includes:

Temperature, rainfall, humidity

 Soil pH, moisture

 NDVI vegetation index

 Crop type

 Yield (target variable)

Preprocessing:

One-hot encoding for categorical features

StandardScaler for normalization

Stratified 80/20 train-test split
Model Architecture

The ADLF architecture is implemented using TensorFlow-Keras Functional API.

 Branches:

Image Input Branch (64×64×3)

Dense-Inception layers

Spatial Pyramid Pooling

Weather Branch

Soil Branch

Satellite NDVI Branch

Tabular Features Branch

Crop Category Embedding Branch

Fusion Layer

All branches are concatenated →
Passed through Attention Modules →
Fed into two heads:

Regression Head: Predicts numeric yield

Classification Head: Predicts yield class (Low/Medium/High)

 Optimization

Adam optimizer

Loss: Combined MAE + Categorical Cross-Entropy

Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

The final ADLF model achieved strong performance on the test dataset:

 Improved accuracy over Random Forest and SVM

 Stable training with attention-enhanced feature fusion

 Successful yield classification & regression

F1-Score

MAE (for regression)
