# Fingerprint Matching Using CNN Project

With the growing ease of access to everyday technology, acts of hacking and identity theft are becoming increasingly prevalent. AI tools have become crucial in combating these crimes, and biometric identities offer a safer alternative to text-based passwords. As people shift towards biometric-based security, the need for effective identification and matching of biometrics, such as fingerprints, becomes paramount. This project aims to build an accurate and efficient model for identifying and matching fingerprint records using Convolutional Neural Networks (CNNs).

## Problem Statement

- **Growing Threat of Hacking and Identity Theft**: With the rise in technological advancements, hacking and identity theft have become dominant forces in society.
  
- **Shift Towards Biometric Security**: People are gradually moving from text-based passwords to biometric-based security solutions.
  
- **Need for Effective Biometric Identification**: The preference for biometric security underscores the importance of developing reliable methods for identifying and matching biometrics like fingerprints.

## Data

We used the Sokoto Coventry Fingerprint Dataset (SOCOFing) from Kaggle for this project. SOCOFing is a biometric fingerprint database designed for academic research purposes.

- **Dataset**: 6,000 grayscale fingerprint images from 600 African subjects.
  
- **Attributes**: The dataset includes labels for gender, hand, finger name, and synthetically altered versions with three different levels of alteration (Easy, Medium, and Hard) for obliteration, central rotation, and z-cut.
  
- **Data Link**: [SOCOFing Dataset on Kaggle](https://www.kaggle.com/datasets/ruizgara/socofing)

## Approach

### CNN Architecture

We employed a Deep Learning approach using Convolutional Neural Networks (CNNs) to process and analyze the fingerprint images.

### Model Architecture

The model architecture involves two input layers for two images, followed by a feature extraction branch, where convolutional layers with batch normalization and dropout are applied. The extracted features from both inputs are then subtracted, and the resulting features are passed through a classification head to output a binary classification (match or non-match).

### Preprocessing

A preprocessing Python file (`Fingerprint_CNN_Preprocess.py`) is provided for dataset creation, and multiple versions of the train file for model training are available, with the latest being `Fingerprint_CNN_Train_V4.py`.

## Results

### Performance Evaluation

The model's accuracy and efficiency were assessed through rigorous testing and validation.

### Accuracy Metrics

- **Identification**: The model's accuracy in correctly identifying and matching fingerprints against the ground truth was evaluated.

### Validation Results

- **Metrics**: Detailed analysis of the model's performance, including precision, recall, and F1-score, highlighted its effectiveness in fingerprint recognition tasks.

## Future Scope

Once the model is fine-tuned for further accuracy, it can be deployed in various real-world applications:

- **Criminal Records Matching**: A matching system for criminal records.
  
- **At-Home Identity Document Updation**: An easier way for individuals to update their identity documents at home, similar to Aadhar biometrics updation.
  
- **Biometric-Based Security Systems**: Enhanced biometric security systems for smartphones and other devices.

## Note

This project is still under development as we strive to achieve higher validation accuracy and take the model to a deployable state. Contributions and feedback are welcome!
