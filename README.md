# Epileptic-Seizures-Detection

## Seizures Detection using Machine Learning

## Project Overview

This project focuses on the automated detection of epileptic seizures using machine learning models applied to EEG signal data. The goal is to develop a robust classification system that can distinguish between seizure and non-seizure states, aiding in the analysis of neurological data.

## Methodology

The analysis follows these key steps:

1.  **Data Preprocessing**: Raw EEG data from patients is preprocessed using the MNE-Python library. This includes extracting parameters, handling channel information, and segmenting the data into epochs for training.

2.  **Model Training & Evaluation**: Several machine learning and deep learning models are trained and evaluated for their ability to classify seizure activity. The models include:
    * **Random Forest Classifier**: A powerful ensemble learning method for classification.
    * **XGBoost Classifier**: A gradient-boosting framework known for its performance and speed.
    * **Convolutional Neural Network (CNN)**: A deep learning model designed to automatically learn spatial hierarchies of features from the data.

3.  **Performance Assessment**: The models' performance is measured using accuracy on a test set. The models are trained on pre-processed data, with a training set of 271 samples and a test set of 31 samples.

## Technologies & Libraries

* **Python**: The core programming language used for the project.
* **MNE-Python**: A library for EEG/MEG analysis, used for preprocessing raw `.edf` files.
* **Scikit-learn**: Used for implementing the Random Forest classifier.
* **XGBoost**: For the XGBoost classification model.
* **TensorFlow/Keras**: Used to build and train the CNN model.
* **NumPy**: For numerical operations and handling array data.
