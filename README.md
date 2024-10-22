# Classification of Health Risks and Speech Disorders
Predicting maternal health risks and classifying speech disorders through feature selection and various classification algorithms.

## Overview
Machine learning algorithms are computational methods that enable computers to learn patterns from data and make predictions or decisions without being explicitly programmed for each specific task. These algorithms use statistical techniques to analyze and interpret data, allowing them to improve their performance over time based on experience. 

Here’s a brief overview of key types of machine learning algorithms:

1. **Supervised Learning**: Involves training a model on a labeled dataset, where the desired output is known. The algorithm learns to map inputs to outputs, making it suitable for tasks like classification and regression.
   - **Examples**: 
     - **Decision Trees**: A model that splits data into branches based on feature values, leading to predictions.
     - **Support Vector Machines (SVM)**: Finds the hyperplane that best separates different classes in the feature space.
     - **K-Nearest Neighbors (KNN)**: Classifies data points based on the majority class among their nearest neighbors.
     - **Random Forest**: An ensemble method that combines multiple decision trees for better accuracy and robustness.

2. **Unsupervised Learning**: Deals with data that has no labeled outputs. The goal is to identify patterns or groupings within the data.
   - **Examples**: 
     - **Clustering Algorithms**: Such as K-Means, which groups similar data points together.
     - **Dimensionality Reduction Techniques**: Such as PCA (Principal Component Analysis), which reduces the number of features while retaining important information.

3. **Reinforcement Learning**: Involves training an agent to make decisions by rewarding it for correct actions and penalizing it for incorrect ones. The agent learns through trial and error in an environment.
   - **Example**: Algorithms that train robots or game-playing AI (like AlphaGo) to maximize a reward signal.

4. **Semi-Supervised Learning**: Combines both labeled and unlabeled data to improve learning accuracy. This is useful when acquiring labeled data is expensive or time-consuming.

Overall, machine learning algorithms are powerful tools for automating tasks, making predictions, and uncovering insights from data across various domains, including healthcare, finance, marketing, and more.



## Part One: Maternal Health Risk Analysis

This project involves analyzing maternal health risk data using various machine learning algorithms. The primary goal is to build predictive models for risk assessment based on the dataset named **"Maternal Health Risk Data Set.csv."**

### 1. Data Preprocessing

#### 1.1 Feature Selection
A subset of relevant features is selected and stored in a new DataFrame named `new_data_frame`. This subset includes columns 0 to 5 from the original dataset.

#### 1.2 Encoding Target Variable
The target variable, representing maternal health risk levels, is encoded into numerical values as follows:
- **'low risk'** is encoded as 0
- **'mid risk'** is encoded as 1
- **'high risk'** is encoded as 2

### 2. Train-Test Split
The dataset is split into training and test sets. Approximately 70% of the data is used for training, and 30% is reserved for testing. Random indices are selected to achieve this split.

### 3. Model Building and Evaluation
The project employs several machine learning algorithms for modeling and evaluation. Each algorithm is explored with different parameter settings:
- **Decision Tree**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Random Forest**

### 4. Model Evaluation Metrics
For each model, the following evaluation metrics are calculated:
- Accuracy on the test set
- Accuracy on the training set
- Cross-validation score (average accuracy)
- Confusion matrix

## Part Two: Speech Disorder Classification

This project involves the analysis of speech feature data for the classification of speech disorders. The primary objective is to build and evaluate machine learning models to classify speech samples into two categories: healthy (1) and speech disorders (0). The dataset used for this project is named **"pd_speech_features.csv."**

### 1. Data Preprocessing

#### 1.1 Feature Selection
A subset of relevant features is selected and stored in a new DataFrame named `new_data_frame`. This subset includes all columns except the first one, which is assumed to be an identifier.

#### 1.2 Train-Test Split
The dataset is divided into a training set and a test set. Approximately 70% of the data is used for training, and 30% is set aside for testing. Randomly selected rows are used for each split.

#### 1.3 Encoding Target Variable
The target variable, representing the classification labels, is encoded as follows:
- **Healthy speech (Class 1)** is encoded as 1.
- **Speech disorders (Class 0)** are encoded as 0.

### 2. Model Building and Evaluation
Machine learning models are built and evaluated using the training and test datasets. Three types of models are explored:
- **Decision Trees**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machines (SVM)**
- **Random Forests**

Different parameter settings are tested for each model.

### 3. Model Evaluation Metrics
For each model, the following evaluation metrics are calculated:
- Accuracy on the test set
- Accuracy on the training set
- Cross-validation score (average accuracy)
- Confusion matrix
