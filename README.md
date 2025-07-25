# Iris Flower Classification Project

This project demonstrates a simple machine learning classification task using the famous Iris flower dataset. The model predicts the species of Iris flowers based on their measurements.

This project is part of my AI learning course and was created with assistance from GitHub Copilot.

## Project Overview

The classifier uses Logistic Regression to predict three species of Iris flowers:
- Setosa
- Versicolor
- Virginica

based on four features:
- Sepal length
- Sepal width
- Petal length
- Petal width

## Technical Details

### Dependencies
- pandas
- scikit-learn
- seaborn
- matplotlib

### Data Processing Steps
1. Load the Iris dataset using scikit-learn
2. Split data into training (80%) and test (20%) sets
3. Scale features using StandardScaler
4. Train a Logistic Regression model
5. Evaluate model performance

### Visualization
The project includes:
- Pairplot showing relationships between features
- Confusion matrix visualization
- Detailed classification report

## How to Run

1. Install required packages:
```bash
pip install pandas scikit-learn seaborn matplotlib
```

2. Run the classifier:
```bash
python iris_classifier.py
```

## Model Performance

The model typically achieves accuracy above 90% on the test set. Detailed metrics including precision, recall, and F1-score are displayed when running the script.

## Pipeline

The project includes a scikit-learn pipeline that combines:
1. StandardScaler for feature scaling
2. LogisticRegression for classification

This pipeline can be used for future predictions on new data.

## Author

[Your Name]

##