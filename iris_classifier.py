import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# Load the Iris dataset as a pandas DataFrame
iris = load_iris(as_frame=True)
df = iris.frame

# Visualize pairwise relationships in the dataset, colored by target class
sns.pairplot(df, hue='target')

# Print the first few rows of the DataFrame
print(df.head())
# Print the count of each class in the target column
print("\nClass counts:\n", df['target'].value_counts())

# Separate features (X) and target labels (y)
X = df.drop(columns='target')
y = df['target']

# Split the data into training and test sets (80% train, 20% test), stratified by class
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit a standard scaler on the training data and transform both train and test sets
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train a logistic regression model on the scaled training data
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train_scaled, y_train)  # <-- Added missing fit step

# Predict the labels for the test set
y_pred = model.predict(X_test_scaled)

# Print the accuracy score of the model on the test set
print("Accuracy:", accuracy_score(y_test, y_pred))
# Print a detailed classification report
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Display the confusion matrix for the test predictions
ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test)
plt.show()

# Create a pipeline that combines scaling and logistic regression for future use
pipeline = Pipeline([
    ('scaler',   StandardScaler()),
    ('classifier', LogisticRegression(max_iter=200, random_state=42))
])