# MLdemos
Demo: Classification problem - classify flower type based on attributes
#Ensure python is installed

#Install ML packags
pip install scikit-learn pandas matplotlib

#Create python file 
nano ml_demo.py

#ML code
----------------------------------------------------------------------------------
# Importing libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()   #Built-in dataset
X = iris.data  #Features(inputs) - 4 nos per flower
y = iris.target   # Labels (outputs) - flower type (0,1,2)
target_names = iris.target_names  # Supervised classification

# Train-test split   --- 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model ---Random forest builds many decision trees , each tree learns patterns and final prediction made based on majority vote
model = RandomForestClassifier()
model.fit(X_train, y_train)    # Model studies relationships between feature & labels - learning phase

# Evaluate once  ---- model predicts test data , actual vs predicted compared and accuracy calculated
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Model trained successfully.")
print("Accuracy on test data:", accuracy)
print("\n--- Interactive Prediction Mode ---")

# User input loop
while True:
    try:
        print("\nEnter flower measurements:")

        sepal_length = float(input("Sepal Length: "))
        sepal_width = float(input("Sepal Width: "))
        petal_length = float(input("Petal Length: "))
        petal_width = float(input("Petal Width: "))
# User input converted to 2 D array as expected by model as scikit-learn expects (samples, fatures)
        user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
# Preidcting
        prediction = model.predict(user_input)
        predicted_class = target_names[prediction[0]]

        print(f"\nPredicted Flower Type: {predicted_class}")

    except ValueError:
        print("Invalid input. Please enter numeric values only.")

    again = input("\nDo you want to try again? (yes/no): ")
    if again.lower() != "yes":
        print("Exiting program.")
        break
  -------------------------------------------------------------

# Run
python3 ml_demo.py

# Sample input
Sepal Length: 5.1
Sepal Width: 3.5
Petal Length: 1.4
Petal Width: 0.2

# Sample output
Predicted Flower Type: setosa

# What is happening internally
Random Forest:
Creates many decision trees
Each tree splits data based on feature thresholds
Each tree predicts class
Final answer = majority vote

So when you input:5.1, 3.5, 1.4, 0.2
Trees check:
Is petal_length < 2?
Is sepal_width > 3?etc.
Based on splits, majority votes.

# Architecture
We built:
Offline training
Interactive inference
CLI-based ML service

In production this would become:
Training pipeline
Saved model artifact
REST API
Endpoint

We trained model every time script runs.
In real production:
Train once
Save model
Load model for inference

