import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate once
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

        user_input = [[sepal_length, sepal_width, petal_length, petal_width]]

        prediction = model.predict(user_input)
        predicted_class = target_names[prediction[0]]

        print(f"\nPredicted Flower Type: {predicted_class}")

    except ValueError:
        print("Invalid input. Please enter numeric values only.")

    again = input("\nDo you want to try again? (yes/no): ")
    if again.lower() != "yes":
        print("Exiting program.")
        break
