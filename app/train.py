from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

def main():
    # Load the 150-row Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Train the classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # Save the trained model
    project_root = Path(__file__).resolve().parent.parent
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "iris_model.joblib"
    joblib.dump(model, model_path)

    print("Training completed successfully!")
    print(f"Number of samples: {len(X)}")
    print(f"Flower classes: {list(iris.target_names)}")
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()
