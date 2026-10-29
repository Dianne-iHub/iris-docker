from pathlib import Path

import joblib
from sklearn.datasets import load_iris

def main():
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "iris_model.joblib"

    model = joblib.load(model_path)
    iris = load_iris()

    # Example measurements: sepal length, sepal width
    # petal length, petal width (centimeters)
    sample = [[5.1, 3.5, 1.4, 0.2]]

    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]

    print(f"Predicted flower: {iris.target_names[prediction]}")
    print(f"Class probabilities: {probabilities.round(3)}")
    

if __name__ == "__main__":
    main()


