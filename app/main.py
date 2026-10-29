
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris


# Locate and load the existing trained model.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "iris_model.joblib"

model = joblib.load(MODEL_PATH)

# Get the names of the three flower species.
iris = load_iris()
class_names = iris.target_names.tolist()

# Create the API application.
app = FastAPI(
    title="Iris Flower Prediction API",
    description="Predict an Iris flower species using a trained ML model.",
    version="1.0.0",
)


# Define the measurements the API accepts.
class FlowerMeasurements(BaseModel):
    sepal_length: float = Field(gt=0, description="Sepal length in cm")
    sepal_width: float = Field(gt=0, description="Sepal width in cm")
    petal_length: float = Field(gt=0, description="Petal length in cm")
    petal_width: float = Field(gt=0, description="Petal width in cm")


@app.get("/")
def home():
    return {
        "message": "Welcome to the Iris Flower Prediction API",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(flower: FlowerMeasurements):
    # Arrange measurements in the order expected by the model.
    sample = [[
        flower.sepal_length,
        flower.sepal_width,
        flower.petal_length,
        flower.petal_width,
    ]]

    # Ask the saved model to predict the species.
    prediction = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]

    # Return the result as JSON.
    return {
        "predicted_class": class_names[prediction],
        "class_probabilities": {
            name: round(float(probability), 4)
            for name, probability in zip(class_names, probabilities)
        },
    }