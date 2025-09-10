# Importing necessary libraries
import numpy as np
import pandas as pd
import pathlib
import logging
import joblib
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from typing import Tuple


# Loading Data
def load_data(data_dir: str) -> pd.DataFrame:
    logger = logging.getLogger(__name__)

    # Forming file paths
    test_path = data_dir / "features" / "test.csv"
    logger.info(f"Loading dataset from {data_dir / "features"}")

    # Loading datasets
    test_df = pd.read_csv(filepath_or_buffer=test_path)

    # Returning datasets
    return test_df.dropna()

# Loading Model
def load_model(model_dir: str) -> BaggingClassifier:
    logger = logging.getLogger(__name__)

    model_path = model_dir / "models" / "bagging_classifier.joblib"
    logger.info(f"Loading model from {model_path}")

    model = joblib.load(filename=model_path)
    return model

# Evaluating Model
def evaluate_model(test_df: pd.DataFrame, model: BaggingClassifier) -> Tuple[str, np.ndarray]:
    logger = logging.getLogger(__name__)
    logger.info("Evaluating the model")

    X_test = test_df.drop(columns=["sentiment", "content"])
    y_test = test_df["sentiment"]

    y_pred = model.predict(X=X_test)
    report = classification_report(y_true=y_test, y_pred=y_pred)
    clf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    return (report, clf_matrix)

# Forming Logger
def form_logger() -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger


# Main function
def main() -> None:
    # Forming logger
    logger = form_logger()
    logger.info(msg="Started model evaluation pipeline")

    # Forming directory paths
    home_dir = pathlib.Path(__file__).parent.parent.parent
    data_dir = home_dir / "data"
    model_dir = home_dir / "models" 
    logger.info(f"Working directory: {home_dir}")

    # Loading data
    test_df = load_data(data_dir=data_dir)

    # Loading model
    model = load_model(model_dir=model_dir)

    # Evaluating model
    report, clf_matrix = evaluate_model(test_df=test_df, model=model)
    print(report)
    print(clf_matrix)

    logger.info("Model Evaluation completed successfully")

if __name__ == "__main__":
    main()