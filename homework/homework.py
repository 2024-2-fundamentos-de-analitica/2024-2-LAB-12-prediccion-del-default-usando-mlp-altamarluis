import gzip
import json
import os
import pickle
import zipfile
from glob import glob

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

def outputCreation(output_directory):
        if os.path.exists(output_directory):
            for file in glob(f"{output_directory}/*"):
                os.remove(file)
            os.rmdir(output_directory)
        os.makedirs(output_directory)

def save_model(path, estimator):
        outputCreation("files/models/")
        with gzip.open(path, "wb") as f:
            pickle.dump(estimator, f)

def pregunta01():

    def loadData(input_directory):
        dfs = []
        routes = glob(f"{input_directory}/*")
        for route in routes:
            with zipfile.ZipFile(f"{route}", mode="r") as zf:
                for fn in zf.namelist():
                    with zf.open(fn) as f:
                        dfs.append(pd.read_csv(f, sep=",", index_col=0))
        return dfs
    
    def cleanse(df):
        df = df.copy()
        df = df.rename(columns={"default payment next month": "default"})
        df = df.loc[df["MARRIAGE"] != 0]
        df = df.loc[df["EDUCATION"] != 0]
        df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x >= 4 else x)
        return df.dropna()
        
    test_df, train_df = [cleanse(df) for df in loadData("files/input")]

    x_train, y_train = train_df.drop(columns=["default"]), train_df["default"]
    x_test, y_test = test_df.drop(columns=["default"]), test_df["default"]

    def f_pipeline(x_train):
        cat_features = ["SEX", "EDUCATION", "MARRIAGE"]
        num_features = [col for col in x_train.columns if col not in cat_features]

        preprocessor = ColumnTransformer(
            [
                ("cat", OneHotEncoder(), cat_features),
                ("scaler", StandardScaler(), num_features),
            ],
        )
        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("feature_selection", SelectKBest(score_func=f_classif)),
                ("pca", PCA()),
                ("classifier", MLPClassifier(max_iter=15000, random_state=21)),
            ]
        )

    pipeline = f_pipeline(x_train)

    def optimizar_hiperparametros(pipeline):

        param_grid = {
            "pca__n_components": [None],
            "feature_selection__k": [20],
            "classifier__hidden_layer_sizes": [(50, 30, 40, 60)],
            "classifier__alpha": [0.26],
            "classifier__learning_rate_init": [0.001],
        }

        return GridSearchCV(
            pipeline,
            param_grid,
            cv=10,
            n_jobs=-1,
            verbose=2,
            refit=True,
        )

    estimator = optimizar_hiperparametros(pipeline)
    estimator.fit(x_train, y_train)

    save_model(
        os.path.join("files/models/", "model.pkl.gz"),
        estimator,
    )

    
    outputCreation("files/output/metrics/")

    def calc_metrics(dataset_type, y_true, y_pred):
        """Calculate metrics"""
        return {
            "type": "metrics",
            "dataset": dataset_type,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

    y_test_pred = estimator.predict(x_test)
    test_precision_metrics = calc_metrics("test", y_test, y_test_pred)
    y_train_pred = estimator.predict(x_train)
    train_precision_metrics = calc_metrics("train", y_train, y_train_pred)

    def matrix_calc(dataset_type, y_true, y_pred):
        """Confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        return {
            "type": "cm_matrix",
            "dataset": dataset_type,
            "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
            "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
        }
    
    test_confusion_metrics = matrix_calc("test", y_test, y_test_pred)
    train_confusion_metrics = matrix_calc("train", y_train, y_train_pred)

    with open("files/output/metrics.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(train_precision_metrics) + "\n")
        file.write(json.dumps(test_precision_metrics) + "\n")
        file.write(json.dumps(train_confusion_metrics) + "\n")
        file.write(json.dumps(test_confusion_metrics) + "\n")


if __name__ == "__main__":
    pregunta01()