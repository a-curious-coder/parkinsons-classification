import csv
import os
from distutils.util import strtobool
from unicodedata import numeric

import matplotlib
from sklearn.preprocessing import StandardScaler

from plots import *

import math
import random
import time
import warnings
from os.path import exists

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from plotly.figure_factory import create_distplot
from sklearn import linear_model
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import (
    ExtraTreesClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    cross_val_score,
    train_test_split,
    KFold,
)
from sklearn.svm import SVC
from tensorflow.keras import layers, models
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from xgboost import XGBRegressor


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


# ------------- DATA WRANGLING -------------
def data_wrangling(df):
    print("-------------------\nPreprocessing\n-------------------")
    if normalise:
        print("[*]\tNormalize Data")
        # Bring all data values to the same range
        df = normalize_tabular_data(df)

    print("[*]\tDrop 'name' column")
    # Drop columns that hold no value regarding prediction
    del df["name"]

    return df


def dimensionality_reduction(X):
    """Applies dimensionality reduction techniques to data
    Reduces number of features retaining as much information as possible
    Args:
        X (pd.DataFrame): Parkinsons data

    Returns:
        pd.DataFrame: Transformed parkinsons data
    """
    dprint("-------------------\nDimensionality Reduction\n-------------------")
    if apply_pca:
        pca_components = int(os.getenv("PCA_COMPONENTS"))
        pca = PCA(n_components=pca_components, whiten="True")
        pcs = pca.fit(X).transform(X)
        # Store PCA results to DataFrame
        X = pd.DataFrame(
            data=pcs,
            index=range(pcs.shape[0]),
            columns=[f"PC{i}" for i in range(pca_components)],
        )
    elif apply_tsvd:
        svd = TruncatedSVD(n_components=5)
        svds = svd.fit(X).transform(X)
        X = pd.DataFrame(
            data=svds,
            index=range(svds.shape[0]),
            columns=[f"PC{i}" for i in range(svds.shape[1])],
        )
    else:
        dprint("[*]\tNo Dimensionality Reduction Used")

    return X


def normalize_tabular_data(data):
    """Normalizes every numerical value in the given data

    Args:        data (pd.DataFrame): Parkinsons Data

    Returns:
        pd.DataFrame: Normalized Parkinsons data
    """
    # Exclude string column and label
    avoid = ["status", "name"]
    # Min/Max normalisation to numerical columns
    for column in data.columns:
        if column not in avoid:
            data[column] = data[column] / data[column].abs().max()
    return data


# -------- EXPERIMENTAL DIMENSIONALITY REDUCTION
def predict_minus_feature(df, removed_col=""):
    """Takes the dataset, separates the data from the labels and finds the best configuration
    through elimination of features and classifying the data to see if and which column being removed
    provides the best classification accuracy.

    Collects prediction accuracy values over n loops to test classification more than once
    Args:
        df (pd.DataFrame): [description]
        removed_col (str, optional): [description]. Defaults to "".
    """
    if exists("misc/removed_columns.csv"):
        return
    num_trials = 10

    if removed_col != "":
        if removed_col in df.columns:
            del df[removed_col]
        else:
            print(f"Column '{removed_col}' doesn't exist")
    print("---------- Removed Columns ----------")
    removed = 0
    removed_columns = []
    while True:
        columns = []
        for trial in range(num_trials):
            # print(f"{(trial+1)/num_trials*100:.2f}% Complete")
            count = 0
            data = []
            column_names = []
            # For each column in the data
            for i in range(df.shape[1] - 2):
                # Split data into test and training sets
                x_train, x_test, y_train, y_test = split_data_train_test(
                    X, y, test_size
                )

                if i != 0:
                    column_names.append(X.columns[count])
                    del x_train[X.columns[count]]
                    del x_test[X.columns[count]]
                    count += 1

                pred = svm(x_train, x_test, y_train, y_test)
                data.append(pred)
            column_names.insert(0, "alldata")
            columns.append(pd.DataFrame({"removed": column_names, "prediction": data}))
        predictions = pd.concat(columns)
        predictions.to_csv("predictions.csv", index=False)
        column, acc = get_max_prediction_average(predictions)
        predictions = get_prediction_averages(predictions)

        if column == "alldata":
            print("--------")
            print(f"All data: {acc:.2f}%")
            print(*df.columns, sep="\n")
            plot_removed_features(predictions, f"final")
            with open("misc/removed_columns.csv", "w") as f:
                write = csv.writer(f)
                write.writerow(removed_columns)
            print("Saved column names as removed_columns.csv")
            return
        else:
            print(f"Removing '{column}' achieves {acc:.2f}%")
            removed += 1
            file_name = f"(Removing {column})(Removed {removed})"
            plot_removed_features(predictions, file_name)
            removed_columns.append(column)
            del df[column]


def get_max_prediction_average(data):
    features = list(data["removed"])
    predictions = list(data["prediction"])

    max_acc = max(predictions)
    max_acc_index = predictions.index(max_acc)
    max_feature = features[max_acc_index]
    return max_feature, max_acc


def get_prediction_averages(predictions):
    features = predictions["removed"].unique()
    a_predictions = {}
    for feature in features:
        filtered = predictions[predictions["removed"] == feature]
        average = filtered["prediction"]
        val = average.sum() / average.shape[0]
        a_predictions[feature] = round(val, 2)

    # plot_removed_features(a_predictions)
    return a_predictions


# ----------- PREPARE DATA FOR CLASSIFICATION MODELS
def data_label_split(data):
    """Splits the data variables/features from the labels/predictive values

    Args:
        data (pd.DataFrame): [description]

    Returns:
        pd.DataFrame: Data features, data labels
    """
    X = data.drop("status", axis=1)
    y = data["status"]
    return X, y


def split_data_train_test(X, y, test_size):
    """Creates train/test split given a dataframe

    Args:
        data (pd.DataFrame): parkinsons data

    Returns:
        pd.DataFrame: training and test splits
    """

    if stratify:
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
    else:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    if balance_labels:
        x_train, y_train = balance_training_data(x_train, y_train)

    return x_train, x_test, y_train, y_test


def balance_training_data(x_train, y_train):
    """Generates new datapoints to ensure an even split in train class labels

    Args:
        x_train (pd.DataFrame): training data
        y_train (pd.DataFrame): training labels

    Returns:
        pd.DataFrame: new train data
    """
    smote = SMOTE(sampling_strategy="minority", n_jobs=-1)
    X_sm, y_sm = smote.fit_resample(x_train, y_train)

    # Plot new ratio in pie chart
    plot_class_ratio(
        y_sm,
        "class_ratio_training_smote",
        "Balanced Training Data with smote",
    )

    # Return fabricated balanced data
    return X_sm, y_sm


# ------------- BEST SETTINGS -------------
def save_best_params(best_params, file_name):
    """Saves the best parameters of any given model to settings folder in working directory

    Args:
        best_params (dict): [description]
        file_name (str): [description]
    """
    sdf = pd.DataFrame.from_dict(best_params, orient="index").transpose()
    sdf.to_csv(f"settings/{file_name}", index=False)


def find_best_nn_settings(x_train, y_train):
    """Finds the optimal neural network parameters

    Args:
        x_train (pd.DataFrame): Training data
        y_train (pd.Series): Training labels
        file_name (str): File name to save parameters as
    """

    file_name = "best_dl_imbalanced.csv"
    if balance_labels:
        file_name = "best_dl_balanced.csv"

    if not exists(f"settings/{file_name}"):
        print("[*]\tAttempting to find optimal parameters for Neural Network...")
        start = time.perf_counter()
        param_grid = {
            "layer1": [16],
            "layer2": [8],
            "output": ["sigmoid", "softmax"],
            "input_shape": [x_train.shape[1]],
            "optimizer": ["rmsprop", "adam"],
            "loss": ["binary_crossentropy"],
            "metric": ["accuracy"],
            "init": ["glorot_uniform", "normal", "uniform"],
            "dropout_rate": [0.0, 0.15, 0.3],
            "batch_size": [512],
            "epochs": [500],
        }

        model = KerasClassifier(build_fn=create_model, verbose=1)
        x_train, x_val, y_train, y_val = split_data_train_test(
            x_train, y_train, val_size
        )

        if grid_search:
            print("[*]\tTuning Model using Grid Search Cross Validation")
            CV_dl = GridSearchCV(
                estimator=model, param_grid=param_grid, n_jobs=-1, cv=5
            )
            results = CV_dl.fit(x_train, y_train, validation_data=(x_val, y_val))
        else:
            print("[*]\tTuning Model using Random Search Cross Validation")
            CV_dl = RandomizedSearchCV(
                estimator=model, param_distributions=param_grid, n_jobs=-1, cv=5
            )
            results = CV_dl.fit(x_train, y_train, validation_data=(x_val, y_val))

        print(results.best_score_, results.best_params_)
        end = time.perf_counter()
        print(f"{end-start:.2f} seconds")
        sdf = pd.DataFrame.from_dict(results.best_params_, orient="index").transpose()
        sdf.to_csv(f"settings/{file_name}", index=False)


def find_best_if_settings(x_train, y_train):
    """Finds optimal isolation forest model parameters

    Args:
        x_train (pd.DataFrame): Training data
        y_train (pd.Series): Train data-set labels
    """
    print("[*]\tAttempting to find optimal parameters for Isolation Forest...")
    model = IsolationForest(random_state=42)

    param_grid = {
        "n_estimators": range(0, 5, 1),
        "max_samples": range(0, 10, 2),
        "contamination": ["auto", 0.0001, 0.0002],
        "max_features": [5, 10, 15, 20],
        "n_jobs": [-1],
    }

    grid_search = GridSearchCV(
        model,
        param_grid,
        scoring="neg_mean_squared_error",
        refit=True,
        cv=10,
        return_train_score=True,
    )
    grid_search.fit(x_train, y_train)
    best_model = grid_search.fit(x_train, y_train)
    save_best_params(best_model.best_params_, "best_if_settings.csv")


def find_best_rf_settings(x_train, y_train):
    """Finds best random forest settings

    Args:
        x_train ([type]): [description]
        y_train ([type]): [description]
    """
    print("[*]\tAttempting to find optimal parameters for Random Forest...")
    start = time.perf_counter()
    param_grid = {
        "n_estimators": range(20, 260, 20),
        "max_features": ["auto", "sqrt", "log2"],
        "max_depth": [4, 5, 6, 7, 8],
        "criterion": ["gini", "entropy"],
    }

    rfc = RandomForestClassifier(random_state=42)
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid)
    CV_rfc.fit(x_train, y_train)
    end = time.perf_counter()
    print(f"{end-start:.2f} seconds")
    save_best_params(CV_rfc.best_params_, "best_rf_settings.csv")


def find_best_lr_settings(x_train, y_train):
    print("[*]\tAttempting to find optimal parameters for Logistic Regression...")
    model = LogisticRegression()
    param_grid = {
        "C": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
    }
    logreg_cv = GridSearchCV(model, param_grid, cv=3, scoring="roc_auc").fit(
        x_train, y_train
    )
    print("tuned hyperparameters :(best parameters) ", logreg_cv.best_params_)
    # print("AUC_ROC :",logreg_cv.best_score_)
    save_best_params(logreg_cv.best_params_, "best_lr_settings.csv")


def find_best_svm_settings(x_train, y_train):
    print("[*]\tAttempting to find optimal parameters for SVM...")
    param_grid = {
        "C": [0.1, 1, 10, 100, 1000],
        "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
        "kernel": ["rbf", "linear"],
    }

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=0)

    # fitting the model for grid search
    grid.fit(x_train, y_train)
    save_best_params(grid.best_params_, "best_svm_settings.csv")


# ------------- CLASSIFICATION METHODS -------------
def isolation_forest(x_train, x_test, y_train, y_test):
    """Used to detect outliers in data

    Args:
        x_train (pd.DataFrame): train data
        x_test (pd.DataFrame): test data
        y_train (pd.DataFrame): train labels
        y_test (pd.DataFrame): test labels
    """
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    if not exists("settings/best_if_settings.csv"):
        find_best_if_settings(x_train, y_train)

    isf = IsolationForest(
        contamination="auto",
        max_features=5,
        max_samples=2,
        n_estimators=1,
        n_jobs=-1,
    )

    y_pred = isf.fit_predict(x_train)

    X_train_iforest, y_train_iforest = (
        x_train[(y_pred != -1), :],
        y_train[(y_pred != -1)],
    )


def extra_trees_classifier(x_train, x_test, y_train, y_test):
    """Uses Extra trees classifier to calculate each feature's importance in classification
    Creates plots

    Args:
        x_train (pd.DataFrame): Data for training
        x_test (pd.DataFrame): Data for testing
        y_train (pd.Series): Labels for train data
        y_test (pd.Series): Labels for test data
    """
    TOP_FEATURES = x_train.shape[1]
    forest = ExtraTreesClassifier(n_estimators=250, max_depth=5, random_state=42)
    forest.fit(x_train, y_train)

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    indices = np.argsort(importances)[::-1]
    indices = indices[:TOP_FEATURES]

    plot_feature_importance(x_train.columns, indices, importances, std)


def linear_regression(x_train, x_test, y_train, y_test):
    """Linear Regression model trained given parkisons data; prints accuracy results based on test sets

    Args:
        x_train (pd.DataFrame): train data
        x_test (pd.DataFrame): test data
        y_train (pd.Series): train labels
        y_test (pd.Series): test labels
    """
    print("-------------------\nLinear Regression\n-------------------")
    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)
    print(f"[*]\t{reg.score(x_test, y_test)*100:.2f}%\tLinear Regression")


def logistic_regression(x_train, x_test, y_train, y_test):
    print("-------------------\nLogistic Regression\n-------------------")
    if not exists("settings/best_lr_settings.csv"):
        find_best_lr_settings(x_train, y_train)

    # Load best settings
    df = pd.read_csv("settings/best_lr_settings.csv")
    # Initialise best settings in variables
    c = df.iloc[0]["C"]
    penalty = df.iloc[0]["penalty"]
    solver = df.iloc[0]["solver"]

    params = {"C": c, "penalty": penalty, "solver": solver}
    print(params)
    classifier = LogisticRegression(random_state=42, n_jobs=-1, **params)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    y_pred_proba = classifier.predict_proba(x_test)[:, 1]
    print("[*]\tOriginal Data Accuracy")
    print(f"\tPrecision: {precision_score(y_test, y_pred)*100:.2f}%")
    print(f"\tRecall: {recall_score(y_test, y_pred)*100:.2f}%")
    print(f"\tF1 Score: {f1_score(y_test, y_pred)*100:.2f}%")
    print(f"\tAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"\tAUC-ROC: {roc_auc_score(y_test, y_pred_proba)*100:.2f}%")

    weight_vector = list(classifier.coef_[0])
    dist = np.dot(x_train, weight_vector)
    y_dist = dist * [-1 if x == 0 else 1 for x in list(y_train)]
    print(len(y_dist))
    print(f"Original training count: {y_train.value_counts()}")

    # sns.kdeplot(y_dist)
    # plt.xlabel("Distance * Y-class")
    # plt.title("Density plot of y dist")
    # plt.grid()
    # plt.show()

    val = np.percentile(y_dist, 20)
    print(f"Threshold Val: {val}")

    X_train_std_new = x_train[(~(y_dist < val))]
    y_train_new = y_train[(~(y_dist < val))]
    print(f"New training set size: {X_train_std_new.shape}")
    print(f"New training label size: {y_train_new.shape}")

    # print("Label counts")
    # print(y_train_new.value_counts())
    params = {"C": 0.001, "penalty": "l2"}
    X_train_std_new, y_train_new = balance_training_data(X_train_std_new, y_train_new)
    classifier1 = LogisticRegression(random_state=42, n_jobs=-1, **params).fit(
        X_train_std_new, y_train_new
    )

    y_pred = classifier1.predict(x_test)
    y_pred_proba = classifier1.predict_proba(x_test)[:, 1]
    print("[*]\tNew Data Accuracy")
    print(f"\tPrecision: {precision_score(y_test, y_pred)*100:.2f}%")
    print(f"\tRecall: {recall_score(y_test, y_pred)*100:.2f}%")
    print(f"\tF1 Score: {f1_score(y_test, y_pred)*100:.2f}%")
    print(f"\tAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"\tAUC-ROC: {roc_auc_score(y_test, y_pred_proba)*100:.2f}%")

    plot_logistic_regression(classifier1.predict_proba(x_test)[:, 1])


def cross_validation(model, n, X, y):
    """Performs cross validation using a given model with n splits

    Args:
        model (??): classification model
        n (int): number of splits
        X (pd.DataFrame): data
        y (pd.Series): labels

    Returns:
        list: list of scores per fold
    """
    # Initialise a cross validation model for cross_val_score to use
    cv = KFold(n_splits=n, shuffle=True)
    # Execute cross validation
    k_scores = cross_val_score(model, X, y, scoring="accuracy", cv=cv, n_jobs=-1)
    return k_scores


def create_model_rf():
    if not exists("settings/best_rf_settings.csv"):
        find_best_rf_settings(x_train, y_train)
    # Load in optimal settings generated by GridSearchCV
    sdf = pd.read_csv("settings/best_rf_settings.csv")
    # Extract best settings
    criterion = sdf.iloc[0]["criterion"]
    max_depth = sdf.iloc[0]["max_depth"]
    max_features = sdf.iloc[0]["max_features"]
    n_estimators = sdf.iloc[0]["n_estimators"]
    # Initialise model
    rf = RandomForestClassifier(
        criterion=criterion,
        max_depth=max_depth,
        max_features=max_features,
        n_estimators=n_estimators,
        random_state=42,
    )
    return rf


def random_forest(x_train, x_test, y_train, y_test):
    """Random Forest Classification

    Args:
        x_train ([type]): [description]
        x_test ([type]): [description]
        y_train ([type]): [description]
        y_test ([type]): [description]

    Returns:
        [type]: [description]
    """
    rf = create_model_rf()
    # Train model
    rf.fit(x_train, y_train)
    # Predict and
    pred = rf.predict(x_test)
    actual = y_test.values
    features = x_test.columns
    count = 0
    for i in range(len(pred)):
        if pred[i] == actual[i]:
            count = count + 1
    acc = f"{count / float(len(pred))*100:.2f}"
    print(f"[*]\t{acc}%\t\tRandom Forest")
    plot_title = f"[Imbalanced] Random Forest Confusion Matrix : ({acc}%)"
    file_name = f"imbalanced_rf_cm"
    if balance_labels:
        plot_title = f"[Balanced] Random Forest Confusion Matrix : ({acc}%)"
        file_name = f"balanced_rf_cm"
    plot_cm(actual, pred, plot_title, file_name)
    return count / float(len(pred)) * 100


def create_model_svm():
    if not exists("settings/best_svm_settings.csv"):
        find_best_svm_settings(x_train, y_train)

    # Load in optimal settings generated by GridSearchCV
    sdf = pd.read_csv("settings/best_svm_settings.csv")
    # Extract best settings
    C = sdf.iloc[0]["C"]
    gamma = sdf.iloc[0]["gamma"]
    kernel = sdf.iloc[0]["kernel"]

    svm = SVC(kernel=kernel, gamma=gamma, C=C)

    return svm


def svm(x_train, x_test, y_train, y_test):
    """Support Vector Machine Neural Network

    Args:
        x_train ([type]): [description]
        x_test ([type]): [description]
        y_train ([type]): [description]
        y_test ([type]): [description]

    Returns:
        [type]: [description]
    """
    svm = create_model_svm()
    svm.fit(x_train, y_train)
    pred = svm.predict(x_test)
    actual = y_test.values
    count = 0
    for i in range(len(pred)):
        if pred[i] == actual[i]:
            count = count + 1
    acc = f"{count / float(len(pred))*100:.2f}"
    print(f"[*]\t{acc}%\t\tSupport Vector Machine")

    plot_title = f"[Imbalanced] SVM Confusion Matrix : ({acc}%)"
    file_name = f"imbalanced_svm_cm"
    if balance_labels:
        plot_title = f"[Balanced] SVM Confusion Matrix : ({acc}%)"
        file_name = f"balanced_svm_cm"

    plot_cm(actual, pred, plot_title, file_name)
    return count / float(len(pred)) * 100


def create_model(
    layer1, layer2, output, input_shape, optimizer, loss, metric, init, dropout_rate=0.0
):
    # Initialise neural network
    network = models.Sequential()
    network.add(
        layers.Dense(
            layer1,
            activation="relu",
            kernel_initializer=init,
            input_shape=(input_shape,),
        )
    )
    network.add(layers.Dropout(dropout_rate))
    network.add(layers.Dense(layer2, activation="relu"))
    network.add(layers.Dropout(dropout_rate))
    network.add(layers.Dense(1, activation=output))

    network.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    return network


def train_neural_network(x_train, x_test, y_train, y_test):
    """Trains neural network with parkinsons data

    Args:
        x_train (pd.DataFrame): training data
        x_test (pd.DataFrame): test data
        y_train (pd.DataFrame): training labels
        y_test (pd.DataFrame): test labels
        file_name ([type]): Name of best settings file
    """
    # Load in epochs from environment variable file
    find_best_nn_settings(x_train, y_train)

    file_name = "best_dl_imbalanced.csv"
    plot_title_affix = "imbalanced_dl"
    model_title_affix = "imbalanced"
    if balance_labels:
        file_name = "best_dl_balanced.csv"
        plot_title_affix = "balanced_dl"
        model_title_affix = "balanced"

    # Read best neural network settings as dictionary
    parms = pd.read_csv(f"settings/{file_name}").to_dict("index")[0]

    # Load best settings
    batch_size = parms["batch_size"]
    dropout_rate = parms["dropout_rate"]
    epochs = parms["epochs"]
    layer1 = parms["layer1"]
    layer2 = parms["layer2"]
    init = parms["init"]
    optimizer = parms["optimizer"]
    loss = parms["loss"]
    metric = parms["metric"]
    output = parms["output"]
    input_shape = x_test.shape[1]

    # If a model hasn't been trained before...
    if not exists(f"models/parkinsons_model_{epochs}_{model_title_affix}.tf"):
        dprint("[*]\tTraining Neural Network")
        # Prepare network
        model = create_model(
            layer1, layer2, output, x_train.shape[1], optimizer, loss, metric, init
        )

        x_train, x_val, y_train, y_val = split_data_train_test(
            x_train, y_train, val_size
        )

        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            verbose=0,
        )

        # Save parkinsons model
        model.save(f"models/parkinsons_model_{epochs}_{model_title_affix}.tf")

        # Get accuracy / loss data
        history_dict = history.history
        # print(history_dict.keys())
        acc = history_dict["accuracy"]
        val_acc = history_dict["val_accuracy"]
        loss = history_dict["loss"]
        val_loss = history_dict["val_loss"]
        if plots:
            plot_accuracy(
                smooth_curve(acc),
                smooth_curve(val_acc),
                f"parkinsons_accuracy_{epochs}",
            )
            plot_loss(
                smooth_curve(loss), smooth_curve(val_loss), f"parkinsons_loss_{epochs}"
            )
    else:
        dprint("[*]\tModel already trained")


def test_neural_network(x_test, y_test):
    """Tests a trained neural network"""
    file_name = "best_dl_imbalanced.csv"
    model_title_affix = "imbalanced"
    if balance_labels:
        file_name = "best_dl_balanced.csv"
        model_title_affix = "balanced"
    # Read epochs from best neural network settings file
    parms = pd.read_csv(f"settings/{file_name}").to_dict("index")[0]
    epochs = parms["epochs"]
    input_shape = x_test.shape[1]

    # Loads model
    # TODO: Somehow state whether model takes data with reduced dimensions (PCA, LDA)
    model = models.load_model(
        f"models/parkinsons_model_{epochs}_{model_title_affix}.tf"
    )
    # predictions = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x=x_test)
    new = []
    for i in range(len(predictions)):
        new.append(predictions[i][0])

    pred = [round(pred) for pred in new]

    actual = y_test.values
    count = 0
    for i in range(len(pred)):
        if pred[i] == actual[i]:
            count = count + 1
    acc = f"{count / float(len(pred))*100:.2f}"
    print(f"[*]\t{acc}%\t\tNeural Network")

    plot_title = f"[Imbalanced] Neural Network Confusion Matrix : ({acc}%)"
    file_name = f"imbalanced_nn_cm"
    if balance_labels:
        plot_title = f"[Balanced] Neural Network Confusion Matrix : ({acc}%)"
        file_name = f"balanced_nn_cm"

    # Plot confusion matrix of results
    plot_cm(actual, pred, plot_title, file_name)


# ------------- MISC METHODS -------------
def dprint(text):
    verbose = False
    if verbose:
        print(text)


def print_title(title):
    """Prints title

    Args:
        title (str): title
    """
    print("-------------------------------")
    print(f"\t{title}")
    print("-------------------------------")


def cls():
    """Clears console - useful for debugging/testing"""
    os.system("cls" if os.name == "nt" else "clear")


def prepare_directory():
    """Creates necessary folders in preparation for data/models saved"""
    folders = [
        "misc",
        "plots",
        "models",
        "settings",
        "plots/removed_columns",
        "plots/confusion_matrices",
    ]
    for folder in folders:
        if not os.path.isdir(folder):
            os.mkdir(folder)


def main():
    """Main"""
    cls()
    # Load in environment variables from .env
    load_dotenv()

    # Initialise global variables
    global balance_labels
    global test_size
    global val_size
    global manual_settings
    global balance_data
    global normalise
    global apply_pca
    global apply_tsvd
    global stratify
    global experimental
    global rewrite_model
    global plots
    global grid_search

    # set environment variables
    eda = strtobool(os.getenv("EDA"))
    plots = strtobool(os.getenv("PLOTS"))

    normalise = strtobool(os.getenv("NORMALISE"))
    balance_labels = strtobool(os.getenv("BALANCE_TRAINING_LABELS"))
    stratify = strtobool(os.getenv("STRATIFY"))
    dimensionality_reduction = strtobool(os.getenv("DIMENSIONALITY_REDUCTION"))

    manual_settings = strtobool(os.getenv("MANUAL_TRAINING"))
    test_size = float(os.getenv("TEST_SIZE"))
    val_size = float(os.getenv("VALIDATION_SIZE"))
    apply_pca = strtobool(os.getenv("PCA"))
    apply_tsvd = strtobool(os.getenv("SVD"))

    experimental = strtobool(os.getenv("EXPERIMENTAL"))

    grid_search = strtobool(os.getenv("GRID_SEARCH"))
    random_search = strtobool(os.getenv("RANDOM_SEARCH"))

    rewrite_model = strtobool(os.getenv("OVERWRITE_NEURAL_MODEL"))

    # Prepare directories for program
    prepare_directory()

    # Load Dataset
    df = pd.read_csv("data/parkinsons.data")

    # EDA
    if eda:
        # print_t
        print(df.info())
        # Number of names
        print(f"Patient Frequency: {len(df['name'].unique())}")
        # Label distribution
        print(f"Label distribution\n{df['status'].value_counts()}")

        print_title("Generating Plots")
        plot_class_ratio(df["status"], "class_ratio_all_data", "All Data Labels Ratio")
        # Filters dataframe to only include numeric data
        numeric_dataframe = df.select_dtypes(include=np.number)
        # Remove label column
        del numeric_dataframe["status"]

        # if plots:
        #     numeric_dataframe = normalize_tabular_data(numeric_dataframe)
        #     print(
        #         "[*]\tPopulation Density Function PDF of features with target variables"
        #     )
        #     fig = create_distplot(
        #         [numeric_dataframe[column] for column in numeric_dataframe.columns],
        #         numeric_dataframe.columns,
        #         bin_size=[2, 2],
        #         show_rug=False,  # rug
        #         show_hist=False,  # hist
        #         show_curve=True,  # curve
        #     )
        #     fig.show()

    # Transform data based on information discovered through eda
    df = data_wrangling(df)

    if plots:
        # Plots pairwise correlation
        plot_pairplot(df)

    if experimental:
        if not exists("misc/removed_columns.csv"):
            predict_minus_feature(df)

    # Split data into data and labels
    X, y = data_label_split(df)

    # Reduce dimensionality
    if dimensionality_reduction:
        X = dimensionality_reduction(X)

    if experimental:
        with open("misc/removed_columns.csv", newline="") as f:
            reader = csv.reader(f)
            data = list(reader)

        for column in data[0]:
            print(f"[*]\tRemoving {column}")
            del X[column]

    # Split data and labels into training/test sets
    x_train, x_test, y_train, y_test = split_data_train_test(X, y, test_size)

    # # Identify outliers
    # isolation_forest(x_train, x_test, y_train, y_test)

    # # Feature importance
    # extra_trees_classifier(x_train, x_test, y_train, y_test)

    # # logistic_regression(x_train, x_test, y_train, y_test)
    # # linear_regression(x_train, x_test, y_train, y_test)
    # print("------------------------------------------------")
    # print(f"\t\tClassification Models\t\t")
    # print("------------------------------------------------")
    # print("\tAccuracy\tClassification Model\t")
    # print("------------------------------------------------")
    # svm(x_train, x_test, y_train, y_test)
    # random_forest(x_train, x_test, y_train, y_test)
    # train_neural_network(x_train, x_test, y_train, y_test)
    # test_neural_network(x_test, y_test)

    print_title("Cross Validation")

    models = {"Random Forest": create_model_rf(), "SVM": create_model_svm()}
    max_folds = 10
    cv_models = range(2, max_folds + 1)
    for key, model in models.items():
        all_scores = []
        for i in cv_models:
            k_scores = cross_validation(model, i, X, y)
            all_scores.append(k_scores)
            # Print mean accuracy
            print(
                f"[{i}]\tCross Validation Accuracy of {k_scores.mean()*100:.2f}%\t{key}"
            )
        file_name = f"boxplot_{key}_{i}.png"
        plot_title = f"{key} Box Plot Repeated K-Fold Cross Validation"
        plot_cv_box_plot(cv_models, all_scores, file_name, plot_title)


if __name__ == "__main__":
    main()
