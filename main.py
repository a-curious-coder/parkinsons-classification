#!/usr/bin/env python
import csv
import os
import time
import warnings
from distutils.util import strtobool
from os.path import exists

import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from plotly import graph_objects as go
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
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow.keras import layers, models
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from plots import *


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
pd.set_option("display.float_format", lambda x: "%.3f" % x)
warnings.filterwarnings("ignore")


# ------------- DATA WRANGLING -------------
def data_wrangling(data):
    if normalise:
        print("[*]\tNormalise Data")
        # Bring all data values to the same range
        data = normalise_tabular_data(data)
    if remove_correlate_features:
        print("[*]\tRemove correlating features")
        # Remove heavily correlating features from data-set (provide redundant information)
        data = remove_correlating_features(data)

    # Remove correlating features above specified threshold
    return data


def dimensionality_reduction(X):
    """Applies dimensionality reduction techniques to data
    Reduces number of features retaining as much information as possible
    Args:
        X (pd.DataFrame): Parkinsons data

    Returns:
        pd.DataFrame: Transformed parkinsons data
    """
    dprint("-------------------\nDimensionality Reduction\n-------------------")
    # Drop names column as can't reduce dimensions using string column
    names = X.drop("name", axis=1)
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
    # Append name column back to dataframe
    X["name"] = names
    return X


def normalise_tabular_data(data):
    """normalises every numerical value in the given data

    Args:        data (pd.DataFrame): Parkinsons Data

    Returns:
        pd.DataFrame: normalised Parkinsons data
    """
    # Exclude string column and label
    avoid = ["status", "name"]
    # Min/Max normalisation to numerical columns
    for column in data.columns:
        if column not in avoid:
            data[column] = data[column] / data[column].abs().max()
    return data


def remove_correlating_features(data: pd.DataFrame) -> pd.DataFrame:
    """Removes features correlating with other features
    yielding correlation values above a user-specified threshold

    Args:
        data (pd.DataFrame): data

    Returns:
        pd.DataFrame: feature selected data
    """
    # Temporarily store names/labels to append back later
    names = data["name"]
    labels = data["status"]
    # Filter dataframe to only include numerical features
    data = data.select_dtypes(include="number").drop("status", axis=1)
    # Calculate correlation amongst features in data
    matrix = data.corr().abs()
    # Create mask that eliminates mirrored values
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    # Apply mask to matrix, only leaving valuable info
    reduced_matrix = matrix.mask(mask)
    # If correlation between two features exceeds correlation threshold, store first column name to list of sets
    to_drop = [
        c for c in reduced_matrix.columns if any(reduced_matrix[c] > corr_threshold)
    ]

    # Drop all columns with correlation above threshold value
    data = data.drop(to_drop, axis=1)

    # Append original columns that couldn't/shouldn't be processed in this function
    data["name"] = names
    data["status"] = labels

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
            columns.append(pd.DataFrame(
                {"removed": column_names, "prediction": data}))
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
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size)

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
    # plot_label_distribution(
    #     y_sm,
    #     "class_ratio_training_smote",
    #     "Balanced Training Data with smote",
    # )

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

    file_name = f"best_dl_settings_{title_template}.csv"

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
            "batch_size": [24, 96, 256, 512],
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
            results = CV_dl.fit(
                x_train, y_train, validation_data=(x_val, y_val))
        else:
            print("[*]\tTuning Model using Random Search Cross Validation")
            CV_dl = RandomizedSearchCV(
                estimator=model, param_distributions=param_grid, n_jobs=-1, cv=5
            )
            results = CV_dl.fit(
                x_train, y_train, validation_data=(x_val, y_val))

        print(results.best_score_, results.best_params_)
        end = time.perf_counter()
        print(f"{end-start:.2f} seconds")
        sdf = pd.DataFrame.from_dict(
            results.best_params_, orient="index").transpose()
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
    save_best_params(CV_rfc.best_params_,
                     f"best_rf_settings_{title_template}.csv")


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
    save_best_params(grid.best_params_,
                     f"best_svm_settings_{title_template}.csv")


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


def extra_trees(x_train, x_test, y_train, y_test):
    """Uses Extra trees classifier to calculate each feature's importance in classification
    Creates plots

    Args:
        x_train (pd.DataFrame): Data for training
        x_test (pd.DataFrame): Data for testing
        y_train (pd.Series): Labels for train data
        y_test (pd.Series): Labels for test data
    """
    TOP_FEATURES = x_train.shape[1]
    forest = ExtraTreesClassifier(
        n_estimators=250, max_depth=5, random_state=42)
    forest.fit(x_train, y_train)

    importances = forest.feature_importances_
    std = np.std(
        [tree.feature_importances_ for tree in forest.estimators_], axis=0)

    indices = np.argsort(importances)[::-1]
    indices = indices[:TOP_FEATURES]

    return x_train.columns, indices, importances, std


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
    X_train_std_new, y_train_new = balance_training_data(
        X_train_std_new, y_train_new)
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
    cv = StratifiedKFold(n_splits=n, shuffle=True)
    # Execute cross validation
    k_scores = cross_val_score(
        model, X, y, scoring="accuracy", cv=cv, n_jobs=-1)

    return k_scores


def create_model_rf(x_train, y_train):
    if not exists(f"settings/best_rf_settings_{title_template}.csv"):
        find_best_rf_settings(x_train, y_train)
    # Load in optimal settings generated by GridSearchCV
    sdf = pd.read_csv(f"settings/best_rf_settings_{title_template}.csv")
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


def random_forest(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> pd.Series:
    """Random Forest training and classification

    Args:
        x_train (pd.DataFrame): training set
        x_test (pd.DataFrame): test set
        y_train (pd.Series): training labels
        y_test (pd.Series): test labels
    Returns:
        pd.Series: predicted and actual labels
    """
    rf = create_model_rf(x_train, y_train)
    # Train model
    rf.fit(x_train, y_train)
    # Predict labels
    pred = rf.predict(x_test)

    return y_test.values, pred


def create_model_svm(x_train, y_train):
    if not exists(f"settings/best_svm_settings_{title_template}.csv"):
        find_best_svm_settings(x_train, y_train)

    # Load in optimal settings generated by GridSearchCV
    sdf = pd.read_csv(f"settings/best_svm_settings_{title_template}.csv")
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
    svm = create_model_svm(x_train, y_train)
    svm.fit(x_train, y_train)
    pred = svm.predict(x_test)
    return y_test.values, pred


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
    """
    # Load in epochs from environment variable file
    find_best_nn_settings(x_train, y_train)
    settings_title_affix = "balanced" if balance_labels else "imbalanced"
    file_name = f"best_dl_settings_{title_template}.csv"

    # Read best neural network settings as dictionary
    parms = pd.read_csv(f"settings/{file_name}").to_dict("index")[0]

    # Load best settings
    batch_size = parms["batch_size"]
    epochs = parms["epochs"]
    layer1 = parms["layer1"]
    layer2 = parms["layer2"]
    init = parms["init"]
    optimizer = parms["optimizer"]
    loss = parms["loss"]
    metric = parms["metric"]
    output = parms["output"]

    # If a model hasn't been trained before...
    if not exists(
        f"models/parkinsons_model_{epochs}_{title_template}.tf"
    ):
        print("[*]\tTraining Neural Network")
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
            verbose=0
        )

        # Save parkinsons model
        model.save(
            f"models/parkinsons_model_{epochs}_{title_template}.tf"
        )

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
                f"accuracy_smooth_{title_template}",
            )
            df = pd.DataFrame({"acc": acc, "val_acc": val_acc})
            df.to_csv(
                f"plots/neural_network/nn_acc_loss/accuracy_{title_template}.csv",
                index=False,
            )
            df = pd.DataFrame(
                {"acc": smooth_curve(acc), "val_acc": smooth_curve(val_acc)}
            )
            df.to_csv(
                f"plots/neural_network/nn_acc_loss/accuracy_smooth_{title_template}.csv",
                index=False,
            )
            plot_loss(
                smooth_curve(loss), smooth_curve(
                    val_loss), f"loss_smooth_{title_template}"
            )
            df = pd.DataFrame({"loss": loss, "val_loss": val_loss})
            df.to_csv(
                f"plots/neural_network/nn_acc_loss/loss_{title_template}.csv",
                index=False,
            )
            df = pd.DataFrame(
                {"loss": smooth_curve(
                    loss), "val_loss": smooth_curve(val_loss)}
            )
            df.to_csv(
                f"plots/neural_network/nn_acc_loss/loss_smooth_{title_template}.csv",
                index=False,
            )
    else:
        dprint("[*]\tModel already trained")


def test_neural_network(x_test, y_test):
    """Tests a trained neural network"""
    file_name = f"best_dl_settings_{title_template}.csv"
    # Read epochs from best neural network settings file
    parms = pd.read_csv(f"settings/{file_name}").to_dict("index")[0]
    epochs = parms["epochs"]

    # Loads model
    # TODO: Somehow state whether model takes data with reduced dimensions (PCA, LDA)
    model = models.load_model(
        f"models/parkinsons_model_{epochs}_{title_template}.tf"
    )

    # predictions = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x=x_test)
    new = []
    for i in range(len(predictions)):
        new.append(predictions[i][0])

    pred = [round(pred) for pred in new]

    return y_test.values, pred


# ------------- MISC METHODS -------------
def accuracy_table(actual, pred, names, type="all"):
    actual = ["Healthy" if val == 1 else "Alzheimer's" for val in actual]
    pred = ["Healthy" if val == 1 else "Alzheimer's" for val in pred]

    labels = pd.DataFrame({"name": names, "actual": actual, "pred": pred})

    if type == "incorrect":
        # Filters dataframe to only contain incorrect predictions
        labels = labels.query("actual != pred")
        print(labels)
    else:
        print(labels)


def get_accuracy(actual, pred):
    count = 0
    for i in range(len(pred)):
        if pred[i] == actual[i]:
            count = count + 1

    acc = f"{count / float(len(pred))*100:.2f}"
    return acc


def dprint(text):
    verbose = False
    if verbose:
        print(text)


def print_title(title):
    """Prints title

    Args:
        title (str): title
    """
    print("--------------------------------------------------------------")
    print(f"\t{title}")
    print("--------------------------------------------------------------")


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
        "plots/neural_network",
        "plots/neural_network/nn_acc_loss",
        "plots/neural_network/nn_training_stats",
        "plots/SVM",
        "plots/RF",
    ]
    for folder in folders:
        if not os.path.isdir(folder):
            os.mkdir(folder)


def initialise_settings():
    """Initialises global settings for program"""
    # Initialise global variables
    global eda
    global plots

    global normalise
    global remove_correlate_features
    global balance_labels
    global corr_threshold
    global dimensionality_reduction
    global stratify
    global manual_settings
    global test_size
    global val_size
    global apply_pca
    global apply_tsvd
    global experimental
    global grid_search
    global random_search
    global rewrite_model
    global cv

    # set environment variables
    eda = strtobool(os.getenv("EDA"))
    plots = strtobool(os.getenv("PLOTS"))

    normalise = strtobool(os.getenv("NORMALISE"))
    remove_correlate_features = strtobool(
        os.getenv("REMOVE_CORRELATED_FEATURES"))
    corr_threshold = float(os.getenv("CORRELATION_DROP_THRESHOLD"))
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

    cv = strtobool(os.getenv("CROSS_VALIDATION"))

    global title_template
    title_template = "balanced_norm_rcf"
    if not normalise:
        title_template = title_template.replace("norm_", "")
    if not remove_correlate_features:
        title_template = title_template.replace("rcf", "")
    if not balance_labels:
        title_template = title_template.replace("balanced", "imbalanced")


# ------------- MAIN METHODS -------------
def main():
    """Main"""

    cls()

    # Load in environment variables from .env
    load_dotenv()

    # Initialises global settings for program using loaded environment variables
    initialise_settings()

    # Prepare directories for program
    prepare_directory()

    # Load Dataset
    df = pd.read_csv("data/parkinsons.data")

    # EDA
    if eda:
        print_title("Exploratory Data Analysis")
        print(df.info())
        print(df.describe())
        # Number of names
        print(f"Patient Frequency: {len(df['name'].unique())}")
        # Label distribution
        print(f"Label distribution\n{df['status'].value_counts()}")

        # Filters dataframe to only include numeric data
        numeric_dataframe = df.select_dtypes(include=np.number)

        if plots:
            print_title("Generating EDA Plots")
            print("[*]\tPlotted label distribution")
            # Label ratio
            plot_label_distribution(
                df["status"], "class_ratio_all_data", "All Data Labels Ratio"
            )
            print("[*]\tPlotting pair plot")
            # Plots pairwise correlation
            for kind in ["reg", "scatter"]:
                plot_pairplot(
                    numeric_dataframe, file_name="full_pairplot.png", kind=kind
                )
                first_five = list(numeric_dataframe.columns[:5])
                first_five.append("status")
                plot_pairplot(
                    numeric_dataframe[first_five],
                    file_name="partial_pairplot.png",
                    kind=kind,
                )
            print("[*]\tPlotting correlation matrix")
            # Plots matrix of plots showing correlation between features in data
            plot_correlation_matrix(
                numeric_dataframe, file_name="full_correlation_matrix.png"
            )

            plot_correlation_matrix(
                numeric_dataframe.iloc[:4, :4],
                file_name="partial_correlation_matrix.png",
            )
    print_title("Pre-processing")
    # Transform data based on information discovered through eda
    df = data_wrangling(df)

    # Split data into data and labels
    X, y = data_label_split(df)

    # Reduce dimensionality
    if dimensionality_reduction:
        X = dimensionality_reduction(X)

    if experimental:
        print_title("Experimental")
        if not exists("misc/removed_columns.csv"):
            predict_minus_feature(df)

        with open("misc/removed_columns.csv", newline="") as f:
            reader = csv.reader(f)
            data = list(reader)

        for column in data[0]:
            print(f"[*]\tRemoving {column}")
            del X[column]

    # Split data and labels into training/test sets
    x_train, x_test, y_train, y_test = split_data_train_test(X, y, test_size)

    print("[*]\tDrop 'name' column")
    # Drop columns that hold no value regarding prediction
    # Drop names from train/test data, store names from test data to array for later
    x_train.drop("name", axis=1, inplace=True)
    n_test = x_test["name"].tolist()
    x_test.drop("name", axis=1, inplace=True)

    if balance_labels:
        print_title(f"Classification Model [Balanced Training Labels]")
        x_train, y_train = balance_training_data(x_train, y_train)
    else:
        print_title(f"Classification Model [Imbalanced Training Labels]")
    print("------------------------------------------------")

    # Identify outliers
    isolation_forest(x_train, x_test, y_train, y_test)

    # Feature importance
    x_train.columns, indices, importances, std = extra_trees(
        x_train, x_test, y_train, y_test
    )

    plot_feature_importance(
        x_train.columns,
        indices,
        importances,
        std,
        f"feature_importance_{title_template}.png",
    )

    # # logistic_regression(x_train, x_test, y_train, y_test)
    linear_regression(x_train, x_test, y_train, y_test)

    a1, p1 = svm(x_train, x_test, y_train, y_test)
    # accuracy_`table(a1, p1, n_test, "incorrect")

    a2, p2 = random_forest(x_train, x_test, y_train, y_test)
    # accuracy_table(a2, p2, n_test, "incorrect")

    train_neural_network(x_train, x_test, y_train, y_test)
    a3, p3 = test_neural_network(x_test, y_test)
    # accuracy_table(a3, p3, n_test, "incorrect")

    if plots:
        print_title("Post Classification Results")
        model_results = {"SVM": [(a1, p1), "svm"], "Random Forest": [
            (a2, p2), "rf"], "NN": [(a3, p3), "nn"]}
        print("Accuracy\tLog Loss\tROC AUC\tModel")
        # TODO: Scatter plots to compare predicted with actual labels
        # for actual, pred in zip(actuals, predictions):
        for model, results in model_results.items():
            actual = results[0][0]
            pred = results[0][1]
            model_abbrev = results[1][0]
            acc = f"{accuracy_score(actual, pred)*100:.2f}"
            file_name = f"{model_abbrev}_cm_{title_template}"
            plot_title = f"[Imbalanced] {model} Confusion Matrix : ({acc}%)"
            if balance_labels:
                plot_title = f"[Balanced] {model} Confusion Matrix : ({acc}%)"

            plot_cm(actual, pred, plot_title, file_name)
            print(
                f"{acc}%\t\t{log_loss(actual, pred):.2f}\t\t{roc_auc_score(actual, pred):.2f}\t{model}")

    if cv:
        del X["name"]
        print_title("Cross Validation")
        print("Folds\tAcc\tMethod")
        models = {
            "Random Forest": create_model_rf(x_train, y_train),
            "Support Vector Machine": create_model_svm(x_train, y_train),
        }
        max_folds = 10
        cv_models = range(2, max_folds + 2)
        for key, model in models.items():
            all_scores = []
            for i in cv_models:
                k_scores = cross_validation(model, i, X, y)
                all_scores.append(k_scores)
                # Print mean accuracy
                print(f"  {i}\t{k_scores.mean()*100:.2f}%\t{key}")

            file_name = f"boxplot_svm_{i-1}_cv.png"
            if "Random Forest" in key:
                file_name = f"boxplot_rf_{i-1}_cv.png"

            plot_title = f"{key} Repeated K-Fold Cross Validation"

            if plots:
                plot_cv_box_plot(cv_models, all_scores, file_name, plot_title)


if __name__ == "__main__":
    main()
