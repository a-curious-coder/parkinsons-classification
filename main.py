import os
import csv
import matplotlib
from sklearn.preprocessing import StandardScaler
from visualisations import *

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import math
import time
from os.path import exists
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from dotenv import load_dotenv
from plotly.figure_factory import create_distplot
from imblearn.over_sampling import SMOTE
from xgboost import XGBRegressor
from sklearn import linear_model
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
    StratifiedShuffleSplit,
    cross_val_score,
)
from sklearn.metrics import (
    mean_squared_error,
    plot_confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)
from tensorflow.keras import layers, models
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
import warnings

warnings.filterwarnings("ignore")


def prepare_directory():
    """Creates necessary folders in preparation for data/models saved"""
    folders = ["misc", "plots", "models", "settings", "plots/removed_columns", "plots/confusion_matrices"]
    for folder in folders:
        if not os.path.isdir(folder):
            os.mkdir(folder)


def normalize_tabular_data(data):
    """Normalizes every value in the given data

    Args:        data (pd.DataFrame): Parkinsons Data

    Returns:
        pd.DataFrame: Normalized Parkinsons data
    """
    avoid = ["status", "name"]
    # apply normalization techniques
    for column in data.columns:
        if column not in avoid:
            data[column] = data[column] / data[column].abs().max()
    return data


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


def split_data_train_test(data):
    """Creates train/test split given a dataframe

    Args:
        data (pd.DataFrame): parkinsons data

    Returns:
        pd.DataFrame: training and test splits
    """
    consistent_split = bool(int(os.getenv("CONSISTENT_SPLIT")))
    test_size = float(os.getenv("TEST_SIZE"))
    balance_data = bool(int(os.getenv("BALANCE_DATA")))

    X, y = data_wrangling(data)
    if consistent_split:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = test_size, stratify = y, random_state = 42)
    else:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    if balance_data:
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


# ------------- DATA WRANGLING -------------
def data_wrangling(df):
    normalise = bool(int(os.getenv("NORMALISE")))
    print("-------------------\nPreprocessing\n-------------------")
    if normalise:
        print("[*]\tNormalize Data")
        # Bring all data values to the same range
        df = normalize_tabular_data(df)
    # Split data into data and labels
    X, y = data_label_split(df)
    print("[*]\tDrop 'name' column")
    # Drop columns that hold no value regarding prediction
    del X["name"]
    # print("[*]\tFeature Selection")
    # compare_features = False
    # if compare_features:
    #     if not exists("plots/compare_features"):
    #         os.mkdir("plots/compare_features")
        # for column in X.columns:
        #     for column2 in X.columns:
        #         if column == column2:
        #             break
        # fig = go.Figure(
        #     data=go.Scatter(x=X[column], y=X[column2], mode="markers"),
        #     layout=go.Layout(
        #         title=dict(text=f"{column} vs. {column2}", x=0.5),
        #         xaxis=dict(title=f"{column}"),
        #         yaxis=dict(title=f"{column2}"),
        #     ),
        # )
        # fig.write_html(
        #     f"plots/compare_features/scatter_{column}_vs_{column2}.html"
        # )

    dprint("[*]\tPopulation Density Function PDF of features with target variables")
    # del X["spread1"]
    fig = create_distplot(
        [X[column] for column in X.columns],
        X.columns,
        bin_size=[2, 2],
        show_rug=False,  # rug
        show_hist=False,  # hist
        show_curve=True,  # curve
    )
    # fig.show()

    dprint("-------------------\nDimensionality Reduction\n-------------------")
    pca = bool(int(os.getenv("PCA")))
    tsvd = bool(int(os.getenv("SVD")))
    pca_components = int(os.getenv("PCA_COMPONENTS"))

    if pca:
        dprint(f"[*]\tPCA with {pca_components} components")
        pca = PCA(n_components=pca_components, whiten="True")
        pcs = pca.fit(X).transform(X)
        # Store PCA results to DataFrame
        X = pd.DataFrame(
            data=pcs,
            index=range(pcs.shape[0]),
            columns=[f"PC{i}" for i in range(pca_components)],
        )
    elif tsvd:
        svd = TruncatedSVD(n_components=5)
        svds = svd.fit(X).transform(X)
        pd.DataFrame(
            data=svds,
            index=range(svds.shape[0]),
            columns=[f"PC{i}" for i in range(svds.shape[1])],
        )
    else:
        dprint("[*]\tNo Method Used")

    return X, y


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
        marks_a_cunt = []
        for trial in range(num_trials):
            # print(f"{(trial+1)/num_trials*100:.2f}% Complete")
            count = 0
            data = []
            column_names = []
            # For each column in the data
            for i in range(df.shape[1] - 2):
                # Split data into test and training sets
                x_train, x_test, y_train, y_test = split_data_train_test(df)
                print(x_train.head())

                balance_data = bool(int(os.getenv("BALANCE_DATA")))

                if i != 0:
                    column_names.append(X.columns[count])
                    del x_train[X.columns[count]]
                    del x_test[X.columns[count]]
                    count += 1

                pred = svm(x_train, x_test, y_train, y_test)
                data.append(pred)
            column_names.insert(0, "alldata")
            marks_a_cunt.append(
                pd.DataFrame({"removed": column_names, "prediction": data})
            )
        predictions = pd.concat(marks_a_cunt)
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


# ------------- BEST SETTINGS -------------
def save_best_params(best_params, file_name):
    """Saves the best parameters of any given model to settings folder in working directory

    Args:
        best_params (dict): [description]
        file_name (str): [description]
    """
    sdf = pd.DataFrame.from_dict(best_params, orient="index").transpose()
    sdf.to_csv(f"settings/{file_name}", index=False)


def find_best_nn_settings(x_train, y_train, file_name):
    if not exists(f"settings/best_dl_{file_name}.csv"):
        print("-------------------\nGridSearchCV\n-------------------")
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
        CV_dl = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 42, stratify = y_train)
        results = CV_dl.fit(x_train, y_train, validation_data = (x_val, y_val))
        print(results.best_score_, results.best_params_)
        end = time.perf_counter()
        print(f"{end-start:.2f} seconds")
        sdf = pd.DataFrame.from_dict(results.best_params_, orient="index").transpose()
        sdf.to_csv(f"settings/best_dl_{file_name}.csv", index=False)


def find_best_if_settings(x_train, y_train):
    print("[*]\tAttempting to find optimal parameters for Isolation Forest...")
    model = IsolationForest(random_state=47)

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

    regressors = {"XGBRegressor": XGBRegressor(random_state=1)}

    df_models = pd.DataFrame(columns=["model", "run_time", "rmse", "rmse_cv"])

    for key in regressors:

        start_time = time.time()

        regressor = regressors[key]
        model = regressor.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        scores = cross_val_score(
            model, x_train, y_train, scoring="neg_mean_squared_error", cv=10
        )

        row = {
            "model": key,
            "run_time": format(round((time.time() - start_time) / 60, 2)),
            "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred))),
            "rmse_cv": round(np.mean(np.sqrt(-scores))),
        }

        df_models = df_models.append(row, ignore_index=True)

    print(df_models.head())
    isf = IsolationForest(
        contamination="auto",
        max_features=5,
        max_samples=2,
        n_estimators=1,
        n_jobs=-1,
    )

    y_pred = isf.fit_predict(x_train)
    print(y_pred)
    print(x_train.shape, y_train.shape)
    X_train_iforest, y_train_iforest = (
        x_train[(y_pred != -1), :],
        y_train[(y_pred != -1)],
    )
    print(X_train_iforest.shape, y_train_iforest.shape)


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


def random_forest(x_train, x_test, y_train, y_test, plot_title = "Confusion Matrix"):
    """Random Forest Classification

    Args:
        x_train ([type]): [description]
        x_test ([type]): [description]
        y_train ([type]): [description]
        y_test ([type]): [description]

    Returns:
        [type]: [description]
    """
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
    print(f"[*]\t{acc}%\tRandom Forest")
    plot_cm(actual, pred, plot_title + f"({acc}%)")
    return count / float(len(pred)) * 100


def svm(x_train, x_test, y_train, y_test, plot_title = "Confusion Matrix"):
    """Support Vector Machine Neural Network

    Args:
        x_train ([type]): [description]
        x_test ([type]): [description]
        y_train ([type]): [description]
        y_test ([type]): [description]

    Returns:
        [type]: [description]
    """
    if not exists("settings/best_svm_settings.csv"):
        find_best_svm_settings(x_train, y_train)

    # Load in optimal settings generated by GridSearchCV
    sdf = pd.read_csv("settings/best_svm_settings.csv")
    # Extract best settings
    C = sdf.iloc[0]["C"]
    gamma = sdf.iloc[0]["gamma"]
    kernel = sdf.iloc[0]["kernel"]

    svclassifier = SVC(kernel=kernel, gamma = gamma, C = C)
    svclassifier.fit(x_train, y_train)
    pred = svclassifier.predict(x_test)
    actual = y_test.values
    count = 0
    for i in range(len(pred)):
        if pred[i] == actual[i]:
            count = count + 1
    acc = f"{count / float(len(pred))*100:.2f}"
    print(f"[*]\t{acc}%\tSVM")
    plot_cm(actual, pred, plot_title + f"({acc}%)")
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


def train_neural_network(x_train, x_test, y_train, y_test, file_name):
    """Trains neural network with parkinsons data

    Args:
        x_train (pd.DataFrame): training data
        x_test (pd.DataFrame): test data
        y_train (pd.DataFrame): training labels
        y_test (pd.DataFrame): test labels
        file_name ([type]): Name of best settings file
    """
    # Load in epochs from environment variable file
    epochs = int(os.getenv("EPOCHS"))
    print(f"-------------------\nNeural Network ({epochs})\n-------------------")
    find_best_nn_settings(x_train, y_train, file_name)

    # Do we overwrite model file if there is 1
    rewrite_model = bool(int(os.getenv("REWRITE_MODEL")))

    # Load best Deep Learning settings file
    bsdf = pd.read_csv(f"settings/best_dl_{file_name}.csv")

    # Load best settings
    batch_size = bsdf.iloc[0]["batch_size"]
    if not rewrite_model:
        epochs = bsdf.iloc[0]["epochs"]
    layer1 = bsdf.iloc[0]["layer1"]
    layer2 = bsdf.iloc[0]["layer2"]
    init = bsdf.iloc[0]["init"]
    optimizer = bsdf.iloc[0]["optimizer"]
    loss = bsdf.iloc[0]["loss"]
    metric = bsdf.iloc[0]["metric"]
    output = bsdf.iloc[0]["output"]
    input_shape = x_train.shape[1]

    # If a model hasn't been trained before...
    if (
        not exists(f"models/parkinsons_model_{epochs}_{input_shape}.tf")
        or rewrite_model
    ):
        # Prepare network
        model = create_model(
            layer1, layer2, output, x_train.shape[1], optimizer, loss, metric, init
        )

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 42, stratify = y_train)

        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            verbose=0,
        )

        # Save parkinsons model
        model.save(f"models/parkinsons_model_{epochs}_{input_shape}.tf")

        # Get accuracy / loss data
        history_dict = history.history
        # print(history_dict.keys())
        acc = history_dict["accuracy"]
        val_acc = history_dict["val_accuracy"]
        loss = history_dict["loss"]
        val_loss = history_dict["val_loss"]

        plot_accuracy(
            smooth_curve(acc), smooth_curve(val_acc), f"parkinsons_accuracy_{epochs}"
        )
        plot_loss(
            smooth_curve(loss), smooth_curve(val_loss), f"parkinsons_loss_{epochs}"
        )
    else:
        print("[*]\tModel already trained")


def test_neural_network(x_test, y_test, file_name, plot_title = "Confusion Matrix"):
    """Tests a trained neural network"""
    # Load best Deep Learning settings file
    bsdf = pd.read_csv(f"settings/best_dl_{file_name}.csv")

    rewrite_model = True
    # Load best settings
    batch_size = bsdf.iloc[0]["batch_size"]
    epochs = int(os.getenv("EPOCHS"))
    if not rewrite_model:
        epochs = bsdf.iloc[0]["epochs"]
    layer1 = bsdf.iloc[0]["layer1"]
    layer2 = bsdf.iloc[0]["layer2"]
    init = bsdf.iloc[0]["init"]
    optimizer = bsdf.iloc[0]["optimizer"]
    loss = bsdf.iloc[0]["loss"]
    metric = bsdf.iloc[0]["metric"]
    output = bsdf.iloc[0]["output"]
    input_shape = x_test.shape[1]

    # Loads models
    print(f"Testing Neural Network Model ({epochs})")
    # TODO: Somehow state whether model takes data with reduced dimensions (PCA, LDA)
    model = models.load_model(f"models/parkinsons_model_{epochs}_{input_shape}.tf")
    # predictions = model.evaluate(x_test, y_test, verbose=0)
    x_test = (x_test-x_test.mean())/x_test.std()
    predictions = model.predict(x = x_test)
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
    print(f"[*]\t{acc}%\t({layer1}, {layer2})\t{optimizer}\t{loss}")
    plot_title = plot_title + f"({acc}%)"
    plot_cm(y_test, pred, plot_title)


# ------------- MISC METHODS -------------
def dprint(text):
    verbose = False
    if verbose:
        print(text)


def main():
    """Main"""
    # Load in environment variables from .env
    load_dotenv()

    # Prepare directories for program
    prepare_directory()

    # Load Dataset
    print("-------------------\nLoad Data\n-------------------")
    df = pd.read_csv("data/parkinsons.data")

    # EDA
    print("-------------------\nEDA\n-------------------")
    # Number of names
    print(f"Number of names: {len(df['name'].unique())}")
    # Number of observations (rows)
    print(f"Number of rows: {df.shape[0]}")
    # Label distribution
    print(f"Target class distribution\n{df['status'].value_counts()}")

    print("---------\nGenerating Plots\n----------")
    plot_class_ratio(df["status"], "class_ratio_all_data", "All Data Labels Ratio")
    predict_minus_feature(df)

    if not exists("misc/removed_columns.csv"):
        predict_minus_feature(df)
    for i in range(2):
        mode = "All Data"
        x_train, x_test, y_train, y_test = split_data_train_test(df)
        
        if i == 0:
            print("[*]\tAll Data")
        else:
            mode = "Removed Columns"
            with open("misc/removed_columns.csv", newline="") as f:
                reader = csv.reader(f)
                data = list(reader)
            for column in data[0]:
                print(f"[*]\tRemoving {column}")
                del x_train[column]
                del x_test[column]
        # print(x_train.head())
        # Find outliers
        # isolation_forest(x_train, x_test, y_train, y_test)
        # Feature importance
        # extra_trees_classifier(x_train, x_test, y_train, y_test)
        #
        # logistic_regression(x_train, x_test, y_train, y_test)
        # linear_regression(x_train, x_test, y_train, y_test)
        svm(x_train, x_test, y_train, y_test, f"({mode}) SVM Confusion Matrix")
        random_forest(x_train, x_test, y_train, y_test, f"({mode}) Random Forest Confusion Matrix")
        train_neural_network(x_train, x_test, y_train, y_test, "parkinsons")
        test_neural_network(x_test, y_test, "parkinsons", plot_title = f"({mode}) Neural Network Confusion Matrix")

if __name__ == "__main__":
    main()
