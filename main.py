from visualisations import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import random
import tensorflow as tf
import numpy as np
import math
import time

from dotenv import load_dotenv
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import models
from tensorflow.keras import layers
from os.path import exists


def normalize_tabular_data(data):
    """Normalizes the values of each column

    Args:        data (pd.DataFrame): Parkinsons Data

    Returns:
        pd.DataFrame: Normalized Parkinsons data
    """
    avoid = ['status', 'name']
    # apply normalization techniques
    for column in data.columns:
        if column not in avoid:
            data[column] = data[column] / data[column].abs().max()
    return data


def data_label_split(data):
    X = data.drop('status', axis = 1)
    y = data['status']
    return X, y


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
    # model = LogisticRegression()
    # param_grid = {
    #     "C": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    #     "penalty":["l1", "l2"],
    #     "solver":["liblinear"]
    # }
    # logreg_cv = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc').fit(x_train, y_train)
    # print("tuned hyperparameters :(best parameters) ",logreg_cv.best_params_)
    # print("AUC_ROC :",logreg_cv.best_score_)
    params = {'C': 0.7, 'penalty': 'l1', 'solver': 'liblinear'}
    classifier = LogisticRegression(random_state=42, n_jobs=-1, **params).fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    y_pred_proba = classifier.predict_proba(x_test)[:,1]
    print("TEST")
    print("Precision: {}".format(precision_score(y_test, y_pred)))
    print("Recall: {}".format(recall_score(y_test, y_pred)))
    print("F1 Score: {}".format(f1_score(y_test, y_pred)))
    print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
    print("AUC-ROC: {}".format(roc_auc_score(y_test, y_pred_proba)))
    print("Test Confusion Matrix")
    print(confusion_matrix(y_test, classifier.predict(x_test)))
    weight_vector = list(classifier.coef_[0])
    dist = np.dot(x_train, weight_vector)
    y_dist = dist*[-1 if x==0 else 1 for x in list(y_train)]
    print(len(y_dist))
    print(y_train.value_counts())
    import seaborn as sns
    import matplotlib.pyplot as plt

    # sns.kdeplot(y_dist)
    # plt.xlabel("Distance * Y-class")
    # plt.title("Density plot of y dist")
    # plt.grid()
    # plt.show()

    val = np.percentile(y_dist, 20)
    print("Threshold Val: ", val)

    print(y_train[(y_dist < val)].value_counts())

    X_train_std_new = x_train[(~(y_dist < val))]
    y_train_new = y_train[(~(y_dist < val))]
    print(X_train_std_new.shape)
    print(y_train_new.shape)

    print(y_train_new.value_counts())
    params = {'C': 0.001, 'penalty': 'l2'}
    X_train_std_new, y_train_new = balance_training_data(X_train_std_new, y_train_new)
    classifier1 = LogisticRegression(random_state=42, n_jobs=-1, **params).fit(X_train_std_new, y_train_new)
    
    y_pred = classifier1.predict(x_test)
    y_pred_proba = classifier1.predict_proba(x_test)[:,1]
    print("TEST")
    print("Precision: {}".format(precision_score(y_test, y_pred)))
    print("Recall: {}".format(recall_score(y_test, y_pred)))
    print("F1 Score: {}".format(f1_score(y_test, y_pred)))
    print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
    print("AUC-ROC: {}".format(roc_auc_score(y_test, y_pred_proba)))

    plot_logistic_regression(classifier1.predict_proba(x_test)[:,1])


def isolation_forest(x_train, x_test, y_train, y_test):
    from sklearn.ensemble import IsolationForest

    isf = IsolationForest(n_jobs=-1, random_state=1)
    isf.fit(x_train, y_train)

    isf.score_samples(x_train)
    print(isf.predict(x_train))


def extra_trees_classifier(x_train, x_test, y_train, y_test):
    from sklearn.ensemble import ExtraTreesClassifier
    TOP_FEATURES = x_train.shape[1]
    forest = ExtraTreesClassifier(n_estimators = 250, max_depth = 5, random_state = 42)
    forest.fit(x_train, y_train)

    importances = forest.feature_importances_
    std = np.std(
        [tree.feature_importances_ for tree in forest.estimators_],
        axis = 0
    )
    
    indices = np.argsort(importances)[::-1]
    indices = indices[:TOP_FEATURES]

    plot_feature_importance(x_train.columns, indices, importances, std)


def balance_training_data(x_train, y_train):
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(sampling_strategy='minority', n_jobs=-1)
    feature_names = x_train.columns
    X_sm, y_sm = smote.fit_resample(x_train, y_train)

    return X_sm, y_sm


def random_forest(x_train, x_test, y_train, y_test):
    if not exists("settings/best_rf.csv"):
        print("-------------------\nGridSearchCV\n-------------------")
        print("[*]\tAttempting to find optimal parameters for Random Forest...")
        start = time.perf_counter()
        param_grid = { 
            'n_estimators': range(20, 260, 20),
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth' : [4,5,6,7,8],
            'criterion' :['gini', 'entropy']
        }

        rfc = RandomForestClassifier(random_state=42)
        CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
        CV_rfc.fit(x_train, y_train)
        end = time.perf_counter()
        print(f"{end-start:.2f} seconds")
        sdf = pd.DataFrame.from_dict(CV_rfc.best_params_, orient = 'index').transpose()
        sdf.to_csv("settings/best_rf.csv", index = False)
    print("-------------------\nRandom Forest\n-------------------")
    # Load in optimal settings generated by GridSearchCV
    sdf = pd.read_csv("settings/best_rf.csv")
    # Extract best settings
    criterion = sdf.iloc[0]['criterion']
    max_depth = sdf.iloc[0]['max_depth']
    max_features = sdf.iloc[0]['max_features']
    n_estimators = sdf.iloc[0]['n_estimators']
    # Initialise model
    rf = RandomForestClassifier(criterion = criterion, max_depth = max_depth, max_features = max_features, n_estimators = n_estimators)
    # Train model
    rf.fit(x_train, y_train)
    # Predict and 
    pred = rf.predict(x_test)
    labels = y_test.values
    count = 0
    for i in range(len(pred)):
        if pred[i] == labels[i]:
            count = count + 1
    print(f"[*]\t{count / float(len(pred))*100:.2f}%\tRandom Forest")


def create_model(layer1, layer2, output, input_shape, optimizer, loss, metric, init, dropout_rate = 0.0):
    # Initialise neural network
    network = models.Sequential()
    network.add(layers.Dense(layer1, activation='relu', kernel_initializer = init, input_shape=(input_shape, )))
    network.add(layers.Dropout(dropout_rate))
    network.add(layers.Dense(layer2, activation='relu'))
    network.add(layers.Dropout(dropout_rate))
    network.add(layers.Dense(1, activation=output))

    network.compile(optimizer=optimizer,
                                loss=loss,
                                metrics=[metric])
    return network


def keras_network(x_train, x_test, y_train, y_test, dataset):
    # Load in epochs from environment variable file
    epochs = int(os.getenv("EPOCHS"))
    print("-------------------\nNeural Network\n-------------------")
    if not exists("settings/best_dl_{dataset}.csv"):
        print("-------------------\nGridSearchCV\n-------------------")
        print("[*]\tAttempting to find optimal parameters for Neural Network...")
        start = time.perf_counter()
        param_grid = {
            'layer1':[16],
            'layer2':[8],
            'output':['sigmoid', 'softmax'],
            'input_shape':[x_train.shape[1]],
            'optimizer' : ['rmsprop', 'adam'],
            'loss':['binary_crossentropy'],
            'metric':['accuracy'],
            'init' : ['glorot_uniform', 'normal', 'uniform'],
            'dropout_rate':[0.0, 0.15, 0.3],
            'batch_size': [512],
            'epochs' : [500]
        }
        
        model = KerasClassifier(build_fn=create_model, verbose = 1)
        CV_dl = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs = -1,  cv= 5)
        results = CV_dl.fit(x_train, y_train)
        print(results.best_score_, results.best_params_)
        end = time.perf_counter()
        print(f"{end-start:.2f} seconds")
        sdf = pd.DataFrame.from_dict(results.best_params_, orient = 'index').transpose()
        sdf.to_csv(f"settings/best_dl_{dataset}.csv", index = False)

    # Do we overwrite model file if there is 1
    rewrite_model = bool(int(os.getenv("REWRITE_MODEL")))

    # Load Best Deep Learning Settings
    bsdf = pd.read_csv(f"settings/best_dl_{dataset}.csv")

    # Best settings
    batch_size = bsdf.iloc[0]['batch_size']
    epochs = bsdf.iloc[0]['epochs']
    layer1 = bsdf.iloc[0]['layer1']
    layer2 = bsdf.iloc[0]['layer2']
    init = bsdf.iloc[0]['init']
    optimizer = bsdf.iloc[0]['optimizer']
    loss = bsdf.iloc[0]['loss']
    metric = bsdf.iloc[0]['metric']
    output = bsdf.iloc[0]['output']
    input_shape = x_train.shape[1]
    
    # If there's no model files
    if not exists(f"models/parkinsons_model_{epochs}_{input_shape}.tf") or rewrite_model:
        print("Training/Testing Model")
        print("Accuracy\tNodes\tOptimizer\tLoss")
        # Validation set size
        val_size = math.floor((x_train.shape[0] / 10) * 9)
        with tf.device('/cpu:0'):
            # Prepare network
            model = create_model(layer1, layer2, output, x_train.shape[1], optimizer, loss, metric, init)

            x_val = x_train[val_size:]
            partial_x_train = x_train[:val_size]
            y_val = y_train[val_size:]
            partial_y_train = y_train[:val_size]

            history = model.fit(partial_x_train,
                                partial_y_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data=(x_val, y_val),
                                verbose = 0)
                                
            # Save parkinsons model
            model.save(f"models/parkinsons_model_{epochs}_{input_shape}.tf")
        history_dict = history.history
        # print(history_dict.keys())
        acc = history_dict['accuracy']
        val_acc = history_dict['val_accuracy']
        # print(f"Accuracy: {acc}")
        # print(f"Validation Accuracy: {val_acc}")
        plot_accuracy(acc[10:], val_acc[10:], f"parkinsons_accuracy_{epochs}")
        predictions = model.evaluate(x_test, y_test, verbose = 0)
        print(f"[*]\t{predictions[1]*100:.2f}%\t({layer1}, {layer2})\t{optimizer}\t{loss}")
    else:
        print("Loading Model(s)")
        print("Accuracy\tNodes\tOptimizer\tLoss")
        # TODO: Somehow state whether model takes data with reduced dimensions (PCA, LDA)
        model = models.load_model(f"models/parkinsons_model_{epochs}_{input_shape}.tf")
        predictions = model.evaluate(x_test, y_test, verbose = 0)
        print(f"[*]\t{predictions[1]*100:.2f}%\t({layer1}, {layer2})\t{optimizer}\t{loss}")


def prepare_directory():
    """Creates necessary folders in preparation for data/models saved
    """
    folders = ["plots", "models", "settings"]
    for folder in folders:
        if not os.path.isdir(folder):
            os.mkdir(folder)


def main():
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
    # Number of observations
    print(f"Number of rows: {df.shape[0]}")
    # TODO: Add donut chart of this distribution
    print(f"Target class distribution\n{df['status'].value_counts()}")
    plot_class_ratio(df['status'], "class_ratio_all_data.html", "All Data Labels Ratio")   
    #  ------------------- PREPROCESSING -------------------  
    # Bring all data values to the same range
    data = normalize_tabular_data(df)
    # Split data into data and labels
    X, y = data_label_split(data)
    # Drop columns that hold no value regarding prediction
    X = X.drop('name', axis = 1)
    # Dimensionality Reduction
    pca = bool(int(os.getenv("PCA")))
    tsvd = bool(int(os.getenv("SVD")))
    pca_components = int(os.getenv("PCA_COMPONENTS"))
    print("-------------------\nDimensionality Reduction\n-------------------")
    if pca:
        print(f"[*]\tPCA with {pca_components} components")
        pca = PCA(n_components=pca_components, whiten='True')
        pcs = pca.fit(X).transform(X)
        # Store PCA results to DataFrame
        X = pd.DataFrame(data = pcs, 
                        index = range(pcs.shape[0]), 
                        columns = [f"PC{i}" for i in range(pca_components)])
    elif tsvd:
        svd = TruncatedSVD(n_components=5)
        svds = svd.fit(X).transform(X)
        pd.DataFrame(data = svds, 
                        index = range(svds.shape[0]), 
                        columns = [f"PC{i}" for i in range(svds.shape[1])])
    else:
        print("[*]\tNo Method Used")
    # Split data into test and training sets
    x_train, x_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=4)
    
    plot_class_ratio(y_train, "class_ratio_training_raw.html", "Unbalanced Training Data")                                              
    isolation_forest(x_train, x_test, y_train, y_test)
    extra_trees_classifier(x_train, x_test, y_train, y_test)
    smote = bool(int(os.getenv("SMOTE")))

    x_train, y_train = balance_training_data(x_train, y_train)
    logistic_regression(x_train, x_test, y_train, y_test)
    plot_class_ratio(y_train, "class_ratio_training_smote.html", "Balanced Training Data with SMOTE")   
    # linear_regression(x_train, x_test, y_train, y_test)
    # random_forest(x_train, x_test, y_train, y_test)
    # keras_network(x_train, x_test, y_train, y_test, "parkinsons")


if __name__ == "__main__":
    main()