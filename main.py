import visualisations as v

import os
import pandas as pd
import random
import tensorflow as tf
import math

from dotenv import load_dotenv
from sklearn import linear_model
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras import layers
from os.path import exists


def normalize_tabular_data(data):
    """Normalizes the values of each column

    Args:
        data (pd.DataFrame): Parkinsons Data

    Returns:
        pd.DataFrame: Normalized Parkinsons data
    """
    avoid = ['status']
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


def random_forest(x_train, x_test, y_train, y_test):
    print("-------------------\nRandom Forest\n-------------------")
    rf = RandomForestClassifier(n_estimators=50)
    rf.fit(x_train, y_train)
    pred = rf.predict(x_test)
    labels = y_test.values
    count = 0
    for i in range(len(pred)):
        if pred[i] == labels[i]:
            count = count + 1
    print(f"[*]\t{count / float(len(pred))*100:.2f}%\tRandom Forest PCA")


def keras_network(x_train, x_test, y_train, y_test):
    # Load in epochs from environment variable file
    epochs = int(os.getenv("EPOCHS"))
    print("-------------------\nNeural Network\n-------------------")
    settings = [['rmsprop', 'binary_crossentropy', 'accuracy'], ['adam', 'binary_crossentropy', 'accuracy']]
    i = 16
    j = 8
    rewrite_model = bool(int(os.getenv("REWRITE_MODEL")))
    if not exists(f"models/mri_model_{epochs}_{settings[0][0]}.tf") or not exists(f"models/mri_model_{epochs}_{settings[1][0]}.tf") or rewrite_model:
        print("Training/Testing Model")
        print("Accuracy\tNodes\tOptimizer\tLoss")
        for setting in settings:
            # Validation set size
            val_size = math.floor((x_train.shape[0] / 10) * 9)
            with tf.device('/cpu:0'):
                # Prepare network
                network = models.Sequential()
                network.add(layers.Dense(i, activation='relu', input_shape=(x_train.shape[1], )))
                network.add(layers.Dense(j, activation='relu'))
                network.add(layers.Dense(1, activation='sigmoid'))

                network.compile(optimizer=setting[0],
                                loss=setting[1],
                                metrics=[setting[2]])

                x_val = x_train[:val_size]
                partial_x_train = x_train[val_size:]
                y_val = y_train[:val_size]
                partial_y_train = y_train[val_size:]

                history = network.fit(partial_x_train,
                                    partial_y_train,
                                    epochs=epochs,
                                    batch_size=16,
                                    validation_data=(x_val, y_val),
                                    verbose = 0)
                # Save mri network
                network.save(f"models/mri_model_{epochs}_{setting[0]}.tf")
            history_dict = history.history
            # print(history_dict.keys())
            acc = history_dict['accuracy']
            val_acc = history_dict['val_accuracy']
            # print(f"Accuracy: {acc}")
            # print(f"Validation Accuracy: {val_acc}")
            v.plot_accuracy(acc[10:], val_acc[10:], f"mri_accuracy_{epochs}_{i}_{j}_{setting[0]}")
            predictions = network.evaluate(x_test, y_test, verbose = 0)
            print(f"[*]\t{predictions[1]*100:.2f}%\t({i}, {j})\t{setting[0]}\t{setting[1]}")
    else:
        print("Loading Model(s)")
        print("Accuracy\tNodes\tOptimizer\tLoss")
        for setting in settings:
            network = models.load_model(f"models/mri_model_{epochs}_{setting[0]}.tf")
            predictions = network.evaluate(x_test, y_test, verbose = 0)
            print(f"[*]\t{predictions[1]*100:.2f}%\t({i}, {j})\t{setting[0]}\t{setting[1]}")


def prepare_directory():
    """Creates necessary folders in preparation for data/models saved
    """
    if not os.path.isdir("plots"):
        os.mkdir("plots")
    if not os.path.isdir("models"):
        os.mkdir("models")


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Load in environment variables from .env
    load_dotenv()
    # Prepare directories for 
    prepare_directory()

    # Load Dataset
    print("-------------------\nLoad Data\n-------------------")
    df = pd.read_csv("data/parkinsons.data")
    # df = df.drop('name', axis = 1)
    # EDA
    print("-------------------\nEDA\n-------------------")
    X, y = data_label_split(df)
    # Number of names
    print(f"Number of names: {len(X['name'].unique())}")
    # Number of observations
    print(f"Number of rows: {X.shape[0]}")
    X = X.drop('name', axis = 1)

    # Dimensionality Reduction
    pca = True
    tsvd = False
    if pca or tsvd:
        print("-------------------\nDimensionality Reduction\n-------------------")
        if pca:
            pca = PCA(n_components=5, whiten='True')
            X = pca.fit(X).transform(X)
        elif tsvd:
            svd = TruncatedSVD(n_components=5)
            X = svd.fit(X).transform(X)

    
    # Split data into test and training sets
    x_train, x_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.4,
                                                        random_state=4)
                                                        
    
    linear_regression(x_train, x_test, y_train, y_test)
    random_forest(x_train, x_test, y_train, y_test)
    keras_network(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()