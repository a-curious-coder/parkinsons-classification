import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import exists
from seaborn import pairplot
from plotly.subplots import make_subplots
from sklearn import metrics
import pandas as pd


def verify_import():
    print("[*]\tvisualisations.py successfully imported")
    pass


def plot_loss(loss, val_loss, file_name, title="Training/Validation Loss"):

    epochs_range = list(range(1, len(loss) + 1))

    loss = [i * 100 for i in loss]
    val_loss = [i * 100 for i in val_loss]

    training_loss = go.Scatter(
        x=epochs_range,
        y=loss,
        mode="markers",
        name="Training Loss",
        marker=dict(color="red"),
        hovertemplate="Epoch %{x}<br>Training Loss: %{y:.2f}%<extra></extra>",
    )

    validation_loss = go.Scatter(
        x=epochs_range,
        y=val_loss,
        mode="markers+lines",
        name="Validation Loss",
        marker=dict(color="blue"),
        hovertemplate="Epoch %{x}<br>Validation Loss: %{y:.2f}%<extra></extra>",
    )

    data = [training_loss, validation_loss]
    layout = go.Layout(
        autosize=False,
        width=1920,
        height=1080,
        xaxis=dict(title="Epochs", linecolor="lightgrey", showgrid=False),
        yaxis=dict(
            range=(0, 100), title="Accuracy (%)", linecolor="lightgrey", showgrid=False
        ),
        title=dict(text=title, x=0.5),
        paper_bgcolor="rgba(255,255,255,255)",
        plot_bgcolor="rgba(255,255,255,255)",
        font=dict(
            color="black",
            size=28,  # can change the size of font here
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    # fig.write_html(f"plots/{file_name}.html")
    fig.write_image(f"plots/nn_training_stats/{file_name}.png")

    return fig


def plot_accuracy(acc, val_acc, file_name, title="Training/Validation Accuracy"):
    """Plots accuracy statistics for neural net models

    Args:
        acc ([type]): [description]
        val_acc ([type]): [description]

    Returns:
        [type]: [description]
    """
    epochs_range = list(range(1, len(acc) + 1))

    acc = [i * 100 for i in acc]
    val_acc = [i * 100 for i in val_acc]

    training_acc = go.Scatter(
        x=epochs_range,
        y=acc,
        mode="markers",
        name="Training Accuracy",
        marker=dict(size=4, color="red", line=dict(width=0.5)),
        hovertemplate="Epoch %{x}<br>Training accuracy: %{y:.2f}%<extra></extra>",
    )
    validation_acc = go.Scatter(
        x=epochs_range,
        y=val_acc,
        mode="markers+lines",
        name="Validation Accuracy",
        marker=dict(size=4, color="blue", line=dict(width=0.5)),
        hovertemplate="Epoch %{x}<br>Validation accuracy: %{y:.2f}%<extra></extra>",
    )
    data = [training_acc, validation_acc]
    layout = go.Layout(
        autosize=False,
        width=1920,
        height=1080,
        xaxis=dict(title="Epochs", linecolor="lightgrey", showgrid=False),
        yaxis=dict(
            range=(0, 100), title="Accuracy (%)", linecolor="lightgrey", showgrid=False
        ),
        title=dict(text=title, x=0.5),
        paper_bgcolor="rgba(255,255,255,255)",
        plot_bgcolor="rgba(255,255,255,255)",
        font=dict(
            color="black",
            size=28,  # can change the size of font here
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    # fig.write_html(f"plots/{file_name}.html")
    fig.write_image(f"plots/nn_training_stats/{file_name}.png")
    return fig


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def plot_all_graphs(loss, val_loss, acc, val_acc):
    """Plots the neural network model's accuracy and loss on same figure
    Args:
        loss ([type]): [description]
        val_loss ([type]): [description]
        acc ([type]): [description]
        val_acc ([type]): [description]
    """
    fig = make_subplots(rows=1, cols=2)

    # Get loss and accuracy statistics
    fig1 = plot_loss(loss, val_loss)
    fig2 = plot_accuracy(acc, val_acc)

    fig.append_trace(fig1, row=1, col=1)
    fig.append_trace(fig2, row=1, col=2)

    fig.update_layout(
        height=600, width=1000, title=dict(text="Deep Learning Model Statistics", x=0.5)
    )


def plot_all_graphs():
    files = [
        "balanced_accuracy",
        "balanced_accuracy_smooth",
        "balanced_loss",
        "balanced_loss_smooth",
        "imbalanced_accuracy",
        "imbalanced_accuracy_smooth",
        "imbalanced_loss",
        "imbalanced_loss_smooth",
    ]
    # Checks that all files exist
    for file in files:
        if not exists(f"plots/nn_acc_loss/{file}.csv"):
            return

    plot_titles = [
        "[Balanced] Training | Validation Accuracy",
        "[Balanced] Smooth Training | Validation Accuracy",
        "[Balanced] Training | Validation Loss",
        "[Balanced] Smooth Training | Validation Loss",
        "[Imbalanced] Training | Validation Accuracy",
        "[Imbalanced] Smooth Training | Validation Accuracy",
        "[Imbalanced] Training | Validation Loss",
        "[Imbalanced] Smooth Training | Validation Loss",
    ]

    for i, file_name in enumerate(files):
        stats = pd.read_csv(f"plots/nn_acc_loss/{file_name}.csv")
        if "Accuracy" in plot_titles[i]:
            plot_accuracy(
                stats["acc"].tolist(),
                stats["val_acc"].tolist(),
                file_name,
                title=plot_titles[i],
            )
        else:
            plot_loss(
                stats["loss"].tolist(),
                stats["val_loss"].tolist(),
                file_name,
                title=plot_titles[i],
            )


def plot_label_distribution(counts, file_name, title):
    df = counts.to_frame()
    # Format labels
    labels = df["status"].value_counts().index.tolist()
    labels[0] = "Healthy" if labels[0] == 0 else "Parkinsons"
    labels[1] = "Healthy" if labels[1] == 0 else "Parkinsons"

    # Gets value counts
    values = [df["status"].value_counts()[i] for i in range(df.shape[1] + 1)]

    colors = ["red", "mediumturquoise"]
    fig = go.Figure(
        data=go.Pie(
            title="Label Distribution",
            labels=labels,
            values=values,
            hole=0.5,
            hoverinfo="label+percent",
            textinfo="value",
            textfont_size=20,
            marker=dict(colors=colors, line=dict(color="grey", width=1)),
        ),
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        ),
    )
    # fig.write_html(f"plots/{file_name}.html")
    fig.write_image(f"plots/{file_name}.png")


def plot_logistic_regression(points):
    fig = go.Figure(
        data=go.Scatter(y=points, mode="markers"),
        layout=go.Layout(
            title="Logistic Regression",
            xaxis=dict(title="Datapoints"),
            yaxis=dict(title="Value", range=[0, 1]),
        ),
    )
    # fig.write_html("plots/logistic_regression.html")


def plot_pairplot(data, file_name="pairplot.png", title="Pairplot"):
    if not plot_exists(file_name):
        plt.figure(figsize=(10, 12))
        # Sorts labels so colours are consistent for each label every time it plots
        data = data.sort_values(by="status")
        # Replaces labels with meaning
        data["status"].replace({0: "Alzheimer's", 1: "Healthy"}, inplace=True)
        # Pairplot with colour representing labels
        sns_plot = sns.pairplot(data, hue="status")
        plt.suptitle(title)
        sns_plot.figure.savefig(f"plots/{file_name}")


def plot_correlation_matrix(df: pd.DataFrame, file_name="correlation_matrix.png"):
    """
    A function to calculate and plot
    correlation matrix of a DataFrame.
    """
    plt.rcParams.update({"font.size": 16})
    df = df.iloc[:, :5]
    # Create the matrix
    matrix = df.corr()
    # Create cmap
    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)

    # Create a mask
    mask = np.triu(np.ones_like(matrix, dtype=bool))

    # Make figsize bigger
    fig, ax = plt.subplots(figsize=(16, 16))

    # Plot the matrix
    _ = sns.heatmap(
        matrix,
        mask=mask,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        cmap=cmap,
        ax=ax,
    )
    # Plot the matrix
    sns.heatmap(
        matrix,
        mask=mask,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        cmap=cmap,
        ax=ax,
    )
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    _.figure.savefig(f"plots/{file_name}", transparent=True)


def plot_cv_box_plot(cv_models, all_scores, file_name, plot_title):
    # fig = go.Figure()
    # print(all_scores)
    # i = 0
    # for scores in all_scores:
    #     i += 1
    #     fig.add_trace(go.Box(y=scores, quartilemethod="linear", name = f"CV{i}"))
    # fig.update_traces(boxpoints="all", jitter=0)
    # fig.update_layout()
    # fig.show()

    # Colours
    c = [
        "hsl(" + str(h) + ",50%" + ",50%)" for h in np.linspace(0, 360, len(all_scores))
    ]

    fig = go.Figure(
        data=[
            go.Box(y=all_scores[i] * 100, marker_color=c[i], name=f"{i+2}")
            for i in range(len(all_scores))
        ]
    )

    # format the layout
    # fig.update_layout(
    #     title=f"Repeated K-Fold Cross Validation Range[{min(cv_models)}, {max(cv_models)}]",
    #     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    #     yaxis=dict(zeroline=True),
    #     paper_bgcolor="rgb(233,233,233)",
    #     plot_bgcolor="rgb(233,233,233)",
    # )

    fig.update_layout(
        autosize=False,
        width=1920,
        height=1080,
        xaxis=dict(
            title="Cross Validation Folds", linecolor="lightgrey", showgrid=False
        ),
        yaxis=dict(title="Accuracy (%)", linecolor="lightgrey", showgrid=False),
        title=dict(
            text=plot_title,
            x=0.5,
        ),
        paper_bgcolor="rgba(255,255,255,255)",
        plot_bgcolor="rgba(255,255,255,255)",
        font=dict(
            color="black",
            size=24,  # can change the size of font here
        ),
        legend=dict(
            x=1,
            y=1,
            traceorder="normal",
            font=dict(family="sans-serif", size=18, color="#000"),
            bgcolor="#E2E2E2",
            bordercolor="#FFFFFF",
            borderwidth=2,
        ),
        annotations=[
            dict(
                x=1.02,
                y=1.05,
                xref="paper",
                yref="paper",
                text="K",
                showarrow=False,
            )
        ],
    )

    fig.write_image(f"plots/{file_name}")
    # if not plot_exists(file_name):
    #     plt.figure(figsize=(24, 10))
    #     sns.set_style("whitegrid", {"axes.grid": False})
    #     sns.boxplot(data=all_scores).set_title(plot_title)
    #     plt.xlabel("Cross Validation #")
    #     plt.ylabel("Accuracy")
    #     plt.savefig(f"plots/{file_name}")


def plot_feature_importance(feature_names, indices, importances, std):
    """Plots feature importance

    Args:
        feature_names (list): column names of numerical values
        indices (_type_): _description_
        importances (_type_): _description_
        std (_type_): _description_
    """
    file_name = "feature_importance"
    if not plot_exists(f"{file_name}.png"):
        hovertexts = []
        # Scale up elements to get better representation of percentage value
        importances = importances * 100
        std = std * 100
        reorder_feature_names = []
        for i in range(len(importances)):
            hovertexts.append(
                "<br>".join(
                    [
                        f"Feature {indices[i]} : ({feature_names[indices[i]]})",
                        f"Importance {importances[indices[i]]:.2f}%",
                    ]
                )
            )
            reorder_feature_names.append(feature_names[indices[i]])

        fig = go.Figure(
            data=go.Bar(
                y=importances[indices],
                error_y=dict(type="data", array=std[indices]),
                hoverinfo="text",
                hovertext=hovertexts,
                marker=dict(color="#006400"),
            ),
            layout=go.Layout(
                # title=dict(text="Feature Importance", x=0.5),
                xaxis=dict(
                    title="Feature",
                    tickmode="array",
                    tickvals=list(range(len(importances))),
                    ticktext=reorder_feature_names,
                    tickangle=45,
                    linecolor="lightgrey",
                    showgrid=False,
                ),
                yaxis=dict(
                    title="Importance (%)", linecolor="lightgrey", showgrid=False
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            ),
        )
        # fig.write_html(f"plots/{file_name}.html")
        fig.write_image(f"plots/{file_name}.png")


def plot_removed_features(a_predictions, removed_column="Unknown"):
    alldata_acc = a_predictions["alldata"]
    # Extract accuracy for all data
    print(alldata_acc)
    # Deletes alldata entry
    del a_predictions["alldata"]
    features = list(a_predictions.keys())
    predictions = list(a_predictions.values())
    explode = np.zeros(len(predictions))
    # Explodes out slice with highest accuracy value
    explode[predictions.index(max(predictions))] = 0.2
    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(a_predictions.keys()),
                values=predictions,
                hole=0.3,
                pull=explode,
                hovertemplate="<br>".join(
                    [
                        "Classifer run with '%{label}' column <b>Removed</b>",
                        "<b>Predicted: %{value}% Accuracy over average of 20 trials</b>",
                        "<extra></extra>",
                    ]
                ),
                textinfo="value",
            )
        ],
        layout=go.Layout(
            title=dict(
                text=f"Highest accuracy when {features[predictions.index(max(predictions))]} is removed of {predictions.index(max(predictions))} %",
                x=0.5,
            )
        ),
    )

    fig.add_annotation(
        font=dict(color="black"),
        text=f"Classification Accuracy for All Data: {alldata_acc}%",
        xref="paper",
        yref="paper",
        x=0.1,
        y=0.1,
        showarrow=False,
    )
    # fig.write_html(f"plots/removed_columns/{removed_column}_removed.html")


def plot_cm(actual, pred, plot_title, file_name):
    labels = ["Parkinsons", "Healthy"]
    cm = metrics.confusion_matrix(actual, pred)

    data = go.Heatmap(
        z=cm,
        y=labels,
        x=labels,
        hovertemplate="<br>".join(
            [
                "Predicted: <b>%{x}</b>",
                "Actual: <b>%{y}</b>",
                "Occurences %{z}",
                "<extra></extra>",
            ]
        ),
        colorscale="rdylgn",
    )
    annotations = []
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            annotations.append(
                {
                    "x": labels[j],
                    "y": labels[i],
                    "text": str(value),
                    "font": {"color": "black", "size": 20},
                    "xref": "x1",
                    "yref": "y1",
                    "showarrow": False,
                }
            )
    layout = go.Layout(
        title=dict(text=plot_title, x=0.5),
        xaxis=dict(title="Predicted"),
        yaxis=dict(title="Actual"),
        annotations=annotations,
    )
    fig = go.Figure(data=data, layout=layout)
    # fig.write_html(f"plots/confusion_matrices/{file_name}.html")
    fig.write_image(f"plots/confusion_matrices/{file_name}.png")


def plot_exists(file_name):
    if exists(f"plots/{file_name}"):
        return True
    return False
