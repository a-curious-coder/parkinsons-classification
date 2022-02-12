import plotly.graph_objs as go
import numpy as np
from plotly.subplots import make_subplots
from sklearn import metrics


def verify_import():
    print("[*]\tvisualisations.py successfully imported")
    pass


def plot_loss(loss, val_loss, file_name):
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
        # yaxis = dict(range = (0, 100)),
        title=dict(text="Training/Validation Loss", x=0.5)
    )
    fig = go.Figure(data=data, layout=layout)
    fig.write_html(f"plots/{file_name}.html")
    return fig


def plot_accuracy(acc, val_acc, file_name):
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
        marker=dict(color="red"),
        hovertemplate="Epoch %{x}<br>Training accuracy: %{y:.2f}%<extra></extra>",
    )
    validation_acc = go.Scatter(
        x=epochs_range,
        y=val_acc,
        mode="markers+lines",
        name="Validation Accuracy",
        marker=dict(color="blue"),
        hovertemplate="Epoch %{x}<br>Validation accuracy: %{y:.2f}%<extra></extra>",
    )
    data = [training_acc, validation_acc]
    layout = go.Layout(
        yaxis=dict(range=(0, 100)),
        title=dict(text="Training/Validation Accuracy", x=0.5),
    )
    fig = go.Figure(data=data, layout=layout)
    fig.write_html(f"plots/{file_name}.html")
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


def plot_class_ratio(counts, filename, title):
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
            title="Class Distribution",
            labels=labels,
            values=values,
            hole=0.3,
            hoverinfo="label+percent",
            textinfo="value",
            textfont_size=20,
            marker=dict(colors=colors, line=dict(color="#000000", width=2)),
        ),
        layout=go.Layout(title=title),
    )
    fig.write_html(f"plots/{filename}.html")
    # fig.show()


def plot_logistic_regression(points):
    fig = go.Figure(
        data=go.Scatter(y=points, mode="markers"),
        layout=go.Layout(
            title="Logistic Regression",
            xaxis=dict(title="Datapoints"),
            yaxis=dict(title="Value", range=[0, 1]),
        ),
    )

    fig.write_html("plots/logistic_regression.html")
    # fig.show()


def plot_feature_importance(feature_names, indices, importances, std):
    hovertexts = []
    # Scale up elements to get better representation of percentage value
    importances = importances * 100
    std = std * 100

    for i in range(len(importances)):
        hovertexts.append(
            "<br>".join(
                [
                    f"Feature {indices[i]} : ({feature_names[indices[i]]})",
                    f"Importance {importances[indices[i]]:.2f}%",
                ]
            )
        )

    fig = go.Figure(
        data=go.Bar(
            y=importances[indices],
            error_y=dict(type="data", array=std[indices]),
            hoverinfo="text",
            hovertext=hovertexts,
        ),
        layout=go.Layout(
            title=dict(text="Feature Importance", x=0.5),
            xaxis=dict(
                title="Feature",
                tickmode="array",
                tickvals=list(range(len(importances))),
                ticktext=indices,
            ),
            yaxis=dict(title="Importance"),
        ),
    )
    fig.write_html("plots/feature_importance.html")
    # fig.show()


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

    fig.write_html(f"plots/removed_columns/{removed_column}_removed.html")
    # fig.show()


def plot_cm(actual, pred, plot_title):
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
        colorscale = 'rdylgn'
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
    # fig.show()
    fig.write_html(f"plots/confusion_matrices/{plot_title}.html")
