import plotly.graph_objs as go
from plotly.subplots import make_subplots


def verify_import():
    print("[*]\tvisualisations.py successfully imported")
    pass


def plot_loss(loss, val_loss):
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
    fig.write_html("plots/mri_loss.html")
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
            marker=dict(colors=colors, line=dict(color="#000000", width=2)),),
        layout = go.Layout(
            title = title
        )
    )
    fig.write_html(f"plots/{filename}.html")
    # fig.show()


def plot_logistic_regression(points):
    fig = go.Figure(
        data=go.Scatter(y=points, mode="markers"),
        layout = go.Layout(
            title = "Logistic Regression",
            xaxis = dict(title = "Datapoints"),
            yaxis = dict(title = "Value", range = [0, 1])
        )
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
                    f"Importance {importances[indices[i]]:.2f}%"
                ]
            )
        )

    fig = go.Figure(
        data=go.Bar(
            y=importances[indices],
            error_y=dict(type="data", array=std[indices]),
            hoverinfo = 'text',
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
