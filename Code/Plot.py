import plotly.graph_objs as go
import plotly.offline as pyo
import numpy as np

import importingtest

importingtest.tester()


def v(vini, x, delta_v):
    a, b = x[0], x[1]
    if a == 0 and b == 0:
        return vini
    elif a == 0 and b == 1:
        return vini + delta_v
    elif a == 1 and b == 0:
        return vini - delta_v
    else:
        return vini


def plot_speed(y, delta_v):
    N = len(y)
    speed = [0]

    for i in range(N // 2):
        speed.append(f(speed[i - 1], y[2 * i : 2 * i + 1], delta_v))
    plt.plot(speed)
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("Evolution of the speed of a train during a trip")
    plt.show()

    # Create a scatter plot
    trace = go.Scatter(
        x=time,
        y=speed,
        mode="lines",
        name="Speed vs Time",
        line=dict(color="royalblue", width=2),
        marker=dict(size=8, symbol="circle", line=dict(width=2, color="darkblue")),
    )

    # Create the layout
    layout = go.Layout(
        title="Speed vs Time",
        titlefont=dict(size=24, family="Arial, sans-serif"),
        xaxis=dict(
            title="Time (s)",
            titlefont=dict(size=18, family="Arial, sans-serif"),
            showgrid=True,
            gridcolor="lightgray",
            zeroline=True,
            zerolinecolor="gray",
            tickfont=dict(size=14),
        ),
        yaxis=dict(
            title="Speed (m/s)",
            titlefont=dict(size=18, family="Arial, sans-serif"),
            showgrid=True,
            gridcolor="lightgray",
            zeroline=True,
            zerolinecolor="gray",
            tickfont=dict(size=14),
        ),
        plot_bgcolor="white",
        paper_bgcolor="rgba(255, 255, 255, 0.9)",
        showlegend=True,
    )

    # Create the figure
    fig = go.Figure(data=[trace], layout=layout)

    # Plot the graph
    pyo.plot(fig, filename="speed_vs_time.html")
