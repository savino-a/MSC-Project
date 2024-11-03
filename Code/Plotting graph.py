import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as pyo
import numpy as np

a = 10
t1 = 100
t2 = 1000
T = 1100


def f(t):
    if t < t1:
        return a * t
    elif t < t2:
        return t1 * a
    elif t < T:
        return t1 * a - (t - t2) * a


time = np.arange(0, 1101)
speed = [f(i) for i in time]
plt.plot(speed)
plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.title("Evolution of the speed of a train during a trip")
plt.show()
import plotly.graph_objs as go
import plotly.offline as pyo


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
