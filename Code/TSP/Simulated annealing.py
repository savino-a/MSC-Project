import numpy as np
import numpy
import matplotlib.pyplot as plt
import os
from random import random
from random import randint

"""import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.express as px"""
import pandas as pd


def representer(p, N, t):
    plt.figure()
    temp1, temp2 = 0, 0
    rep1, rep2 = [], []
    for i in range(N):
        (temp1, temp2) = p[i]
        rep1.append(temp1)
        rep2.append(temp2)
    plt.plot(rep2, rep1, marker="*", markersize=t / 80, color="r", linestyle="none")


def representer2(p, N, t):
    data = {
        "City": [
            "Los Angeles",
            "Houston",
            "Miami",
            "New York",
            "Chicago",
            "Phoenix",
            "Philadelphia",
            "San Antonio",
            "San Diego",
            "Dallas",
            "San Jose",
            "Austin",
            "Jacksonville",
            "Fort Worth",
            "Columbus",
            "San Francisco",
            "Charlotte",
            "Indianapolis",
            "Seattle",
            "Denver",
            "Washington D.C.",
            "Boston",
            "El Paso",
            "Nashville",
            "Detroit",
            "Oklahoma City",
        ],
        "Latitude": [
            34.0522,
            29.7604,
            25.7617,
            40.7128,
            41.8781,
            33.4484,
            39.9526,
            29.4241,
            32.7157,
            32.7767,
            37.3382,
            30.2672,
            30.3322,
            32.7555,
            39.9612,
            37.7749,
            35.2271,
            39.7684,
            47.6062,
            39.7392,
            38.9072,
            42.3601,
            31.7619,
            36.1627,
            42.3314,
            35.4676,
        ],
        "Longitude": [
            -118.2437,
            -95.3698,
            -80.1918,
            -74.0060,
            -87.6298,
            -112.0740,
            -75.1652,
            -98.4936,
            -117.1611,
            -96.7969,
            -121.8863,
            -97.7431,
            -81.6557,
            -97.3308,
            -82.9988,
            -122.4194,
            -80.8431,
            -86.1581,
            -122.3321,
            -104.9903,
            -77.0369,
            -71.0589,
            -106.4850,
            -86.7816,
            -83.0458,
            -97.5164,
        ],
    }

    # Convert to a DataFrame
    df = pd.DataFrame(data)

    # Create a scatter map
    fig = px.scatter_geo(
        df,
        lat="Latitude",
        lon="Longitude",
        text="City",  # City names as hover text
        title="Cities in the US",
        scope="usa",  # Limit map to the USA
    )

    # Update the trace to make the markers and text larger
    fig.update_traces(
        marker=dict(size=16),  # Adjust marker size
        textfont=dict(size=26),  # Adjust text size
    )

    # Show the figure
    fig.show()


def circ_ini(N, p):
    l = [0]
    dist = 0
    for i in range(1, N):
        l.append(i)
        dist += distance(p, i - 1, i)
    dist += distance(p, 0, N - 1)
    return l, dist


def poser_les_villes(N, t):
    p = []
    for i in range(N):
        x = t * random()
        y = t * random()
        p.append((x, y))
    return p


def poser_les_villes(N, t):
    data = {
        "City": [
            "Los Angeles",
            "Houston",
            "Miami",
            "New York",
            "Chicago",
            "Phoenix",
            "Philadelphia",
            "San Antonio",
            "San Diego",
            "Dallas",
            "San Jose",
            "Austin",
            "Jacksonville",
            "Fort Worth",
            "Columbus",
            "San Francisco",
            "Charlotte",
            "Indianapolis",
            "Seattle",
            "Denver",
            "Washington D.C.",
            "Boston",
            "El Paso",
            "Nashville",
            "Detroit",
            "Oklahoma City",
        ],
        "Latitude": [
            34.0522,
            29.7604,
            25.7617,
            40.7128,
            41.8781,
            33.4484,
            39.9526,
            29.4241,
            32.7157,
            32.7767,
            37.3382,
            30.2672,
            30.3322,
            32.7555,
            39.9612,
            37.7749,
            35.2271,
            39.7684,
            47.6062,
            39.7392,
            38.9072,
            42.3601,
            31.7619,
            36.1627,
            42.3314,
            35.4676,
        ],
        "Longitude": [
            -118.2437,
            -95.3698,
            -80.1918,
            -74.0060,
            -87.6298,
            -112.0740,
            -75.1652,
            -98.4936,
            -117.1611,
            -96.7969,
            -121.8863,
            -97.7431,
            -81.6557,
            -97.3308,
            -82.9988,
            -122.4194,
            -80.8431,
            -86.1581,
            -122.3321,
            -104.9903,
            -77.0369,
            -71.0589,
            -106.4850,
            -86.7816,
            -83.0458,
            -97.5164,
        ],
    }
    # Convert to a DataFrame
    df = pd.DataFrame(data)

    # Create a list of tuples (latitude, longitude)
    p = list(zip(df["Latitude"], df["Longitude"]))
    return p


def distance(p, i, j):
    (x1, y1) = p[i]
    (x2, y2) = p[j]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1 / 2)


def voisin2(circ, i, j):
    circ[i], circ[j] = circ[j], circ[i]
    return circ


def matrice_distance(p):
    a = []
    for i in range(len(p)):
        ltemp = []
        for j in range(len(p)):
            ltemp.append(distance(p, i, j))
        a.append(ltemp)
    return a


def voisin(circ, i):
    if i == len(circ) - 1:
        circ[i], circ[0] = circ[0], circ[i]
    else:
        circ[i], circ[i + 1] = circ[i + 1], circ[i]
    return circ


def diff(circ, i, dist):
    if i == len(circ) - 1:
        s1 = (
            dist[circ[i - 2]][circ[i - 1]]
            + dist[circ[i - 1]][circ[i]]
            + dist[circ[i]][circ[0]]
        )
        s2 = (
            dist[circ[i - 2]][circ[i]]
            + dist[circ[i - 1]][circ[i]]
            + dist[circ[i - 1]][circ[0]]
        )
    elif i == 1:
        s1 = (
            dist[circ[len(circ) - 1]][circ[i - 1]]
            + dist[circ[i - 1]][circ[i]]
            + dist[circ[i]][circ[2]]
        )
        s2 = (
            dist[circ[len(circ) - 1]][circ[i]]
            + dist[circ[i - 1]][circ[i]]
            + dist[circ[i - 1]][circ[2]]
        )
    elif i == 0:
        s1 = (
            dist[circ[len(circ) - 2]][circ[len(circ) - 1]]
            + dist[circ[len(circ) - 1]][circ[i]]
            + dist[circ[i]][circ[1]]
        )
        s2 = (
            dist[circ[len(circ) - 2]][circ[i]]
            + dist[circ[len(circ) - 1]][circ[i]]
            + dist[circ[len(circ) - 1]][circ[1]]
        )
    else:
        s1 = (
            dist[circ[i - 2]][circ[i - 1]]
            + dist[circ[i - 1]][circ[i]]
            + dist[circ[i]][circ[i + 1]]
        )
        s2 = (
            dist[circ[i - 2]][circ[i]]
            + dist[circ[i - 1]][circ[i]]
            + dist[circ[i - 1]][circ[i + 1]]
        )
    return s1 - s2


def diff2(circ, i, j, dist):
    N = len(circ)
    if i == j:
        return 0
    elif i == j + 1 or (j == N - 1 and i == 0):
        return diff(circ, i, dist)
    elif i == j - 1 or (j == 0 and i == N - 1):
        return diff(circ, j, dist)
    elif i == 0:
        s1 = (
            dist[circ[N - 1]][circ[i]]
            + dist[circ[i]][circ[i + 1]]
            + dist[circ[j - 1]][circ[j]]
            + dist[circ[j]][circ[j + 1]]
        )
        s2 = (
            dist[circ[N - 1]][circ[j]]
            + dist[circ[j]][circ[i + 1]]
            + dist[circ[j - 1]][circ[i]]
            + dist[circ[i]][circ[j + 1]]
        )
    elif i == N - 1:
        s1 = (
            dist[circ[i - 1]][circ[i]]
            + dist[circ[i]][circ[0]]
            + dist[circ[j - 1]][circ[j]]
            + dist[circ[j]][circ[j + 1]]
        )
        s2 = (
            dist[circ[i - 1]][circ[j]]
            + dist[circ[j]][circ[0]]
            + dist[circ[j - 1]][circ[i]]
            + dist[circ[i]][circ[j + 1]]
        )
    elif j == 0:
        s1 = (
            dist[circ[i - 1]][circ[i]]
            + dist[circ[i]][circ[i + 1]]
            + dist[circ[N - 1]][circ[j]]
            + dist[circ[j]][circ[j + 1]]
        )
        s2 = (
            dist[circ[i - 1]][circ[j]]
            + dist[circ[j]][circ[i + 1]]
            + dist[circ[N - 1]][circ[i]]
            + dist[circ[i]][circ[j + 1]]
        )
    elif j == N - 1:
        s1 = (
            dist[circ[i - 1]][circ[i]]
            + dist[circ[i]][circ[i + 1]]
            + dist[circ[j - 1]][circ[j]]
            + dist[circ[j]][circ[0]]
        )
        s2 = (
            dist[circ[i - 1]][circ[j]]
            + dist[circ[j]][circ[i + 1]]
            + dist[circ[j - 1]][circ[i]]
            + dist[circ[i]][circ[0]]
        )
    else:
        s1 = (
            dist[circ[i - 1]][circ[i]]
            + dist[circ[i]][circ[i + 1]]
            + dist[circ[j - 1]][circ[j]]
            + dist[circ[j]][circ[j + 1]]
        )
        s2 = (
            dist[circ[i - 1]][circ[j]]
            + dist[circ[j]][circ[i + 1]]
            + dist[circ[j - 1]][circ[i]]
            + dist[circ[i]][circ[j + 1]]
        )
    return s1 - s2


def probach(T, d):
    return np.exp(d / T)


def evo_T(T, compteur, N):
    if compteur % (10 * N) == 0:
        return 0.999 * T
    else:
        return T


def T_init(N, t):
    if (1000 * t / N) < 100 * T_fin(N):
        return 1000
    else:
        return 1000 * t / N


def T_fin(N):
    return 0.01


def tri_brute(p, N, t):
    mat_dist = matrice_distance(p)
    circ, d = circ_ini(N, p)
    s = 0
    for i in range(0, N - 1):
        x = circ[i]
        y = circ[i + 1]
        v = i + 1
        min = mat_dist[x][y]
        for j in range(i + 1, N):
            if mat_dist[x][circ[j]] < min:
                v = j
                min = mat_dist[x][circ[j]]
        s += min
        circ = voisin2(circ, i + 1, v)
    s += mat_dist[circ[0]][circ[N - 1]]
    return (circ, s)


def copy(circ):
    N = len(circ)
    p = []
    for i in range(N):
        p.append(circ[i])
    return p


def calcul(circ, mat_dist):
    N = len(circ)
    d = 0
    for i in range(N - 1):
        d = d + mat_dist[circ[i]][circ[i + 1]]
    d = d + mat_dist[circ[0]][circ[N - 1]]
    return d


def recuit(N, t):
    compteur = 0
    p = poser_les_villes(N, t)
    mat_dist = matrice_distance(p)
    representer(p, N, t)
    circ, d = circ_ini(N, p)
    T = T_init(N, t)
    temp3 = 0
    i1, i2 = 0, 0
    Tf = T_fin(N)
    x1, y1, x2, y2 = 0, 0, 0, 0
    circbr, dbr = tri_brute(p, N, t)
    circmin, dmin = copy(circ), d
    print(d)
    while T > Tf:
        i = randint(0, N - 1)
        j = randint(0, N - 1)
        dif = diff2(circ, i, j, mat_dist)
        proba = probach(T, dif)
        p1 = random()
        if dif >= 0:
            d = d - dif
            circ = voisin2(circ, i, j)
            if d < dmin:
                dmin = d
                circmin = copy(circ)
        else:
            if p1 <= proba:
                circ = voisin2(circ, i, j)
                d = d - dif
        compteur += 1
        T = evo_T(T, compteur, N)

    for i in range(1, N):
        i2 = circbr[i]
        i1 = circbr[i - 1]
        x1, y1 = p[i1]
        x2, y2 = p[i2]
        plt.arrow(
            y1, x1, y2 - y1, x2 - x1, head_width=t / 800, width=t / 5000, color="g"
        )
    i2 = circbr[0]
    i1 = circbr[N - 1]
    x1, y1 = p[i1]
    x2, y2 = p[i2]
    plt.arrow(y1, x1, y2 - y1, x2 - x1, head_width=t / 800, width=t / 5000, color="g")
    for i in range(1, N):
        i2 = circmin[i]
        i1 = circmin[i - 1]
        x1, y1 = p[i1]
        x2, y2 = p[i2]
        plt.arrow(
            y1, x1, y2 - y1, x2 - x1, head_width=t / 800, width=t / 5000, color="b"
        )
    i2 = circmin[0]
    i1 = circmin[N - 1]
    x1, y1 = p[i1]
    x2, y2 = p[i2]
    plt.arrow(y1, x1, y2 - y1, x2 - x1, head_width=t / 800, width=t / 5000, color="b")
    plt.show()
    return (circ, dmin, dbr)


res = recuit(20, 1000)
print(res)
