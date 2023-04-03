"""
    Script con distintos tipos de plots para la práctica: Diagrama de Voronoi, clasificación en clusters de los puntos,
    comparciones entre nº de clusters (o épsilon) y el coef. de silouette...
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

from funciones import load_data, createKMeans, classify


# init
X, Y = load_data()

# KMEANS
N_clusters_Kmeans = range(2, 16)
Silhouette_Kmeans = [createKMeans(X, n)[2] for n in N_clusters_Kmeans]

# DBSCAN
N = 15 # nº de epsilons para probar 
metrics = ['manhattan', 'euclidean']
epsilons = np.linspace(0.1, 0.4, N)

classification = np.array(                   # [0] -> labels 
    [[classify(X, Y, eps, metric)            # [1] -> n_clusters
                for eps in epsilons]         # [2] -> core_samples_mask
                    for metric in metrics])  # [3] -> silhouette

N_clusters_DBSCAN = classification[:,:,1].tolist()
Silhouette_DBSCAN = classification[:,:,3].tolist()


# ------------------------------ PLOTS ------------------------------


# esto es una copia modificada de la función: from scipy.spatial import voronoi_plot_2d
def vor_plot_2d(vor, ax=None, **kw):
    from matplotlib.collections import LineCollection

    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi diagram is not 2-D")

    if kw.get('show_points', True):
        point_size = kw.get('point_size', None)
        point_size_2 = point_size // 2
        ax.plot(vor.points[:, 0], vor.points[:, 1], 'o', markersize=point_size, markerfacecolor="white", markeredgecolor="black")
        ax.plot(vor.points[:, 0], vor.points[:, 1], 'xk', markersize=point_size_2, label="cluster centers")
    if kw.get('show_vertices', True):
        ax.plot(vor.vertices[:, 0], vor.vertices[:, 1], 'ok')

    line_colors = kw.get('line_colors', 'k')
    line_width = kw.get('line_width', 1.0)
    line_alpha = kw.get('line_alpha', 1.0)

    center = vor.points.mean(axis=0)
    ptp_bound = vor.points.ptp(axis=0)

    finite_segments = []
    infinite_segments = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            finite_segments.append(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            if (vor.furthest_site):
                direction = -direction
            far_point = vor.vertices[i] + direction * ptp_bound.max() * 2

            infinite_segments.append([vor.vertices[i], far_point])


    ax.add_collection(LineCollection(finite_segments,
                                     colors=line_colors,
                                     lw=line_width,
                                     alpha=line_alpha,
                                     linestyle='solid'))
    ax.add_collection(LineCollection(infinite_segments,
                                     colors=line_colors,
                                     lw=line_width,
                                     alpha=line_alpha,
                                     linestyle='solid')) # dashed

    return ax.figure


def plot_points_and_voronoi_diagram(X, kmeans, n_clusters, problems=None, lp=None):
    
    # hacer el diagrama de voronoi
    centers = kmeans.cluster_centers_
    vor = Voronoi(centers)
    fig, ax = plt.subplots(1, 1, figsize=(8,4))
    
    # guardar los labels y generar un color para cada clase
    labels = kmeans.labels_
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
    
    alpha = 1 if problems == None else 0.7
    
    # graficar cada cluster (con su color respectivo)
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # negro : para el ruido
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=5, alpha=alpha)
    
    # para el "enunciado 3" graficamos los puntos (0,0) y (0,-1) un poco más grandes y del color que predecimos con "kmeans.predict"
    if problems != None:
        for p, l in zip(problems, lp): # lp = "label predict"
            i = list(unique_labels).index(l)
            color = colors[i]
            ax.plot(p[0], p[1],'s', markersize=15, markerfacecolor=color, markeredgecolor='k', label=p)
        ax.legend()
    
    # graficar el diagrama de voronoi (las líneas que separan las clases)
    vor_plot_2d(vor, ax=ax, line_width=1.5, point_size=14, show_vertices=False)
    
    ax.legend()
    ax.set_title('KMeans - Diagrama de Voronoi | %d clusters' % n_clusters)
    plt.show()


# es una función muy similar a "plot_points_and_voronoi_diagram()" pero no grafica el diagram de voronoi solo la separación por clusters
def plot_points_KMeans(X, labels, n_clusters, problems=None, lp=None):
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
    
    alpha = 1 if problems == None else 0.7
    
    # plt.figure(figsize=(8,4))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=5, alpha=alpha)
    
    if problems != None:
        for p, l in zip(problems, lp):
            i = list(unique_labels).index(l)
            color = colors[i]
            plt.plot(p[0], p[1],'s', markersize=15, markerfacecolor=color, markeredgecolor='k', label=p)
        plt.legend()
    
    plt.title('Fixed number of KMeans clusters: %d' % n_clusters)
    plt.show()


def plot_points_DBSCAN(X, labels, n_clusters, core_samples_mask, axes=False):
    
    # guardar los labels y generar un color para cada clase
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]

    # axes = True -> indica que este plot lo estamos metiendo en otro gráfico por tanto
    # no hace falta generar un figura ni poner titulos
    if not axes:
        plt.figure(figsize=(8,4))
    
    # graficar cada cluster (con su color respectivo)
    for k, col in zip(unique_labels, colors):
        s1, s2 = 5, 3
        mark = "o"
        alpha = 1
        if k == -1:
            # ruido -> color negro, más pequeños y transparentes, y dibujados con triángulos
            col = (0, 0, 0, 1)
            s1, s2 = 3, 3
            alpha = 0.5
            mark = "^"

        class_member_mask = (labels == k)

        # los puntos que pertenezcan al núcleo del cluster se dibujan algo más grandes
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], mark, markerfacecolor=col,
                markeredgecolor='k', markersize=s1, alpha=alpha)

        # los puntos frontera, más pequeños
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], mark, markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=s2, alpha=alpha)

    if not axes:
        plt.title('Estimated number of DBSCAN clusters: %d' % n_clusters)
        plt.show()


def plot_Epsilon_vs_Silouette():

    N = len(metrics)
    colors = [plt.cm.Spectral(n) for n in np.linspace(0,1,N)]

    # graficar con líneas diferentes para cada métrica : manhatan / euclidean
    for i in range(N):
        i_max = np.argmax(Silhouette_DBSCAN[i])
        plt.plot(epsilons, Silhouette_DBSCAN[i], "-", label=metrics[i], color=colors[i])
        plt.plot(epsilons[i_max], Silhouette_DBSCAN[i][i_max], "ro")

    plt.xlabel("epsilon")
    plt.ylabel("Silhouette coef")
    plt.legend()
    plt.title("Classify : DBSCAN")
    plt.show()


def plot_Clusters_vs_Silouette():

    N = len(metrics)
    colors = [plt.cm.Spectral(n) for n in np.linspace(0,1,N+1)]

    # dbscan
    for i in range(N): # N = 2 metrics
        plt.plot(N_clusters_DBSCAN[i], Silhouette_DBSCAN[i], "o", markeredgecolor="k", label=f"DBSCAN [{metrics[i]}]", color=colors[i])

    # kmeans
    plt.plot(N_clusters_Kmeans, Silhouette_Kmeans, "-o", markerfacecolor=colors[-1], markeredgecolor="k", label="KMeans")

    xticks = list(set(N_clusters_DBSCAN[0] + N_clusters_DBSCAN[1] + list(N_clusters_Kmeans)))
    plt.xticks(xticks)
    plt.xlabel("n clusters")
    plt.ylabel("Silhouette coef")
    plt.legend()
    plt.title("Classify : KMEANS & DBSCAN")
    plt.show()


# función extra para hacer un estudio más amplio del algoritmo DBSCAN con diferente epsilons y n0
# graficaremos
#     2 filas : una para cada métrica
#     2 columnas : a la izq. Epsilon vs Silouette y a la der. la distribución en clusters del mejor resultado. 
def plot_Epsilon_vs_Silouette_maspruebas():

    # intervalos a estudiar
    epsilons = np.linspace(0.05, 0.6, 50)
    n0s = np.linspace(10,50,5, dtype=np.int8)
    colors = [[plt.cm.Greens(n) for n in np.linspace(0,1,len(n0s))], [plt.cm.Reds(n) for n in np.linspace(0,1,len(n0s))]]
    
    # clasificación de todos los resultados
    _silhouette = np.array(                           
        [[[classify(X, Y, eps, metric, n0)[3]            # [0] -> labels 
                    for eps in epsilons]                 # [1] -> n_clusters
                        for n0 in n0s]                   # [2] -> core_samples_mask
                            for metric in metrics])      # [3] -> silhouette
    
    for i in range(len(metrics)):

        # subplot : Epsilon vs Silouette
        plt.subplot(2,2,2*i+1)
        for j in range(len(n0s)):
            plt.plot(epsilons, _silhouette[i,j,:], "-", label=r"$n_0 = $" + str(n0s[j]), color=colors[i][j])

        # indicar el mejor resultado

        i_max = np.argmax(_silhouette[i].reshape(-1))
        eps = epsilons[i_max % len(epsilons)]
        n0 = n0s[i_max // len(epsilons)]
        plt.plot(eps, _silhouette[i].reshape(-1)[i_max], "ro", label="max")

        plt.title(r"Algoritmo DBSACN : Estudio de $\varepsilon$" + f" [{metrics[i]}]")
        plt.ylabel("Silhouette coef")
        plt.legend(loc="upper left")

        # subplot : mejor cluster
        plt.subplot(2,2,2*i+2)
        _labels, _n_clusters, _core_samples_mask, _ = classify(X, Y, eps, metrics[i], n0=n0)
        plt.title(f"[{metrics[i]}] " + r"$ n_0 $ = " + str(n0) + r", $ \varepsilon $ = " + "{:.2f}, n_clusters = ".format(eps) + str(_n_clusters))
        plot_points_DBSCAN(X, _labels, _n_clusters, _core_samples_mask, True)

    plt.show()

