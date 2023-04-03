import matplotlib.pyplot as plt
import numpy as np 

from funciones import createKMeans, predict
from plots import *

        
def pregunta1():

    # plot all the results
    i_max = np.argmax(Silhouette_Kmeans)
    plt.plot(N_clusters_Kmeans, Silhouette_Kmeans, "k-", label="Silhouette Coef")
    plt.plot(N_clusters_Kmeans[i_max], Silhouette_Kmeans[i_max], "ro")
    plt.xlabel("n clusters")
    plt.ylabel("Silhouette coef")    
    plt.title("Classify : KMeans")
    plt.xticks(N_clusters_Kmeans)
    plt.show()

    # plot best result
    n = N_clusters_Kmeans[i_max]
    kmeans, _, _ = createKMeans(X, n)

    plot_points_and_voronoi_diagram(X, kmeans, n)

def pregunta2():

    plot_Epsilon_vs_Silouette()

    plot_Clusters_vs_Silouette()

    # plot best results
    i_max = np.argmax(np.array(Silhouette_DBSCAN).reshape(-1))
    eps = epsilons[i_max % N]
    metric = metrics[i_max // N]
    labels, n_clusters, core_samples_mask, _ = classify(X, Y, eps, metric)

    plot_points_DBSCAN(X, labels, n_clusters, core_samples_mask)

def pregunta3():
    
    problems = [(0,0), (0,-1)]

    i_max = np.argmax(Silhouette_Kmeans)
    n_max = N_clusters_Kmeans[i_max]
    
    kmeans, labels, _ = createKMeans(X, n_max)
    preds = [predict(kmeans, problem) for problem in problems]

    plot_points_KMeans(X, labels, n_max, problems, preds)


def preguntas():
    print("Elige una opcion:")
    print(" 1) Pregunta 1")
    print(" 2) Pregunta 2")
    print(" 3) Pregunta 3")
    print(" 4) Estudio extra (de la pregunta 2)")
    print(" 5) Salir")
    option = input(" > ")
    if option == "1":
        pregunta1()
    elif option == "2":
        pregunta2()
    elif option == "3":
        pregunta3()
    elif option == "4":
        plot_Epsilon_vs_Silouette_maspruebas()
    else:
        return False
    return True


def main():
    active = True
    while active:
        active = preguntas()

if __name__ == "__main__":
    main()
