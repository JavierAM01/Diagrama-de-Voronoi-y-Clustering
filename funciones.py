"""
    Script con las funciones de clasificación de los algoritmos KMeans y DBSCAN
"""

from sklearn.cluster import KMeans, DBSCAN
from sklearn import metrics

import numpy as np

# Aquí cargamos el sistema X de 1500 elementos (personas) con dos estados
def load_data():
    archivo1 = "Personas_en_la_facultad_matematicas.txt"
    archivo2 = "Grados_en_la_facultad_matematicas.txt"
    Y = np.loadtxt(archivo2, encoding="latin-1")
    X = np.loadtxt(archivo1, encoding="latin-1")
    return X, Y 


# Los clasificamos mediante el algoritmo KMeans
def createKMeans(X, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    labels = kmeans.labels_
    silhouette = metrics.silhouette_score(X, labels)
    return kmeans, labels, silhouette


# Predicción (Kmeans) de elementos para pertenecer a una clase:
def predict(kmeans, problem): 
    problem = np.array(problem).reshape(1,-1) # tiene que recibir un np.array de una dimensión aparte del propio vector (reshape)
    pred = kmeans.predict(problem)[0]  # [0] es porque: resultado = np.array([pred]) => resultado[0] = pred
    return pred

# Los clasificamos mediante el algoritmo DBSCAN
def classify(X, Y, epsilon=0.1, metric='manhattan', n0=10): # 'euclidean'
    
    db = DBSCAN(eps=epsilon, min_samples=n0, metric=metric).fit(X)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    try:
        silhouette = metrics.silhouette_score(X, labels)
    except: # only 1 label
        silhouette = 0
    return labels, n_clusters, core_samples_mask, silhouette



