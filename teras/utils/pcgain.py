from sklearn.cluster import (AgglomerativeClustering,
                             KMeans,
                             MiniBatchKMeans,
                             SpectralClustering,
                             SpectralBiclustering)


SUPPORTED_CLUSTERING_METHODS = ["Agglomerative", "KMeans", "MiniBatchKMeans", "Spectral", "SpectralBiclustering"]


def cluster(x, num_clusters=5, method="kmeans"):
    """
    Clusters the data into `num_clusters` clusters using
    the specified cluster method.

    Args:
        x: Dataset to cluster
        num_clusters: Number of clusters to cluster the data into.
        method: Should be one of the following,
            ["Agglomerative", "KMeans", "MiniBatchKMeans", "Spectral", "SpectralBiclustering"]
            The names are case in-sensitive.
            Defaults to "kmeans".

    Returns:
        Predicted cluster index for each sample.
    """
    method = method.lower()
    if method == "agglomerative":
        cluster_idx = AgglomerativeClustering(n_clusters=num_clusters).fit_predict(x)
    elif method == "kmeans":
        cluster_idx = KMeans(n_clusters=num_clusters).fit_predict(x)
    elif method == "minibatchkmeans":
        cluster_idx = MiniBatchKMeans(n_clusters=num_clusters).fit_predict(x)
    elif method == "spectral":
        cluster_idx = SpectralClustering(n_clusters=num_clusters).fit_predict(x)
    elif method == "spectralbiclustering":
        cluster_idx = SpectralBiclustering(n_clusters=num_clusters).fit(x).row_labels_
    else:
        raise ValueError(f"The clustering method name passed `{method}` is either not "
                         f"currently supported or you made a typo in the name."
                         f"Please note that ")
    return cluster_idx


