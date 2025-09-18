import sys
import numpy as np

sys.path.insert(0, "../recipe")

from recipe.i6_experiments.users.enrique.jobs.initializations.k_means.returnn_kmeans.k_means import NnOutputClusteringCallback

def plot_clusters(data, centroids):
    """Plot the clusters with their centers."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    plt.scatter(x=data[:,0], y=data[:,1], label=f'Data')
    plt.scatter(x=centroids[:,0], y=centroids[:,1], color='red', marker='x', label='Centroids')
    
    plt.title('Clusters Visualization')
    plt.legend()
    plt.savefig('clusters.png')

def main():
    # Example usage of NnOutputClusteringCallback
    callback = NnOutputClusteringCallback(
        num_clusters=10,
        writer_batch_size=1,
    )
    callback.init()  # Assuming no model is needed for this example
    
    # generate gaussian data around 10 clusters
    # first generate 10 random centers
    centers = np.random.rand(10, 2) * 1000

    # initialize
    while not callback.is_initialized:
        # generate a random point around one of the centers
        center = centers[np.random.randint(0, 10)]
        point = center + np.random.randn(2) * 5
        callback.process_point(x=point)
    
    print("Initialization complete. Starting to process points...")

    data = []
    for _ in range(1000 * 1024):
        # generate a random point around one of the centers
        center = centers[np.random.randint(0, 10)]
        point = center + np.random.randn(2) * 5
        data.append(point)

        # process the point
        callback.process_point(point)
    
    callback.finish()

    plot_clusters(np.array(data), callback.centroids)
    
if __name__ == "__main__":
    main()