import faiss
import numpy as np

# Generate some random data
d = 64  # dimension
nb = 10000  # database size
nq = 100  # nb of queries
np.random.seed(1234)  # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.  # make dataset more distinguishable
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

# Define k-means parameters
ncentroids = 100
niter = 20
verbose = True

# Move to GPU using cuVS resources
res = faiss.ResourcesCuvs()  # Use cuVS resources instead of StandardGpuResources

# Initialize k-means on GPU with cuVS
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
kmeans.cuvs_handle = res  # Set the cuVS handle

# Train the k-means model
kmeans.train(xb)

# Get the cluster centroids
centroids = kmeans.centroids

# To search on GPU with cuVS, create a cuVS index
index_flat = faiss.IndexFlatL2(d)  # the other index
gpu_index_flat = faiss.index_cpu_to_cuvs(res, index_flat)  # Use index_cpu_to_cuvs instead
gpu_index_flat.add(centroids)

# Search on GPU
D, I = gpu_index_flat.search(xb, 1)

# Print some results
print("Centroids shape:", centroids.shape)
print("First 5 cluster assignments:", I[:5])
print("First 5 distances to centroids:", D[:5])