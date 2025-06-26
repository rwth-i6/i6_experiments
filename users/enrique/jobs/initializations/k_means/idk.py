# Objective: maximize over cluster means {μ_c} and class sequence {c_1^N} the following expression:
# 
#     sum over n of:
#         log P(c_n | c_{n-m}^{n-1})        # log-probability of current class given previous m classes
#         - (1 / v^2) * || x_n - μ_{c_n} ||^2  # squared Euclidean distance penalty between observation x_n and class mean μ_{c_n}
#
# In full:
# max_{μ_c} max_{c_1^N} sum_n [ log P(c_n | c_{n-m}^{n-1}) - (1 / v^2) * || x_n - μ_{c_n} ||^2 ]


import torch
from  recipe.i6_experiments.users.enrique.jobs.initializations.k_means.lm  import log_lm_pr

variance = 0.01  # Example variance for the distance penalty
variance_inverse_squared = variance ** -2

def custom_distance(a, b):
    # L2 distance
    l2 = torch.norm(a - b, dim=1)

    # LM probability of the 
    log_lm_prior = log_lm_pr(a, b)
    #return log_lm_prior - variance_inverse_squared * l2
    return l2

def kmeans_torch(X, k, num_iters=10):
    n, d = X.shape
    # Initialize centroids randomly
    centroids = X[torch.randperm(n)[:k]]

    for _ in range(num_iters):
        # Compute distances (all X to all centroids)
        distances = torch.stack([custom_distance(X, centroid.unsqueeze(0).expand_as(X)) for centroid in centroids])
        # Assign clusters
        labels = torch.argmin(distances, dim=0)
        # Update centroids
        for i in range(k):
            mask = (labels == i)
            if mask.any():
                centroids[i] = X[mask].mean(dim=0)
    return labels, centroids

# Usage
X = torch.randn(10000, 128).cuda()   # Example data on GPU
labels, centroids = kmeans_torch(X, k=20)


X_cpu = X.cpu()
centroids_cpu = centroids.cpu()
labels_cpu = labels.cpu()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Reduce to 2D
X_2d = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(X_cpu)
centroids_2d = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(centroids_cpu)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_cpu, cmap='tab20', alpha=0.5, s=10, label='Points')
plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='black', s=100, marker='X', label='Centroids')
plt.legend()
plt.title("Clustered Points and Centroids")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)

# Save the figure
plt.savefig("cluster_plot.png", dpi=300, bbox_inches='tight')

# Show it (optional)
plt.show()

