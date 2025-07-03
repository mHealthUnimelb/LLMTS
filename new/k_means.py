import torch
from cuml.cluster import KMeans as cuKMeans
import cupy as cp

# Convert embeddings to GPU array
wte_gpu = cp.asarray(wte)

# Apply GPU-accelerated K-Means clustering
kmeans = cuKMeans(n_clusters=1000, random_state=0)
kmeans.fit(wte_gpu)

# Convert the cluster centers back to numpy array
wte_reduced = cp.asnumpy(kmeans.cluster_centers_)

# Save the reduced token embeddings
torch.save(wte_reduced, "wte_kmeans_1000_tokens.pt")
