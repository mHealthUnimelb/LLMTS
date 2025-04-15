import torch
import numpy as np
from sklearn.cluster import KMeans


class VocabClusterer:
    def __init__(self, num_clusters=2000, random_state=42):
        self.num_clusters = num_clusters
        self.random_state = random_state
        self.kmeans = None
        self.cluster_assignments = None
        self.cluster_centers = None

    def fit(self, word_embeddings: torch.Tensor):
        """
        word_embeddings: shape [vocab_size, embed_dim]
        """
        vocab_size, embed_dim = word_embeddings.shape
        # Convert to numpy for sklearn
        embeddings_np = word_embeddings.cpu().numpy()

        # Fit KMeans
        self.kmeans = KMeans(n_clusters=self.num_clusters,
                             random_state=self.random_state)
        self.kmeans.fit(embeddings_np)

        # Store cluster assignments and centers
        self.cluster_assignments = self.kmeans.labels_  # shape: [vocab_size]
        self.cluster_centers = torch.tensor(self.kmeans.cluster_centers_,
                                            dtype=word_embeddings.dtype)
        print(f"[VocabClusterer] Fitted k-means with {self.num_clusters} clusters.")

    def get_cluster(self, token_id: int) -> int:
        """
        Returns which cluster this token belongs to.
        """
        return self.cluster_assignments[token_id]

    def get_cluster_centroid(self, cluster_id: int) -> torch.Tensor:
        """
        Returns the centroid of a given cluster.
        """
        return self.cluster_centers[cluster_id]

    def get_cluster_assignments(self) -> np.ndarray:
        """
        Returns all cluster assignments, shape: [vocab_size].
        """
        return self.cluster_assignments

    def get_cluster_members(self, cluster_id: int) -> np.ndarray:
        """
        Returns all token_ids in the specified cluster.
        """
        return np.where(self.cluster_assignments == cluster_id)[0]
