import numpy as np

class MaskGenerator:
    """Cluster dense DINO features into superpixel-like masks using k-means."""

    def __init__(self, num_clusters=100, spatial_weight=0.25, num_iters=10, seed=0):
        self.num_clusters = num_clusters
        self.spatial_weight = spatial_weight
        self.num_iters = num_iters
        self.rng = np.random.RandomState(seed)

    def _kmeans(self, X):
        """Simple k-means clustering implemented with numpy."""
        N, C = X.shape
        centers = X[self.rng.choice(N, self.num_clusters, replace=False)]
        labels = np.zeros(N, dtype=np.int32)
        for _ in range(self.num_iters):
            dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            labels = dists.argmin(axis=1)
            for k in range(self.num_clusters):
                idx = np.where(labels == k)[0]
                if len(idx) > 0:
                    centers[k] = X[idx].mean(axis=0)
        return labels

    def generate(self, dino_feat, feat_shape):
        """Return HxW mask of cluster IDs for given features."""
        H, W = feat_shape
        N, C = dino_feat.shape
        assert N == H * W, "Feature count does not match shape"
        y = np.repeat(np.arange(H), W)
        x = np.tile(np.arange(W), H)
        spatial = np.stack([y / H, x / W], axis=1) * self.spatial_weight
        X = np.concatenate([dino_feat, spatial], axis=1)
        labels = self._kmeans(X)
        return labels.reshape(H, W)
