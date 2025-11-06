import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y_true = make_blobs(
    n_samples=300,
    centers=[[0, 0], [6, 6], [0, 6]],
    cluster_std=0.8,
    random_state=42,
)


def run_kmeans(X, initial_centroids, prefix):
    centroids = initial_centroids.copy()
    K = centroids.shape[0]

    for iteration in range(5):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        plt.figure(figsize=(6, 5))
        colors = ["r", "g", "b", "c", "m", "y"]
        for k in range(K):
            plt.scatter(
                X[labels == k, 0],
                X[labels == k, 1],
                s=30,
                color=colors[k],
                label=f"Cluster {k+1}",
            )
            plt.scatter(
                centroids[k, 0],
                centroids[k, 1],
                color="black",
                marker="x",
                s=200,
                linewidths=3,
            )
        plt.title(f"{prefix} Iteration {iteration + 1}")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{prefix}{iteration + 1}.png", dpi=120)
        plt.close()

        new_centroids = np.array(
            [
                X[labels == k].mean(axis=0) if len(X[labels == k]) > 0 else centroids[k]
                for k in range(K)
            ]
        )
        centroids = new_centroids


good_init = np.array([[0, 0], [6, 6], [0, 6]])
run_kmeans(X, good_init, prefix="A")


bad_init = np.array([[5, 5], [6, 4], [2, 2]])
run_kmeans(X, bad_init, prefix="B")

print("✅ 已生成 A1–A5（正确收敛）和 B1–B5（典型错误收敛）")
