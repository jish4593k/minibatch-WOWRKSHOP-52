import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import KDTree
import plotly.express as px
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

sns.set(style="whitegrid")

def kmeans(X, k):
    model = keras.Sequential([
        keras.layers.Input(shape=(X.shape[1],)),
        keras.layers.Dense(k, activation='linear', use_bias=False)
    ])

    model.layers[1].set_weights([X[np.random.choice(X.shape[0], k, replace=False)]])
    
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    model.fit(X, X, epochs=10, batch_size=32, verbose=0)

    return model.layers[1].get_weights()[0]

def mini_batch_kmeans(X, k, b, t, replacement=True):
    model = keras.Sequential([
        keras.layers.Input(shape=(X.shape[1],)),
        keras.layers.Dense(k, activation='linear', use_bias=False)
    ])

    model.layers[1].set_weights([X[np.random.choice(X.shape[0], k, replace=False)]])
    
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    for _ in range(t):
        if replacement:
            X_batch = X[np.random.choice(X.shape[0], b, replace=True)]
        else:
            X_batch = X[np.random.choice(X.shape[0], b, replace=False)]
        
        model.fit(X_batch, X_batch, epochs=1, batch_size=32, verbose=0)

    return model.layers[1].get_weights()[0]

def plot_clusters(X, labels, centers, title):
    df = np.column_stack((X, labels))
    df = pd.DataFrame(df, columns=['X1', 'X2', 'Cluster'])
    
    fig = px.scatter(df, x='X1', y='X2', color='Cluster', title=title,
                     template='plotly', width=800, height=500, color_continuous_scale='viridis')
    
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    fig.update_layout(showlegend=False)

    for i, center in enumerate(centers):
        fig.add_trace(go.Scatter(x=[center[0]], y=[center[1]], mode='markers', marker=dict(color='red', size=14),
                         showlegend=False, name=f'Cluster {i+1} Center'))

    fig.show()


if __name__ == '__main__':
    np.random.seed(1)

    n = 1000
    d = 2
    X, y = make_blobs(n, d, centers=3)

    k = 3
    b = 50
    t = 10

    C_init = X[:k]
    C_kmeans = kmeans(X, k)
    C_mbkm = mini_batch_kmeans(X, k, b, t, replacement=False)
    C_mbkm_wr = mini_batch_kmeans(X, k, b, t, replacement=True)

    labels_init = compute_labels(X, C_init)
    labels_kmeans = compute_labels(X, C_kmeans)
    labels_mbkm = compute_labels(X, C_mbkm)
    labels_mbkm_wr = compute_labels(X, C_mbkm_wr)

    print("Adjusted rand scores:")
    print("labels_kmeans, labels_init =", adjusted_rand_score(labels_kmeans, labels_init))
    print("labels_kmeans, labels_mbkm =", adjusted_rand_score(labels_kmeans, labels_mbkm))
    print("labels_kmeans, labels_mbkm_wr =", adjusted_rand_score(labels_kmeans, labels_mbkm_wr))

    plot_clusters(X, labels_init, C_init, 'Initial Clusters')
    plot_clusters(X, labels_kmeans, C_kmeans, 'K-Means Clusters')
    plot_clusters(X, labels_mbkm, C_mbkm, 'Mini-Batch K-Means Clusters (Without Replacement)')
    plot_clusters(X, labels_mbkm_wr, C_mbkm_wr, 'Mini-Batch K-Means Clusters (With Replacement)')
