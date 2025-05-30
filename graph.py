import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt 
from torch_geometric.utils import to_networkx
import umap

def create_graph(features, k=6):
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(features)
    distances, indices = knn.kneighbors(features)

    edge_index = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:
            edge_index.append([i, neighbor])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    x = torch.tensor(features, dtype=torch.float)
    graph_data = Data(x=x, edge_index=edge_index)

    return graph_data

def visualize_graph(graph, features, reduction_method):
    # G = to_networkx(graph, to_undirected=True)
    # plt.figure(figsize=(8,6))
    # nx.draw(G, with_labels=False, node_size=100, node_color='skyblue', edge_color='gray')
    # plt.show()
    if reduction_method == 'tsne':
        reduced_features = TSNE(n_components=2, random_state=42).fit_transform(features)
        title = "KNN Graph Visualization with t-SNE Dimensionality Reduction"
    else:
        reduced_features = umap.UMAP(n_components=2, random_state=42).fit_transform(features)
        title = "KNN Graph Visualization with UMAP Dimensionality Reduction"
    
    kmeans = KMeans(n_clusters = 5, random_state=42)
    cluster_labels = kmeans.fit_predict(features)

    pos = {i: reduced_features[i] for i in range(graph.num_nodes)}

    edge_list = graph.edge_index.t().numpy()
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(edge_list)
    cmap = plt.get_cmap('viridis', 5)
    plt.figure(figsize=(12, 10))
    nx.draw(
        nx_graph, pos,
        node_size=50, 
        node_color=cluster_labels, 
        cmap=cmap,
        with_labels=False,
        alpha=0.7, 
        edge_color='gray'
    )
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])  
    plt.colorbar(sm, label="Cluster", ax=plt.gca())
    plt.title(title)
    plt.savefig(f"{reduction_method} graph.png", dpi=300, bbox_inches='tight')
    plt.close()




if __name__=='__main__':
    feature_file = "image_features.npy"
    filename_file = "image_filenames.npy"

    features = np.load(feature_file)
    filenames = np.load(filename_file)
    graph = create_graph(features)
    #torch.save(graph, "graph_full.pt")
    #print("Graph saved to graph_full.pt")
    visualize_graph(graph, features, 'tsne')
    visualize_graph(graph, features, "umap")
    print("done vis.")

