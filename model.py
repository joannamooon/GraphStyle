import torch 
import torch.nn as nn 
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 

def load_graph(pth="graph_full.pt"):
    graph = torch.load(pth, weights_only=False)
    print("graph loaded")
    return graph

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, heads=4, dropout=0.3):
        super(GAT, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, embedding_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim * heads)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.layer_norm1(self.gat1(x, edge_index)))
        x = self.dropout(x)
        x = self.layer_norm2(self.gat2(x, edge_index))
        return x 
    
def train_GAT(graph, input_dim, hidden_dim, embedding_dim, n_clusters, epochs=100, lr=0.01):
    model = GAT(input_dim, hidden_dim, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        embeddings = model(graph)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        clusters = kmeans.fit_predict(embeddings.detach().cpu().numpy())
        
        # calculate clustering loss
        clusters = torch.tensor(clusters, device=embeddings.device)
        row, col = graph.edge_index
        pos_sim = torch.sum(embeddings[row] * embeddings[col], dim=1)
        pos_loss = -torch.log(torch.sigmoid(pos_sim) + 1e-15).mean()
        cluster_center = torch.stack([embeddings[clusters == c].mean(dim=0) for c in range(clusters.max().item() + 1)])
        cluster_loss = torch.norm(embeddings - cluster_center[clusters], dim=1).mean()
        embedding_reg = torch.norm(embeddings, dim=1).mean()
        loss = 0.8 * pos_loss + (1 - 0.8) * cluster_loss + 0.05 * embedding_reg
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
    
    print("Done Training. ")
    return model, kmeans

def save_model(model, save_path="gat_model.pt"):
    torch.save(model, save_path)
    print("Model Saved. ")

if __name__ == '__main__':
    graph_path = "graph_full.pt"
    graph_data = load_graph(graph_path)
    graph_data.x = torch.tensor(StandardScaler().fit_transform(graph_data.x.numpy()), dtype=torch.float32)

    input_dim = graph_data.x.size(1)
    hidden_dim = 128
    embedding_dim = 256
    heads = 8
    n_clusters = 5

    trained_model, kmeans = train_GAT(graph_data, input_dim, hidden_dim, embedding_dim, n_clusters, lr=0.001)
    save_model(trained_model, "gat_model.pt")
    embeddings = trained_model(graph_data).detach().cpu().numpy()
    pca = PCA(n_components=2)
    reduced_emb = pca.fit_transform(embeddings)
    plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], c='blue')
    plt.savefig("Embedding Visualization", dpi=300, bbox_inches='tight')
    plt.close()



