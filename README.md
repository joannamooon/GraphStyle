This project explores a new method for making fashion recommendations by combining user-created visual boards (such as moodboards) with graph neural network techniques.
My approach links deep image feature extraction with graph-based learning to recommend products that match a user’s style both contextually and aesthetically.

🔍 How It Works

Image Representation

Product and scene images are processed through a fine-tuned ResNet-50 model.
The base model is pre-trained on Fashion MNIST and adapted to capture richer style-specific features.

Graph Building

Using cosine similarity between embeddings, we build a K-Nearest Neighbors (KNN) graph.
Each node represents an image; edges connect visually similar items.

Learning with GAT

I use a two-layer Graph Attention Network:
First layer learns hidden node embeddings.
Second layer outputs classification predictions.
Training uses cross-entropy loss with the Adam optimizer.

| Metric                   | Value  | Meaning                                                                                                      |
| ------------------------ | ------ | ------------------------------------------------------------------------------------------------------------ |
| **Silhouette Score**     | 0.9394 | Very strong — clusters are highly separated and consistent internally.                                       |
| **Davies-Bouldin Index** | 0.4614 | Low — indicates compact clusters with clear boundaries.                                                      |
| **Graph Density**        | 0.0009 | Sparse — only a small number of meaningful item connections remain, which is expected in recommender graphs. |
