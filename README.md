This project makes pinterest-like fashion recommendations by fashion/clothing item photos with graph neural network techniques. I've created a pipeline from feature extraction to modelling, training, evaluation, and metrics! 

🔍 How It Works

Image Representation

A dataset that has both products and fashion images with detailed annotations that allow learning contextual recommendations is used and is passed through a fine-tuned ResNet-50 model. The ResNet-50 model is trained on fashion items that captures style-specific features (design, texture). 2048 dimensions are processed by the ResNet-50 model, and an extra feed-forward neural network compresses it to 128 dimensional embedding, so it is more efficient along the pipeline. 

Graph Building

Using cosine similarity between embeddings, a K-Nearest Neighbors (KNN) graph is built.
Each node represents an image; edges connect visually similar items.
KNNs maintainins intra-cluster similarity, enhance inter-cluster separation, and encourage diversity in the embeddings. 

Learning with GAT

I use a two-layer Graph Attention Network: 
First layer learns hidden node embeddings.
Second layer outputs classification predictions.
Training uses cross-entropy loss with the Adam optimizer.
During training, K-Means clustering is applied iteratively on the embeddings to refine cluster assignments, enabling the GAT to adapt and learn meaningful groupings. 


| Metric                   | Value  | Meaning                                                                                                      |
| ------------------------ | ------ | ------------------------------------------------------------------------------------------------------------ |
| **Silhouette Score**     | 0.8923 | Very strong — clusters are highly separated and consistent internally.                                       |
| **Davies-Bouldin Index** | 0.4234 | Low — indicates compact clusters with clear boundaries.                                                      |
| **Graph Density**        | 0.0089 | Sparse — only a small number of meaningful item connections remain, which is expected in recommender graphs. |

Future Work

While the current system demonstrates strong clustering performance and effective use of graph neural networks for fashion recommendations, there are several directions to enhance its capabilities:
- User Preference Integration
  - Incorporate user interaction data (clicks, saves, purchases) alongside visual embeddings to create a hybrid recommendation system.
  - Explore personalized embeddings that adapt graph connections based on individual tastes.

- Multi-Modal Feature Fusion
  - Extend beyond visual data by integrating product metadata (brand, material, season) and textual descriptions.
  - Use transformer-based models (e.g., CLIP, BERT) to capture cross-modal similarities between images and text.
- Improved Evaluation Metrics
  - Conduct A/B testing with user studies to validate real-world recommendation impact.
