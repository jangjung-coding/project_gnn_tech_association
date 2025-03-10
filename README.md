# GNN_Tech_Association 🌕🌑

This repository contains the code for analyzing and visualizing the relationships between technical documents using Graph Neural Networks (GNNs). 
The project focuses on constructing a knowledge graph for space base construction technologies and exploring various scenarios such as technology clustering, roadmap generation, and keyword analysis.

---
**1. Load and preprocess documents:**
- Load markdown documents from the specified directory.
- Preprocess text by removing punctuation, numbers, and stop words.

**2. Extract keywords:**
- Use TF-IDF and KRWordRank to extract keywords from the documents.

**3. Calculate similarities and create graph:**
- Calculate cosine similarity for document embeddings.
- Calculate Jaccard similarity for keywords.
- Combine similarities to create a weighted graph.

**4. Visualize the graph:**
- Use NetworkX and Matplotlib to visualize the knowledge graph.

**5. Train and evaluate GAT model:**
- Define and train a Graph Attention Network (GAT) model using K-Fold cross-validation.
- Evaluate the model using AUC, accuracy, and F1-score.

**6. Generate technology roadmap:**
- Use PageRank scores to generate a weighted technology roadmap.

**7. Cluster technologies:**
- Use K-Means clustering to group technologies.

**8. Analyze keywords:**
- Extract and analyze keywords from the documents.

**9. Find new relationships:**
- Compare the existing knowledge graph with the GAT-based graph to find new relationships.
