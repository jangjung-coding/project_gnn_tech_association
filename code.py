# %%
# Standard libraries
import re
from pathlib import Path

# Data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from matplotlib import rc

# Graph and network analysis
import networkx as nx

# Machine learning and NLP
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from krwordrank.word import KRWordRank

# Deep learning and graph neural networks
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import negative_sampling

# %%
# Font settings for plots
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Directory paths
MODEL_DIR = r"yourModelPath"
VAULT_PATH = r"yourPath"
SIMILARITY_THRESHOLD = 0

# Korean stop words
korean_stop_words = set([
    '의', '가', '이', '은', '는', '을', '를', '에', '와', '과', '도', '으로', '로', '에서', '에게', '뿐', '만', '하다', '하는', '하거나',
    '및', '등', '그리고', '하지만', '또한'
])

# %%
# Load markdown documents from the specified directory
def load_documents():
    md_files = {}
    for path in Path(VAULT_PATH).rglob("*.md"):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().split('---', 2)[-1].strip()
            md_files[Path(path).stem] = content
        except Exception as e:
            print(f"Error: {path.name} - {str(e)}")
    return md_files

docs = load_documents()
print(f"Loaded {len(docs)} documents")

# Preprocess text by removing punctuation, numbers, and stop words
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [word for word in words if word not in korean_stop_words]
    return ' '.join(words)

preprocessed_docs = {name: preprocess_text(content) for name, content in docs.items()}

# Load pre-trained SentenceTransformer model
model = SentenceTransformer(MODEL_DIR)

# Encode documents in batches
def batch_encode(model, texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

embeddings = batch_encode(model, list(docs.values()))

# Initialize TF-IDF vectorizer and fit on preprocessed documents
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_vectorizer.fit(preprocessed_docs.values())

# Extract keywords using TF-IDF
def extract_keywords_tfidf(text, top_n=5):
    tfidf_vector = tfidf_vectorizer.transform([text])
    sorted_indices = np.argsort(tfidf_vector.toarray()[0])[::-1]
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
    return feature_names[sorted_indices][:top_n].tolist()

# Extract keywords using KRWordRank
def extract_keywords_krwordrank(texts, top_n=5):
    wordrank_extractor = KRWordRank(min_count=5, max_length=10, verbose=False)
    beta = 0.85
    max_iter = 10
    keywords, rank, graph = wordrank_extractor.extract(texts, beta, max_iter)
    sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [keyword for keyword, _ in sorted_keywords]

# Combine keywords from TF-IDF and KRWordRank
doc_keywords = {}
for name, content in preprocessed_docs.items():
    tfidf_keywords = extract_keywords_tfidf(content, top_n=5)
    kr_keywords = extract_keywords_krwordrank([content], top_n=5)
    combined_keywords = list(set(tfidf_keywords + kr_keywords))
    combined_keywords = [kw for kw in combined_keywords if kw != '기술']
    doc_keywords[name] = combined_keywords[:10]

# Calculate Jaccard similarity between two lists
def jaccard_similarity(list1, list2):
    s1, s2 = set(list1), set(list2)
    if not s1 and not s2:
        return 0
    return len(s1.intersection(s2)) / len(s1.union(s2))

# Extract major and minor categories from document content
def extract_categories(content):
    lines = content.split('\n')
    major_category = None
    minor_category = None
    for line in lines:
        if line.startswith("대분류:"):
            major_category = line.replace("대분류:", "").strip()
        elif line.startswith("중분류:"):
            minor_category = line.replace("중분류:", "").strip()
    return major_category, minor_category

# Store categories for each document
doc_categories = {}
for name, content in docs.items():
    major_cat, minor_cat = extract_categories(content)
    doc_categories[name] = {'major': major_cat, 'minor': minor_cat}

# %%
# Calculate cosine similarity matrix for embeddings
sim_matrix = cosine_similarity(embeddings)

# Weights for combined similarity calculation
w_embedding = 0.5
w_keyword = 0.25
w_category = 0.25

# Initialize graph
doc_keys = list(docs.keys())
G = nx.Graph()
for idx, name in enumerate(doc_keys):
    G.add_node(name, embedding=embeddings[idx])

# Add edges based on combined similarity
for i in range(len(doc_keys)):
    for j in range(i + 1, len(doc_keys)):
        cos_sim = sim_matrix[i][j]
        jaccard_sim = jaccard_similarity(doc_keywords[doc_keys[i]], doc_keywords[doc_keys[j]])
        
        if doc_categories[doc_keys[i]]['major'] == doc_categories[doc_keys[j]]['major']:
            category_sim = 1 if doc_categories[doc_keys[i]]['minor'] == doc_categories[doc_keys[j]]['minor'] else 0.5
        else:
            category_sim = 0
        
        combined_sim = w_embedding * cos_sim + w_keyword * jaccard_sim + w_category * category_sim
        
        if combined_sim > SIMILARITY_THRESHOLD:
            G.add_edge(doc_keys[i], doc_keys[j], weight=combined_sim)

# Print graph statistics
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()} (Based on similarity threshold of {SIMILARITY_THRESHOLD})")
weights = [G.edges[e]['weight'] for e in G.edges()]
print(f"Average edge weight: {np.mean(weights):.3f}")

# %%
plt.figure(figsize=(60, 36))  #20, 12
pos = nx.spring_layout(G, seed=42, k=0.3)

# Draw nodes
nodes = nx.draw_networkx_nodes(G, pos,
                               node_size=2500,
                               node_color='#FF8000',
                               alpha=0.9)

# Draw edges
edges = nx.draw_networkx_edges(G, pos,
                               edge_color=[G.edges[e]['weight'] for e in G.edges()],
                               edge_cmap=plt.cm.Oranges,
                               edge_vmin=0.5,
                               edge_vmax=1.0,
                               width=2.5,
                               alpha=0.7)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.Oranges,
                           norm=plt.Normalize(vmin=SIMILARITY_THRESHOLD, vmax=1.0))
sm.set_array([])
ax = plt.gca()
cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
cbar.set_label('기술 간 유사도', fontsize=30)

# Draw labels
nx.draw_networkx_labels(G, pos,
                        labels={n: n for n in G.nodes},
                        font_size=10,
                        font_family='Malgun Gothic')

# Title and layout
plt.title("유인 우주기지 건설기술 지식 그래프", fontsize=50, pad=20)
plt.axis('off')
plt.tight_layout()
plt.show()

# %%
# Convert NetworkX graph object to PyTorch Geometric data object
def nx_to_pyg_data(G):
    node_list = list(G.nodes)
    embeddings = np.array([G.nodes[n]['embedding'] for n in node_list])
    edge_index = []
    edge_weight = []

    node_idx_map = {node: idx for idx, node in enumerate(node_list)}

    for u, v, data in G.edges(data=True):
        edge_index.append([node_idx_map[u], node_idx_map[v]])
        edge_index.append([node_idx_map[v], node_idx_map[u]])  # Handle undirected graph
        edge_weight.append(data['weight'])
        edge_weight.append(data['weight'])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    x = torch.tensor(embeddings, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight), node_list

data, node_list = nx_to_pyg_data(G)

# Define GAT model
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, heads=8, dropout=0.4):
        super(GAT, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
        self.layers.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        for conv in self.layers[:-1]:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=0.4, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x

# %%
# Prepare for K-Fold cross-validation
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Space to store performance metrics
fold_results = []
auc_scores = []
accuracy_scores = []
f1_scores = []

data = data.to(device)  # Move data to the appropriate device

# Perform K-Fold cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(data.x.cpu())):
    print(f'Fold {fold+1}/{k_folds}')
    
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    
    # Create edge masks for training and validation
    train_edge_mask = train_mask[data.edge_index[0].cpu()] & train_mask[data.edge_index[1].cpu()]
    val_edge_mask = val_mask[data.edge_index[0].cpu()] & val_mask[data.edge_index[1].cpu()]
    
    train_edges = data.edge_index[:, train_edge_mask].to(device)
    val_edges = data.edge_index[:, val_edge_mask].to(device)
    
    model = GAT(in_channels=data.num_features, hidden_channels=64, out_channels=32, num_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
    
    # Training
    model.train()
    epochs = 20
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        embeddings = model(data.x, data.edge_index)
        
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index, num_nodes=data.num_nodes, num_neg_samples=train_edges.size(1), method='sparse'
        ).to(device)
        
        pos_pred = (embeddings[train_edges[0]] * embeddings[train_edges[1]]).sum(dim=-1)
        neg_pred = (embeddings[neg_edge_index[0]] * embeddings[neg_edge_index[1]]).sum(dim=-1)
        
        loss_pos = F.binary_cross_entropy_with_logits(pos_pred, torch.ones_like(pos_pred))
        loss_neg = F.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))
        
        loss = loss_pos + loss_neg
        loss.backward()
        optimizer.step()
        
        if epoch % 1 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}')
        
        # Early stopping check
        model.eval()
        with torch.no_grad():
            val_embeddings = model(data.x, data.edge_index)
            pos_val_pred = (val_embeddings[val_edges[0]] * val_embeddings[val_edges[1]]).sum(dim=-1)
            loss_val = F.binary_cross_entropy_with_logits(pos_val_pred, torch.ones_like(pos_val_pred))
            
            if loss_val.item() < best_val_loss:
                best_val_loss = loss_val.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_embeddings = model(data.x, data.edge_index)
        pos_val_pred = (val_embeddings[val_edges[0]] * val_embeddings[val_edges[1]]).sum(dim=-1)
        
        loss_val = F.binary_cross_entropy_with_logits(pos_val_pred, torch.ones_like(pos_val_pred))
        
        print(f'Fold {fold+1} Validation Loss: {loss_val.item():.4f}')
        fold_results.append(loss_val.item())
        
        # Convert logits to probabilities
        pos_probs = torch.sigmoid(pos_val_pred).cpu().numpy()
        labels = torch.ones_like(pos_val_pred).cpu().numpy()
        
        # Calculate evaluation metrics
        try:
            auc = roc_auc_score(labels, pos_probs)
        except ValueError:
            auc = float('nan')
        
        accuracy = accuracy_score(labels, (pos_probs > 0.5).astype(int))
        f1 = f1_score(labels, (pos_probs > 0.5).astype(int), zero_division=1)
        
        auc_scores.append(auc)
        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
        
        print(f'Fold {fold+1} AUC: {auc:.4f}, Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}')
        
print('Cross-validation completed')
print(f'Average validation loss: {sum(fold_results) / k_folds:.4f}')
print(f'Average AUC: {sum([x for x in auc_scores if not torch.isnan(torch.tensor(x))]) / k_folds:.4f}')
print(f'Average accuracy: {sum(accuracy_scores) / k_folds:.4f}')
print(f'Average F1-score: {sum(f1_scores) / k_folds:.4f}')

model.eval()
with torch.no_grad():
    final_embeddings = model(data.x, data.edge_index.to(torch.long)).cpu().numpy()

# %%
# Combined similarity calculation function
def calculate_combined_similarity(embeddings, weights=(0.5, 0.25, 0.25)):
    cosine_sim = cosine_similarity(embeddings)
    euclidean_dist = euclidean_distances(embeddings)
    manhattan_dist = manhattan_distances(embeddings)
    
    # Normalize distances
    scaler = MinMaxScaler()
    euclidean_dist = scaler.fit_transform(euclidean_dist)
    manhattan_dist = scaler.fit_transform(manhattan_dist)
    
    # Combine similarities/distances with weights
    combined_sim = (weights[0] * cosine_sim) - (weights[1] * euclidean_dist + weights[2] * manhattan_dist)
    
    return combined_sim

# Create graph using combined similarity
G_viz = nx.Graph()

for idx, node_name in enumerate(node_list):
    G_viz.add_node(node_name, embedding=final_embeddings[idx])

combined_sim_matrix = calculate_combined_similarity(final_embeddings)

SIMILARITY_THRESHOLD = 0

for i in range(len(node_list)):
    for j in range(i + 1, len(node_list)):
        similarity = combined_sim_matrix[i, j]
        if similarity >= SIMILARITY_THRESHOLD:
            G_viz.add_edge(node_list[i], node_list[j], weight=similarity)

plt.figure(figsize=(60, 36))
pos = nx.spring_layout(G_viz, seed=42, k=0.3)
nx.draw_networkx_nodes(G_viz, pos,
                       node_size=2500,
                       node_color='#FF8000',
                       alpha=0.9)
edges = G_viz.edges(data=True)
nx.draw_networkx_edges(G_viz, pos,
                       edgelist=[(u, v) for u, v, d in edges],
                       width=[d['weight'] * 3 for u, v, d in edges],
                       edge_cmap=plt.cm.Oranges,
                       edge_vmin=0.5,
                       edge_vmax=1.0,
                       edge_color='orange',
                       alpha=0.6)
nx.draw_networkx_labels(G_viz, pos,
                        font_size=10,
                        font_family='Malgun Gothic')
sm = plt.cm.ScalarMappable(cmap=plt.cm.Oranges,
                           norm=plt.Normalize(vmin=SIMILARITY_THRESHOLD, vmax=1.0))
sm.set_array([])
ax = plt.gca()
cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
cbar.set_label('Similarity between technologies', fontsize=30) 
plt.title("GAT-based Knowledge Graph Visualization", fontsize=50)
plt.axis('off')
plt.tight_layout()
plt.show()

# %%
# Scenario 1: Calculating relationships between technical documents
combined_sim_matrix = calculate_combined_similarity(final_embeddings)
mean_combined_sim = np.mean(combined_sim_matrix)
print(f"Average combined similarity between final embeddings: {mean_combined_sim:.4f}")

def predict_association(doc_name1, doc_name2, threshold=mean_combined_sim):
    idx1 = node_list.index(doc_name1)
    idx2 = node_list.index(doc_name2)
    
    similarity = combined_sim_matrix[idx1, idx2]
    
    print(f"'{doc_name1}' and '{doc_name2}'")
    print(f"📌 Similarity: {similarity:.4f}")
    
    if similarity >= threshold:
        print("✅ These documents are likely to be related.")
        return True
    else:
        print("❌ These documents are unlikely to be related.")
        return False

# Example:
predict_association(node_list[0], node_list[0])

# %%
# Scenario 2: Finding similar documents based on a given document
def find_similar_documents(doc_name, top_n=5):
    idx = node_list.index(doc_name)
    similarities = combined_sim_matrix[idx]
    sorted_indices = np.argsort(similarities)[::-1]
    
    print(f"Documents similar to '{doc_name}':")
    for i in range(1, top_n + 1):
        similar_doc = node_list[sorted_indices[i]]
        similarity = similarities[sorted_indices[i]]
        print(f"📄 '{similar_doc}' with similarity {similarity:.4f}")
        
    return [node_list[i] for i in sorted_indices[1:top_n + 1]]

# Example:
similar_docs = find_similar_documents(node_list[0])

# %%
# Scenario 3: Identify the most important technologies by analyzing network centrality
def calculate_centralities(G):
    degree_centrality = nx.degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G)
    
    centrality_df = pd.DataFrame([degree_centrality, closeness_centrality, betweenness_centrality, pagerank]).T
    centrality_df.columns = ['Degree', 'Closeness', 'Betweenness', 'PageRank']
    
    return centrality_df

# Calculate centralities for the graph
centrality_df = calculate_centralities(G_viz)
centrality_df['Total'] = centrality_df.sum(axis=1)

# Identify the top technologies based on total centrality
top_techs = centrality_df.sort_values('Total', ascending=False).head(20)

# Print the top technologies
print("\n📌 Top technologies based on network centrality:")
display(top_techs.style.background_gradient(cmap='viridis').format(precision=4, subset=['Total']))

# %%
# Scenario 3: Generate a weighted technology roadmap
def generate_weighted_tech_roadmap_with_pagerank(start_techs, G, depth=5):
    # Step 1: Calculate PageRank scores for all nodes in the graph
    pagerank_scores = nx.pagerank(G)

    roadmap = []
    current_layer = start_techs
    visited = set(start_techs)

    for level in range(depth):
        next_layer = []
        tech_scores = {}

        # Step 2: Collect neighbors and prioritize using PageRank scores
        for tech in current_layer:
            neighbors = G.neighbors(tech)
            for neighbor in neighbors:
                if neighbor not in visited:
                    if neighbor not in tech_scores:
                        tech_scores[neighbor] = 0
                    # Add the product of edge weight and PageRank score
                    tech_scores[neighbor] += G[tech][neighbor]['weight'] * pagerank_scores[neighbor]

        # Step 3: Sort neighbors by their combined score (edge weight * PageRank)
        sorted_neighbors = sorted(tech_scores.items(), key=lambda x: x[1], reverse=True)

        # Step 4: Select top neighbors based on the current layer size
        next_layer = [neighbor for neighbor, score in sorted_neighbors[:max(1, len(current_layer) - 1)]]

        # Add selected neighbors to the roadmap and mark them as visited
        for tech in next_layer:
            roadmap.append((tech, level + 1))
            visited.add(tech)

        current_layer = next_layer

    # Step 5: Print the roadmap
    print(f"\n📌 Technology Roadmap (Depth with {depth+1})")
    print(' || '.join(start_techs))
    for tech, level in sorted(roadmap, key=lambda x: x[1]):
        if level == 0:
            print(f"{'  ' * level} - {tech}")
        else:
            print(f"{'  ' * level} - {tech}")

    return roadmap

# Visualize the roadmap on the knowledge graph
def visualize_roadmap_on_graph(G, roadmap, start_techs):
    pos = nx.spring_layout(G, seed=42, k=0.3)
    plt.figure(figsize=(60, 36))

    # Draw all nodes and edges
    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color='#FF8000', alpha=0.5)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)

    # Highlight the roadmap nodes and edges with gradient colors
    roadmap_nodes = set(start_techs)
    roadmap_edges = []
    node_colors = {}
    base_color = np.array([0, 0, 1])  # Blue color
    for tech, level in roadmap:
        roadmap_nodes.add(tech)
        for neighbor in G.neighbors(tech):
            if neighbor in roadmap_nodes:
                roadmap_edges.append((tech, neighbor))
        # Calculate color based on level
        color = base_color + (1 - base_color) * (level / (len(start_techs) - 1))
        node_colors[tech] = color

    # Draw roadmap nodes with gradient colors
    for tech in start_techs:
        nx.draw_networkx_nodes(G, pos, nodelist=[tech], node_size=3000, node_color=[base_color], alpha=0.9)
    for tech, color in node_colors.items():
        nx.draw_networkx_nodes(G, pos, nodelist=[tech], node_size=3000, node_color=[color], alpha=0.9)

    nx.draw_networkx_edges(G, pos, edgelist=roadmap_edges, edge_color='blue', width=2.5, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='Malgun Gothic')

    plt.title("Technology Roadmap on Knowledge Graph(Descending)", fontsize=50)
    plt.axis('off')
    plt.show()

# Example Usage
start_techs = [node_list[1], node_list[10], node_list[50], node_list[70]]  # Set multiple starting technologies
roadmap = generate_weighted_tech_roadmap_with_pagerank(start_techs, G_viz, depth=len(start_techs)-1)
visualize_roadmap_on_graph(G_viz, roadmap, start_techs)
visualize_roadmap_on_graph(G, roadmap, start_techs)

'''
Features
- PageRank-based prioritization: Combines edge weights with global PageRank scores to rank neighbors.
- Multiple starting points: Allows setting multiple starting points for more diverse roadmaps.
- Flexibility: Adjust depth and start_techs to set the desired range and detail level.
'''

# %%
# Scenario 3.5: Generate a weighted technology roadmap
def generate_weighted_tech_roadmap(start_tech, G, depth=5):
    # Step 1: Calculate PageRank scores for all nodes in the graph
    pagerank_scores = nx.pagerank(G)

    roadmap = [(start_tech, 0)]
    current_layer = [start_tech]
    visited = set(current_layer)

    for level in range(1, depth + 1):
        next_layer = []
        tech_scores = {}

        # Step 2: Collect neighbors and prioritize using PageRank scores
        for tech in current_layer:
            neighbors = G.neighbors(tech)
            for neighbor in neighbors:
                if neighbor not in visited and neighbor[0] != tech[0]:  # Skip nodes starting with the same letter
                    if neighbor not in tech_scores:
                        tech_scores[neighbor] = 0
                    # Add the product of edge weight and PageRank score
                    tech_scores[neighbor] += G[tech][neighbor]['weight'] * pagerank_scores[neighbor]

        # Step 3: Sort neighbors by their combined score (edge weight * PageRank)
        sorted_neighbors = sorted(tech_scores.items(), key=lambda x: x[1], reverse=True)

        # Step 4: Select top neighbors based on the current layer size
        next_layer = [neighbor for neighbor, score in sorted_neighbors[:max(2, level)]]

        # Add selected neighbors to the roadmap and mark them as visited
        for tech in next_layer:
            roadmap.append((tech, level))
            visited.add(tech)

        current_layer = next_layer

    # Step 5: Print the roadmap
    print(f"\n📌 Technology Roadmap (Depth with {depth})")
    for tech, level in sorted(roadmap, key=lambda x: x[1]):
        print(f"{'  ' * level}- {tech}")

    return roadmap

# Visualize the roadmap on the knowledge graph
def visualize_roadmap_on_graph(G, roadmap, start_tech, depth=5):
    pos = nx.spring_layout(G, seed=42, k=0.3)
    plt.figure(figsize=(60, 36))

    # Draw all nodes and edges
    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color='#FF8000', alpha=0.5)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)

    # Highlight the roadmap nodes and edges with gradient colors
    roadmap_nodes = {start_tech}
    roadmap_edges = []
    node_colors = {}
    base_color = np.array([0, 0, 1])  # Blue color
    for tech, level in roadmap:
        roadmap_nodes.add(tech)
        for neighbor in G.neighbors(tech):
            if neighbor in roadmap_nodes:
                roadmap_edges.append((tech, neighbor))
        # Calculate color based on level
        color = base_color + (1 - base_color) * (level / depth)
        node_colors[tech] = color

    # Draw roadmap nodes with gradient colors
    for tech, color in node_colors.items():
        nx.draw_networkx_nodes(G, pos, nodelist=[tech], node_size=3000, node_color=[color], alpha=0.9)

    nx.draw_networkx_edges(G, pos, edgelist=roadmap_edges, edge_color='blue', width=2.5, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='Malgun Gothic')

    plt.title("Technology Roadmap on Knowledge Graph(Ascending)", fontsize=50)
    plt.axis('off')
    plt.show()

# Example Usage
start_tech = 'Tech Document'  # Set a single starting technology
roadmap = generate_weighted_tech_roadmap(start_tech, G_viz, depth=5)
visualize_roadmap_on_graph(G_viz, roadmap, start_tech, depth=5)
visualize_roadmap_on_graph(G, roadmap, start_tech, depth=5)

# %%
# Scenario 4: Technology cluster exploration
num_clusters = 6
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(final_embeddings)

# Group technologies by clusters
cluster_groups = {i: [] for i in range(num_clusters)}
for idx, cluster_label in enumerate(clusters):
    cluster_groups[cluster_label].append(node_list[idx])

# Print technology clusters for roadmap planning
print("\n📌 Technology Clusters for Roadmap Planning:")
for cluster_id, tech_names in cluster_groups.items():
    print(f"\nCluster {cluster_id + 1}:")
    for tech in tech_names:
        print(f"- {tech}")

# %%
# Scenario 5: Technology keyword analysis
def extract_keywords_from_docs(documents):
    all_keywords = []
    for doc in documents:
        all_keywords.extend(doc_keywords[doc])
    return all_keywords

all_tech_keywords = extract_keywords_from_docs(node_list)

# Calculate keyword frequencies
keyword_freq = {kw: all_tech_keywords.count(kw) for kw in set(all_tech_keywords)}
sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)

# Print top 10 keywords from technology documents
print("\n📌 Top 10 Keywords from Technology Documents:"
      "\n(Keyword, Frequency)")
for kw, freq in sorted_keywords[:10]:
    print(f"- {kw}: {freq}")

# %%
# Scenario 6: Find new relationships using GAT-based G_viz that were not found in the existing knowledge graph G

# Compare the existing knowledge graph G with the GAT-based graph G_viz
def find_new_relationships(G, G_viz):
    new_relationships = []
    for edge in G_viz.edges(data=True):
        if not G.has_edge(edge[0], edge[1]) and edge[0][0] != edge[1][0]:  # Exclude relationships within the same major category
            new_relationships.append(edge)
    # Sort by similarity in descending order
    new_relationships = sorted(new_relationships, key=lambda x: x[2]['weight'], reverse=True)
    return new_relationships

# Find new relationships
new_relationships = find_new_relationships(G, G_viz)

# Convert new relationships to a pandas DataFrame for better visualization
new_relationships_df = pd.DataFrame(new_relationships, columns=['Node 1', 'Node 2', 'Attributes'])
new_relationships_df['Similarity'] = new_relationships_df['Attributes'].apply(lambda x: x['weight'])
new_relationships_df = new_relationships_df.drop(columns=['Attributes'])

# Print new relationships
print("📌 New relationships found in the GAT-based graph:")
display(new_relationships_df)


