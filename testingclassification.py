import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from torch_geometric.data import Dataset
from tqdm import tqdm
from torch_geometric.nn.pool import global_mean_pool

from torch import nn
from graphormer.model import Graphormer
from graphormer.functional import precalculate_custom_attributes, precalculate_paths

import time
import sys
import os

# --- 1. DATASET SETUP ---
# Using MUTAG (Graph Classification)
dataset = TUDataset(root="./", name="MUTAG")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# HYPER-PARAMETERS
NUM_LAYERS = 3
NODE_DIM = 128
FF_DIM = 256
N_HEADS = 4
MAX_IN_DEGREE = 5
MAX_OUT_DEGREE = 5
MAX_PATH_DISTANCE = 5

# --- 2. PREPROCESSING (Centrality Encoding) ---
print("Precalculating custom attributes (Centrality Encoding)...")
modified_data_list = []
for data in tqdm(dataset):
    # This step adds in_degree and out_degree to each individual graph
    modified_data = precalculate_custom_attributes(data, max_in_degree=MAX_IN_DEGREE, max_out_degree=MAX_OUT_DEGREE)
    modified_data_list.append(modified_data)

class ModifiedDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list        
    def len(self):
        return len(self.data_list)    
    def get(self, idx):
        return self.data_list[idx]

modified_dataset = ModifiedDataset(modified_data_list)

# --- 3. DATA LOADERS ---
from sklearn.model_selection import train_test_split
train_ids, test_ids = train_test_split(list(range(len(modified_dataset))), test_size=0.2, random_state=42)
train_loader = DataLoader(Subset(modified_dataset, train_ids), batch_size=8, shuffle=True)
test_loader = DataLoader(Subset(modified_dataset, test_ids), batch_size=8)

# --- 4. MODEL SETUP ---
model = Graphormer(
    num_layers=NUM_LAYERS,
    input_node_dim=dataset.num_node_features,
    node_dim=NODE_DIM,
    input_edge_dim=dataset.num_edge_features,
    edge_dim=NODE_DIM,
    output_dim=dataset.num_classes, 
    n_heads=N_HEADS,
    ff_dim=FF_DIM,
    max_in_degree=MAX_IN_DEGREE,
    max_out_degree=MAX_OUT_DEGREE,
    max_path_distance=MAX_PATH_DISTANCE,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_function = nn.CrossEntropyLoss()

# --- 5. TRAINING LOOP (With Dynamic Path Calculation) ---
# 

for epoch in range(10):
    model.train()
    total_loss = 0.0
    correct = 0
    epoch_start = time.time()
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        # DYNAMIC PATH CALCULATION: This fixes the IndexError by ensuring 
        # path tensors match the specific batch dimensions exactly.
        _, _, node_paths_length, edge_paths_tensor, edge_paths_length = precalculate_paths(
            batch, max_path_distance=MAX_PATH_DISTANCE
        )
        
        batch.node_paths_length = node_paths_length.to(DEVICE)
        batch.edge_paths_tensor = edge_paths_tensor.to(DEVICE)
        batch.edge_paths_length = edge_paths_length.to(DEVICE)
        batch = batch.to(DEVICE)

        optimizer.zero_grad()

        # Forward Pass
        # Graphormer output shape: [Total_Nodes, Node_Dim]
        node_embeddings = model(batch)
        
        # Pooling to Graph Level: [Batch_Size, Output_Dim]
        logits = global_mean_pool(node_embeddings, batch.batch)
        
        loss = loss_function(logits, batch.y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == batch.y).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    train_acc = correct / len(train_ids)
    print(f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Time: {time.time()-epoch_start:.2f}s")

    # --- 6. EVALUATION ---
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for batch in test_loader:
            _, _, n_p_l, e_p_t, e_p_l = precalculate_paths(batch, max_path_distance=MAX_PATH_DISTANCE)
            batch.node_paths_length = n_p_l.to(DEVICE)
            batch.edge_paths_tensor = e_p_t.to(DEVICE)
            batch.edge_paths_length = e_p_l.to(DEVICE)
            batch = batch.to(DEVICE)

            logits = global_mean_pool(model(batch), batch.batch)
            preds = logits.argmax(dim=1)
            test_correct += (preds == batch.y).sum().item()

    print(f"--> TEST ACCURACY: {test_correct / len(test_ids):.4f}\n")