import torch
import torch.nn as nn
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from torch_geometric.data import Dataset
from tqdm import tqdm
from torch_geometric.nn.pool import global_mean_pool
from sklearn.model_selection import train_test_split
import time
import os

# Import your models from model.py
from graphormer.model import MoEGraphormer
from graphormer.functional import precalculate_custom_attributes, precalculate_paths

# --- 1. SETTINGS & DEVICE ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Hyperparameters
NUM_LAYERS = 3
NODE_DIM = 128
FF_DIM = 256
N_HEADS = 4
MAX_IN_DEGREE = 5
MAX_OUT_DEGREE = 5
MAX_PATH_DISTANCE = 5
NUM_EXPERTS = 4
TOP_K = 2

# --- 2. DATASET PREPARATION ---
dataset = TUDataset(root="./", name="MUTAG")

print("Precalculating Centrality Encodings...")
modified_data_list = []
for data in tqdm(dataset):
    # Adds in_degree and out_degree to the data object
    mod = precalculate_custom_attributes(data, max_in_degree=MAX_IN_DEGREE, max_out_degree=MAX_OUT_DEGREE)
    modified_data_list.append(mod)

class ModifiedDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list        
    def len(self):
        return len(self.data_list)    
    def get(self, idx):
        return self.data_list[idx]

modified_dataset = ModifiedDataset(modified_data_list)
train_ids, test_ids = train_test_split(list(range(len(modified_dataset))), test_size=0.2, random_state=42)

train_loader = DataLoader(Subset(modified_dataset, train_ids), batch_size=8, shuffle=True)
test_loader = DataLoader(Subset(modified_dataset, test_ids), batch_size=8)

# --- 3. INITIALIZE MoE MODEL ---
model = MoEGraphormer(
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
    num_experts=NUM_EXPERTS,
    top_k=TOP_K
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_function = nn.CrossEntropyLoss()

# --- 4. TRAINING LOOP ---


for epoch in range(10):
    model.train()
    total_loss = 0.0
    correct = 0
    epoch_start = time.time()
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        # Dynamic path calculation to avoid IndexError
        _, _, npl, ept, epl = precalculate_paths(batch, max_path_distance=MAX_PATH_DISTANCE)
        batch.node_paths_length = npl.to(DEVICE)
        batch.edge_paths_tensor = ept.to(DEVICE)
        batch.edge_paths_length = epl.to(DEVICE)
        batch = batch.to(DEVICE)

        optimizer.zero_grad()

        # Handle potential MoE return values (x, and list of router_logits)
        # Note: If your forward returns only x, remove the router_logits part
        result = model(batch)
        if isinstance(result, tuple):
            node_embeddings, _ = result
        else:
            node_embeddings = result
        
        logits = global_mean_pool(node_embeddings, batch.batch)
        loss = loss_function(logits, batch.y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == batch.y).sum().item()
    
    print(f"Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {correct/len(train_ids):.4f}")

    # --- 5. EVALUATION ---
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for batch in test_loader:
            _, _, npl, ept, epl = precalculate_paths(batch, max_path_distance=MAX_PATH_DISTANCE)
            batch.node_paths_length, batch.edge_paths_tensor, batch.edge_paths_length = npl.to(DEVICE), ept.to(DEVICE), epl.to(DEVICE)
            batch = batch.to(DEVICE)

            result = model(batch)
            node_embeddings = result[0] if isinstance(result, tuple) else result
            
            logits = global_mean_pool(node_embeddings, batch.batch)
            test_correct += (logits.argmax(dim=1) == batch.y).sum().item()

    print(f"--> TEST ACCURACY: {test_correct / len(test_ids):.4f}\n")