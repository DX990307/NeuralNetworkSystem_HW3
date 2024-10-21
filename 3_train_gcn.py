import argparse
import torch
import torch as th
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
import dgl
import gp_apis  # Ensure this is correctly imported
from pytorch_apis import gspmmv  # Import the modified gspmmv function
import graphpy  # Import the graphpy module

# Define the model using gspmmv
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, graph, reverse=False, norm=True, device='cuda:0'):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_feats, out_feats))
        self.graph = graph
        self.reverse = reverse
        self.norm = norm
        self.device = device
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features):
        # features: (N, in_feats)
        h = torch.mm(features, self.weight)  # Linear transformation
        h = h.to(self.device)

        # Call our custom gspmmv function
        output = gspmmv(self.graph, h, h.shape[0], h.shape[1], self.reverse, self.norm, self.device)
        return output

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, graph, device='cuda:0'):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(in_feats, hidden_size, graph, device=device)
        self.layer2 = GCNLayer(hidden_size, num_classes, graph, device=device)

    def forward(self, features):
        x = self.layer1(features)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

def main():
    parser = argparse.ArgumentParser(description='Test gspmmv function')
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # Load dataset
    if args.dataset == 'cora':
        data = dgl.data.CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = dgl.data.CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = dgl.data.PubmedGraphDataset()
    elif args.dataset == 'reddit':
        data = dgl.data.RedditDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    g = data[0]
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Get edges in COO format and convert to CSR and CSC
    src, dst = g.edges(order='srcdst')
    num_nodes = g.num_nodes()
    num_edges = g.num_edges()

    # Build adjacency matrix
    adj = sp.coo_matrix((np.ones(num_edges, dtype=np.int32), (src.numpy(), dst.numpy())), shape=(num_nodes, num_nodes))
    adj_csr = adj.tocsr()
    adj_csc = adj.tocsc()

    # Convert CSR and CSC matrices to numpy arrays
    row_ptr = adj_csr.indptr.astype(np.int32)
    col_ind = adj_csr.indices.astype(np.int32)
    col_ptr = adj_csc.indptr.astype(np.int32)
    row_ind = adj_csc.indices.astype(np.int32)

    # Initialize the graph_t object
    graph = graphpy.init_graph(
        row_ptr,     # CSR offsets
        col_ind,     # CSR neighbors
        col_ptr,     # CSC offsets
        row_ind,     # CSC neighbors
        0,           # flag (unused)
        num_nodes    # Number of vertices
    )

    # Prepare features and labels
    features = g.ndata['feat'].to(device)
    labels = g.ndata['label'].to(device)
    train_mask = g.ndata['train_mask'].to(device)
    test_mask = g.ndata['test_mask'].to(device)

    # Define the model and optimizer
    model = GCN(features.shape[1], 16, data.num_labels, graph, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        logits = model(features)
        loss = loss_func(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(features)
        _, indices = torch.max(logits[test_mask], dim=1)
        correct = torch.sum(indices == labels[test_mask])
        accuracy = correct.item() * 1.0 / len(labels[test_mask])
        print('Test Accuracy: {:.4f}'.format(accuracy))

if __name__ == '__main__':
    main()
