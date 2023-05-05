import torch
from torch_geometric.loader import DataLoader

from torch_geometric.nn import GAE, VGAE

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


from GNN.dataset import GraphData
from GNN.autoencoder import VariationalLinearEncoder, VariationalGCNEncoder, LinearEncoder, GCNEncoder


CUDA_LAUNCH_BLOCKING=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = GraphData(root="./data")
loader = DataLoader(dataset, batch_size=10)

in_channels, out_channels = dataset.num_features, 64

model = GAE(LinearEncoder(in_channels, out_channels))

model = torch.load('checkpoint/model.pt').to(device)
model.eval()
encoder = model.encoder

graph_embeddings = []

for i in range(9999):
    batch = dataset.get(i)

    batch = batch.to(device)

    out = encoder(batch.x, batch.edge_index)
    
    d = out.view(-1,71,64)
    edge_indexes = batch.edge_index.flatten()  #reshape(2*71,-1).transpose(1,0)

    # for idx, edges in enumerate(edge_indexes):
    x = torch.sum(d[:, edge_indexes, :], dim=1)  #calulating dimesnions by suming all the nodes.

    graph_embeddings.append(x) 


graph_embeddings = torch.stack(graph_embeddings).detach().cpu().squeeze()


dim_reduced = TSNE(n_components=2).fit_transform(graph_embeddings)
# dim_reduced = PCA(n_components=2).fit_transform(graph_embeddings)


x = [i[0] for i in dim_reduced]
y = [i[1] for i in dim_reduced]

plt.figure(figsize=(10,10))
plt.scatter(x, y)
plt.show()