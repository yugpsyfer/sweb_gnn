import torch
from torch_geometric.loader import DataLoader
from GNN.dataset import GraphData
from torch_geometric.nn import GAE, VGAE

from GNN.autoencoder import VariationalLinearEncoder, VariationalGCNEncoder, LinearEncoder, GCNEncoder


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 30
lr = 1e-2


dataset = GraphData(root="./data")
loader = DataLoader(dataset, batch_size=64)

in_channels, out_channels = dataset.num_features, 16

model = GAE(GCNEncoder(in_channels, out_channels))

# model = GAE(LinearEncoder(in_channels, out_channels))

# model = VGAE(VariationalGCNEncoder(in_channels, out_channels))

# model = VGAE(VariationalLinearEncoder(in_channels, out_channels))


model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    l=0
    model.train()
    optimizer.zero_grad()
    for idx in range(1000):
        batch = dataset.get(idx)
        batch = batch.to(device)
        batch.x = batch.x.to(torch.float)
        z = model.encode(batch.x, batch.edge_index)
        loss = model.recon_loss(z, batch.edge_index)  #POSITIVE EDGE TO TRAIN AGAINST
        # if args.variational:
        #     loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        l+=loss.item()
    return float(l)


# @torch.no_grad()
# def test(data):
#     model.eval()
#     z = model.encode(data.x, data.edge_index)
#     return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)


for epoch in range(1, epochs + 1):
    loss = train()
    # auc, ap = test(test_data)
    print(f'Epoch: {epoch:03d}, loss: {loss:.4f}')