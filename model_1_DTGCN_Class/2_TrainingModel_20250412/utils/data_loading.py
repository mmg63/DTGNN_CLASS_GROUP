# utils/data_loading.py
import torch

def load_datasets(train_path, test_path):
    train_dataset = torch.load(train_path)
    test_dataset = torch.load(test_path)
    return train_dataset, test_dataset

def normalize_graphs(dataset):
    for data, _ in dataset:
        data.x[:, 0:2] -= data.x[:, 0:2].min()
        data.x[:, 0:2] /= data.x[:, 0:2].max()
        data.x[:, 2:5] -= data.x[:, 2:5].min()
        data.x[:, 2:5] /= data.x[:, 2:5].max()

def make_graphs_undirected(dataset):
    from torch_geometric.utils import to_undirected, add_self_loops
    for data, _ in dataset:
        data.edge_index = to_undirected(data.edge_index)
        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))
        # data.y = data.y[1:]  # remove background class
