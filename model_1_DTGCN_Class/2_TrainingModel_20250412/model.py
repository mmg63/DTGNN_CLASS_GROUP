from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import SAGEConv, BatchNorm
from torch.nn import Linear
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_,xavier_normal_, kaiming_uniform_, kaiming_normal_

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Enabling MPS for the operators that are not currently implemented for mps devices


class GraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GraphConv, self).__init__(aggr='add')  # "Add" aggregation
        self.lin = nn.Linear(in_channels, out_channels, bias=True)#.to(gpu_run())

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        x = self.lin(x)#.to(gpu_run())

        row, col = edge_index#.to(gpu_run())
        deg = degree(row, x.size(0), dtype=x.dtype)#.to(gpu_run())
        deg_inv_sqrt = deg.pow(-0.5)#.to(gpu_run())
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]#.to(gpu_run())

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)#.to(gpu_run())

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class Net(nn.Module):
    def __init__(self,num_node_features, num_classes):
        super(Net, self).__init__()
        self.conv1 = GraphConv(num_node_features, 32)
        # self.dropout = nn.Dropout(p=0.5)
        self.conv2 = GraphConv(32, 64)
        self.fc = nn.Linear(64, num_classes)
        # self.apply(self._init_weights)    

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, data):
        # x, edge_index = data.x[:, 2].unsqueeze(-1), data.edge_index
        x, edge_index = data.x, data.edge_index
        # x[:,0] = (x[:,0] - torch.min(x[:,0])) / torch.max(x[:,0])
        # x[:,1] = (x[:,1] - torch.min(x[:,1])) / torch.max(x[:,1])
        x = F.relu(self.conv1(x, edge_index))#.to(gpu_run())
        # x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))#.to(gpu_run())
        x = global_mean_pool(x, data.batch)#.to(gpu_run())
        x = self.fc(x)#.to(gpu_run())
        # return F.softmax(x, dim=1)#.to(gpu_run())
        return F.softmax(x, dim=1)#.to(gpu_run())


class ComplexGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ComplexGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        self.pool = global_mean_pool
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x[:,0] = (x[:,0] - torch.min(x[:,0])) / torch.max(x[:,0])
        x[:,1] = (x[:,1] - torch.min(x[:,1])) / torch.max(x[:,1])

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.pool(x, batch)  # Aggregate node embeddings to graph-level
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)


class GATMultiHead(torch.nn.Module):
    def __init__(self, in_channels=5, hidden_channels=None, out_channels=10, heads=None):
        super(GATMultiHead, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        if heads is None:
            heads = [2, 2, 2]
        if hidden_channels is None:
            hidden_channels = [32, 64, 64]
        self.conv1 = GATConv(in_channels=self.in_channels,
                             out_channels=hidden_channels[0],
                             heads=heads[0],)
                             # dropout=0.6)
        self.bn1 = BatchNorm(hidden_channels[0] * heads[0])
        self.conv2 = GATConv(in_channels=hidden_channels[0] * heads[0],
                             out_channels=hidden_channels[1],
                             heads=heads[1],)
                             # dropout=0.6)
        self.bn2 = BatchNorm(hidden_channels[1] * heads[1])
        self.conv3 = GATConv(in_channels=hidden_channels[1] * heads[1],
                             out_channels=hidden_channels[2],
                             heads=heads[2],)
                             # dropout=0.6)
        self.bn3 = BatchNorm(hidden_channels[2] * heads[2])
        self.flatten = Linear(in_features=hidden_channels[2] * heads[2],
                              out_features=32)
        self.lin1 = Linear(in_features=32,
                           out_features=32)
        self.bn4 = BatchNorm(32)
        self.lin2 = Linear(in_features=32,
                           out_features=self.out_channels)

    def forward(self, data):
        x, edge_index = data.x[:,0:3], data.edge_index
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)  # Adding BatchNormalization to check that this can repair the arrucary rate or not?!!!
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.3)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.3)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.3)
        
        x_out_conv = global_mean_pool(x, data.batch)
        x = self.flatten(x_out_conv)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.3) 
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x_out_conv, F.log_softmax(x, dim=1)
    

class HGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=10, heads=None):
        super(HGNN, self).__init__()
        if heads is None:
            heads = [2, 2, 2]
        if hidden_channels is None:
            hidden_channels = [32, 64, 64]
        self.conv1 = GATConv(in_channels=in_channels, out_channels=hidden_channels[0], heads=heads[0], concat=True)
        self.conv2 = GATConv(in_channels=hidden_channels[0] * heads[0], out_channels=hidden_channels[1], heads=heads[1], concat=True)
        self.conv3 = GATConv(in_channels=(hidden_channels[1] * heads[1]) * 2, out_channels=hidden_channels[2], heads=heads[2], concat=False)
        self.residual = nn.Linear(in_channels, hidden_channels[2] * heads[2])
        # Calculate the total concatenated feature size
        total_concat_features = (in_channels + (hidden_channels[0] * heads[0]) +
                                 (hidden_channels[1] * heads[1]) +
                                 hidden_channels[2])
        self.classifier = nn.Sequential(
            # Adjust the input feature size of the first linear layer
            nn.Linear(in_features=total_concat_features, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=out_channels)
        )
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Initial GAT layers with residual connection
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        # Residual connection and concatenation
        res = self.residual(x)
        # x2_res = torch.cat([x2, res], dim=1)
        x2_res = torch.cat([x2, res], dim=1)
        # Third GAT layer after concatenation
        x3 = F.relu(self.conv3(x2_res, edge_index))
        # Concatenation of original, first two GATs, and third GAT outputs
        x_final = torch.cat([x, x1, x2, x3], dim=1)
        # Readout layer
        x_readout = global_mean_pool(x_final, data.batch)
        # Classifier
        x_out = self.classifier(x_readout)
        return F.log_softmax(x_out, dim=-1)


class ArcFaceLoss(nn.Module):
    def __init__(self, s=64.0, m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cos_theta, labels):
        theta = torch.acos(torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = one_hot * target_logits + (1.0 - one_hot) * cos_theta
        output *= self.s
        return output

class CombinedLoss(nn.Module):
    def __init__(self, s=64.0, m=0.5, weight=None, reduction='mean'):
        super(CombinedLoss, self).__init__()
        self.arcface_loss = ArcFaceLoss(s, m)
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, logits, labels):
        cos_theta = F.normalize(logits)
        arcface_logits = self.arcface_loss(cos_theta, labels)
        ce_loss = self.cross_entropy(logits, labels)
        return ce_loss + arcface_logits.mean()


def compute_norm(self, edge_index, num_ent):
    row, col = edge_index
    deg = degree(col, num_ent)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
    return norm


def get_param(shape):
    param = Parameter(torch.Tensor(*shape));
    #kaiming_uniform_(param.data, a=0, mode='fan_out', nonlinearity='leaky_relu')
    kaiming_uniform_(param.data)
    return param

class SuperpixelGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, num_ent=None, act=lambda x: x, params=None):
        super(self.__class__, self).__init__()

        self.p = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ent =num_ent
        self.num_heads=2
        self.act = act
        self.device = None
 
#         self.w1_loop = get_param((in_channels, out_channels))
        self.w1_out = get_param((in_channels, out_channels))
        self.w2_out = get_param((in_channels, out_channels))
        self.w3_out = get_param((in_channels, out_channels))
        self.w4_out = get_param((in_channels, out_channels))
        self.w5_out = get_param((in_channels, out_channels))
        self.w6_out = get_param((in_channels, out_channels))
        self.w7_out = get_param((in_channels, out_channels))
        self.w8_out = get_param((in_channels, out_channels))
        self.dropout = torch.nn.Dropout(0.3)
        self.bn1 = torch.nn.BatchNorm1d(in_channels)
#         self.projection = nn.Linear(in_channels, 1)
#         self.a = get_param(torch.Tensor(2*out_channels, 1))
        self.leakyrelu = nn.LeakyReLU(0.2)


    def forward(self, x,  edge_index, batch):
        #self.edge_index = edge_index
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        
        # Compute normalization
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # self.norm =  self.compute_norm(edge_index, self.num_ent)
        
        
        out = self.propagate(edge_index, x=x, norm=norm)
#         out= self.agg_multi_head(out, 1)
        
        
        out=self.gelu(out)
        
        out=self.bn1(out)
        

        out1=torch.mm(out,self.w1_out)
        out2=torch.mm(out,self.w2_out)
        out3=torch.mm(out,self.w3_out)
        out4=torch.mm(out,self.w4_out)
        out5=torch.mm(out,self.w1_out)
        out6=torch.mm(out,self.w2_out)
        out7=torch.mm(out,self.w3_out)
        out8=torch.mm(out,self.w4_out)
        
        out=(out1+out2+out3+out4+out5+out6+out7+out8)/8
        
        out = global_mean_pool(out, batch=batch)
        #out_prime=torch.cat((out1, out2, out3,out4), 1)
        #out_prime=out_prime.view(-1,4,8)
        return out
    
    def gelu(self, x):
        return 0.5 * x * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * x**3)))
    
    def message(self, x_j, norm):
        
        return norm.view(-1, 1) * x_j


class GATMultiHead_residual(torch.nn.Module):
    def __init__(self, in_channels=5, hidden_channels=None, out_channels=10, heads=None):
        super(GATMultiHead_residual, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        if heads is None:
            heads = [2, 2, 2]
        if hidden_channels is None:
            hidden_channels = [32, 64, 64]
        
        self.conv1 = GATConv(in_channels=self.in_channels,
                             out_channels=hidden_channels[0],
                             heads=heads[0])
        self.bn1 = BatchNorm(hidden_channels[0] * heads[0])
        
        self.conv2 = GATConv(in_channels=hidden_channels[0] * heads[0],
                             out_channels=hidden_channels[1],
                             heads=heads[1])
        self.bn2 = BatchNorm(hidden_channels[1] * heads[1])
        
        self.conv3 = GATConv(in_channels=hidden_channels[1] * heads[1],
                             out_channels=hidden_channels[2],
                             heads=heads[2])
        self.bn3 = BatchNorm(hidden_channels[2] * heads[2])
        
        self.flatten = nn.Linear(in_features=hidden_channels[2] * heads[2],
                                 out_features=32)
        self.bn4 = BatchNorm(32)
        self.lin1 = nn.Linear(in_features=32, out_features=32)
        self.lin2 = nn.Linear(in_features=32, out_features=self.out_channels)

        # Adding projection layers for residual connections
        self.project1 = nn.Linear(self.in_channels, hidden_channels[0] * heads[0]) if self.in_channels != hidden_channels[0] * heads[0] else None
        self.project2 = nn.Linear(hidden_channels[0] * heads[0], hidden_channels[1] * heads[1]) if hidden_channels[0] * heads[0] != hidden_channels[1] * heads[1] else None
        self.project3 = nn.Linear(hidden_channels[1] * heads[1], hidden_channels[2] * heads[2]) if hidden_channels[1] * heads[1] != hidden_channels[2] else None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        residual = x
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.3)
        if self.project1 is not None:
            residual = self.project1(residual)
        x += residual  # Residual connection

        residual = x
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.3)
        if self.project2 is not None:
            residual = self.project2(residual)
        x += residual  # Residual connection

        residual = x
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.3)
        if self.project3 is not None:
            residual = self.project3(residual)
        x += residual  # Residual connection
        
        x_out_conv = global_mean_pool(x, data.batch)
        x = self.flatten(x_out_conv)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.3)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.3)  # Adding dropout after lin1
        x = self.lin2(x)
        # x = F.relu(self.lin1(x))
        # x = self.lin2(x)
        
        # return x_out_conv, F.log_softmax(x, dim=1)
        # we just return the logits and use the BCEwithlogits loss function which combines sigmoid and binary cross entropy in one step, for better numerical stability
        return x  