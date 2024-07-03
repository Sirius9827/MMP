import torch  
import torch.nn as nn  
import torch_geometric.nn as pyg_nn  
import dgl.nn as dglnn

class CNNModel(nn.Module):  
    def __init__(self, input_dim, num_classes):  
        super(CNNModel, self).__init__()  
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3)  
        self.relu = nn.ReLU()  
        self.maxpool = nn.MaxPool1d(kernel_size=2)  
        self.fc1 = nn.Linear(32, 64)  
        self.fc2 = nn.Linear(64, num_classes)  
  
    def forward(self, x):  
        x = self.conv1(x)  
        x = self.relu(x)  
        x = self.maxpool(x)  
        x = x.view(x.size(0), -1)  
        x = self.fc1(x)  
        x = self.relu(x)  
        x = self.fc2(x)  
        return x  
  
class RNNModel(nn.Module):  
    def __init__(self, input_dim, hidden_dim, num_classes):  
        super(RNNModel, self).__init__()  
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)  
        self.fc1 = nn.Linear(hidden_dim, 64)  
        self.relu = nn.ReLU()  
        self.fc2 = nn.Linear(64, num_classes)  
  
    def forward(self, x):  
        _, h_n = self.rnn(x)  
        x = h_n.squeeze(0)  
        x = self.fc1(x)  
        x = self.relu(x)  
        x = self.fc2(x)  
        return x    

class Transformer(nn.Module):  
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim):  
        super(Transformer, self).__init__()  
        self.embedding = nn.Linear(input_dim, d_model)  
        self.transformer = nn.Transformer(d_model, nhead, num_layers, dim_feedforward)  
        self.fc = nn.Linear(d_model, output_dim)  
  
    def forward(self, x):  
        x = self.embedding(x)  
        x = self.transformer(x)  
        x = self.fc(x)  
        return x  
    
class MLPModel(nn.Module):  
    def __init__(self, input_dim, num_classes):  
        super(MLPModel, self).__init__()  
        self.fc1 = nn.Linear(input_dim, 64)  
        self.relu = nn.ReLU()  
        self.fc2 = nn.Linear(64, 64)  
        self.fc3 = nn.Linear(64, num_classes)  
  
    def forward(self, x):  
        x = self.fc1(x)  
        x = self.relu(x)  
        x = self.fc2(x)  
        x = self.relu(x)  
        x = self.fc3(x)  
        return x  
    
class GCNModel(nn.Module):  
    def __init__(self, input_dim, hidden_dim, num_classes):  
        super(GCNModel, self).__init__()  
        self.conv1 = pyg_nn.GCNConv(input_dim, hidden_dim)  
        self.conv2 = pyg_nn.GCNConv(hidden_dim, num_classes)  
        self.relu = nn.ReLU()  
  
    def forward(self, x, edge_index):  
        x = self.conv1(x, edge_index)  
        x = self.relu(x)  
        x = self.conv2(x, edge_index)  
        return x 

class MPNNModel(nn.Module):  
    def __init__(self, input_dim, hidden_dim, num_classes):  
        super(MPNNModel, self).__init__()  
        self.edge_update = dglnn.EdgeUpdateNetwork(input_dim, hidden_dim)  
        self.node_update = dglnn.NodeUpdateNetwork(hidden_dim, hidden_dim)  
        self.readout = dglnn.GlobalAttentionPooling(nn.Linear(hidden_dim, 1))  
        self.fc = nn.Linear(hidden_dim, num_classes)  
  
    def forward(self, g, x):  
        g = g.local_var()  
        g.ndata['h'] = x  
        g.apply_edges(self.edge_update)  
        g.update_all(message_func=dgl.function.u_mul_e('h', 'm', 'm'), reduce_func=dgl.function.sum('m', 'h_neigh'))  
        g.apply_nodes(self.node_update)  
        x = self.readout(g, g.ndata['h'])  
        x = self.fc(x)  
        return x  
    
class GATModel(nn.Module):  
    def __init__(self, input_dim, hidden_dim, num_classes):  
        super(GATModel, self).__init__()  
        self.conv1 = pyg_nn.GATConv(input_dim, hidden_dim, heads=4)  
        self.conv2 = pyg_nn.GATConv(4 * hidden_dim, num_classes, heads=1, concat=False)  
        self.relu = nn.ReLU()  
  
    def forward(self, x, edge_index):  
        x = self.conv1(x, edge_index)  
        x = self.relu(x)  
        x = self.conv2(x, edge_index)  
        return x  
    
class FFNModel(nn.Module):  
    def __init__(self, input_dim, num_classes):  
        super(FFNModel, self).__init__()  
        self.fc1 = nn.Linear(input_dim, 64)  
        self.relu = nn.ReLU()  
        self.fc2 = nn.Linear(64, 64)  
        self.fc3 = nn.Linear(64, num_classes)  
  
    def forward(self, x):  
        x = self.fc1(x)  
        x = self.relu(x)  
        x = self.fc2(x)  
        x = self.relu(x)  
        x = self.fc3(x)  
        return x  
    
class SPNModel(nn.Module):  
    def __init__(self, input_dim, num_classes):  
        super(SPNModel, self).__init__()  
        self.fc1 = nn.Linear(input_dim, 64)  
        self.prod = nn.Parameter(torch.randn(64, num_classes))  
        self.sumw = nn.Parameter(torch.randn(1, num_classes))  
        self.relu = nn.ReLU()  
  
    def forward(self, x):  
        x = self.fc1(x)  
        x = self.relu(x)  
        x = torch.matmul(x, self.prod)  
        x = torch.sum(torch.exp(x), dim=1, keepdim=True)  
        x = torch.log(x) + self.sumw  
        return x 