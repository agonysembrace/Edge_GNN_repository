import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from dgl.data import DGLDataset
import dgl
import dgl.function as fn
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, ELU, Sigmoid, BatchNorm1d as BN, ReLU6 as ReLU6
from dgl.nn import HeteroGraphConv
import scipy
from dgl.utils import expand_as_pair
import matplotlib.pyplot as plt
# import tensorflow as tf
# 维度梳理：（忽略batch_size维度，也就是不管第一维，并且因为mlp不能处理矩阵，要展平，处理逻辑时恢复回来）
# 初始数据：train_B*1，train_K*1, train_B*train_K*2 (实虚部)
# 进入pre_mlp: train_B*dim, train_K*dim, train_B*train_K*2 *dim
# 更新node： 
# 对于每一个AP: 邻居node 1*dim，邻居边：2*dim，合并起来进mlp，->3*dim，所有结果聚合 ->1*dim，和自己拼接进mlp ->2*dim,输出 1*dim
# 对于每一个UE: 同理
# 对于每一条边: …….
# torch.autograd.set_detect_anomaly(True)
# torch.autograd.set_detect_anomaly(True)

# 500 epoch 3.9
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())
torch.set_printoptions(precision=8)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
# torch.set_default_dtype(torch.double)
bias = True
dimension = 64
# 指定训练轮数
epoch_num = 5000
# 指定批大小
batch_size = 200
# 指定学习率
learning_rate = 1e-3
# 指定高斯白噪声
var_noise = 1
# 指定激活函数
# active_fun = nn.Tanh()
# active_fun = nn.LeakyReLU()
active_fun = nn.ReLU()
# 瑞丽信道系数
c = 1/np.sqrt(2)
# PowerBudget /W
P_max = 1
# backhaulBudget /bps
# C_max = 1.1
# sigma
sigma = var_noise
# AP数量，单天线 train和test显然是可以不一样的
train_B = 16
test_B = 16
# UE数量
train_K = 8
test_K = 8
# 训练集
train_layouts = 100
# 测试集
test_layouts = 100
# 模拟路损
beta = 0.6

hid_dim = 64
# 指示函数因子
delta = 1e-5
data_from_mat = scipy.io.loadmat('lable.mat')
Hd = np.transpose(data_from_mat['Hd1'],(0,2,1))
train_channel_rel = np.real(Hd[0:train_layouts,:,:])
train_channel_ima = np.imag(Hd[0:train_layouts,:,:])
train_C_max = data_from_mat['C_max'][0:train_layouts]
train_lable = data_from_mat['binarys'][0:train_layouts]

test_channel_rel = np.real(Hd[train_layouts:train_layouts+test_layouts,:,:])
test_channel_ima = np.imag(Hd[train_layouts:train_layouts+test_layouts,:,:])
test_C_max = data_from_mat['C_max'][train_layouts:train_layouts+test_layouts]
test_lable = data_from_mat['binarys'][train_layouts:train_layouts+test_layouts]
# 创建信道，因为不能直接输入复数进入神经网络，我们输入信道的模值
# 创建信道实部
# train_channel_rel = beta * np.random.randn(train_layouts, train_B, train_K)
# # 创建信道虚部
# train_channel_ima = beta * np.random.randn(train_layouts, train_B, train_K)
# train_lable =  np.random.rand(train_layouts,train_B,train_K).round()
# train_C_max = 10+np.random.randn(train_layouts, train_B)
# test_channel_rel = beta * np.random.randn(test_layouts, test_B, test_K)
# test_channel_ima = beta * np.random.randn(test_layouts, test_B, test_K)
# test_lable =  np.random.rand(test_layouts,test_B, test_K).round()
# test_C_max =  10+np.random.randn(test_layouts, test_B)

# # scipy.io.savemat('test_channel.mat',{'test_channel':test_channel_rel + 1j* test_channel_ima})
# train_channel = scipy.io.loadmat('test_200_channel.mat')
# test_channel_rel = np.transpose(np.real(train_channel['Hd1']),axes=(0, 2, 1))
# test_channel_ima = np.transpose(np.imag(train_channel['Hd1']),axes=(0, 2, 1))

def normalize_one_tensor(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor
# 构建数据集，数据构成为：信道实虚部，APUE数量
class MyDataset(DGLDataset):
    def __init__(self, rel, ima, lable, C_max, BK):
        self.rel = rel
        self.ima = ima
        self.lable = lable
        self.C_max = C_max
        # sinr 分子（信道增益
        # self.direct = torch.tensor(direct, dtype = torch.float)
        self.BK = BK
        self.get_cg()
        super().__init__(name='beamformer_design')

    def build_graph(self, idx):
        edge_feature_rel = self.rel[idx,:,:].reshape((self.BK[0]*self.BK[1],1))
        edge_feature_ima = self.ima[idx,:,:].reshape((self.BK[0]*self.BK[1],1))
        # edge_features = (torch.tensor(np.concatenate((edge_feature_rel, edge_feature_ima), axis = -1), dtype = torch.float))
        # 归一化
        edge_features = normalize_one_tensor(torch.tensor(np.concatenate((edge_feature_rel, edge_feature_ima), axis = -1), dtype = torch.float))
        graph = dgl.heterograph( {('AP','AP2UE','UE'): self.adj,
                                  ('UE','UE2AP','AP'): self.adj_t}) 
        
        ## AP数据为功率budget
        graph.nodes['AP'].data['feat'] = torch.unsqueeze(normalize_one_tensor(torch.tensor(self.C_max[idx,:], dtype = torch.float)),dim=-1)
        ## AP数据为功率budget
        # graph.nodes['AP'].data['feat'] = torch.unsqueeze((torch.tensor(self.C_max[idx,:], dtype = torch.float)),dim=-1)
        ## UE数据为sigma^2
        graph.nodes['UE'].data['feat'] = torch.unsqueeze(1 * torch.ones(self.BK[1]),dim=-1) 
        # bool_tensor=torch.ones((8,2),dtype=bool)
        # bool_tensor[1,0] = False
        # bool_tensor[4,0] = False
        # edge_features = edge_features[bool_tensor].view(7,2)
        graph.edges['AP2UE'].data['feat'] = edge_features
        graph.edges['AP2UE'].data['eid'] = torch.arange(1,self.BK[0]*self.BK[1]+1)
        graph.edges['UE2AP'].data['feat'] = edge_features
        graph.edges['UE2AP'].data['eid'] = torch.arange(1,self.BK[0]*self.BK[1]+1)
        return graph
    # 构建出边的链接关系，表示从每一个AP到每一个UE都有连接，
    def get_cg(self):
        self.adj = []; 
        self.adj_t = []
        for i in range(0,self.BK[0]):
            for j in range(0,self.BK[1]):
                self.adj.append([i,j])
                self.adj_t.append([j,i])
        # self.adj.remove([1,0])
        # self.adj_t.remove([0,1])

    def __len__(self):
        # TODO 之前是len(direct)，替换成信道可以吗？
        return len(self.ima)
    # 获取一条数据
    def __getitem__(self, index):
       # 返回指定索引的图、信道实虚部
        return self.graph_list[index], self.lable[index]
    # 创建一个大的图list的函数
    def process(self):
        n = len(self.rel)
        self.graph_list = []
        for i in range(n):
            graph = self.build_graph(i)
            self.graph_list.append(graph)


# DGL collate function 用于批处理时，构成一个小批次     
def collate(samples):
    graphs, label  = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs) 
    return batched_graph, torch.stack([torch.from_numpy(i) for i in label]) 

# 以上构建完图结构后，下面开始构建图神经网络，这里我们把神经网络的输出看做两个向量拼接成的矩阵，所以维度(:,:,0)表示实部 (:,:,1)表示虚部
# 根据神经网络的输出（beamformer）和我们固定的信道信息 计算合速率
def rate_loss(link, label, test_mode = False):

    threshold = 0.5 # 设置阈值

    binary_output = torch.zeros_like(link) # 创建与模型输出相同大小的全零矩阵

    # 使用条件判断将大于等于阈值的元素赋予1，小于阈值的元素赋予0
    binary_output[link >= threshold] = 1
    binary_output = binary_output.squeeze()

    a = (binary_output==label)
    false_count = (~a).sum().item()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(link.squeeze(),label.to(torch.float32))
    if test_mode:
        return loss
    else:
        return torch.mean(loss)

# 定义整体的神经网络结构，输入层MLP（将数据转换成APnode、UEnode、edge，维度全是64）->多个中间更新层MLP（保持维度为64）->输出层MLP
class HetGNN(nn.Module):
    def __init__(self):
        super(HetGNN, self).__init__()
        
        self.preLayer = PreLayer() # 预处理层
        self.update_layers = nn.ModuleList([UpdateLayer()] * 2)  # 更新层，这里使用了2个更新层
        self.postprocess_layer = PostLayer(mlp_post) # 后处理层
    
    # 输入网络时，节点特征数量为batchsize*AP batchsize*UE，边特征数量为
    def forward(self, graph):
        self.preLayer(graph)
        for update_layer in self.update_layers:
            update_layer(graph)
        output = self.postprocess_layer(graph)
        output = output.view(graph.batch_size,
                            graph.number_of_nodes('AP')//graph.batch_size,
                            graph.number_of_nodes('UE')//graph.batch_size,
                            -1)
        return output
    
# 本网络中的所有MLP保持为3个线性层，配合指定的激活函数，active_fun在开头配置
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        hidden_dim = hid_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias)
        self.relu1 = active_fun
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias)
        self.linear3 = nn.Linear(hidden_dim, output_dim, bias)
        self.bn = nn.BatchNorm1d(output_dim,momentum=0.01,)
        # self.bn2 = nn.BatchNorm1d(output_dim,momentum=0.05,)
        self.ln = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu1(x)
        x = self.linear3(x)
        # x = self.ln(x)
        # shape = x.size()
        # if(len(shape)>2):
        #     for i in range (shape[1]):
        #         tmp = x[:,i,:].clone()
        #         x[:,i,:] = self.bn(tmp.squeeze()).clone()
        x = self.relu1(x)
        shape = x.size()
        x = x.view(-1,shape[-1])
        x = self.bn(x)
        x = x.view(shape)
        # if(len(shape)>2):
        #     for i in range (shape[1]):
        #         tmp = x[:,i,:].clone()
        #         x[:,i,:] = self.bn(tmp.squeeze()).clone()
        return x
    
class MLP_post(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_post, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim, bias)    
        self.linear2 = nn.Linear(output_dim, output_dim, False)    
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        # x  = self.relu(x)
        x = self.linear1(x)
        x = self.sigmoid(x)
        # x = self.tanh(x)
        # x = self.dropout(x)
        # x  = self.relu(x)
        # x = self.linear2(x)
        # x = GLUA.forward(x)
        return x
    

class MLP_One_Layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_One_Layer, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim, bias)
        self.active_fun = active_fun
        self.bn = nn.BatchNorm1d(output_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.active_fun(x)
        x = self.bn(x)
        # shape = x.size()
        # x = x.view(-1,shape[-1])
        # x = self.bn(x)
        # x = x.view(shape)
        return x
# 自定义可学习的掩码层（仅允许取值为0或1）
class BinaryMaskLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BinaryMaskLayer, self).__init__()
        self.mask = nn.Parameter(torch.randn(test_K),True)

    def forward(self, x):
        binary_mask = torch.clamp(torch.sign(self.mask), min=0)   # 使用torch.clamp和torch.sign进行二值化处理
        # binary_mask_expanded = binary_mask.unsqueeze(0).unsqueeze(-1)
        masked_output = x * binary_mask   # 将输入张量与二进制掩码相乘得到输出结果
        return masked_output
MyBinaryMaskLayer = BinaryMaskLayer(dimension,dimension)

# 本网络中的mlp涉及到多个，每一个mlp应该为不同的实例，这样才能保证参数分别进行更新
mlp_pre_ap = MLP_One_Layer(1,1 * dimension) 
mlp_pre_ue = MLP_One_Layer(1,1 * dimension) 
mlp_pre_edge = MLP_One_Layer(2,dimension) 

# mlp for AP update
mlp_update_1 = MLP(dimension*2,dimension) 
mlp_update_2 = MLP(dimension*2,dimension) 
# mlp for UE update
mlp_update_3 = MLP(dimension*2,dimension) 
mlp_update_4 = MLP(dimension*2,dimension) 

# mlp for EDGE_UE
mlp_update_5 = MLP(dimension*2,dimension) 
# mlp for EDGE_AP
mlp_update_6 = MLP(dimension*2,dimension) 
# mlp for EDGE
mlp_update_7 = MLP(dimension*2,dimension) 

mlp_post = MLP_post(dimension,1) 
 
# 预处理层，将初始的节点特征和边特征，进行维度映射
class PreLayer(nn.Module):
    def __init__(self):
        super(PreLayer, self).__init__()
        self.AP_pre_MLP = mlp_pre_ap
        self.UE_pre_MLP = mlp_pre_ue
        self.EDGE_pre_MLP = mlp_pre_edge
    def forward(self, graph):
        graph.nodes['AP'].data['hid'] = self.AP_pre_MLP(graph.nodes['AP'].data['feat'])
        graph.nodes['UE'].data['hid'] = self.UE_pre_MLP(graph.nodes['UE'].data['feat'])
        graph.edges['AP2UE'].data['hid'] = self.EDGE_pre_MLP(graph.edges['AP2UE'].data['feat'])
        graph.edges['UE2AP'].data['hid'] = self.EDGE_pre_MLP(graph.edges['UE2AP'].data['feat'])

class PostLayer(nn.Module):
    def __init__(self, mlp):
        super(PostLayer, self).__init__()
        self.post_mlp = mlp
    def forward(self, graph):
        return self.post_mlp(graph.edata['hid'][('AP','AP2UE','UE')])

class APConv(nn.Module):
    def __init__(self, mlp1, mlp2, **kwargs):
        super(APConv, self).__init__()
        self.mlp1 = mlp1
        self.mlp2 = mlp2

    def forward(self, g):
        # 对每个AP节点，获取其临边
        def message_func(edges):
            neighbor_ue_features = edges.src['hid']
            edge_feature = edges.data['hid']
            return {'neighbor_ue_features': neighbor_ue_features,
                    'edge_feature': edge_feature}

        def reduce_func(nodes):
            AP_mlp_result = self.mlp1(torch.cat((nodes.mailbox['edge_feature'],
                                                  nodes.mailbox['neighbor_ue_features']), dim=-1))
            # 聚合邻居ue数据，取Max
            # agg, _ = torch.max(AP_mlp_result, dim=1)
            agg = torch.sum(AP_mlp_result, dim=1)

            new_AP_feat = self.mlp2(torch.cat((nodes.data['hid'], agg), dim=-1))
            return {'new': new_AP_feat}
        # g.send_and_recv(g.edges(etype=('AP','AP2UE','UE')), message_func=message_func,reduce_func = reduce_func,apply_node_func = None,etype= ('AP','AP2UE','UE'))
        g.send_and_recv(g.edges(etype=('UE2AP')), message_func=message_func,reduce_func = reduce_func,apply_node_func = None,etype= ('UE','UE2AP','AP'))

            
class UEConv(nn.Module):
    def __init__(self, mlp1, mlp2, **kwargs):
        super(UEConv, self).__init__()
        self.mlp1 = mlp1
        self.mlp2 = mlp2

    def forward(self, g):
        # 对每个UE节点，获取其临边
        def message_func(edges):
            neighbor_ap_features = edges.src['hid']
            edge_feature = edges.data['hid']
            return {'neighbor_ap_features': neighbor_ap_features,
                    'edge_feature': edge_feature}

        def reduce_func(nodes):
            UE_mlp_result = self.mlp1(torch.cat((nodes.mailbox['edge_feature'],
                                                  nodes.mailbox['neighbor_ap_features']), dim=-1))
            # agg, _ = torch.max(UE_mlp_result, dim=1)
            agg = torch.sum(UE_mlp_result, dim=1)

            new_UE_feat = self.mlp2(torch.cat((nodes.data['hid'], agg), dim=-1))
            return {'new': new_UE_feat}

        g.send_and_recv(g.edges(etype=('AP2UE')), message_func=message_func,reduce_func = reduce_func,apply_node_func = None,etype= ('AP2UE'))



# 流程：对于一条边来说：
# 1、所连AP与AP的所有边（去除自己）拼接进入mlp
# 2、所连UE与UE的所有边（去除自己) 拼接进入mlp
# 3、所有mlp结果聚合得到agg
# 4、和自己的边特征拼接进入mlp得到新特征
class EgdeConv(nn.Module):
    def __init__(self, mlp1, mlp2, mlp3,  **kwargs):
        super(EgdeConv, self).__init__()
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.mlp3 = mlp3
    
    def forward(self, g):
        # 对每个AP节点，获取其临边
        def message_func_ap(edges):
            edge_feature = edges.data['hid']
            return {'edge_feature': edge_feature,'eid':edges.data['eid']}

        def reduce_func_ap(nodes):
            AP_mlp_result = self.mlp1(torch.cat((nodes.data['hid'].unsqueeze(1).expand(-1,nodes.mailbox['edge_feature'].size(1),-1),
                                                    nodes.mailbox['edge_feature']), dim=-1)).squeeze()
            # 每个节点存储自己临边和自己特征的聚合最大值
            # agg, _ = torch.max(AP_mlp_result, dim=1)
            # 每个节点存储自己特征和邻边特征的最大拼接聚合
            return {'ap_result': AP_mlp_result,'eid':nodes.mailbox['eid'].squeeze()}
        def message_func_ue(edges):
            edge_feature = edges.data['hid']
            return {'edge_feature': edge_feature,'eid':edges.data['eid']}

        def reduce_func_ue(nodes):
            UE_mlp_result = self.mlp2(torch.cat((nodes.data['hid'].unsqueeze(1).expand(-1,nodes.mailbox['edge_feature'].size(1),-1),
                                                    nodes.mailbox['edge_feature']), dim=-1)).squeeze()
            # 每个节点存储自己临边和自己特征的聚合最大值
            # agg, _ = torch.max(UE_mlp_result, dim=1)
            # 每个节点存储自己特征和邻边特征的最大拼接聚合
            return {'ue_result':UE_mlp_result,'eid':nodes.mailbox['eid'].squeeze()}
        def message_func_edge(edges):
            # 获取掩码，消息的边来源不能等于当前边
            mask_ap = torch.transpose((torch.transpose(edges.src['eid'],0,1) == edges.data['eid']),0,1)
            mask_ap = mask_ap.unsqueeze(-1).expand(-1,-1,dimension)
            ap_info = edges.src['ap_result']
            ap_info[mask_ap] = 0
            mask_ue = torch.transpose((torch.transpose(edges.dst['eid'],0,1) == edges.data['eid']),0,1)
            mask_ue = mask_ue.unsqueeze(-1).expand(-1,-1,dimension)
            ue_info = edges.dst['ue_result']
            ue_info[mask_ue] = 0
            # agg = torch.max(torch.cat((ap_info,ue_info),dim = 1),dim = 1)[0]
            agg = torch.sum(torch.cat((ap_info,ue_info),dim = 1),dim = 1)
            return {'new': self.mlp3(torch.cat((edges.data['hid'],agg),dim = -1))}
    # g.send_and_recv(g.edges(etype=('AP','AP2UE','UE')), message_func=message_func,reduce_func = reduce_func,apply_node_func = None,etype= ('AP','AP2UE','UE'))
        g.send_and_recv(g.edges(etype=('UE', 'UE2AP', 'AP')), message_func = message_func_ap, reduce_func = reduce_func_ap, apply_node_func = None,etype= ('UE', 'UE2AP', 'AP'))
        g.send_and_recv(g.edges(etype=('AP','AP2UE','UE')), message_func = message_func_ue, reduce_func = reduce_func_ue, apply_node_func = None,etype= ('AP','AP2UE','UE'))
        g.apply_edges(message_func_edge ,etype =('AP','AP2UE','UE'))
        # g.apply_edges(message_func_edge_1 ,etype =('UE', 'UE2AP', 'AP'))
        g.edges['UE2AP'].data['new'] = g.edges['AP2UE'].data['new']
        g.nodes['UE'].data['hid'] = g.nodes['UE'].data['new']
        g.nodes['AP'].data['hid'] = g.nodes['AP'].data['new']
        g.edges['UE2AP'].data['hid'] = g.edges['UE2AP'].data['new']
        g.edges['AP2UE'].data['hid'] = g.edges['AP2UE'].data['new']



# 更新层，对于节点和边的更新有着不同的策略,这与messagePassingGNN产生了明显的不同
class UpdateLayer(nn.Module):
    def __init__(self):
        super(UpdateLayer, self).__init__()
        self.APConv = APConv(mlp_update_1, mlp_update_2)
        self.UEConv = UEConv(mlp_update_3, mlp_update_4)
        self.EgdeConv = EgdeConv(mlp_update_5,mlp_update_6,mlp_update_7)
    
    def forward(self, graph):
        self.APConv(graph)
        self.UEConv(graph)
        self.EgdeConv(graph)
        

# 定义归一化函数，用于控制最后一层mlp的输出，不超过最大功率
def norm_func(real, imag):
    real_normalized = real
    imag_normalized = imag
    beamformer_complex = torch.complex(real, imag)
    dims = beamformer_complex.size()
    for i in range(0,dims[0]):
        # if(torch.norm(beamformer_complex[i,:,:].clone(), dim=1,p=2)!=0):
        beamformer_complex[i,:,:] = beamformer_complex[i,:,:].clone() / torch.unsqueeze(torch.norm(beamformer_complex[i,:,:].clone(), dim=1,p=2),dim=-1)
    real_normalized = torch.real(beamformer_complex)
    imag_normalized = torch.imag(beamformer_complex)
    return real_normalized, imag_normalized
# 定义输入维度、隐藏层维度和输出维度




# 创建HetGNN模型实例
hetgnn_model = HetGNN()
# print(hetgnn_model)

model = hetgnn_model

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-3)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.00001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)




def test(loader):
    model.eval()
    correct = 0
    for (g, lable) in loader:
        optimizer.zero_grad()

        K = lable.shape[1] # 6
        bs = len(g.nodes['AP'].data['feat'])//K
        output = model(g)
        loss = rate_loss(output, lable)
        correct += loss.item() * bs
    return correct / len(loader.dataset)
def main():
    # 创建数据集，实际上是把训练数据和测试数据构建成图的过程
    train_data = MyDataset(train_channel_rel, train_channel_ima, train_lable,train_C_max,(train_B, train_K))
    test_data = MyDataset(test_channel_rel, test_channel_ima,test_lable, test_C_max ,(test_B, test_K))

    # train_data = train_data 
    # 检查一下异构图的两类结点名称
    print(train_data[0][0].ntypes)

    # 批处理，这里调用之前定义的collate函数
    
    train_loader = DataLoader(train_data, batch_size, shuffle=True, collate_fn=collate)
    # train_loader = DataLoader(train_data, batch_size, shuffle=True, collate_fn=collate,num_workers=4,pin_memory=True)、
    test_loader = DataLoader(test_data, test_layouts, shuffle=True, collate_fn=collate)
    def train(epoch):
        """ Train for one epoch. """
        model.train()
        loss_all = 0
        for batch_idx, (g, lable) in enumerate(train_loader):
            K = lable.shape[1] # 6
            bs = len(g.nodes['AP'].data['feat'])//K
            output = model(g)
            loss = rate_loss(output, lable)
            loss.requires_grad_(True)
            optimizer.zero_grad()
            loss.backward()
        
            loss_all += loss.item() * bs
            optimizer.step()
        return loss_all / len(train_loader.dataset)
    
    for epoch in range(0, epoch_num):
        if(epoch % 10 == 1):
            with torch.no_grad():
                # train_rate = train(train_loader)
                test_rate = test(test_loader)
            print('Epoch {:03d},  ——————————  Test Rate: {:.4f}'.format(epoch,test_rate))
        train_rate = train(epoch)
        print('Epoch {:03d}, Train Rate: {:.4f}'.format(epoch, train_rate))
        scheduler.step()

    # beamformer= torch.zeros()
    ## For CDF Plot
    import matplotlib.pyplot as plt
    for  (g, lable) in test_loader:
        index = 0
        K = lable.shape[-1] # 6
        bs = len(g.nodes['AP'].data['feat'])//K
      
        output = model(g)
        # print(output)
        # scipy.io.savemat('beamformer.mat',{'beamformer':record.detach().cpu().numpy()})
        gnn_rates = rate_loss(output, lable, True).flatten().detach().cpu().numpy()
        full = 0.5*torch.ones_like(output)
        all_one_rates= rate_loss(full,lable, True).flatten().cpu().numpy()

    # beamformer = beamformer.cpu()

    # scipy.io.savemat('beamformer.mat',{'beamformer':beamformer})
    scipy.io.savemat('gnn_rate.mat',{'gnn_rate':gnn_rates})
    scipy.io.savemat('all_one_rates.mat',{'all_one_rates':all_one_rates})

    # min_rate, max_rate = 0, 10
    y_axis = np.arange(0, 1.0, 1/test_layouts)
    gnn_rates.sort()
    all_one_rates.sort()
    # opt_rates.sort()
    # gnn_rates = np.insert(gnn_rates, 0, min_rate); gnn_rates = np.insert(gnn_rates,201,max_rate)
    # all_one_rates = np.insert(all_one_rates, 0, min_rate); all_one_rates = np.insert(all_one_rates,201,max_rate)
    # opt_rates = np.insert(opt_rates, 0, min_rate); opt_rates = np.insert(opt_rates,201,max_rate)
    plt.plot(gnn_rates, y_axis, label = 'GNN')
    # plt.plot(opt_rates, y_axis, label = 'Optimal')
    plt.plot(all_one_rates, y_axis, label = 'Random beamformer')
    plt.xlabel('Minimum rate [bps/Hz]', {'fontsize':16})
    plt.ylabel('Empirical CDF', {'fontsize':16})
    plt.legend(fontsize = 12)
    plt.grid()
    plt.show(block=True)
    # np.save()
    torch.save(hetgnn_model, 'hetgnn_model.pt')
if __name__ == '__main__':
    main()





