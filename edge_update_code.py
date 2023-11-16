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
epoch_num = 600
# 指定批大小
batch_size = 100000
# 指定学习率
learning_rate = 1e-3
lambda_lr = 1
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
C_max = 4
# sigma
sigma = var_noise
# AP数量，单天线 train和test显然是可以不一样的
train_B = 4
test_B = 4
# UE数量
train_K = 4
test_K = 4
# 训练2
train_layouts = 10
test_layouts = 200
# 模拟路损
beta = 0.6

hid_dim = 64
# 指示函数因子
delta = 1e-3

# 创建信道，因为不能直接输入复数进入神经网络，我们输入信道的模值
# 创建信道实部
train_channel_rel = beta * np.random.randn(train_layouts, train_B, train_K)
# 创建信道虚部
train_channel_ima = beta * np.random.randn(train_layouts, train_B, train_K)
# print(train_channel_rel + 1j * train_channel_ima)
# test
test_channel_rel = beta * np.random.randn(test_layouts, test_B, test_K)
test_channel_ima = beta * np.random.randn(test_layouts, test_B, test_K)
# lambda_init = torch.zeros(train_layouts, train_B)
train_Cmax = 5*np.random.rand(train_layouts, train_B)
test_Cmax = 5*np.random.rand(test_layouts, test_B)

scipy.io.savemat('train_channel_1112.mat',{'train_channel_rel':train_channel_rel, 'train_channel_ima': train_channel_ima})
scipy.io.savemat('train_backhaul_1112.mat',{'train_Cmax':train_Cmax})
scipy.io.savemat('test_channel_1112.mat',{'test_channel_rel':test_channel_rel, 'test_channel_ima': test_channel_ima})
scipy.io.savemat('test_backhaul_1112.mat',{'test_Cmax':train_Cmax})
# scipy.io.savemat('test_channel.mat',{'test_channel':test_channel_rel + 1j* test_channel_ima})
train_channel = scipy.io.loadmat('train_channel_1112.mat')
backhaul = scipy.io.loadmat('train_backhaul_1112.mat')
train_channel_rel = train_channel['train_channel_rel']
train_channel_ima = train_channel['train_channel_ima']
train_Cmax = backhaul['train_Cmax']
# # # test_channel_rel = np.transpose(np.real(train_channel['Hd1']),axes=(0, 2, 1))
# # # test_channel_ima = np.transpose(np.imag(train_channel['Hd1']),axes=(0, 2, 1))
# train_channel_rel = np.transpose(np.real(train_channel['Hd']),axes=(0, 2, 1))
# train_channel_ima = np.transpose(np.imag(train_channel['Hd']),axes=(0, 2, 1))
def normalize_one_tensor(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor
# 构建数据集，数据构成为：信道实虚部，APUE数量
class MyDataset(DGLDataset):
    def __init__(self, rel, ima, BK, Cmax):
        self.rel = rel
        self.ima = ima
        # sinr 分子（信道增益
        # self.direct = torch.tensor(direct, dtype = torch.float)
        self.BK = BK
        self.get_cg()
        self.Cmax = Cmax
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
        graph.nodes['AP'].data['feat'] = torch.unsqueeze(1 * torch.ones(self.BK[0]),dim=-1) 
        ## AP数据还有backhaul budget
        graph.nodes['AP'].data['bakchaul'] = torch.unsqueeze(C_max * torch.ones(self.BK[0]),dim=-1) 
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
        return self.graph_list[index], self.rel[index], self.ima[index] , self.Cmax[index]
    # 创建一个大的图list的函数
    def process(self):
        n = len(self.rel)
        self.graph_list = []
        for i in range(n):
            graph = self.build_graph(i)
            self.graph_list.append(graph)


# DGL collate function 用于批处理时，构成一个小批次     
def collate(samples):
    graphs, rel, ima, cmax  = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs) 
    return batched_graph, torch.stack([torch.from_numpy(i) for i in rel]) , torch.stack([torch.from_numpy(i) for i in ima]),torch.stack([torch.from_numpy(i) for i in cmax])

# relu = nn.Softplus()
relu = nn.ReLU()

def caculate_rate(x, beamformer, rel, ima, test_mode = False):
    
    beamformer_all = beamformer[:, :, :, 0].float() + 1j * beamformer[:, :, :, 1].float() 
    # 引入新的association变量
    beamformer_all = x.squeeze() * beamformer_all
    # 测试第一个基站不为第一个用户服务
    # beamformer_all[:,0,0] = 0
    beamformer_all = norm_func_2(beamformer_all)   
    channel_all = rel.float() + 1j * ima.float() 
    B_cur = beamformer.size()[1]
    K_cur = beamformer.size()[2]
    batch_cur = rel.shape[0]
    SINRs_numerators = torch.zeros((batch_cur, K_cur)) 
    SINRs_denominators = torch.zeros(batch_cur, K_cur) 
    # 分子，表示有用信号
    for i in range(0, K_cur):
            SINRs_numerators[:,i] = torch.squeeze(torch.abs(torch.matmul(beamformer_all[:,:,i].unsqueeze(1), torch.transpose(channel_all[:,:,i].unsqueeze(1),dim0=1,dim1=2))) ** 2)
    # 分母是干扰波束*自己的信道
    for i in range(0, K_cur):
            for j in range(0, K_cur):
                if(i != j):
                    SINRs_denominators[:,i] = SINRs_denominators[:,i] + torch.squeeze(torch.abs(torch.matmul(beamformer_all[:,:,j].unsqueeze(1) , torch.transpose(channel_all[:,:,i].unsqueeze(1),dim0=1,dim1=2))) ** 2)
    SINRs_denominators += var_noise
    SINRs = SINRs_numerators / SINRs_denominators 
    rates = torch.log2(1 + SINRs) 
    sum_rate = torch.sum(rates, dim = 1)  # take sum
    return rates
    # return sum_rate
def caculate_backhaul(rates, sampled_x):
    B_cur = sampled_x.size()[1]
    K_cur = sampled_x.size()[2]
    batch_cur = sampled_x.shape[0]
    backhaul = torch.zeros(batch_cur,B_cur)
    # backhaul_x = torch.zeros(batch_cur,B_cur)
    rates_unsqueeze = rates.unsqueeze(1).expand(batch_cur,B_cur,K_cur)
    sampled_rates = torch.mul(sampled_x.squeeze(), rates_unsqueeze)
    backhaul = torch.sum(sampled_rates,dim = -1)
    return backhaul
# 以上构建完图结构后，下面开始构建图神经网络，这里我们把神经网络的输出看做两个向量拼接成的矩阵，所以维度(:,:,0)表示实部 (:,:,1)表示虚部
# 根据神经网络的输出（beamformer）和我们固定的信道信息 计算合速率
def rate_loss(lambda_gnn, x, beamformer, rel, ima, test_mode = False):
    
    x = torch.squeeze(x)
    # 采样x（量化）将大于0.5的x量化为1
    sampled_x = sample_x_from_x(x)
    policy_x = x ** sampled_x

    # x = (x > 0.5).float()
    beamformer_all = beamformer[:, :, :, 0].float() + 1j * beamformer[:, :, :, 1].float() 
    # 引入新的association变量
    # beamformer_all = x.squeeze() * beamformer_all
    # 测试第一个基站不为第一个用户服务
    # beamformer_all[:,0,0] = 0
    beamformer_all = norm_func_2(beamformer_all)
    norm_beam = torch.norm(beamformer_all.unsqueeze(-1),dim = -1)
    mask = (norm_beam > 0.01).float()
    # beamformer_all = mask * beamformer_all
    
    channel_all = rel.float() + 1j * ima.float() 
    B_cur = beamformer.size()[1]
    K_cur = beamformer.size()[2]
    batch_cur = rel.shape[0]
    SINRs_numerators = torch.zeros((batch_cur, K_cur)) 
    SINRs_denominators = torch.zeros(batch_cur, K_cur) 
    # 分子，表示有用信号
    for i in range(0, K_cur):
            SINRs_numerators[:,i] = torch.squeeze(torch.abs(torch.matmul(beamformer_all[:,:,i].unsqueeze(1), torch.transpose(channel_all[:,:,i].unsqueeze(1),dim0=1,dim1=2))) ** 2)
    # 分母是干扰波束*自己的信道
    for i in range(0, K_cur):
            for j in range(0, K_cur):
                if(i != j):
                    # bb = beamformer_all[b,:,j].unsqueeze(0)
                    # bbb = torch.transpose(channel_all[b,:,i].unsqueeze(0),0,1)
                    SINRs_denominators[:,i] = SINRs_denominators[:,i] + torch.squeeze(torch.abs(torch.matmul(beamformer_all[:,:,j].unsqueeze(1) , torch.transpose(channel_all[:,:,i].unsqueeze(1),dim0=1,dim1=2))) ** 2)
    SINRs_denominators += var_noise
    SINRs = SINRs_numerators / SINRs_denominators 
    rates = torch.log2(1 + SINRs) 
    sum_rate = torch.sum(rates, dim = 1)  # take sum
    # sum_rate = torch.min(rates, dim = 1)[0]  # take sum
    # 计算每个AP的backhaul放入目标函数
    backhaul = torch.zeros(batch_cur,B_cur)
    # backhaul_x = torch.zeros(batch_cur,B_cur)
    for i in range(0, B_cur):
        for k in range(0,K_cur):
            # backhaul[:,i] = backhaul[:,i]+caculate_backhaul(beamformer_all[:,i,k].unsqueeze(1))*rates[:,k]
            backhaul[:,i] = backhaul[:,i]+rates[:,k]
            # backhaul[:,i] = backhaul[:,i]+x[:,i,k].squeeze()  *rates[:,k]
    # for i in range(0, B_cur):
    #     for k in range(0,K_cur):
    #         # 关注速率和
    #         # backhaul_x[:,i] = backhaul_x[:,i]+torch.sum(x[:,i,k]*rates[:,k],dim = -1)     
    #         # 只关注连接数   
    #         backhaul_x[:,i] = backhaul_x[:,i]+x[:,i,k].squeeze()  
    backhaul_x = torch.sum(x,dim=-1)
    #         backhaul[:,i] = backhaul[:,i]+caculate_backhaul(beamformer_all[:,i,k].unsqueeze(1))
    # sum_rate = 0*sum_rate
    # sum_rate -= 10000*torch.norm(beamformer_all,dim = (1,2))
    # sum_rate -= 1000*torch.sum(torch.abs(torch.log(torch.norm(beamformer_all,dim = (2))/delta+1)/torch.log(torch.tensor(1/delta+1))-C_max),dim=1)
    # sum_rate -= 1000*(torch.sum(torch.abs((torch.norm(beamformer_all,dim = (2))/(delta+torch.norm(beamformer_all,dim = (2))))-C_max),dim=1))
    # sum_rate -= torch.sum( 10* (backhaul),dim = 1)
    # sum_rate = sum_rate - torch.sum( 1* torch.abs(backhaul-C_max),dim = 1)
    # sum_rate = sum_rate - torch.sum(x*(1-x)) - torch.sum( 1* torch.abs(backhaul_x-C_max),dim = 1)
    C = C_max*torch.ones_like(backhaul)
    # C = 16*torch.ones_like(backhaul)
    zero = torch.zeros_like(backhaul)
    rates_min = 0.2*torch.ones_like(rates)
    zero_rates = torch.zeros_like(rates)
    
    power_cur = torch.norm(beamformer_all[:,:,:].clone(), dim=2,p=2)
    
    # sum_rate = sum_rate - torch.sum(x*(1-x)) - torch.sum( 100* torch.abs(backhaul_x-2),dim = 1)
    # sum_rate -= torch.sum( lambda1* torch.abs(backhaul-C_max),dim = 1) +torch.sum( 1000*relu((backhaul-C_max)),dim=1)
    # sum_rate -= torch.sum( 1000*relu((backhaul-C_max)),dim=1)+torch.sum(1000*relu(torch.norm(beamformer_all[:,:,:], dim=1,p=2)-P_max),dim=1)
    # sum_rate = sum_rate - torch.sum(x*(1-x)) - torch.sum( torch.maximum(backhaul_x-C,zero)) -torch.sum(torch.maximum(rates_min-rates,zero_rates))
    # sum_rate -= torch.sum( 1000*torch.maximum(backhaul-C,zero),dim=1)
    # sum_rate -= torch.sum(1000*torch.abs(torch.norm(beamformer_all[:,:,:], dim=2,p=2)-P_max),dim=1)
    # sum_rate -= 100*torch.sum(bs_backhaul, dim = 1)
    # backhaul_constraint =torch.sum( torch.maximum(backhaul_x-C,zero)) 
    # backhaul_constraint =torch.sum( torch.abs(backhaul-20))
    # backhaul_constraint =torch.sum( 1* custom_maxmum_function(backhaul-20,zero))
    # backhaul_constraint = torch.mean( custom_maxmum_function(backhaul_x-C,zero))
    backhaul_nograd = backhaul.data
    backhaul_constraint2 = torch.sum(lambda_gnn * (backhaul_nograd - C),dim = 1)
    backhaul_constraint =  (backhaul - C)
    # backhaul_constraint = (power_cur - P_max)
    # sum_rate -= backhaul_constraint

    # 用于损失函数的项
    return_for_policy_gradient = sum_rate * torch.log(1e-10 + policy_x)
    if test_mode:
        return sum_rate
        
    else:
        # return -torch.mean(sum_rate),backhaul_constraint,torch.sum(x*(1-x))
        return -torch.mean(sum_rate), backhaul_constraint, sum_rate
def sample_x_from_x(x):
    sampled_x = torch.multinomial(x,1)
    return sampled_x
class CustomMaximum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        maximum_values = torch.maximum(x, y)
        ctx.save_for_backward(maximum_values, x, y)  # 保存前向传播中需要用到的变量
        return maximum_values
    
    @staticmethod
    def backward(ctx, grad_output): 
        maximum_values, x ,y = ctx.saved_tensors
        
        # gradient_x = (x >= y).float() * grad_output   # 较大值对应的梯度为grad_output，较小值对应的梯度为0
        # gradient_y = (y > x).float() * grad_output     # 较大值对应的梯度为grad_output，较小值对应的梯度为0
        
         # 如果想让较小元素具有固定大小（例如0.1） 的非零梯度，则将上述语句修改如下：
        gradient_x = ((x >= y).float() - (x < y).float()) * grad_output  
        gradient_y = ((y > x).float() - (y <= x).float()) * grad_output  
        return gradient_x, gradient_y
custom_maxmum_function = CustomMaximum.apply
class Htanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, epsilon=1):
        ctx.save_for_backward(x.data, torch.tensor(epsilon))
        return (x.sign()+1)/2

    @staticmethod
    def backward(ctx, dy):
        x, epsilon = ctx.saved_tensors
        dx = torch.where((x < - epsilon) | (x > epsilon), torch.zeros_like(dy), dy)
        dx = torch.rand_like(x)
        dx = (dx * 2) - 1
        return dx, None
htanh = Htanh()
x = torch.tensor([-2.5, -1.6, 0, 3.6, 4.7])
y = htanh.apply(x)
print(y)

# 定义整体的神经网络结构，输入层MLP（将数据转换成APnode、UEnode、edge，维度全是64）->多个中间更新层MLP（保持维度为64）->输出层MLP
class HetGNN(nn.Module):
    def __init__(self):
        super(HetGNN, self).__init__()
        
        self.preLayer = PreLayer() # 预处理层
        self.update_layers = nn.ModuleList([UpdateLayer()] * 2)  # 更新层，这里使用了2个更新层
        self.postprocess_layer = PostLayer(mlp_post, mlp_post_lambda, mlp_post_x) # 后处理层
        self.softmax = nn.Softmax(dim = -1)
    # 输入网络时，节点特征数量为batchsize*AP batchsize*UE，边特征数量为
    def forward(self, graph):
        self.preLayer(graph)
        for update_layer in self.update_layers:
            update_layer(graph)
        lambda_gnn,output,x = self.postprocess_layer(graph)
        lambda_gnn = lambda_gnn.view(graph.batch_size,-1)
        output = output.view(graph.batch_size,
                            graph.number_of_nodes('AP')//graph.batch_size,
                            graph.number_of_nodes('UE')//graph.batch_size,
                            -1)
        # self.preLayer_x(graph)
        # for update_layer_x in self.update_layers_x:
        #     update_layer_x(graph)
        # x = self.postprocess_layer_x(graph)
        x = x.view(graph.batch_size,
                            graph.number_of_nodes('AP')//graph.batch_size,
                            graph.number_of_nodes('UE')//graph.batch_size,
                            1)
        x = torch.concatenate((x, 1-x), axis = -1)
        # x = self.softmax(x)
        x = nn.functional.normalize(x,p = 1, dim = -1)

        output_real = output[:,:,:,0]
        output_imag = output[:,:,:,1]
        # P_max = 1
        norm_output_real, norm_output_imag = output_real, output_imag
        # bool_tensor=torch.ones((4,2))
        # bool_tensor = bool_tensor.unsqueeze(0).expand(output_real.size()[0],-1,-1)
        # bool_tensor[:,1,0] = 0
        # output_real = output_real*bool_tensor
        # output_imag = output_imag*bool_tensor
        # bool_tensor[1,0] = False
        # 归一化 功率约束
        # norm_output_real, norm_output_imag = norm_func(output_real, output_imag)
        # norm_output_real = MyBinaryMaskLayer.forward(norm_output_real)
        # norm_output_imag = MyBinaryMaskLayer.forward(norm_output_imag)
        norm_output = torch.cat((torch.unsqueeze(norm_output_real,dim = 3),
                                 torch.unsqueeze(norm_output_imag,dim = 3)),dim = 3)
        # norm_output = MyBinaryMaskLayer.forward(norm_output)
        return lambda_gnn, x, norm_output
    
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
        # x = self.tanh(x)
        # x = self.dropout(x)
        # x  = self.relu(x)
        # x = self.linear2(x)
        # x = GLUA.forward(x)
        return x
class MLP_post_lambda(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_post_lambda, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim, bias)    
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.linear1(x)
        x  = self.relu(x)
        return x
class MLP_post_x(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_post_x, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim, bias)    
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # x  = self.relu(x)
        x = self.linear1(x)
        x = self.sigmoid(x)
        return x    
def threshold_fun(x):
    return 1./(1+torch.exp(-50*(x-0.5)))
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

mlp_post = MLP_post(dimension,2) 
mlp_post_lambda = MLP_post_lambda(dimension,1) 
mlp_post_x = MLP_post_x(dimension,1)
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
    def __init__(self, mlp, mlp_lambda, mlp_post_x):
        super(PostLayer, self).__init__()
        self.post_mlp = mlp
        self.post_mlp_lambda = mlp_lambda
        self.post_mlp_x =  mlp_post_x
    def forward(self, graph):
        # a = graph.ndata['hid']['UE']
        return self.post_mlp_lambda(graph.ndata['hid'][('AP')]), self.post_mlp(graph.edata['hid'][('AP','AP2UE','UE')]), self.post_mlp_x(graph.edata['hid'][('AP','AP2UE','UE')])

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

        # 对每个AP节点，获取其
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
    beamformer_complex[:,:,:] = beamformer_complex[:,:,:].clone() / (torch.unsqueeze(torch.norm(beamformer_complex[:,:,:].clone(), dim=2,p=2),dim=-1)+1e-10)

    # for i in range(0,dims[0]):
        # if(torch.norm(beamformer_complex[i,:,:].clone(), dim=1,p=2)!=0):
        # beamformer_complex[i,:,:] = beamformer_complex[i,:,:].clone() / (torch.unsqueeze(torch.norm(beamformer_complex[i,:,:].clone(), dim=1,p=2),dim=-1)+0.0001)
    real_normalized = torch.real(beamformer_complex)
    imag_normalized = torch.imag(beamformer_complex)
    return real_normalized, imag_normalized
# 定义输入维度、隐藏层维度和输出维度
def norm_func_2(beamformer_complex):
    beamformer_complex[:,:,:] = beamformer_complex[:,:,:].clone() / (torch.unsqueeze(torch.norm(beamformer_complex[:,:,:].clone(), dim=2,p=2),dim=-1)+1e-10)
    return beamformer_complex

# 创建HetGNN模型实例
hetgnn_model = HetGNN()
# print(hetgnn_model)

model = hetgnn_model

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,weight_decay=1e-3)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.00001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

def test(loader):
    model.eval()
    correct = 0
    for (g, rel, ima, lambda_init) in loader:
        optimizer.zero_grad()

        K = rel.shape[-1] # 6
        bs = len(g.nodes['UE'].data['feat'])//K
        lambda_gnn, x, output = model(g)
        loss,loss2,loss3 = rate_loss(lambda_gnn, x, output, rel, ima)
        correct += loss.item() * bs
    return correct / len(loader.dataset)
# global lambda_init2
lambda_init2 = torch.zeros(train_layouts,train_B, requires_grad=True, device=device)

def cal_rate_WMMSE(x, backhaul):
    batch_size_cur = x.size()[0]
    B_cur = x.size()[1]
    K_cur = x.size()[2]
    # rate = torch.ones(batch_size_cur,B_cur)
    rate = (torch.sum(x,dim = -1))
    # rate[backhaul > C_max] = 0
    return x
def main():
    # 创建数据集，实际上是把训练数据和测试数据构建成图的过程
    train_data = MyDataset(train_channel_rel, train_channel_ima, (train_B, train_K), train_Cmax)
    test_data = MyDataset(test_channel_rel, test_channel_ima,  (test_B, test_K), test_Cmax)

    # train_data = train_data 
    # 检查一下异构图的两类结点名称
    print(train_data[0][0].ntypes)

    # 批处理，这里调用之前定义的collate函数
    
    train_loader = DataLoader(train_data, batch_size, shuffle=True, collate_fn=collate)
    # train_loader = DataLoader(train_data, batch_size, shuffle=True, collate_fn=collate,num_workers=4,pin_memory=True)、
    test_loader = DataLoader(test_data, test_layouts, shuffle=True, collate_fn=collate)
    # def train(epoch):
    #     """ Train for one epoch. """
       
    #     return loss_all2 / len(train_loader.dataset)
    objective_val = np.zeros(epoch_num)
    avg_rates_epoch = []
    # 开始训练，训练epoch_num次
    AAAA = torch.randint(2, (10, 4, 4))
    for epoch in range(0, epoch_num):
        # if(epoch % 10 == 1):
        #     with torch.no_grad():
        #         # train_rate = train(train_loader)
        #         test_rate = test(test_loader)
        #     print('Epoch {:03d},  ——————————  Test Rate: {:.4f}'.format(epoch,test_rate))
        global learning_rate
        global lambda_init2
        global lambda_lr
        model.train()
        loss_all = 0
        loss_all2 = 0
        avg_rates_batch = []
        
        for batch_idx, (g, rel, ima, Cmax) in enumerate(train_loader):
            model.zero_grad()

            # 清零上一个batch的梯度
            K = rel.shape[-1] # 6
            bs = len(g.nodes['UE'].data['feat'])//K
            lambda_gnn,x,beamformer = model(g)
            # x_with_one_minus_x = torch.concat((x,(1-x)),dim = -1)
            batch_size_cur = x.size()[0]
            B_cur = x.size()[1]
            K_cur = x.size()[2]
            # xx = x_with_one_minus_x.view(-1,2)
            x = x.view(-1,2)
            all_rates = []
            all_rates_WMMSE = []
            avg_rates = []
            all_sampled_xx = []
            all_sampled_x = []
            all_xx = []
            all_backhaul = []
            sample_times = 20
            threshold_epoch = 1
            if(epoch == threshold_epoch):
                sampled_x_ones = 1-sample_x_from_x(x).detach()
            ## 对于输出的一个x，对其进行多次的采样
            for sample_idx in range(0,sample_times):
                # 因为采样输出下标，所以1减去下标正好表示此处是否是1
                sampled_x = 1-sample_x_from_x(x)
                sampled_x = AAAA
                # if(epoch < 100):
                #     sampled_x = 0*sampled_x.view(batch_size_cur,B_cur,K_cur)+1
                # else:
                #     sampled_x = sampled_x.view(batch_size_cur,B_cur,K_cur)
                # sampled_x = 0*sampled_x.view(batch_size_cur,B_cur,K_cur)+1
                # sampled_x[:,1,:] = 0
                # if(epoch > threshold_epoch):
                #     sampled_x = 0*sampled_x.view(batch_size_cur,B_cur,K_cur)+1
                sampled_x = sampled_x.view(batch_size_cur,B_cur,K_cur)
                sampled_xx = torch.concat((sampled_x.unsqueeze(-1),(1-sampled_x.unsqueeze(-1))),dim = -1)
                # 使用采样后的x计算和速率
                rates = caculate_rate(sampled_x, beamformer, rel, ima)
                
                all_rates.append(rates)
                backhaul = caculate_backhaul(rates, sampled_x)
                WMMSE_rates = cal_rate_WMMSE(sampled_x,backhaul)
                all_backhaul.append(backhaul)
                all_rates_WMMSE.append(WMMSE_rates)
                all_sampled_xx.append(sampled_xx) 
                all_sampled_x.append(sampled_x)
            # 获取这100次遍历的所有数据，
            all_rates = torch.stack(all_rates, dim=0)
            all_rates_WMMSE = torch.stack(all_rates_WMMSE,dim = 0)
            all_sampled_xx = torch.stack(all_sampled_xx, dim=0)
            avg_rates = torch.mean(all_rates, dim=0) # ergodic average rates
            all_xx = x.view(batch_size_cur,B_cur,K_cur,2).unsqueeze(0).expand(sample_times,batch_size_cur,B_cur,K_cur,2)
            all_backhaul = torch.stack(all_backhaul, dim = 0)
            all_sampled_x = torch.stack(all_sampled_x)
            # 使用采样后的x计算每个基站的backhaul
            backhaul = caculate_backhaul(avg_rates, sampled_x)
            sum_rate = torch.sum(all_rates,dim = -1)
            lambda_now = lambda_init2[batch_idx*batch_size:(batch_idx+1)*batch_size,:]
            # 将滞后的lambda清零
            mask = torch.ones_like(lambda_now)
            mask[backhaul<Cmax] = 0
            lambda_now= mask * lambda_now
            constraint_gap = (torch.sum((lambda_now * (backhaul-Cmax)),dim=1))
            # constraint_gap = (torch.sum((lambda_now * torch.maximum((backhaul-C_max),torch.zeros(1))),dim=1))
            # 本次采样带来的性能函数具体值-用于策略梯度项
            # 多次采样带来的奖励函数值，该函数将直接决定X的输出，我们需要考虑上backhaul约束，而不能单单看和速率
            # func_value_for_policy_gradient = torch.sum(all_rates_WMMSE,dim = (-1)).detach()-torch.max(torch.sum(all_backhaul-C_max,dim = -1),0)[0].detach()
            func_value_for_policy_gradient = torch.sum(all_rates,dim = (-1)).detach()-(torch.sum((lambda_now * (all_backhaul-Cmax)),dim=-1)).detach()

            # func_value_for_policy_gradient = torch.sum(all_rates_WMMSE,dim = (-1)).detach()
            # func_value_for_policy_gradient = torch.sum(all_rates,dim = (-1)).detach()
            # func_value_for_policy_gradient = all_rates_WMMSE.detach()
            # ** 表示乘方操作，这里使用采样前的概率为底，采样结果为幂

            # sampled_user_x = (all_xx**all_sampled_xx).view(sample_times,batch_size_cur,-1)
            sampled_user_x = (all_xx**all_sampled_xx)
            sampled_user_probability = torch.prod(sampled_user_x.view(sample_times,batch_size_cur,train_B,train_K, -1),dim = -1)
            sampled_user_probability = torch.prod(sampled_user_x.view(sample_times,batch_size_cur,-1),dim = -1)
            policy_gradient_term = func_value_for_policy_gradient.detach()*torch.log(1e-10+sampled_user_probability)
            # policy_gradient_term = func_value_for_policy_gradient.detach()*torch.log(1e-10+sampled_user_probability)
            # policy_gradient_term = func_value_for_policy_gradient.detach()*sampled_user_probability
            # loss_func = -torch.mean(sum_rate - 10*constraint_gap)  + torch.mean(policy_gradient_term)
            # loss_func = -torch.mean(torch.sum(all_rates,dim = (-1)) - 1*constraint_gap) 
            loss_func = -torch.mean(torch.sum(avg_rates,dim = (-1))- 1*constraint_gap) - torch.mean(policy_gradient_term)
            # loss_func =  -torch.mean(policy_gradient_term)
            # loss_func = torch.sum(x[:,1])-torch.sum(x[:,0])
            # f = func_value_for_policy_gradient.type(torch.float64)
            # print(all_sampled_x)
            
            # print(f.mean())
            # loss_func = -torch.mean(torch.sum(all_rates,dim = (-1))) - torch.mean(policy_gradient_term)
            # loss_func = -torch.mean(torch.sum(all_rates,dim = (-1))- 1*constraint_gap)
            # print(torch.mean(sampled_user_probability))
            # loss_func = -torch.mean(torch.sum(all_rates,dim = (-1)))
            # # 使用采样后的x计算backhaul
            # loss,constraint,sum_rate = rate_loss(lambda_gnn, x,output, rel, ima)
            # loss.requires_grad_(True)
            # loss.requires_grad_(True)
            # loss2.requires_grad_(True)
           
            # optimizer.zero_grad()
            # loss.backward(retain_graph=True)
            # loss3.backward()
            # print(loss3)
            # model.zero_grad()
            # lambda_init = 1
            
            lambda_init2.retain_grad()

            # mask = constraint < 0
            # grad_direct = torch.ones_like(lambda_now)
            # grad_direct[mask] = -1
            # loss.backward(retain_graph=True)
            # loss2.backward(retain_graph=True)
            # loss3.backward()
            # constraint_gap = (torch.sum((lambda_now * constraint),dim=1))
            x.retain_grad()
            # # print(constraint_gap)
            # Utility_func = -torch.mean(sum_rate - 1*constraint_gap)
            loss_func.backward()
            # lambda_cur.requires_grad=False
            # Utility_func.backward()
            
            # 不允许负值
            # perform gradient ascent/descent
            print(x.grad)
            ##
            with torch.no_grad():
            # # # primal GNN parameters
            # #     for i, theta_main in enumerate(list(model.parameters())):
            # #         # theta_main += lr_main * torch.clamp(dtheta_main[i], min=-1, max=1)
            # #         if theta_main.grad is not None:
            # #             theta_main -= learning_rate * theta_main.grad
                lambda_init2 += lambda_lr * lambda_init2.grad
            lambda_init2.data.clamp_(0)  
            lambda_init2.grad.zero_()
            ##


            # # # 清空梯度
            # for theta_ in list(model.parameters()) + [lambda_init2]:
            #     if theta_.grad is not None:
            #         theta_.grad.zero_()
            ##
            loss_all += torch.mean(torch.sum(all_rates,dim = -1)).item()*bs
            loss_all2 += loss_func.item()*bs
            # print(torch.mean(torch.sum(all_rates,dim = -1)).item())

            # lambda_init2.requires_grad = False
            optimizer.step()    
            # print(backhaul)
        # train_rate = train(epoch)
        
        # avg_rates_epoch.append(np.mean(avg_rates_batch).item())
        if (epoch + 1) % 10 == 0:
            # print(Cmax)
            lambda_lr *= 0.95
            learning_rate *= 0.8 #0.8
        # loss_all += loss.item() * bs
        train_rate = loss_all / len(train_loader.dataset)
        loss_val = loss_all2 / len(train_loader.dataset)
        objective_val[epoch] = train_rate
        print('Epoch {:03d}, Train Rate: {:.4f}, Loss :{:.4f}, Constraint Gap: {:.4f}'.format(epoch, train_rate, loss_val,  torch.mean(constraint_gap)))
        # print(Cmax)
        # print(backhaul)
        # scheduler.step()
    scipy.io.savemat('obj_val.mat',{'obj_val':objective_val})
    # beamformer= torch.zeros()
    ## For CDF Plot
    import matplotlib.pyplot as plt
    for  (g, rel, ima, Cmax) in test_loader:
        index = 0
        model.eval()
        K = rel.shape[-1] # 6
        bs = len(g.nodes['UE'].data['feat'])//K
        lambda_gnn,x,beamformer = model(g)
        # x_with_one_minus_x = torch.concat((x,(1-x)),dim = -1)
        batch_size_cur = x.size()[0]
        B_cur = x.size()[1]
        K_cur = x.size()[2]
        x = x.view(-1,2)
        sampled_x = 1-sample_x_from_x(x)   
        sampled_x = sampled_x.view(batch_size_cur,B_cur,K_cur)
        rates = caculate_rate(sampled_x, beamformer, rel, ima)
        # 使用采样后的x计算和速率
        gnn_rates = torch.sum(rates,dim = -1).detach()
        backhaul = caculate_backhaul(rates, sampled_x)
        # print(output)
        scipy.io.savemat('test_backhaul.mat',{'Cmax':Cmax.detach().numpy(),'backhaul':backhaul.detach().numpy()})
        # all_one_rates= rate_loss(x,full, rel, ima, True).flatten().detach().cpu().numpy()

    # beamformer = beamformer.cpu()

    # scipy.io.savemat('beamformer.mat',{'beamformer':beamformer})
    
    # scipy.io.savemat('all_one_rates.mat',{'all_one_rates':all_one_rates})

    # min_rate, max_rate = 0, 10
    y_axis = np.arange(0, 1.0, 1/test_layouts)
    gnn_rates = gnn_rates.sort().values
    scipy.io.savemat('gnn_rate.mat',{'gnn_rate':gnn_rates})
    # all_one_rates.sort()
    # opt_rates.sort()
    # gnn_rates = np.insert(gnn_rates, 0, min_rate); gnn_rates = np.insert(gnn_rates,201,max_rate)
    # all_one_rates = np.insert(all_one_rates, 0, min_rate); all_one_rates = np.insert(all_one_rates,201,max_rate)
    # opt_rates = np.insert(opt_rates, 0, min_rate); opt_rates = np.insert(opt_rates,201,max_rate)
    plt.plot(gnn_rates, y_axis, label = 'GNN')
    # plt.plot(opt_rates, y_axis, label = 'Optimal')
    # plt.plot(all_one_rates, y_axis, label = 'Random beamformer')
    plt.xlabel('Minimum rate [bps/Hz]', {'fontsize':16})
    plt.ylabel('Empirical CDF', {'fontsize':16})
    plt.legend(fontsize = 12)
    plt.grid()
    plt.show(block=True)
    # np.save()
    torch.save(hetgnn_model, 'hetgnn_model.pt')
if __name__ == '__main__':
    main()