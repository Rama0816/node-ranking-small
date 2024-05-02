 
import numpy as np
import pickle
import networkx as nx
import torch
from utils import *
import random
import torch.nn as nn
from model_bet import GNN_Bet
torch.manual_seed(20)
import argparse

#Loading graph data
parser = argparse.ArgumentParser()
parser.add_argument("--g",default="SF")
args = parser.parse_args()
gtype = args.g
print(gtype)
if gtype == "SF":
    data_path = "./datasets/data_splits/SF/betweenness/"
    print("Scale-free graphs selected.")

elif gtype == "ER":
    data_path = "./datasets/data_splits/ER/betweenness/"
    print("Erdos-Renyi random graphs selected.")
elif gtype == "GRP":
    data_path = "./datasets/data_splits/GRP/betweenness/"
    print("Gaussian Random Partition graphs selected.")



#Load training data
print(f"Loading data...")
with open(data_path+"training.pickle","rb") as fopen:
    list_graph_train,list_n_seq_train,list_num_node_train,bc_mat_train = pickle.load(fopen)


with open(data_path+"test.pickle","rb") as fopen:
    list_graph_test,list_n_seq_test,list_num_node_test,bc_mat_test = pickle.load(fopen)

model_size = 100
#Get adjacency matrices from graphs
print(f"Graphs to adjacency conversion.")

list_adj_train,list_adj_t_train = graph_to_adj_bet(list_graph_train,list_n_seq_train,list_num_node_train,model_size)
list_adj_test,list_adj_t_test = graph_to_adj_bet(list_graph_test,list_n_seq_test,list_num_node_test,model_size)



def train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train):
    model.train()
    total_count_train = list()
    loss_train = 0
    num_samples_train = len(list_adj_train)
    for i in range(num_samples_train):
        adj = list_adj_train[i]
        num_nodes = list_num_node_train[i]
        adj_t = list_adj_t_train[i]
        adj = adj.to(device)
        adj_t = adj_t.to(device)

        optimizer.zero_grad()
            
        y_out = model(adj,adj_t)
        true_arr = torch.from_numpy(bc_mat_train[:,i]).float()
        true_val = true_arr.to(device)
        
        loss_rank = loss_cal(y_out,true_val,num_nodes,device,model_size)
        loss_train = loss_train + float(loss_rank)
        loss_rank.backward()
        optimizer.step()

def test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test):
    model.eval()
    loss_val = 0
    list_kt = list()
    num_samples_test = len(list_adj_test)
    for j in range(num_samples_test):
        adj = list_adj_test[j]
        adj_t = list_adj_t_test[j]
        adj=adj.to(device)
        adj_t = adj_t.to(device)
        num_nodes = list_num_node_test[j]
        
        y_out = model(adj,adj_t)
    
        
        true_arr = torch.from_numpy(bc_mat_test[:,j]).float()
        true_val = true_arr.to(device)
    
        kt = ranking_correlation(y_out,true_val,num_nodes,model_size)
        list_kt.append(kt)
        #g_tmp = list_graph_test[j]
        #print(f"Graph stats:{g_tmp.number_of_nodes()}/{g_tmp.number_of_edges()},  KT:{kt}")

    print(f"   Average KT score on test graphs is: {np.mean(np.array(list_kt))} and std: {np.std(np.array(list_kt))}")



#Model parameters
hidden = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNN_Bet(ninput=model_size,nhid=hidden,dropout=0.6)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)
num_epoch = 10

print("Training")
print(f"Total Number of epoches: {num_epoch}")
for e in range(num_epoch):
    print(f"Epoch number: {e+1}/{num_epoch}")
    train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train)

    #to check test loss while training
    # with torch.no_grad():
    #     test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test)
#test on 10 test graphs and print average KT Score and its stanard deviation
#with torch.no_grad():
#    test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test)

# nx_karate = nx.karate_club_graph()
# edges = list(nx_karate.edges())
# graph = nx.MultiDiGraph()
# graph.add_edges_from(edges)

# self_loops = list(nx.selfloop_edges(graph))
# graph.remove_edges_from(self_loops)

# node_sequence = graph.nodes()
# adj_temp = nx.adjacency_matrix(graph)
# adj_temp_t = adj_temp.transpose()

# arr_temp1 = np.sum(adj_temp,axis=1)
# arr_temp2 = np.sum(adj_temp_t,axis=1)

# arr_multi = np.multiply(arr_temp1,arr_temp2)
# arr_multi = np.where(arr_multi>0,1.0,0.0)

# degree_arr = arr_multi

# non_zero_ind = np.nonzero(degree_arr.flatten())
# non_zero_ind = non_zero_ind[0]

# g_nkit = nx2nkit(graph)

# in_n_seq = [node_sequence[nz_ind] for nz_ind in non_zero_ind]
# all_out_dict = get_out_edges(g_nkit, node_sequence)
# all_in_dict = get_in_edges(g_nkit,in_n_seq)

# for index in non_zero_ind:

#     is_zero = clique_check(index,node_sequence,all_out_dict,all_in_dict)
#     if is_zero == True:
#         degree_arr[index,0] = 0.0

# adj_temp = adj_temp.multiply(csr_matrix(degree_arr))
# adj_temp_t = adj_temp_t.multiply(csr_matrix(degree_arr))

# rand_pos = 0
# top_mat = csr_matrix((rand_pos,rand_pos))
# remain_ind = model_size - rand_pos - len(graph)
# bottom_mat = csr_matrix((remain_ind,remain_ind))

# adj_temp = csr_matrix(adj_temp)
# adj_mat = sp.block_diag((top_mat,adj_temp,bottom_mat))

# adj_temp_t = csr_matrix(adj_temp_t)
# adj_mat_t = sp.block_diag((top_mat,adj_temp_t,bottom_mat))

# adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat)
# adj_mat_t = sparse_mx_to_torch_sparse_tensor(adj_mat_t)

# y_out = model(adj_mat,adj_mat_t)

# print(y_out)

# nx_karate = nx.karate_club_graph()
# edges = list(nx_karate.edges())
# graph = nx.MultiDiGraph()
# graph.add_edges_from(edges)

graph = nx.karate_club_graph()
# exact_bet = dict()
# exact_bet = nx.betweenness_centrality(graph)
# print("Exact betweenness centrality of karate club: ", exact_bet)

adj_temp = nx.adjacency_matrix(graph)
adj_temp_t = adj_temp.transpose()

rand_pos = 0
top_mat = csr_matrix((rand_pos,rand_pos))
remain_ind = model_size - rand_pos - len(graph)
bottom_mat = csr_matrix((remain_ind,remain_ind))

adj_temp = csr_matrix(adj_temp)
adj_mat = sp.block_diag((top_mat,adj_temp,bottom_mat))

adj_temp_t = csr_matrix(adj_temp_t)
adj_mat_t = sp.block_diag((top_mat,adj_temp_t,bottom_mat))

adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat)
adj_mat_t = sparse_mx_to_torch_sparse_tensor(adj_mat_t)

y_out = model(adj_mat,adj_mat_t)
output = y_out[0:34].detach().cpu().numpy()
for i in range(34):
    print(output[i][0])
