import torch
import torch.nn as nn
from constants import *
import torch.optim as optim
from edge_generator import EdgeGenerator
from matrix_generator import MatrixGenerator
from combined_generator import CombinedGenerator

edge_generator = EdgeGenerator(INPUT_SIZE_GEN, HIDDEN_SIZE_GEN, OUTPUT_SIZE_EDGE_GEN)
matrix_generator = MatrixGenerator(INPUT_SIZE_GEN, HIDDEN_SIZE_GEN, OUTPUT_SIZE_MAT_GEN)

model = CombinedGenerator(edge_generator, matrix_generator)
model.load_state_dict(torch.load("model.pth"))
model.eval()

def get_fake_data(batch_size, combined):
    fake_graphs=[]
    for i in range(batch_size):
        rand_noise = torch.randn(1, INPUT_SIZE_GEN)
        fake_graphs.append(combined(rand_noise))
    return fake_graphs 
fake_data = get_fake_data(64,model)

def deconstructor(matrix):
    mat1 = matrix[:, :3]  
    mat2 = matrix[:, 3:]  
    return mat1, mat2

def adj_matrix_to_dict(adj_matrix):
    adj_dict = {}
    for i, row in enumerate(adj_matrix):
        adj_dict[i] = []
        for j, edge in enumerate(row):
            if edge != 0:
                adj_dict[i].append(j)
    return adj_dict

dict_list = []
for data in fake_data:
    dict_list.append(adj_matrix_to_dict(deconstructor(data)[1]))