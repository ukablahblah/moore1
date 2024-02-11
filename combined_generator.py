import torch
import torch.nn as nn

class CombinedGenerator(nn.Module):
    def __init__(self, edge_generator, matrix_generator):
        super(CombinedGenerator, self).__init__()
        self.edge_generator = edge_generator
        self.matrix_generator = matrix_generator

    def forward(self, rand_noise):
        adj = self.edge_generator(rand_noise)
        
        matrices = self.matrix_generator(rand_noise)

        comb = torch.cat((matrices, adj), dim=1)
        return comb