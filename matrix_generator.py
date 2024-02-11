import torch
import torch.nn as nn
from constants import NUM_ROWS, NODE_FEATURES

class MatrixGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MatrixGenerator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, noise):
        gen_matrix = self.generator(noise)

        gen_matrix = torch.reshape(gen_matrix, (NUM_ROWS,NODE_FEATURES))

        return gen_matrix