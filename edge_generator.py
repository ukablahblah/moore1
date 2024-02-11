import torch
import torch.nn as nn
from constants import NUM_ROWS

class EdgeGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EdgeGenerator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )

    def forward(self, noise):
        matrix_for_edge = self.generator(noise)
        matrix_for_edge = torch.where(matrix_for_edge >= 0, torch.tensor(1.0), torch.tensor(0.0))

        matrix_for_edge = torch.reshape(matrix_for_edge, (NUM_ROWS,NUM_ROWS))

        matrix_for_edge = matrix_for_edge.fill_diagonal_(0)

        return matrix_for_edge