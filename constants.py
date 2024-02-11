# Constants for the generator
NUM_ROWS = 16
NODE_FEATURES = 3
INPUT_SIZE_GEN = 100  # Size of the random noise vector for the generator
HIDDEN_SIZE_GEN = 200
OUTPUT_SIZE_EDGE_GEN = NUM_ROWS * NUM_ROWS   # Output size for the matrix used to generate edge in edge generator
OUTPUT_SIZE_MAT_GEN = NUM_ROWS * NODE_FEATURES   # Output size for the matrix generator

# Constants for the discriminator
INPUT_SIZE_DISCRIM = NUM_ROWS * (NUM_ROWS + NODE_FEATURES)
OUTPUT_SIZE_DISCRIM = 1
HIDDEN_SIZE_DISCRIM = 100
