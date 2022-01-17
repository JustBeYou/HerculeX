NUM_SIMULATIONS = 100
EPSILON = 0.3
CONSTANT = 0.5

REGULARIZER = 0.00001
LEARNING_RATE = 0.1
INPUT_DIM = (11, 11, 1)
OUTPUT_DIM = 121
MOMENTUM = 0.9
HIDDEN = [{'filters': 64, 'kernel_size': (4, 4)},
          {'filters': 64, 'kernel_size': (4, 4)}]

EPOCHS = 15
VERBOSE = 3
VALIDATION_SPLIT = 0.2
BATCH_SIZE = None