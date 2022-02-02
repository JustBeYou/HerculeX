NUM_SIMULATIONS = 300
EPSILON = 0.2
CONSTANT = 0.5

BOARD_SIZE = 3

REGULARIZER = 0.0001
LEARNING_RATE = 0.001
INPUT_DIM = (BOARD_SIZE, BOARD_SIZE, 7)
OUTPUT_DIM = BOARD_SIZE ** 2
MOMENTUM = 0.8
HIDDEN = [{'filters': 128, 'kernel_size': (3, 3)},
          {'filters': 128, 'kernel_size': (3, 3)},
          {'filters': 128, 'kernel_size': (4, 4)},
          {'filters': 128, 'kernel_size': (4, 4)},
          {'filters': 128, 'kernel_size': (4, 4)}]

TRAIN_ITERS = 100
EPOCHS = 50
VERBOSE = 2
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 256