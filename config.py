
TRAIN_CSV = "data/train_balanced.csv"
TEST_CSV = "data/test.csv"
MODEL_PATH = "transformer_model.pth"

# Model architecture
SRC_VOCAB_SIZE = 36000
TGT_VOCAB_SIZE = 3  # Positive, Negative, Neutral
SRC_SEQ_LEN = 256
TGT_SEQ_LEN = 3
D_MODEL = 256
N_LAYERS = 4
H = 8
D_FF = 1024
DROPOUT = 0.5

# Training settings
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 1e-4  #3e-4
DEVICE = "cuda"

# Extras
EARLY_STOPPING_PATIENCE = 3
WEIGHTED_LOSS = True
