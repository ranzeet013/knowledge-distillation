import torch
from torchvision import transforms

# Device configuration: Use GPU if available, else fallback to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing: Convert images to tensors and normalize with MNIST mean and std
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Data loader and training setup
BATCH_SIZE = 128        # Number of samples per batch
TEST_SIZE = 0.1         # Validation set size as a fraction of training data
RANDOM_STATE = 42       # Seed for reproducibility
EPOCHS = 10             # Number of training epochs

# Knowledge distillation hyperparameters
TEMPERATURE = 5         # Temperature for softening probability distributions
ALPHA = 0.5             # Weight between hard loss and soft loss (0.5 = equal weight)
