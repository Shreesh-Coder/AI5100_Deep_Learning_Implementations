import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

plt.style.use('seaborn')

# Set random seed for reproducibility
np.random.seed(8999)
torch.manual_seed(8999)

# Define input arrays
X, Y = [], []

# Define length limits for individual data points
LOW_LIM = 2
UPR_LIM = 9

# Generate 5000 data points
for _ in range(5000):
    LEN = np.random.randint(LOW_LIM, UPR_LIM)
    D1 = np.random.random((LEN)).astype(np.double)
    D2 = np.zeros(LEN)
    ONEs = np.random.randint(LEN, size=(2))
    while ONEs[0] == ONEs[1]:
        ONEs = np.random.randint(LEN, size=(2))
    D2[ONEs] = 1
    X.append(torch.from_numpy(np.array([(D1[i], D2[i]) for i in range(LEN)])).double())
    Y.append(torch.from_numpy(np.array([D1[ONEs[0]] + D1[ONEs[1]]])))

# Convert to numpy array
X = np.array(X)
Y = np.array(Y)

# Print shapes of the arrays
print(X.shape, Y.shape)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
max_length = max(len(sequence) for sequence in X)
X_train = np.array([sequence + [0]*(max_length - len(sequence)) for sequence in X])

# Select the device (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Define the Elman RNN model
class ElmanRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ElmanRNN, self).__init__()
        self.U = nn.Linear(input_size, hidden_size, bias=False)
        self.W = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, output_size)
        self.double()

    def forward(self, input, hidden_state):
        Ux = self.U(input)
        Wh = self.W(hidden_state)
        Ht = torch.tanh(Ux + Wh)
        output = self.V(Ht)
        return output, Ht

# Initialize parameters
MAX_EPOCH = 25
INPUT_SIZE = 2
HIDDEN_SIZE = 10
OUTPUT_SIZE = 1
LEARNING_RATE = 0.005

# Instantiate the model
model = ElmanRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
print(model)

# Training process
train_loss_history = {}
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in tqdm(range(MAX_EPOCH)):
    epoch_loss_history = []
    for X_batch, Y_batch in zip(X_train, y_train):
        model.zero_grad()
        loss = 0
        hidden = torch.zeros(1, HIDDEN_SIZE, requires_grad=False, dtype=torch.float64).to(device)
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        output = 0
        for i in range(X_batch.shape[0]):
            output, hidden = model(X_batch[i, :], hidden)
        loss = mse_loss(output, Y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()
        epoch_loss_history.append(loss.detach().item())
    train_loss_history[epoch] = torch.tensor(epoch_loss_history).mean()

# Function to plot training loss
def plot_loss_graph(history):
    plt.plot(list(history.keys()), list(history.values()), label='Loss')
    plt.title('Loss at every epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

plot_loss_graph(train_loss_history)

# Evaluate the model's accuracy
torch.manual_seed(8999)
total_test_points = len(X_test)
correct_predictions = 0

for X_batch, Y_batch in zip(X_test, y_test):
    X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
    hidden, output = torch.zeros(1, HIDDEN_SIZE, requires_grad=False, dtype=torch.float64).to(device), 0
    for i in range(X_batch.shape[0]):
        output, hidden = model(X_batch[i, :], hidden)
    if abs(output.item() - Y_batch.item()) < 0.02:
        correct_predictions += 1

accuracy = correct_predictions / total_test_points
print(f"Accuracy: {accuracy:.4f}")
