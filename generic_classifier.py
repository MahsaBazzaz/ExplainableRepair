import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import CAVE_CHANNELS, CAVE_COLS, CAVE_ROWS, MARIO_CHANNELS, MARIO_COLS, MARIO_ROWS, SUPERCAT_CHANNELS, SUPERCAT_COLS, SUPERCAT_ROWS, mario_chars_unique
from constants import TOMB_COLS, TOMB_ROWS, TOMB_CHANNELS, tomb_chars_unique
from utility import get_dataset
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):
    def __init__(self, cols, rows, channels):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(cols * rows * channels, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

def predict(model, input_tensor):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
        return predicted_class

def predict_proba(model, input_tensor):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        prob = F.softmax(output, dim=1)
        return prob

