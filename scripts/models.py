import torch
import torch.nn as nn
import torch.nn.functional as F

class TeacherModel(nn.Module):
    """
    A larger CNN model used as the teacher in knowledge distillation.
    Architecture:
        - 3 convolutional layers with ReLU and MaxPooling
        - 2 fully connected layers with Dropout
    Input:
        Image tensor of shape (batch_size, 1, 28, 28)
    Output:
        Logits for 10 classes
    """
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class StudentModel(nn.Module):
    """
    A smaller, fully connected neural network used as the student.
    Architecture:
        - 2 fully connected layers
    Input:
        Image tensor of shape (batch_size, 1, 28, 28)
    Output:
        Logits for 10 classes
    """
    def __init__(self):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
