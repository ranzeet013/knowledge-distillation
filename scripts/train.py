import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from distillation import distillation_loss

def train_model(model, train_loader, val_loader, epochs, optimizer, criterion, device):
    """
    Trains a model using standard training or knowledge distillation.

    Args:
        model (nn.Module): The neural network to be trained.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        epochs (int): Number of epochs for training.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (function): Loss function (supports both standard and distillation loss).
        device (torch.device): Device to run training on (CPU/GPU).

    Returns:
        tuple: Lists of training loss, validation loss, training accuracy, and validation accuracy per epoch.
    """
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            if len(batch) == 3:  # Distillation case
                data, target, teacher_logits = batch
                data, target, teacher_logits = data.to(device), target.to(device), teacher_logits.to(device)
                output = model(data)
                loss = criterion(output, target, teacher_logits)
            else:  # Regular training case
                data, target = batch
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_acc.append(train_accuracy)

        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_acc.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    return train_losses, val_losses, train_acc, val_acc

def evaluate(model, loader, criterion, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model (nn.Module): The trained model to be evaluated.
        loader (DataLoader): DataLoader for the dataset.
        criterion (function): Loss function (supports both standard and distillation loss).
        device (torch.device): Device to perform evaluation on (CPU/GPU).

    Returns:
        tuple: Average loss and accuracy on the given dataset.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:  # Distillation case
                data, target, teacher_logits = batch
                data, target, teacher_logits = data.to(device), target.to(device), teacher_logits.to(device)
                output = model(data)
                loss = criterion(output, target, teacher_logits)
            else:  # Regular case
                data, target = batch
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    loss = running_loss / len(loader)
    accuracy = correct / total
    return loss, accuracy
