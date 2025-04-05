"""
This script orchestrates the complete knowledge distillation workflow:
1. Data preparation and splitting
2. Teacher model training
3. Student model training (regular and distilled)
4. Model evaluation and comparison
5. Visualization and result saving
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# Import custom modules
from models import TeacherModel, StudentModel
from train import train_model, evaluate
from distillation import DistillDataset, get_teacher_logits, distillation_loss
from visualization import save_training_curves, save_accuracy_comparison, visualize_predictions
from config import DEVICE, BATCH_SIZE, TEST_SIZE, RANDOM_STATE, EPOCHS, TEMPERATURE, ALPHA

# Initialize directories
def initialize_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/predictions', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)

def load_and_split_data():
    """Load MNIST data and create train/val/test splits"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load datasets
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # Split train into train and validation
    train_idx, val_idx = train_test_split(
        np.arange(len(train_dataset)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=train_dataset.targets
    )

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    print(f"\nDataset sizes:")
    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, train_subset, val_subset, test_dataset

def train_teacher(train_loader, val_loader):
    """Train the teacher model"""
    print("\nTraining Teacher Model...")
    teacher = TeacherModel().to(DEVICE)
    optimizer = torch.optim.Adam(teacher.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    train_loss, val_loss, train_acc, val_acc = train_model(
        teacher, train_loader, val_loader, 
        EPOCHS, optimizer, criterion, DEVICE
    )

    # Save training curves
    save_training_curves(
        train_acc, val_acc, [], [], [], [],
        train_loss, val_loss, [], [], [], [],
        save_path='results/plots/teacher_training.png'
    )

    return teacher, train_loss, val_loss, train_acc, val_acc

def prepare_distillation_data(teacher, train_loader, val_loader, train_subset, val_subset):
    """Prepare datasets with teacher logits for distillation"""
    print("\nPreparing distillation data...")
    train_logits = get_teacher_logits(teacher, train_loader, DEVICE)
    val_logits = get_teacher_logits(teacher, val_loader, DEVICE)

    # Create distillation datasets
    train_distill = DistillDataset(train_subset, train_logits)
    val_distill = DistillDataset(val_subset, val_logits)

    # Create data loaders
    train_distill_loader = DataLoader(train_distill, batch_size=BATCH_SIZE, shuffle=True)
    val_distill_loader = DataLoader(val_distill, batch_size=BATCH_SIZE)

    return train_distill_loader, val_distill_loader

def train_students(teacher, train_loader, val_loader, train_distill_loader, val_distill_loader):
    """Train both regular and distilled student models"""
    print("\nTraining Student Models...")
    
    # Initialize students
    student_reg = StudentModel().to(DEVICE)
    student_dist = StudentModel().to(DEVICE)

    # Train regular student
    print("\nTraining Regular Student...")
    reg_optimizer = torch.optim.Adam(student_reg.parameters())
    reg_criterion = torch.nn.CrossEntropyLoss()
    
    reg_train_loss, reg_val_loss, reg_train_acc, reg_val_acc = train_model(
        student_reg, train_loader, val_loader,
        EPOCHS, reg_optimizer, reg_criterion, DEVICE
    )

    # Train distilled student
    print("\nTraining Distilled Student...")
    dist_optimizer = torch.optim.Adam(student_dist.parameters())
    
    dist_train_loss, dist_val_loss, dist_train_acc, dist_val_acc = train_model(
        student_dist, train_distill_loader, val_distill_loader,
        EPOCHS, dist_optimizer, distillation_loss, DEVICE
    )

    return (
        student_reg, student_dist,
        reg_train_loss, reg_val_loss, reg_train_acc, reg_val_acc,
        dist_train_loss, dist_val_loss, dist_train_acc, dist_val_acc
    )

def evaluate_all_models(teacher, student_reg, student_dist, test_loader):
    """Evaluate all models on test set"""
    print("\nEvaluating Models on Test Set...")
    
    # Evaluate teacher
    teacher_acc, teacher_preds, teacher_targets = evaluate(teacher, test_loader, torch.nn.CrossEntropyLoss(), DEVICE)
    
    # Evaluate students
    reg_acc, reg_preds, reg_targets = evaluate(student_reg, test_loader, torch.nn.CrossEntropyLoss(), DEVICE)
    dist_acc, dist_preds, dist_targets = evaluate(student_dist, test_loader, torch.nn.CrossEntropyLoss(), DEVICE)

    print("\nTest Accuracies:")
    print(f"Teacher: {teacher_acc:.4f}")
    print(f"Regular Student: {reg_acc:.4f}")
    print(f"Distilled Student: {dist_acc:.4f}")

    return (
        teacher_acc, reg_acc, dist_acc,
        teacher_preds, reg_preds, dist_preds,
        teacher_targets
    )

def save_and_visualize_results(teacher, student_reg, student_dist, 
                             train_metrics, test_metrics, test_loader):
    """Save models and visualize results"""
    print("\nSaving models and results...")
    
    # Save models
    torch.save(teacher.state_dict(), 'models/teacher_model.pth')
    torch.save(student_reg.state_dict(), 'models/student_regular.pth')
    torch.save(student_dist.state_dict(), 'models/student_distilled.pth')

    # Unpack metrics
    (teacher_train_loss, teacher_val_loss, teacher_train_acc, teacher_val_acc,
     reg_train_loss, reg_val_loss, reg_train_acc, reg_val_acc,
     dist_train_loss, dist_val_loss, dist_train_acc, dist_val_acc) = train_metrics
    
    (teacher_acc, reg_acc, dist_acc, 
     teacher_preds, reg_preds, dist_preds, 
     targets) = test_metrics

    # Save training curves
    save_training_curves(
        teacher_train_acc, teacher_val_acc,
        reg_train_acc, reg_val_acc,
        dist_train_acc, dist_val_acc,
        teacher_train_loss, teacher_val_loss,
        reg_train_loss, reg_val_loss,
        dist_train_loss, dist_val_loss,
        save_path='results/plots/training_curves.png'
    )

    # Save accuracy comparison
    save_accuracy_comparison(
        teacher_acc, reg_acc, dist_acc,
        save_path='results/plots/accuracy_comparison.png'
    )

    # Visualize sample predictions
    visualize_predictions(
        teacher, student_reg, student_dist,
        test_loader,
        save_path='results/plots/prediction_examples.png'
    )

def main():
    """Main execution function"""
    # Initialize
    initialize_directories()
    
    # Load and prepare data
    train_loader, val_loader, test_loader, train_subset, val_subset, _ = load_and_split_data()

    # Train teacher
    teacher, teacher_train_loss, teacher_val_loss, teacher_train_acc, teacher_val_acc = train_teacher(train_loader, val_loader)

    # Prepare distillation data
    train_distill_loader, val_distill_loader = prepare_distillation_data(
        teacher, train_loader, val_loader, train_subset, val_subset
    )

    # Train students
    (student_reg, student_dist,
     reg_train_loss, reg_val_loss, reg_train_acc, reg_val_acc,
     dist_train_loss, dist_val_loss, dist_train_acc, dist_val_acc) = train_students(
        teacher, train_loader, val_loader, train_distill_loader, val_distill_loader
    )

    # Evaluate models
    test_metrics = evaluate_all_models(teacher, student_reg, student_dist, test_loader)

    # Package training metrics
    train_metrics = (
        teacher_train_loss, teacher_val_loss, teacher_train_acc, teacher_val_acc,
        reg_train_loss, reg_val_loss, reg_train_acc, reg_val_acc,
        dist_train_loss, dist_val_loss, dist_train_acc, dist_val_acc
    )

    # Save and visualize results
    save_and_visualize_results(
        teacher, student_reg, student_dist,
        train_metrics, test_metrics,
        test_loader
    )

    print("\nKnowledge distillation pipeline completed successfully!")

if __name__ == '__main__':
    main()