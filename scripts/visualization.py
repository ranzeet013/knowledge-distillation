import matplotlib.pyplot as plt
import torch
from config import DEVICE

def save_training_curves(teacher_train_acc, teacher_val_acc, 
                        regular_train_acc, regular_val_acc,
                        distilled_train_acc, distilled_val_acc,
                        teacher_train_loss, teacher_val_loss,
                        regular_train_loss, regular_val_loss,
                        distilled_train_loss, distilled_val_loss,
                        save_path='/content/drive/MyDrive/kd/plots/training_curves.png'):
    """
    Plot and save training curves for accuracy and loss.
    
    Args:
        teacher_train_acc: List of teacher training accuracies
        teacher_val_acc: List of teacher validation accuracies
        regular_train_acc: List of regular student training accuracies
        regular_val_acc: List of regular student validation accuracies
        distilled_train_acc: List of distilled student training accuracies
        distilled_val_acc: List of distilled student validation accuracies
        teacher_train_loss: List of teacher training losses
        teacher_val_loss: List of teacher validation losses
        regular_train_loss: List of regular student training losses
        regular_val_loss: List of regular student validation losses
        distilled_train_loss: List of distilled student training losses
        distilled_val_loss: List of distilled student validation losses
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(teacher_train_acc, label='Teacher Train')
    plt.plot(teacher_val_acc, label='Teacher Val')
    plt.plot(regular_train_acc, label='Student Regular Train')
    plt.plot(regular_val_acc, label='Student Regular Val')
    plt.plot(distilled_train_acc, label='Student Distilled Train')
    plt.plot(distilled_val_acc, label='Student Distilled Val')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(teacher_train_loss, label='Teacher Train')
    plt.plot(teacher_val_loss, label='Teacher Val')
    plt.plot(regular_train_loss, label='Student Regular Train')
    plt.plot(regular_val_loss, label='Student Regular Val')
    plt.plot(distilled_train_loss, label='Student Distilled Train')
    plt.plot(distilled_val_loss, label='Student Distilled Val')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def save_accuracy_comparison(teacher_test_acc, regular_test_acc, distilled_test_acc,
                           save_path='/content/drive/MyDrive/kd/plots/accuracy_comparison.png'):
    """
    Create and save a bar plot comparing final test accuracies.
    
    Args:
        teacher_test_acc: Teacher model test accuracy
        regular_test_acc: Regular student test accuracy
        distilled_test_acc: Distilled student test accuracy
        save_path: Path to save the plot
    """
    models = ['Teacher', 'Student Regular', 'Student Distilled']
    accuracies = [teacher_test_acc, regular_test_acc, distilled_test_acc]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, accuracies, color=['blue', 'orange', 'green'])
    plt.ylabel('Test Accuracy')
    plt.title('Model Comparison on Test Set')
    plt.ylim(0.9, 1.0)

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')

    plt.savefig(save_path)
    plt.show()

def save_results_summary(teacher_test_acc, regular_test_acc, distilled_test_acc,
                       save_path='/content/drive/MyDrive/kd/results/summary.txt'):
    """
    Save model performance results to a text file.
    
    Args:
        teacher_test_acc: Teacher model test accuracy
        regular_test_acc: Regular student test accuracy
        distilled_test_acc: Distilled student test accuracy
        save_path: Path to save the summary file
    """
    with open(save_path, 'w') as f:
        f.write(f"Teacher test accuracy: {teacher_test_acc:.4f}\n")
        f.write(f"Regular student test accuracy: {regular_test_acc:.4f}\n")
        f.write(f"Distilled student test accuracy: {distilled_test_acc:.4f}\n")
        f.write(f"Improvement from distillation: {distilled_test_acc - regular_test_acc:.4f}\n")

def visualize_predictions(teacher_model, student_regular, student_distilled, 
                        loader, num_samples=5, 
                        save_path='/content/drive/MyDrive/kd/plots/prediction_examples.png'):
    """
    Visualize and save sample predictions from all models.
    
    Args:
        teacher_model: Trained teacher model
        student_regular: Regular student model
        student_distilled: Distilled student model
        loader: DataLoader for getting samples
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
    """
    dataiter = iter(loader)
    images, labels = next(dataiter)
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    # Get predictions
    with torch.no_grad():
        teacher_preds = torch.argmax(teacher_model(images[:num_samples]), 1)
        regular_preds = torch.argmax(student_regular(images[:num_samples]), 1)
        distilled_preds = torch.argmax(student_distilled(images[:num_samples]), 1)

    # Plot
    plt.figure(figsize=(15, 3 * num_samples))
    for i in range(num_samples):
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(images[i].cpu().squeeze(), cmap='gray')
        plt.title(f"True: {labels[i].item()}\nTeacher: {teacher_preds[i].item()}")
        plt.axis('off')

        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(images[i].cpu().squeeze(), cmap='gray')
        plt.title(f"True: {labels[i].item()}\nRegular: {regular_preds[i].item()}")
        plt.axis('off')

        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(images[i].cpu().squeeze(), cmap='gray')
        plt.title(f"True: {labels[i].item()}\nDistilled: {distilled_preds[i].item()}")
        plt.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def save_all_visualizations(teacher_model, student_regular, student_distilled,
                          teacher_train_acc, teacher_val_acc,
                          regular_train_acc, regular_val_acc,
                          distilled_train_acc, distilled_val_acc,
                          teacher_train_loss, teacher_val_loss,
                          regular_train_loss, regular_val_loss,
                          distilled_train_loss, distilled_val_loss,
                          teacher_test_acc, regular_test_acc, distilled_test_acc,
                          test_loader):
    """
    Save all visualizations and results in one call.
    
    Args:
        All model objects and their training metrics
        test_loader: DataLoader for test set
    """
    # Save training curves
    save_training_curves(
        teacher_train_acc, teacher_val_acc,
        regular_train_acc, regular_val_acc,
        distilled_train_acc, distilled_val_acc,
        teacher_train_loss, teacher_val_loss,
        regular_train_loss, regular_val_loss,
        distilled_train_loss, distilled_val_loss
    )
    
    # Save accuracy comparison
    save_accuracy_comparison(teacher_test_acc, regular_test_acc, distilled_test_acc)
    
    # Save results summary
    save_results_summary(teacher_test_acc, regular_test_acc, distilled_test_acc)
    
    # Save prediction examples
    visualize_predictions(
        teacher_model, student_regular, student_distilled,
        test_loader
    )
    
    print("All models, plots, and results have been saved in their respective folders.")