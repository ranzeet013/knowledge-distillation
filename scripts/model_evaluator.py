"""
A module for evaluating trained teacher and student models on the MNIST test set.
Includes functionality for calculating accuracy, saving results, and visualizing comparisons.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from google.colab import drive

class TeacherModel(torch.nn.Module):
    """The teacher model architecture used for training"""
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1)
        self.fc1 = torch.nn.Linear(64*3*3, 64)
        self.fc2 = torch.nn.Linear(64, 10)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.maxpool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.maxpool(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.maxpool(x)
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class StudentModel(torch.nn.Module):
    """The student model architecture used for training"""
    def __init__(self):
        super(StudentModel, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 32)
        self.fc2 = torch.nn.Linear(32, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ModelEvaluator:
    """
    A class for evaluating trained models on the MNIST test set.
    
    Attributes:
        device (torch.device): The device (CPU/GPU) being used
        model_paths (dict): Paths to saved model checkpoints
        results_dir (str): Directory to save evaluation results
        test_loader (DataLoader): Loader for MNIST test set
    """
    
    def __init__(self, model_dir='/content/drive/MyDrive/kd/models',
                results_dir='/content/drive/MyDrive/kd/results/metrics'):
        """
        Initialize the evaluator and load test data.
        
        Args:
            model_dir (str): Directory containing saved models
            results_dir (str): Directory to save evaluation results
        """
        # Mount Google Drive if running in Colab
        try:
            drive.mount('/content/drive', force_remount=True)
            self.in_colab = True
        except:
            self.in_colab = False
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Configure paths
        self.model_paths = {
            'teacher': os.path.join(model_dir, 'teacher_model.pth'),
            'student_regular': os.path.join(model_dir, 'student_regular.pth'),
            'student_distilled': os.path.join(model_dir, 'student_distilled.pth')
        }
        self.results_dir = results_dir
        
        # Create directories if they don't exist
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Load test data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        self.test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    def load_model(self, model_type):
        """
        Load a pretrained model from disk.
        
        Args:
            model_type (str): Type of model to load ('teacher', 'student_regular', 'student_distilled')
            
        Returns:
            torch.nn.Module: The loaded model
        """
        if model_type == 'teacher':
            model = TeacherModel().to(self.device)
        else:
            model = StudentModel().to(self.device)

        try:
            model.load_state_dict(torch.load(self.model_paths[model_type], map_location=self.device))
            model.eval()
            return model
        except FileNotFoundError:
            print(f"Error: Model file not found at {self.model_paths[model_type]}")
            return None
        except Exception as e:
            print(f"Error loading {model_type} model: {str(e)}")
            return None

    def evaluate_model(self, model):
        """
        Evaluate a model on the test set.
        
        Args:
            model (torch.nn.Module): Model to evaluate
            
        Returns:
            tuple: (accuracy, predictions, targets) where:
                - accuracy (float): Test accuracy percentage
                - predictions (np.array): Array of model predictions
                - targets (np.array): Array of ground truth labels
        """
        if model is None:
            return 0.0, None, None
            
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        accuracy = 100 * correct / total
        return accuracy, np.array(all_preds), np.array(all_targets)

    def evaluate_all_models(self):
        """
        Evaluate all models and save results.
        
        Returns:
            dict: Dictionary containing evaluation results for all models
        """
        results = {}

        for model_name in self.model_paths.keys():
            print(f"\nEvaluating {model_name}...")
            model = self.load_model(model_name)
            accuracy, preds, targets = self.evaluate_model(model)
            
            results[model_name] = {
                'accuracy': accuracy,
                'predictions': preds,
                'targets': targets
            }
            
            print(f"{model_name} accuracy: {accuracy:.2f}%")

        # Save results
        results_path = os.path.join(self.results_dir, 'test_results.pt')
        torch.save(results, results_path)
        print(f"\nTest results saved to {results_path}")

        # Plot and save comparison
        self.plot_results(results)
        
        return results

    def plot_results(self, results):
        """
        Plot and save a comparison of model accuracies.
        
        Args:
            results (dict): Evaluation results from evaluate_all_models()
        """
        models = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in models]

        plt.figure(figsize=(10, 5))
        bars = plt.bar(models, accuracies, color=['blue', 'orange', 'green'])
        plt.ylabel('Accuracy (%)')
        plt.title('Model Comparison on Test Set')
        plt.ylim(0, 100)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom')

        plot_path = os.path.join(self.results_dir, 'accuracy_comparison.png')
        plt.savefig(plot_path)
        print(f"Accuracy plot saved to {plot_path}")
        plt.show()

def main():
    """Main function to run model evaluation"""
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all_models()
    
    # Print final results
    print("\nFinal Evaluation Results:")
    for model_name, data in results.items():
        print(f"{model_name:>15}: {data['accuracy']:.2f}% accuracy")

if __name__ == '__main__':
    main()