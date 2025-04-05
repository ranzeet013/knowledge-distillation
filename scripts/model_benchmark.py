"""
A comprehensive model evaluation module that calculates multiple metrics,
generates visualizations, and saves detailed reports for comparing model performance.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report,
                           roc_auc_score, accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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

class ModelBenchmark:
    """
    A comprehensive model evaluation class that calculates multiple metrics,
    generates visualizations, and saves detailed reports.
    """
    
    def __init__(self, model_dir='/content/drive/MyDrive/kd/models',
                results_dir='/content/drive/MyDrive/kd/results/metrics'):
        """
        Initialize the benchmark with model paths and results directory.
        
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
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.test_dataset = datasets.MNIST('./data', train=False, download=True, 
                                         transform=self.transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False)

    def load_model(self, model_type):
        """
        Load a pretrained model from disk with error handling.
        
        Args:
            model_type (str): Type of model to load ('teacher', 'student_regular', 'student_distilled')
            
        Returns:
            torch.nn.Module: The loaded model or None if loading failed
        """
        try:
            if model_type == 'teacher':
                model = TeacherModel().to(self.device)
            else:
                model = StudentModel().to(self.device)

            model.load_state_dict(torch.load(self.model_paths[model_type], 
                                  map_location=self.device))
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading {model_type}: {str(e)}")
            return None

    def calculate_metrics(self, y_true, y_pred, y_probs=None):
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true (np.array): Ground truth labels
            y_pred (np.array): Model predictions
            y_probs (np.array): Model predicted probabilities (optional)
            
        Returns:
            dict: Dictionary of calculated metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        }

        if y_probs is not None and len(np.unique(y_true)) > 2:  # ROC AUC for multi-class
            try:
                metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_probs, multi_class='ovo')
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_probs, multi_class='ovr')
            except Exception as e:
                print(f"Could not calculate ROC AUC: {str(e)}")

        return metrics

    def evaluate_model(self, model):
        """
        Extended evaluation with probabilities and metrics.
        
        Args:
            model (torch.nn.Module): Model to evaluate
            
        Returns:
            dict: Dictionary containing:
                - metrics: Various performance metrics
                - confusion_matrix: Confusion matrix
                - classification_report: Detailed classification report
                - predictions: Model predictions
                - probabilities: Prediction probabilities
                - targets: Ground truth labels
        """
        if model is None:
            return None
            
        model.eval()
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_probs = np.array(all_probs)

        metrics = self.calculate_metrics(y_true, y_pred, y_probs)
        cm = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, output_dict=True)

        return {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions': y_pred,
            'probabilities': y_probs,
            'targets': y_true
        }

    def plot_confusion_matrix(self, cm, model_name):
        """
        Plot and save a confusion matrix.
        
        Args:
            cm (np.array): Confusion matrix
            model_name (str): Name of the model
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=range(10), yticklabels=range(10))
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        save_path = os.path.join(self.results_dir, f'{model_name}_cm.png')
        plt.savefig(save_path)
        plt.show()

    def save_metrics_report(self, results):
        """
        Save comprehensive metrics report to CSV and Excel.
        
        Args:
            results (dict): Evaluation results from evaluate_model()
            
        Returns:
            pd.DataFrame: DataFrame containing the metrics
        """
        metrics_data = []

        for model_name, result in results.items():
            if result is None:
                continue
                
            row = {'model': model_name}
            row.update(result['metrics'])

            # Add per-class precision/recall/f1
            class_report = result['classification_report']
            for class_id in range(10):
                if str(class_id) in class_report:
                    row[f'class_{class_id}_precision'] = class_report[str(class_id)]['precision']
                    row[f'class_{class_id}_recall'] = class_report[str(class_id)]['recall']
                    row[f'class_{class_id}_f1'] = class_report[str(class_id)]['f1-score']

            metrics_data.append(row)

        df = pd.DataFrame(metrics_data)

        # Save to multiple formats
        csv_path = os.path.join(self.results_dir, 'detailed_metrics.csv')
        excel_path = os.path.join(self.results_dir, 'detailed_metrics.xlsx')

        df.to_csv(csv_path, index=False)
        df.to_excel(excel_path, index=False)

        print(f"\nSaved detailed metrics to:\n{csv_path}\n{excel_path}")
        return df

    def plot_comparative_metrics(self, results):
        """
        Plot comparison of key metrics across models.
        
        Args:
            results (dict): Evaluation results from evaluate_model()
        """
        metrics_to_plot = ['accuracy', 'f1_macro', 'f1_weighted']

        plt.figure(figsize=(15, 5))
        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(1, len(metrics_to_plot), i+1)
            values = [results[m]['metrics'][metric] for m in results if results[m] is not None]
            model_names = [m for m in results if results[m] is not None]
            
            bars = plt.bar(model_names, values)
            plt.title(metric.replace('_', ' ').title())
            plt.ylabel('Score')
            plt.ylim(0, 1)

            # Add values on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f"{height:.4f}", ha='center')

        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'metrics_comparison.png')
        plt.savefig(save_path)
        plt.show()

    def run_benchmark(self):
        """
        Run comprehensive benchmark tests on all models.
        
        Returns:
            tuple: (results, metrics_df) where:
                - results: Dictionary of full evaluation results
                - metrics_df: DataFrame of key metrics
        """
        results = {}

        for model_name in self.model_paths.keys():
            print(f"\nEvaluating {model_name}...")
            model = self.load_model(model_name)
            result = self.evaluate_model(model)
            results[model_name] = result

            if result is None:
                continue

            # Print summary
            print(f"\n{model_name} Performance:")
            print(f"Accuracy: {result['metrics']['accuracy']:.4f}")
            print(f"Macro F1: {result['metrics']['f1_macro']:.4f}")
            print(f"Weighted F1: {result['metrics']['f1_weighted']:.4f}")

            # Plot confusion matrix
            self.plot_confusion_matrix(result['confusion_matrix'], model_name)

        # Save comprehensive results
        results_path = os.path.join(self.results_dir, 'full_test_results.pt')
        torch.save(results, results_path)
        print(f"\nSaved full results to {results_path}")

        # Generate and save detailed metrics report
        metrics_df = self.save_metrics_report(results)

        # Plot comparative metrics
        self.plot_comparative_metrics(results)

        return results, metrics_df

def main():
    """Main function to run the benchmark"""
    benchmark = ModelBenchmark()
    results, metrics_df = benchmark.run_benchmark()
    
    # Display top-level metrics
    print("\nFinal Benchmark Results:")
    print(metrics_df[['model', 'accuracy', 'f1_macro', 'f1_weighted']])

if __name__ == '__main__':
    main()