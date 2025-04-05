"""
A module for loading trained teacher and student models and making digit predictions.
Includes functionality for visualizing predictions and comparing model outputs.
"""

import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import files
import os

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

class DigitPredictor:
    """
    A class for loading trained models and making digit predictions.
    
    Attributes:
        device (torch.device): The device (CPU/GPU) being used
        transform (transforms.Compose): Image transformations
        models (dict): Dictionary of loaded models
    """
    
    def __init__(self, model_dir='/content/drive/MyDrive/kd/models'):
        """
        Initialize the predictor and load models.
        
        Args:
            model_dir (str): Path to directory containing saved models
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.model_dir = model_dir
        
        # Create directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load models
        self.models = {
            'teacher': self._load_model('teacher_model'),
            'student_regular': self._load_model('student_regular'),
            'student_distilled': self._load_model('student_distilled')
        }

    def _load_model(self, model_name):
        """
        Helper method to load a specific model.
        
        Args:
            model_name (str): Name of the model to load
            
        Returns:
            torch.nn.Module: The loaded model
        """
        if 'teacher' in model_name:
            model = TeacherModel().to(self.device)
        else:
            model = StudentModel().to(self.device)

        model_path = f'{self.model_dir}/{model_name}.pth'
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            return model
        except FileNotFoundError:
            print(f"Warning: Model {model_name} not found at {model_path}")
            return None
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            return None

    def predict(self, image):
        """
        Make predictions on a single image using all loaded models.
        
        Args:
            image: PIL Image or numpy array (28x28)
            
        Returns:
            dict: Dictionary with predictions and probabilities for each model
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Preprocess
        image = self.transform(image).unsqueeze(0).to(self.device)

        results = {}
        with torch.no_grad():
            for name, model in self.models.items():
                if model is None:
                    continue
                    
                output = model(image)
                probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
                pred = torch.argmax(output).item()

                results[name] = {
                    'prediction': pred,
                    'probabilities': probs,
                    'confidence': probs[pred]
                }

        return results

    def visualize_prediction(self, image, results, save_dir=None):
        """
        Visualize predictions from all models.
        
        Args:
            image: Input image (PIL or numpy array)
            results: Prediction results from predict() method
            save_dir: Directory to save visualization (optional)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        plt.figure(figsize=(15, 5))

        # Show image
        plt.subplot(1, len(results)+1, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Input Image')
        plt.axis('off')

        # Show predictions
        for i, (name, result) in enumerate(results.items()):
            plt.subplot(1, len(results)+1, i+2)
            bars = plt.bar(range(10), result['probabilities'], color='skyblue')
            bars[result['prediction']].set_color('orange')
            plt.title(f"{name}\nPred: {result['prediction']} ({result['confidence']:.2f})")
            plt.ylim(0, 1)
            plt.xlabel('Digit')
            plt.ylabel('Probability')

        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'prediction_comparison.png')
            plt.savefig(save_path)
            print(f"Saved visualization to {save_path}")
        plt.show()

def upload_image():
    """
    Helper function to upload images in Colab.
    
    Returns:
        numpy.ndarray: The uploaded image as a numpy array
    """
    uploaded = files.upload()
    for filename in uploaded.keys():
        try:
            img = Image.open(filename).convert('L')  # Convert to grayscale
            img = img.resize((28, 28))  # Resize to MNIST dimensions
            return np.array(img)
        except Exception as e:
            print(f"Error processing image: {str(e)}")
    return None

def demo_predictor(model_dir='/content/drive/MyDrive/kd/models', 
                  save_dir='/content/drive/MyDrive/kd/results/predictions'):
    """
    Demonstration function showing how to use the DigitPredictor.
    
    Args:
        model_dir: Path to directory containing saved models
        save_dir: Directory to save output visualizations
    """
    # Initialize predictor
    predictor = DigitPredictor(model_dir=model_dir)
    
    print("Digit Prediction Demo")
    print("Choose an option:")
    print("1. Use a random test image")
    print("2. Upload your own image")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        # Create a random digit-like array for demonstration
        sample_image = np.random.rand(28, 28) * 0.2  # Mostly dark
        sample_image[10:18, 10:18] = 0.8  # Add a bright square
    elif choice == '2':
        print("Please upload an image file (it will be resized to 28x28):")
        sample_image = upload_image()
        if sample_image is None:
            print("No image uploaded, using random image instead")
            sample_image = np.random.rand(28, 28) * 0.2
            sample_image[10:18, 10:18] = 0.8
    else:
        print("Invalid choice, using random image")
        sample_image = np.random.rand(28, 28) * 0.2
        sample_image[10:18, 10:18] = 0.8

    # Predict and visualize
    predictions = predictor.predict(sample_image)
    predictor.visualize_prediction(sample_image, predictions, save_dir=save_dir)

    print("\nPrediction Summary:")
    for model, result in predictions.items():
        print(f"{model:>15}: Predicted {result['prediction']} with confidence {result['confidence']:.2f}")

if __name__ == '__main__':
    # Mount Google Drive if running in Colab
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        IN_COLAB = True
    except:
        IN_COLAB = False
    
    # Run the demo
    demo_predictor()