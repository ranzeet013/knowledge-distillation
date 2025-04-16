# 🧠 Knowledge Distillation Pipeline

A complete pipeline for training and evaluating models on the MNIST dataset using knowledge distillation. It involves a teacher model, a regular student model, and a distilled student model.

---

## 📚 Table of Contents
- [Overview](#overview)  
- [Pipeline Structure](#pipeline-structure)  
- [Execution Flow](#execution-flow)  
- [Output Files](#output-files)  
- [Configuration](#configuration)   
- [License](#license)  

---

## 📝 Overview

This pipeline performs:
- Training of a **teacher model**  
- Training of a **regular student model**  
- Training of a **distilled student model** using teacher logits  
- Evaluation of all models  
- Saving of models, metrics, and visualizations  

---

## 🔁 Pipeline Structure

### Initialization  
Creates folders for saving models and results:
- `models/`
- `results/metrics/`, `results/predictions/`, `results/plots/`

### Data Preparation  
Loads MNIST, splits into training/validation/test sets, and returns `DataLoader` instances.

### Teacher Training  
Trains a high-capacity model using standard cross-entropy loss.

### Distillation Setup  
Uses the trained teacher to generate logits for training samples. Creates custom datasets for soft label training.

### Student Training  
- Trains a **regular student** on hard labels.  
- Trains a **distilled student** on a mix of hard and soft labels from the teacher.

### Evaluation  
Evaluates all models on the MNIST test set and computes performance metrics.

### Result Saving  
Saves models, training curves, accuracy plots, and prediction examples.

---

## ⚙️ Execution Flow

1. Initialize directories  
2. Load and split data  
3. Train teacher model  
4. Prepare distillation data  
5. Train student models  
6. Evaluate all models  
7. Save models and visualizations  

---

## 📤 Output Files

**Models (`models/`)**  
- `teacher_model.pth`  
- `student_regular.pth`  
- `student_distilled.pth`  

**Plots (`results/plots/`)**  
- `teacher_training.png`  
- `training_curves.png`  
- `accuracy_comparison.png`  
- `prediction_examples.png`  

**Metrics (`results/metrics/`)**  
- Evaluation results for all models  

---

## ⚙️ Configuration

Defined in `config.py`:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
TEST_SIZE = 0.1
RANDOM_STATE = 42
EPOCHS = 10
TEMPERATURE = 5
ALPHA = 0.5
```
## ⚙️ License 
This project is licensed under the MIT License. See the LICENSE file for details.
