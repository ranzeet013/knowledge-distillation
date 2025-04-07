Knowledge Distillation Pipeline Documentation
============================================

.. contents:: Table of Contents
   :depth: 3
   :local:

Overview
--------
This documentation covers the complete knowledge distillation pipeline implemented in ``main.py``. The system trains:

1. A complex teacher model
2. A regular student model
3. A distilled student model (using knowledge from the teacher)

and evaluates their performance on the MNIST dataset.

Project Structure
-----------------
::

    knowledge-distillation/
    ├── main.py                # Main pipeline script
    ├── models.py              # Model architectures
    ├── train.py               # Training utilities
    ├── distillation.py        # Distillation components
    ├── visualization.py       # Plotting functions
    ├── config.py              # Configuration settings
    ├── docs/
    │   └── pipeline.rst       # This documentation file
    ├── models/                # Saved models
    ├── results/               # Output files
    │   ├── metrics/           # Evaluation metrics
    │   ├── predictions/       # Prediction examples
    │   └── plots/             # Training curves and comparisons

Main Pipeline (main.py)
-----------------------

The main script orchestrates the complete workflow through several key functions:

Initialization
~~~~~~~~~~~~~~

.. py:function:: initialize_directories()
   
   Creates necessary directories for storing:
   
   - Models (``models/``)
   - Results (``results/{metrics,predictions,plots}``)

Data Preparation
~~~~~~~~~~~~~~~

.. py:function:: load_and_split_data()
   
   - Loads MNIST dataset
   - Splits training data into train/validation sets
   - Creates DataLoader instances
   - Returns: (train_loader, val_loader, test_loader, train_subset, val_subset, test_dataset)

Teacher Training
~~~~~~~~~~~~~~~

.. py:function:: train_teacher(train_loader, val_loader)
   
   - Initializes TeacherModel
   - Trains using Adam optimizer and CrossEntropyLoss
   - Saves training curves
   - Returns: (model, train_loss, val_loss, train_acc, val_acc)

Distillation Setup
~~~~~~~~~~~~~~~~~

.. py:function:: prepare_distillation_data(teacher, train_loader, val_loader, train_subset, val_subset)
   
   - Gets teacher logits for training data
   - Creates DistillDataset instances
   - Returns: (train_distill_loader, val_distill_loader)

Student Training
~~~~~~~~~~~~~~~

.. py:function:: train_students(teacher, train_loader, val_loader, train_distill_loader, val_distill_loader)
   
   Trains both:
   
   1. Regular student (on original data)
   2. Distilled student (using teacher logits)
   
   Returns tuple containing both models and their training metrics

Evaluation
~~~~~~~~~

.. py:function:: evaluate_all_models(teacher, student_reg, student_dist, test_loader)
   
   Evaluates all models on test set and prints:
   
   - Test accuracy for each model
   - Prediction distributions
   
   Returns test metrics tuple

Result Saving
~~~~~~~~~~~~

.. py:function:: save_and_visualize_results(teacher, student_reg, student_dist, train_metrics, test_metrics, test_loader)
   
   Saves:
   
   - Model checkpoints (``.pth`` files)
   - Training curves
   - Accuracy comparisons
   - Prediction visualizations

Execution Flow
--------------

The main execution follows this sequence:

1. Initialize directories
2. Load and split MNIST data
3. Train teacher model
4. Prepare distillation datasets
5. Train both student models
6. Evaluate all models
7. Save results and visualizations

Usage Examples
--------------

Command Line Execution
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python main.py

Programmatic Usage
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from main import load_and_split_data, train_teacher
   
   # Get data loaders
   train_loader, val_loader, *_ = load_and_split_data()
   
   # Train teacher model
   teacher, *_ = train_teacher(train_loader, val_loader)

Output Files
------------

After successful execution, these files are generated:

- ``models/``
  - ``teacher_model.pth``
  - ``student_regular.pth``
  - ``student_distilled.pth``
  
- ``results/plots/``
  - ``teacher_training.png``
  - ``training_curves.png``
  - ``accuracy_comparison.png``
  - ``prediction_examples.png``
  
- ``results/metrics/``
  - Various evaluation metrics files

Configuration
-------------

All parameters are configured in ``config.py``:

.. code-block:: python

   # Device configuration
   DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   # Training parameters
   BATCH_SIZE = 128
   TEST_SIZE = 0.1
   RANDOM_STATE = 42
   EPOCHS = 10
   
   # Distillation parameters
   TEMPERATURE = 5
   ALPHA = 0.5  # Weight for hard vs soft loss

Dependencies
------------

Required Python packages:

- torch
- torchvision
- numpy
- matplotlib
- scikit-learn
- seaborn (for some visualizations)

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.