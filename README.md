# Hybrid Quantum-Classical Neural Network (QCNN) for Breast Cancer Classification

## Overview

This project implements a hybrid quantum-classical neural network to classify breast cancer histopathology images. By integrating **PyTorch** for classical feature extraction and **Qiskit** for quantum circuit execution, this pipeline evaluates the potential of Quantum Machine Learning (QML) in oncology and medical image analysis.

## Architecture

Based on the `Hybrid_QCNN_BreakHis` model state dictionary, the pipeline consists of four integrated stages:

1. **Classical Feature Extraction (`feature_extractor`):** A classical Convolutional Neural Network processes the raw, high-resolution histopathology images to extract a feature map.

2. **Dimensionality Reduction (`reducer`):** A linear layer compresses the extracted classical features into a lower-dimensional vector that matches the number of available qubits for quantum encoding.

3. **Quantum Computation (`quantum`):** The scaled features are embedded into a quantum state. A parameterized quantum circuit (ansatz) processes the state, and a measurement is taken to output expectation values. 

4. **Classical Classification Head (`classifier`):** A final fully connected classical layer maps the quantum measurements to the final target classes (Benign vs. Malignant).

## Dataset

This project utilizes the **BreaKHis** (Breast Cancer Histopathological Database) dataset. It is a standard benchmark collection of microscopic biopsy images of breast tumors:

* **Classes:** The dataset is broadly categorized into **Benign** and **Malignant** tumors, with further sub-classifications for specific tumor types (e.g., adenosis, fibroadenoma, ductal carcinoma).

* **Magnification Factors:** The clinical images are captured at four different objective magnifications: 40X, 100X, 200X, and 400X.

* **Volume:** It contains 7,909 high-resolution histopathological images.

## Requirements

Ensure you have Python 3.8+ installed. The following core libraries are required:

* PyTorch & torchvision
* Qiskit, qiskit-machine-learning
* NumPy
* Pandas
* scikit-learn
* Matplotlib

## Installation & Usage

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install the required dependencies:
   ```bash
   python -m pip install --upgrade pip
pip install numpy pandas pillow tqdm matplotlib torch scikit-learn torchvision qiskit qiskit-machine-learning```

3. Download the BreaKHis dataset and extract it into the `data/` directory.

4. Load the compiled PyTorch/Qiskit model weights for inference or fine-tuning:
   ```python
   import torch
   
   # Load the hybrid QCNN state dictionary
   # Ensure the 'Hybrid_QCNN_BreakHis.pth' or extracted directory is in the root path
   model_weights = torch.load('Hybrid_QCNN_BreakHis/data.pkl')
   ```

## Future Work

* **Multi-class Expansion:** Scaling the classification head to identify the specific tumor subtypes rather than just binary Benign/Malignant classification.
* **Hardware Execution:** Running the parameterized quantum circuit on real IBM Quantum hardware.
