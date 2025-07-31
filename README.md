# DTGNN_CLASS_GROUP

**Delaunay Triangulation Graph Neural Network (DTGNN) for Class-Imbalanced Graph Classification**

'D.T-Graph is reffered to Delaunay Triangulation Graphs here.'


## 🔍 Overview

This repository provides the implementation of **DTGNN**, a framework designed to address class imbalance biomedical images in graph classification tasks. The method dynamically adjusts graph topology during training and incorporates class-aware strategies to improve performance, especially on underrepresented classes.

The implementation supports multi-label or binary classification tasks using PyTorch Geometric, and includes tools for preprocessing, evaluation, training, visualization, and reproducibility each of which in their individual folder for easier management.

---

## 📂 Repository Structure

```bash

DTGNN_CLASS_GROUP/
├── model_1_DTGCN_Class/
│   ├── 1_Processing/                    # Processing the dataset and providing the D.T-Graphs
│   ├── 2_TrainingModel_20250412/        # Train and Evaluation modules 
│   └── Gleason_Class.md                 
├── model_2_DTGCN_Group/
│   ├── 0_Augmentation_Using_SMOTE_GAN/  # Augmentation processes
│   ├── 1_Processing/
│   └── 2_TrainingModel_20250412/
└── README.md
```

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/XXXXXXXXXXXXXX/DTGNN_CLASS_GROUP.git
   cd DTGNN_CLASS_GROUP
   ```
2. (Optional) Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirement.txt
   ```


## 📈 Evaluation Metrics

- **Exact Match Accuracy**
- **Label-wise Precision, Recall, and F1**
- **Balanced Accuracy**
- **Geometric Mean Score**
- **ROC Curve / Confusion Matrix** (optional)

## 📊 Results

Our model demonstrates significant improvements on long-tailed graph datasets compared to baseline GNNs. Full experimental results and visualizations are available in the results/ folder.


## 📘 Citation

If you use this codebase in your research, please cite:

```bibtex
@article{dtgnn2025,
    authors  = {},
    title   = {},
    journal = {},
    year    = {},
    note    = {}
}
```

## 🤝 Contribution

Contributions, issues, and feature requests are welcome!  
Feel free to open a pull request or submit an issue on GitHub.

## 📄 License

This code is released **for academic and non-commercial research use only**.

You are **not permitted** to use, reproduce, distribute, or modify any part of this repository for **commercial purposes**, including but not limited to:
- Use in proprietary or industrial projects
- Integration into commercial products or services
- Research sponsored by commercial entities

If you are affiliated with a company, industry partner, or plan to use this code for commercial applications, **you must obtain explicit written permission** from the author.

To request permission, please contact:

📧 MY_EMAIL@EMAIL.COM

🔗 

**© 2025 M....G. All rights reserved.**


## 📬 Contact

For questions or collaboration, please contact:  
**MY NAME**  
Email: MY_EMAILS@EMAIL.COM
