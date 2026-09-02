# 📊 Erasmus Visual Data Analysis Suite

![Python](https://img.shields.io/badge/Language-Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Environment-Jupyter_Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Data Science](https://img.shields.io/badge/Domain-Visual_Data_Analysis-blue?style=for-the-badge)
![Manifold Learning](https://img.shields.io/badge/Algorithms-t--SNE_|_UMAP_|_ISOMAP-purple?style=for-the-badge)
![3D VTK](https://img.shields.io/badge/Visualization-3D_Volume_Rendering_VTK-red?style=for-the-badge)
![Erasmus](https://img.shields.io/badge/Program-Erasmus+_Academic_Coursework-003399?style=for-the-badge)

---

## 📌 Executive Summary

This repository contains the complete portfolio of Jupyter Notebooks, Python analysis scripts, 3D medical volume models, geospatial shapefiles, and academic reports developed for the **Visual Data Analysis (VDA)** course during the **Erasmus+ Academic Program**.

The suite covers 9 major domains in visual analytics and scientific visualization, spanning multi-dimensional projection techniques (PCP, SPLOM, RadViz, PCA), non-linear manifold learning (ISOMAP, Laplacian Eigenmaps, LDA, t-SNE, UMAP), perceptual color spaces, network graph layouts, geospatial GIS dashboards (GeoPandas), 3D scalar field extraction (Marching Cubes), 3D volume ray casting (VTK), and vector field flow visualization (Streamlines & Stream Tubes).

---

## 🧮 Theoretical Framework & Mathematical Formulations

### 1. Principal Component Analysis (PCA)
Linear dimensionality reduction via covariance matrix eigenvalue decomposition:
$$C = \frac{1}{N} X^T X \implies C v_i = \lambda_i v_i$$
Projects high-dimensional vectors onto eigenvectors $v_i$ corresponding to maximal eigenvalues $\lambda_i$.

### 2. t-Distributed Stochastic Neighbor Embedding (t-SNE)
Non-linear manifold learning minimizing Kullback-Leibler (KL) divergence between high-dimensional Gaussian probabilities $p_{ij}$ and low-dimensional Student-t probabilities $q_{ij}$:
$$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}, \quad q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_k \sum_{l \neq k} (1 + \|y_k - y_l\|^2)^{-1}}$$

$$\mathcal{L}_{t-SNE} = KL(P || Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

### 3. Uniform Manifold Approximation and Projection (UMAP)
Preserves local and global non-linear structure using fuzzy simplicial sets and Riemannian geometry:
$$w(x_i, x_j) = \exp\left(-\frac{\max(0, d(x_i, x_j) - \rho_i)}{\sigma_i}\right)$$

### 4. Marching Cubes 3D Isosurface Extraction
3D scalar grid thresholding using trilinear vertex interpolation along voxel edges for target iso-value $i_{target}$:
$$x = x_1 + \frac{i_{target} - i_1}{i_2 - i_1}(x_2 - x_1)$$

---

## 📂 Topic Matrix & Portfolio Structure

```
Erasmus-Visual-Data-Analysis-Suite/
├── notebooks/
│   ├── Multidimensional-Data-Visualization/       # Parallel Coordinates, SPLOMs, Heatmaps (Kidney & Wine Datasets)
│   ├── Dimensionality-Reduction-PCA-RadViz/        # PCA, Star Coordinates, RadViz Projections (Cancer & Cortex Datasets)
│   ├── Manifold-Learning-ISOMAP-tSNE-LDA/          # ISOMAP, Laplacian Eigenmaps, Linear Discriminant Analysis, t-SNE
│   ├── Nonlinear-Embeddings-UMAP-ColorSpaces/      # t-SNE Perplexity Tuning, UMAP Manifolds, Perceptual Colormaps
│   ├── Graph-Network-Visualization/                # Node-Link Diagrams, Network Layouts, Color-to-Meaning Mappings
│   ├── Geospatial-Data-Visualization/              # GeoPandas, Shapefiles (.shp), District Power Generation Analysis
│   ├── Scalar-Fields-Isosurface-Extraction/        # Marching Cubes Algorithm, VTK Isosurface Extraction (Head Model)
│   ├── Volume-Rendering-Ray-Casting/              # 3D Simple Ray Casting & Volume Rendering (Bonsai VTI Model)
│   └── Flow-Visualization-Vector-Fields/           # Vector Field Topology, Streamlines, Stream Tubes (VTK Datasets)
├── data/                                          # Datasets (CSV, XLS, XLSX, VTI, VTK, VTS, VTP, Shapefiles)
│   └── districts/                                 # GeoPandas Shapefiles & District Power Generation Datasets
├── docs/                                          # PDF Documentation
│   ├── specifications/                            # Assignment Problem Specifications & Research Papers
│   └── reports/                                   # Student Submitted PDF Reports
├── requirements.txt                               # Python Dependencies
├── .gitignore
└── .gitattributes
```

### Topic & Algorithm Overview

| Domain Topic | Notebook / Script | Key Algorithms & Methods | Datasets |
| :--- | :--- | :--- | :--- |
| **Multidimensional Vis** | `VDA2.2.ipynb`, `VDA2.3.ipynb` | Parallel Coordinates (PCP), SPLOM, Heatmaps | Kidney Disease, Wine Quality |
| **PCA & RadViz** | `VDA3.1.ipynb`, `VDA3.3.ipynb` | PCA, Star Coordinates, RadViz | Breast Cancer, Data Cortex |
| **Manifold Learning** | `vda4.2.ipynb`, `vda4.3.ipynb` | ISOMAP, Laplacian Eigenmaps, LDA, t-SNE | Synthetic & Benchmark Embeddings |
| **UMAP & Color Spaces** | `VDA5_2.ipynb`, `VDA5.4.ipynb` | UMAP, t-SNE Perplexity Tuning, CIELAB | Canyon Image, Manifold Clusters |
| **Graph Visualization** | `VDA6.1.ipynb`, `VDA6.4.ipynb` | Node-Link Layouts, Color-to-Meaning | Oldtimer Image, Network Graphs |
| **Geospatial Analytics** | `VDA 7.2.ipynb` | GeoPandas, GIS Shapefiles, Dashboards | Germany Power Generation (`.shp`) |
| **Scalar Field Vis** | `VDA9.2.ipynb`, `vda9.3.ipynb` | Marching Cubes, Isosurfaces, Slicing | Head CT Scan (`head.vti`) |
| **3D Volume Rendering** | `SimpleRayCast.py` | Ray Casting, Transfer Functions | Bonsai Volume (`bonsai.vti`) |
| **Flow Visualization** | `stream-tubes.py` | Streamlines, Stream Tubes, Vector Fields | Cuboid Velocity (`.vtk`, `.vts`) |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher.

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/merickiziltas/Erasmus-Visual-Data-Analysis-Suite.git
cd Erasmus-Visual-Data-Analysis-Suite

pip install -r requirements.txt
```

### 2. Launch Jupyter Notebook Server
```bash
jupyter notebook
```

---

## 👨‍💻 Author

**Ahmet Meriç Kızıltaş**  
*Erasmus+ Academic Exchange Student at University of Bonn | Boğaziçi University*  
[Student ID: 2022400225]
