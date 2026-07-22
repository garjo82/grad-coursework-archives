# Graduate Coursework Archives

A centralized repository containing academic artifacts, project implementations, and core theoretical baselines across graduate-level computer science, data science, and unmanned systems coursework.

This archive serves as a reference baseline for system architectures, algorithm implementations, and empirical analyses developed throughout coursework at the University of Texas at Austin, the University of Illinois Urbana-Champaign, and Embry-Riddle Aeronautical University.

---

## University of Texas at Austin (UT Austin)

Coursework from the Master of Science in Computer Science program, focusing on core machine learning foundations, deep learning architectures, and advanced neural network paradigms.

### Machine Learning

A comprehensive series of theoretical problem sets and algorithmic implementations covering foundational machine learning paradigms, theoretical optimization, and probability models.

* **Homework 3:** Mathematical derivations for dimensionality reduction, PCA variance maximization, SVD, and latent variable spaces.
* **Homework 4:** Algorithmic logic and theoretical proofs for frequent itemset lattice traversal, Apriori, and sequential pattern mining.
* **Homework 5:** Theoretical analysis and proofs covering spectral graph theory, Laplacian matrices, and graph-based optimization.

### Deep Learning

Implementation of deep architectures for computer vision, end-to-end control, and trajectory planning in simulated environments.

* **Homework 4 (Trajectory Planning):** End-to-end autonomous driving planners using MLPs, Perceiver Transformers (cross-attention over query embeddings), and CNN vision backbones in PySuperTuxKart.

### Advances in Deep Learning

* **Homework 3 (Reasoned Unit Conversion):** Fine-tuning language models (SmolLM2-1.7B) using batched generation, In-Context Chain-of-Thought (CoT) prompting, Supervised Fine-Tuning (SFT with LoRA adapters), and Rejection Sampling Fine-Tuning (RFT) for structured reasoning and unit conversion.

---

## University of Illinois Urbana-Champaign (UIUC)

Coursework in data mining algorithms, statistical modeling, and applied quantitative analysis.

### Data Mining

* **Hierarchical Clustering (`submission.py`):** Pure Python implementation (standard library only) of agglomerative hierarchical clustering on 2D geographic data, supporting Single, Complete, and Average Linkage metrics using Euclidean distance.

### Applied Statistics

* **Cut-In Risk Analysis Project (`joelg4-project.pdf`):** Empirical safety and statistical risk analysis of vehicle cut-in behavior using nuScenes autonomous driving trajectory datasets.
* **Trajectory Data Pipeline (`nuscenes_preprocessing.py`):** Preprocessing script to extract, filter (`forward_velocity > -1.0`), clean, and produce structured CSV datasets (`clean_cutin_dataset*.csv`) for regression and proximity risk modeling.

---

## Embry-Riddle Aeronautical University (ERAU)

Graduate coursework focused on unmanned systems engineering, autonomous platforms, and aerospace domain implementations.

### Embry-Riddle Aeronautical University

* **UNSY 603 (Tactical UGV Convoy Architecture):** Designed a bolt-on, non-invasive drive-by-wire retrofit system converting standard HMMWV platforms into autonomous uncrewed ground followers for tactical off-road convoy applications[cite: 1].
* **System Design & Trade Studies:** Developed system-level architectures (NVIDIA Jetson AGX, LiDAR, stereo vision, V2V mesh, and dead-reckoning IMU) supporting GNSS-denied navigation, SIL/HIL testing, and multi-UGV scalability[cite: 1].
