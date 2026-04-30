# Motor Imagery BCI : Reproducing Tibrewal et al. (2022)
This project attempts to reproduce the results of Tibrewal et al. 2022. Under active development. The paper essentially tries to show that deep learning based classification of motor imagery data might be more useful than the standard LDA + CSP approach when it comes to BCI inefficient users.

<img src="media/task.png" width=60% height=60%>

More than just reproducing the results of the paper, the repository focuses on showing the users how a BCI pipeline is developed. Including but not limited to:

- [x] **Data Preparation**: preprocessing, **restructuring (epoching)** and **artifact removal** of **EEG data**
- [x] **Deep dive**: math/implementation behind techniques like **Common Spatial Pattern** and **Linear Discriminant Analysis**
- [x] **Evaluation**: methods for evaluating ML models used for EEG data
- [x] **Research Software Standards**: best software development practices such as modularization and readability

---

## Index
**Project Stages**
1. [Stage 1: Data Preparation](#stage-1-data-preparation)
2. [Stage 2: Learning CSP](#stage-2-learning-csp)
3. [Stage 3: CSP + LDA Pipeline](#stage-3-csp--lda-pipeline)

**Setup and Usage**
- coming soon

---

## Current Stage: Stage 3: CSP + LDA Pipeline

---

## Stage 1: Data Preparation
- [x] Load raw calibration EEG data per participant
- [x] Epoch data around task window
- [x] Extract labels

---

## Stage 2: Learning CSP
To really understand whats happening under the hood, there is a folder in this repository called `learn_CSP`. This focuses on showing the math and the implementation of the **CSP** technique.

<img src="media/difference_covar_matrix.png" width=40% height=40%>

- [x] Covariance matrix computation
- [x] Generalized eigenvalue decomposition
- [x] Spatial filter extraction and log-variance features

---

## Stage 3: CSP + LDA Pipeline
- [x] Add filter-bank technique to existing CSP methods
- [x] Train calibration data on CSP + LDA and evaluate using stratified k-fold cross validation
- [ ] Add folder explaining the essential math and implementation behind LDA
