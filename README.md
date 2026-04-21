# Motor imagery, BCI illiteracy, DeepLearning/Machine Learning

This project attempts to reproduce the results of Tibrewal et al 2022. Under active development. The paper essentially tries to show that deep learning based classification of motor imagery data might be more useful than the standard LDA + CSP approach when it comes to BCI inefficient users.

<img src="media/task.png" width=50% height=50%>

More than just reproducing the results of the paper, the repository focuses on showing the users how a BCI pipeline is developed. Including but not limited to:
1. **Data Preparation**: preprocessing, **restructuring (epoching)** and **artifact removal** of **EEG data**.

2. **Deep dive**: math/implementation behind techniques like **Common Spatial Pattern** and **Linear Discriminant Analysis**.

3. **Evaluation**: methods for evaluating ML models used for EEG data. 

4. **Research Software Standards**: Best software development practices such as modularization and readability.

# Current stage:

1. To really understand whats happening under the hood, there is a folder in this repository called `learn_CSP`. This focuses on showing the math and the implementation of the **CSP** technique.
<img src = "media/difference_covar_matrix.png" width=75% height=75%>

2. Logic wise, the script is ready to be trained on the CSP + LDA technique. Preparing the data (epoching, vectorization) has been done.

# Next steps:
1. Add the filter-bank technique to existing CSP methods.
2. Train the calibration data on the CSP + LDA method and evaluate training performance using chronological cross validation. 
3. Add another folder explaining the essential math and implementation behind LDA.