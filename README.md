# BreastCancerML
Analyzing the Breast Cancer Wisconsin (Diagnostic) Data Set using various methods with scikit-learn, an assay

# Introduction
Breast cancer is the most common type of cancer in women worldwide and are diagnosed in 12% of all women in the United States over their lifetimes **[1]**. Early detection of breast cancer is key for improved treatment outcome and using machine learning, including deep learning to facilitate and improve breast cancer detection is an active research field **[2]**.

The dataset to be analyzed is the Breast Cancer Wisconsin (Diagnostic) Data Set from the UCI Machine Learning Repository **[3]** and also kaggle.com **[4]**. Each instance has 30 features and the dataset contains ~600 instances. It includes features extracted 
>from a digitized image of a fine needle aspirate (FNA) of a breast mass which describe characteristics of the cell nuclei present in the image as well as the diagnosis (B: benign, M: malignant) **[3]**. 

# Part I: Comparation between different learners
## 1-1. Decision trees
### 1-1-1 Hyperparameter Tuning
`criterion` defines the function to measure the quality of a split or equivalently, the impurity / information-gain. There are two options, gini or entropy.
`max_depth` is the maximum depth of the tree. A test run with default settings using the training set produced a tree with a depth of 7. Depth lower than 3 was shown to perform consistently worse in a pilot experiment so `range(3, 10)` was chosen as the options.
`min_samples_leaf` defines the minimum number of samples required in a leaf node. And it was set to `range(1, 10)` empirically.
`ccp_alpha` is the complexity parameter used for post pruning. `ccp_alpha = 0` means no pruning. Higher `ccp_alpha` leads to more aggressive pruning causing a higher impurity in the leaf nodes. It reduces accuracy on the training set and can reduce the risk of over-fitting. Based on progressively more aggressive pruning of a tree constructed with default parameters (Figure 1), I set the options to `[0, 0.005, 0.01, 0.015, 0.02]`.