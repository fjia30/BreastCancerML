# BreastCancerML
Analyzing the Breast Cancer Wisconsin (Diagnostic) Data Set using various methods with scikit-learn, an assay

# Introduction
Breast cancer is the most common type of cancer in women worldwide and are diagnosed in 12% of all women in the United States over their lifetimes **[1]**. Early detection of breast cancer is key for improved treatment outcome and using machine learning, including deep learning to facilitate and improve breast cancer detection is an active research field **[2]**.

The dataset to be analyzed is the Breast Cancer Wisconsin (Diagnostic) Data Set from the UCI Machine Learning Repository **[3]** and also kaggle.com **[4]**. Each instance has 30 features and the dataset contains ~600 instances. It includes features extracted
>from a digitized image of a fine needle aspirate (FNA) of a breast mass which describe characteristics of the cell nuclei present in the image as well as the diagnosis (B: benign, M: malignant) **[3]**. 

# Part I: Comparation between different learners
## 1-1 Decision trees
### 1-1-1 Hyperparameter Tuning
`criterion` defines the function to measure the quality of a split or equivalently, the impurity / information-gain. There are two options, gini or entropy.
`max_depth` is the maximum depth of the tree. A test run with default settings using the training set produced a tree with a depth of 7. Depth lower than 3 was shown to perform consistently worse in a pilot experiment so `range(3, 10)` was chosen as the options.
`min_samples_leaf` defines the minimum number of samples required in a leaf node. And it was set to `range(1, 10)` empirically.
`ccp_alpha` is the complexity parameter used for post pruning. `ccp_alpha = 0` means no pruning. Higher `ccp_alpha` leads to more aggressive pruning causing a higher impurity in the leaf nodes. It reduces accuracy on the training set and can reduce the risk of over-fitting. Based on progressively more aggressive pruning of a tree constructed with default parameters **(Figure 1)**, I set the options to `[0, 0.005, 0.01, 0.015, 0.02]`.

![Figure 1. ccp_alpha vs impurity](https://github.com/fjia30/BreastCancerML/blob/master/PartI/Figure1.png)

After setting the range for each hyperparameter, I carried out grid search using `GridSearchCV`. The best parameters are `ccp_alpha = 0.01`, `criterion = entropy`, `max_depth = 5` and `min_samples_leaf = 5` with an accuracy of `0.9472`. Five-fold Cross validation was used in all cases for accuracy score calculation.
### 1-1-2 learning curve
Next, I used the set of hyperparameters to construct the optimized decision tree and graphed its learning curve **(Figure 2)**.

![Figure 2. Learning curve of optimized tree](https://github.com/fjia30/BreastCancerML/blob/master/PartI/Figure2.png)

From this graph it is clear that, as expected, the learner performed consistently better on the training set compared to the test set. However, the accuracy on the test set continued to increase with increasing training set size. This suggest that over-fitting was likely not a problem in this case, on the other side, this learner might have the potential to further improve given more instances to train.
## 1-2 Boosting
### 1-2-1 Hyperparameter Tuning
Decision trees were chosen as weak learners for boosting here. To test the performance of a weak leaner, I first tested the accuracy of a decision stump (decision tree with only 1 depth). The performance was surprisingly well `accuracy: 0.9034 ± 0.0146` and thus pruning was not used (not applicable to a decision stump).
The only hyperparameter need tuning here is the number of weaker leaners defined by `n_estimators`. I tested `range(1, 501)` and found the accuracy increased and stabilized at around 400 **(Figure 3)**, so I used 400 as the final parameter. The accuracy was `0.9754 ± 0.0102`.



### 1-2-2 learning curve
Learning curves were graphed based on the number of learners **(Figure 3)** and the size of the training set **(Figure 4)**. Again, as expected, the learner performed consistently better on the training set compared to the test set and no over-fitting was observed. The fact that the train accuracy remained very high while the test accuracy kept increasing is typical of a boosting algorithm.




