# BreastCancerML
Analyzing the Breast Cancer Wisconsin (Diagnostic) Data Set using various methods with scikit-learn, an assay

# How to test by youself
All the code for each part can be found in the corresponding folder.

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

![Figure 3. # of leaners vs boosting accuracy](https://github.com/fjia30/BreastCancerML/blob/master/PartI/Figure3.png)


### 1-2-2 learning curve
Learning curves were graphed based on the number of learners **(Figure 3)**, above and the size of the training set **(Figure 4)**. 

![Figure 4. Learning curve of Adaboost with decision stumps](https://github.com/fjia30/BreastCancerML/blob/master/PartI/Figure4.png)

Again, as expected, the learner performed consistently better on the training set compared to the test set and no over-fitting was observed. The fact that the train accuracy remained very high while the test accuracy kept increasing is typical of a boosting algorithm.
## 1-3 k-Nearest Neighbors
### 1-3-1 Hyperparameter Tuning
The only hyperparameter needed tuning here is `k`. I chose `range(1, 21)`. Consistently, the learner had a 100% accuracy score on training data when `k = 1` and the accuracy decreased as k increased **(Figure 5)**. 

![Figure 5. k value vs accuracy](https://github.com/fjia30/BreastCancerML/blob/master/PartI/Figure5.png)

On the test set, accuracy increased initially then decreased as k got bigger. A `k = 9` was decided based on the result. It had an accuracy of `0.9315 ± 0.0279`.
## 1-3-2 learning curve
The learning curve of the resulting 9-NN learner was graphed **(Figure 6)**.

![Figure 6. Learning curve of a 9-NN learner ](https://github.com/fjia30/BreastCancerML/blob/master/PartI/Figure6.png)

Accuracy score on the training set remained high while accuracy score on the test set increased sharply initially then stabilized. The initial sharp increase in my opinion is because when the size of training set is small, a 9-NN is more likely to include distant neighbors from a different class due to the sparsity of the instances and perform poorly. As training size increases, more instances “fill in the blank” and 9-NN’s performance increases drastically. To prove this idea, I also graphed the learning curve of a 2-NN learner **(Figure 7)**.

![Figure 7. Learning curve of a 2-NN learner](https://github.com/fjia30/BreastCancerML/blob/master/PartI/Figure7.png)

Consistent with my theory, the 2-NN learner performed much better with a smaller training set because `k = 2` is more local and less affected by a sparse training set.
## 1-4 Support Vector Machines
### 1-4-1 Preprocessing
SVM is sensitive to scaling **[5]**, so features were first standardized by removing the mean and scaling to unit variance using `StandardScaler`.
### 1-4-2 Hyperparameter Tuning
Hyperparameters for SVM in this case are different kernels. I chose three different kernel functions: a radial basis function, a polynomial function and a sigmoid function.
Accuracy of SVM with a radial basis function kernel: `0.9736 ± 0.0147`.
Accuracy of SVM with a sigmoid function kernel: `0.9596 ± 0.0204`.
For polynomial kernel, I had to tune the degree number d. And the result showed that a simple `d = 1`(linear) kernel performed the best **(Figure 8)** with an accuracy of `0.9754 ± 0.0102`. 

![Figure 8. tuning d for SVM with a polynomial kernel](https://github.com/fjia30/BreastCancerML/blob/master/PartI/Figure8.png)

Because this linear kernel performed comparable to the rbf kernel and better than the sigmoid kernel. I decided to use a linear kernel due to its simplicity. 
### 1-4-3 learning curve
Learning curve of this linear SVM shows continued improvement in testing accuracy as training size increased until training size reached about 350 and stabilized after that **(Figure 9)**.

![Figure 9. Learning curve of a linear SVM](https://github.com/fjia30/BreastCancerML/blob/master/PartI/Figure9.png)

## 1-5 Neural Networks
### 1-5-1 Preprocessing
Multi-layer Perceptron is also sensitive to feature scaling **(6)**. Features were standardized the same way as for SVM learning.
### 1-5-2 Hyperparameter Tuning
The hyperparameters include the activation function, which can be a logistic function, a rectified linear unit function `f(x) = max (0, x)` or none; the size of the hidden layer and I chose to have only one hidden layer and set the options to `[3, 10, 30, 100]`; `alpha`, which controls the regularization term. A higher `alpha` reduces the risk of over-fitting and I set the options to `[0.1, 0.01, 0.001, 0.0001, 0.00001]`.
Grid search returned an optimized learner with a logistic activation function, `alpha = 0.1` and 30 perceptrons in the hidden layer. The accuracy score was `0.9824 ± 0.0078`.	
### 1-5-3 learning curve
Learning curve of the optimized ANN is slightly concerning because towards the end the testing score shows a slight drop while the training score continues to rise which is a signature of over-fitting **(Figure 10)**.

![Figure 10. Learning curve of the optimized ANN](https://github.com/fjia30/BreastCancerML/blob/master/PartI/Figure10.png)

Given the fact that the change is very small, and our alpha value is relatively big (0.01). I think it is still mostly well-fitted.
## 1-6 Comparison
Learner|Decision Tree|Boosting|kNN|SVM|ANN
----|-------------|--------|---|---|----
Accuracy|0.9472|0.9754|0.9315|0.9754|0.9824

The table above shows the accuracies of different learners on the testing set. The order is ANN > SVM = Boosting > Decision Tree > kNN. To take a closer look at the best learner ANN and to have a complete understanding of its prediction power, I generated a confusion matrix **(Figure 11)**.

![Figure 11. Confusion matrix of trained ANN](https://github.com/fjia30/BreastCancerML/blob/master/PartI/Figure11.png)

Here a `M` (malignant) diagnosis is a positive prediction. For cancer diagnosis, the critical numbers here are the false discovery rate, which defines the percentage of benign tumors that are identified as malignant, and the false negative rate, which defines the percentage of actual malignant tumors that are missed by the learner. This trained ANN has a false discovery rate of `1/50 = 2%` and a false negative rate of `/54 = 7.5%`. It is clear that although the accuracy is high (~98.2%) the high false negative rate means this learner can miss many tumors that are actually malignant. Therefore, further improvement is needed. In the following parts, I performed addtional analysis and optimization to try to improve the performance of this ANN.

# Part II: tuning ANN with randomized optimization
## 2-1 Introduction
In order to better tuning the hyperparameters of this ANN, I employed three different randomized optimization algorithms. They are Randomized hill climb, Simulated annealing and Genetic Algorithm. I originally used 30 perceptrons in the hidden layer but it turned out to be too resource intensive to optimize with the randomized optimization algorithms, so I reduced the number to 10. This change caused a slight drop of accuracy when I fit the classifier using Back Propagation (from 0.9824 to 0.9649 on the test set), but greatly reduced the time required for the other optimizations.
## 2-2 Optimization
To find out what kind of impact can each parameter have on the performance of each algorithm, I performed a grid search with different combinations of key parameters for each algorithm and checked the performance of the resulting neural network. These parameters were later used to futher tune the optimization algorithms side by side.

Max attempts|Num restarts|Train accuracy|Test accuracy|Train time(s)
------------|--------|--------------|-------------|----------
10|0|0.90|0.85|9.08
10|10|0.91|0.94|30.73
10|20|0.93|0.94|50.60
20|0|0.86|0.83|11.24
20|10|0.93|0.96|122.95
20|20|0.90|0.93|234.07
50|0|0.93|0.88|11.19
50|10|0.95|0.93|123.30
50|20|0.95|0.95|234.63

For randomized hill climb (see above table), increasing both maximal number of attempts and number of random restarts increases the accuracy of the resulting classifier.

Max attempts|Init temp|Min temp|Train accuracy|Test accuracy|Train time(s)
------------|--------|---------|--------------|-------------|----------
50|1|0.01|0.84|0.80|15.31
50|1|0.1|0.09|0.08|15.54
50|5|0.01|0.88|0.91|15.10
50|5|0.1|0.33|0.31|15.43
100|1|0.01|0.45|0.51|15.20
100|1|0.1|0.37|0.39|15.25
100|5|0.01|0.52|0.52|15.29
100|5|0.1|0.35|0.37|15.26

In contrast, changing initial or final temperature had little impact on simulated annealing while increasing maximal number of attempts enhanced the resulting classifier’s performance (see above table).

Max attempts|Pop size|Mutation rate|Train accuracy|Test accuracy|Train time(s)
------------|--------|-------------|--------------|-------------|-------------
10|10|0.1|0.82|0.86|0.33
10|10|0.3|0.88|0.87|0.22
10|50|0.1|0.90|0.92|2.20
10|50|0.3|0.92|0.93|1.71
50|10|0.1|0.85|0.93|0.59
50|10|0.3|0.88|0.88|1.59
50|50|0.1|0.90|0.92|2.99
50|50|0.3|0.93|0.96|4.76

Increasing maximal number of attempts, population size and mutation rate all boosted genetic algorithm as well (see table above). These parameters were later used to futher tune the optimization algorithms side by side.
## 2-3 Comparison
To compare the three algorithms and Back Propagation on their performance on neural network training, I varied key parameters of the three algorithms to achieve an accuracy of 0.9 of the resulting classifiers and plot the accuracy score on either training or test set vs the training time.

![Figure 12A. performance of each algorithm vs Back Propagation](https://github.com/fjia30/BreastCancerML/blob/master/PartII/Figure12A.png)

![Figure 12B. performance of each algorithm vs Back Propagation](https://github.com/fjia30/BreastCancerML/blob/master/PartII/Figure12B.png)

Back Propagation, randomized hill climb and genetic algorithm consistently trained classifiers with an test accuracy score greater than 0.9. In contrast, simulated annealing performed poorly and were highly unpredictable with the results. Between Back Propagation, randomized hill climb and genetic algorithm, Back Propagation performed the best as it is tailor made for training neural nets and has potentially better implementation by the sklearn package. Genetic algorithm performed the best among the three alternative algorithms.
## 2-4 Conclusion 
In conclusion, the performance on training neural network classifier with the breast cancer dataset is in the order of Back Propagation > genetic algorithm >> randomized hill climb >> simulated annealing. It is interesting to note that simulated annealing, which performs very well in many cases, is the worst here. The poor performance of simulated annealing and randomized hill climb in this case is likely due to neural net’s large input space and many local maxima. Genetic algorithm on the other hand, is good at preserving good network structures which makes the search more efficient and therefore performed better in this case.
# Part III: Dimensionality Reduction
## 3-1 Introduction
Our randomized optimization did not significantly improve the ANN learner. Next I applied different undupervised learning techniques to achieve dimensionality reduction and hopefully better performance. Firstly, the dataset was normalized by removing the mean and scaling to unit variance for each attribute. A quarter of the instances picked at random were set aside as the test set and the remaining instances were used for clustering and dimensionality reduction analyses.
## 3-2 Clustering without dimensionality reduction
I first removed labels from the data and carried out clustering using K-means or expectation maximization (EM). Because there were only two different labels, I set the number of centers to 2. After clustering, I added labels back and checked how well different labels were separated by the clustering algorithms. In an ideal situation, clustering would separate the instances into 2 groups each with a different label. Indicating a discernable difference in the attributes between the two classes of instances.

![Figure 13](https://github.com/fjia30/BreastCancerML/blob/master/PartIII/Figure13.png)

For our dataset, this is mostly the case **(Figure 13)**. Specifically, without clustering, about 63% samples were classified as benign and 37% malignant. After clustering, either using K-means or EM, the resulting clusters highly enriched one class or the other. For K-means, ~90% samples were classified as benign in one cluster and ~93% samples were classified as malignant in the other cluster. EM performed slightly better and it was ~96% benign in one cluster and ~92% malignant in the other. Although neither produced a pure cluster, suggesting room for further improvement. Nonetheless, both clustering algorithms produced clusters with much more homogenous labels. This is comparable to decision trees learning where splitting on a “good” attribute would produce well separated subpopulations. The difference is that the clustering algorithms consider all attributes together to make the split.

To better characterize the performance of clustering on separating the classes, I plotted the homogeneity score, which ranges from 0 to 1 with 0 being complete mix-up and 1 being complete separation of labels (the ideal situation) against the number of centers used in clustering. K-means and EM showed similar results so only EM is shown **(Figure 14)**.

![Figure 14. Number of centers vs homogeneity score](https://github.com/fjia30/BreastCancerML/blob/master/PartIII/Figure14.png)

For the number of centers, I chose the range from 2 to 30 (the number of attributes in the dataset) as the eventual goal was to do dimensionality reduction, having more centers than the number of attributes would defeat this purpose. It shows that the score drops first and then raised higher compared to `k = 2`. But because the clustering already performed well with `k = 2` and further increasing the score require a dramatic increase of k, I chose to stick with 2 centers. 

I would like to point out that an “ideal” or “better” clustering result is subjective. In this case, I wanted the clusters to separate instances with different labels well (which is commonly the case). However, having a clustering result that does not achieve this goal, does not mean that the resulting clusters are not meaningful. It still reveals differences between subgroups of instances, but those differences do not contribute to the classification problem and therefore should be considered noise.
## 3-3 Principal Component Analysis (PCA)
PCA was carried out first without dimensionality reduction to examine each component’s contribution to the variance based on their respective eigenvalues **(Figure 15)**.

![Figure 15. PCA component explained variance ratio](https://github.com/fjia30/BreastCancerML/blob/master/PartIII/Figure15.png)

It shows that ~90% of the variance can be nicely captured by 10 out of the 30 components and this number was chosen as the target for dimensionality reduction.
## 3-4 Clustering after PCA
After dimensionality reduction using PCA, I again carried out K-means and EM. The result for Dataset I was similar as before for K-means, a nice separation of the two classes **(Figure 16)**.

![Figure 16](https://github.com/fjia30/BreastCancerML/blob/master/PartIII/Figure16.png)

However, EM, having ~82% benign in one cluster and ~74% malignant in the other, performed considerably worse than before. It was likely because the ~10% variation I dropped corelated strongly to different labels and removing it was detrimental for the algorithms, especially EM. PCA did not improve the outcome of our clustering algorithms. I think the reason lies in the nature of PCA, which essentially combines all variations and picks top components that contribute to said variations. In reality, some variations contribute to the classification problem, other variations are simply noise. For example, one way to separate dogs from people is the number of legs they have, and in this case, variations in hair color, which can be great, is simply noise. But PCA mashes these variations together instead finding the relevant one, which leads to little improvement to the clustering results.
## 3-5 Independent Component Analysis (ICA)
ICA linearly separates multivariable data into random variables with non-Gaussian distributions and are as independent to each other as possible. To check the effect of ICA, I first compared the kurtosis of each component before and after ICA, which is an indicator of non-Gaussianity. ICA significantly increased the kurtosis of the top components, making their distribution less Gaussian **(Figure 17)**.

![Figure 17](https://github.com/fjia30/BreastCancerML/blob/master/PartIII/Figure17.png)

It also showed that the 5 components with the least kurtosis in both datasets did not change much after ICA and should be safe to drop. To look at more details, I plotted the distribution of the 2 components with the highest kurtosis before and after ICA for Dataset 1 **(Figure 18)**. It is very clear to see the non-Gaussianity of the two components, manifested by their peakedness, greatly increased after ICA.
![Figure 18](https://github.com/fjia30/BreastCancerML/blob/master/PartIII/Figure18.png)
## 3-6 PCA vs ICA
To further visualize the clustering results and the effect of PCA and ICA on clustering. I plotted scatter plots. Green and orange represent different clusters. Dots and crosses represent different labels, (benign and malignant, respectively) **(Figure 19)**.

![Figure 19](https://github.com/fjia30/BreastCancerML/blob/master/PartIII/Figure19.png)

For original data, 2 out of the 30 attributes were randomly selected to plot the data points. For PCA, two attributes with the highest variance (Figure 15) was used. For ICA, the number of components was reduced to 2. The results show that although the clustering performance did not change much after PCA or ICA (compare Figure 13, Figure 16 and Figure 20), PCA and ICA greatly improved data visualization, projecting high dimensional space which are hard to graph onto lower dimensions for a better data presentation.
## 3-7 Clustering after ICA
ICA is not inherently a dimensionality reduction algorithm. However, the relevance of each independent component revealed by ICA can be evaluated and the irrelevant ones can be dropped. This can reduce noise and improve the performance of downstream clustering and classification algorithms. Therefore, for the actual dimensionality reduction with ICA, I first carried out PCA as stated above to reduce the number of dimensions to 10, then conducted ICA. The problem of finding the most relevant independent components could be treated as an optimization problem which we practiced in Part II. In this case, my goal was to maximize the purity of resulting clusters which could be measured using a homogeneity score.

![Figure 20](https://github.com/fjia30/BreastCancerML/blob/master/PartIII/Figure20.png)

Using randomized hill climb, I was able to further reduce the number of dimensions to 9 with K-means and to 6 with EM with great separation **(Figure 20)**. The results were significantly better for both algorithms compared to Figure 5, with no ICA. In addition, and impressively, the K-means result was also slightly better than that in **Figure 13**, which had a dimension of 30 compared to 9 here. This result shows that PCA coupled with ICA for K-means successfully removed more than half of the dimensions but at the same time kept and even slightly increased the clustering performance. It was likely due to the successful removal irrelevant variables which only contributed to noise.
## 3-8 Clustering after Randomized Projections
For randomized projections, I tried two different methods, Gaussian random projection and sparse random projection. Sparse random projection produced better results, so I am only showing the results with sparse random projections. I chose to project the original dataset to a 10-dimentioal space, after about 10 trials, I was able to obtain very good clustering result using both algorithms **(Figure 21)**.

![Figure 21](https://github.com/fjia30/BreastCancerML/blob/master/PartIII/Figure21.png)

It is comparable to the original result, with only one third of the dimensions. However, it is still worse compared to Figure 20 where we carried out PCA and ICA and randomized optimization, both in performance and in the degree of dimensionality reduction. This result shows the surprising simplicity and effectiveness of randomized projection.
## 3-9 Clustering after Linear Discriminant Analysis (LDA)
For the last feature selection / dimensionality reduction algorithm, I chose LDA. It is interesting because it is different from all other algorithms used previously and considers the label. Also because of this feature, I had to perform clustering both on the training set and on the test set. 

![Figure 22](https://github.com/fjia30/BreastCancerML/blob/master/PartIII/Figure22.png)

LDA performed very well. Both K-means and EM generated very homogenous clusters. And due to the nature of LDA, there was only at most (class number -1) number of components kept and because I had only 2 types of labels in both datasets, the transformed data had a dimension of only 1. This is the best dimensionality reduction among all algorithms tested and this advantage was mostly due to the utilization of the class labels.
## 3-10 Back to ANN
Now we come back to our orginal goal: to improve the performance of the ANN learner. I coupled all the dimensionality reduction processes described above with ANN training. The table above lists the results.

Num of Dimensions|Train Time|Train Accuracy|Test Accuracy
-----------------|----------|--------------|-------------
Original Data|30|432ms|0.9842|0.9843
PCA|10|443ms|0.9812|0.9860
ICA|13|1517ms|0.9695|0.9720
RP|10|527ms|0.9601|0.9510
LDA|1|541ms|0.9765|0.9860
K-means|10|652ms|0.9742|0.9650
EM|10|578ms|0.9695|0.9650

## 3-10 Conclution

As shown in the table above, LDA or PCA coupled with ANN produced the best accuracy score on the test set `0.9860`.

LDA had the least dimensions, only 1. This great reduction of dimensionality was due to firstly the characteristics of the attributes of the dataset, in which many attributes represented similar or correlated values (e.g. mean of the radius vs mean of the area) and secondly LDA’s ability to utilize the class label which provided additional information for a better dimensionality reduction.

It is interesting to compare PCA vs ICA. While ICA was better at improving the clustering results, PCA was more successful in helping with ANN training. PCA reduced the number of dimensions more significantly than ICA and training the learner was much faster. The superior performance of PCA in ANN training, in this case, in my opinion was due to better conservation of information. As our previous analyses have shown, there is little noise in the data and therefore removing variance would likely lead to loss of information. And PCA is better at conserving variation in the data compared to ICA, which focuses more on the non-Gaussianity rather than variance.

Last but not least, the two clustering algorithms also performed well as dimensionality reduction methods for ANN training. Using clustering for dimensionality reduction has been proven successful (e.g. the centroid method **[7]**). However, it requires a careful calibration of the number of centers and choosing the right distance function, which is less straightforward and intuitive compared to methods such as PCA and ICA.


# References
1. Breast Cancer Treatment, A Review. AG Waks and EP Winer. JAMA. 321(3), 288-300 (2019).
2. Deep Learning to Improve Breast Cancer Detection on Screening Mammography. L Shen et al. Sci Rep 9, 12495 (2019).
3. https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
4. https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
5. Importance of Feature Scaling. https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html
6. https://scikit-learn.org/stable/modules/neural_networks_supervised.html#tips-on-practical-use
7. Park, H., Jeon, M. & Rosen, J.B. Lower Dimensional Representation of Text Data Based on Centroids and Least Squares. BIT Numerical Mathematics 43, 427–448 (2003).









