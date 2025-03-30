# SC5002_assignment3_group1

This is the file for NTU SC5002 Lab Assignment 3 from Group 1 (LAM YUEN IN, WANG WEIQIAN, FANG XINQING)

**Project Explanation and Steps taken:**
 
  In this project, our aim is to examine two models using the same dataset -- k-means model and MLP (Multi-Layer Perceptron) model. We first applied the K-means algorithm from _sklearn.cluster_ to test the data using different k values. This method is to see how data groups themselves naturally, and minimizing the within-cluster sum of squares. Elbow method was used to figure out the optimal number of clusters to have for our data, where we get the result that 3 is the optimal value for k. In addition, silhouette score was applied to evaluate the clustering quality, where k=3 was found to have the highest silhouette score. As for the MLP (Multi-Layer Perceptron) model, we guided the network to learn the mapping from flower measurements to species labels. MLP performes an accuracy score of 96-97% from the 5-fold cross-validation method. Lastly, we used a confusion matrix to evaluate the performance of the classification model by comparing predicted labels with actual labels.
  
**Dataset Explanation:**
  
  The dataset we chose is the Iris dataset from the UCI Machine Learning Repository. The dataset contains data about flowers, and it consists of 150 samples and four numerical features which are categorized into three species. There is no missing values in the data, so all data were properly given.

**Insights**

1. The Elbow method is very efficient at determing the best value for k, where too few clusters (small k value) result in very high WCSS, but too many clusters (big k value) result in overfitting, implying that clusters may not generalize well and may represent noise rather than genuine patterns. Hence, the elbow point provides a good balance and generates the most appropriate result.

2. By setting the number of clusters to 3, the algorithm produced clusters that largely correspond to the three actual species, demonstrating that even without supervision, the data’s natural structure can be effectively uncovered.

3. A Multi-Layer Perceptron (MLP) classifier is used with three hidden layers of 64 neurons each and a maximum of 1024 training iterations. This setup ensures that the results stay the same across different train-test splits and cross-validation runs.

4. MLPClassifier is a supervised learning method that predicts labels using training data, while K-Means is an unsupervised learning method that groups data based on patterns without predefined labels.

Work distribution:
1,LAM YUEN IN: fining data set, implement models
2,WANG WEIQIAN: creating slides, analysis result
3,FANG XINQING: Data profiling, generate README file
