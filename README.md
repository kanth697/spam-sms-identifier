## SPAM SMS DETECTION

Approach

- Loading Data
- Input and Output Data
- Applying Regular Expression
- Each word to lower case
- Splitting words to Tokenize
- Stemming with PorterStemmer handling Stop Words
- Preparing Messages with Remaining Tokens
- Preparing WordVector Corpus
- Applying Classification

## SMS Spam Classification Steps

- Data Preparation
- Exploratory Data Analysis(EDA)
- Text Pre-processing and TF-IDF
- Model Building with Classification Algorithm

We implement three different types of classification algorithms. Among all the three, the one with larger accuracy will be suggested. The three classification algorithms are

- Multinomial Naïve Bayes
- K Nearest Neighbour algorithm
- Decision Tree

## Naive Bayes Theorem

- Supervised Learned Algorithm
- Naïve Bayes classification is easy to implement and fast.
- It will converge faster than discriminative models like logistic regression.
- It requires less training data.
- It is highly scalable in nature, or they scale linearly with the number of predictors and data points.
- It can make probabilistic predictions and can handle continuous as well as discrete data.
- Naïve Bayes classification algorithm can be used for binary as well as multi-class classification problems both.

Naïve Bayes algorithms is a classification technique based on applying Bayes&#39; theorem with a strong assumption that all the predictors are independent to each other. In simple words, the assumption is that the presence of a feature in a class is independent to the presence of any other feature in the same class. For example, a phone may be considered as smart if it is having touch screen, internet facility, good camera etc. Though all these features are dependent on each other, they contribute independently to the probability of that the phone is a smart phone.

In Bayesian classification, the main interest is to find the posterior probabilities i.e. the probability of a label given some observed features, 𝑃(𝐿 | 𝑓𝑒𝑎𝑡𝑢𝑟𝑒𝑠). With the help of Bayes theorem, we can express this in quantitative form as follows –

P(L|features)=[P(L).P(features|L)]/P(features)

Here, (𝐿 | 𝑓𝑒𝑎𝑡𝑢𝑟𝑒𝑠) is the posterior probability of class.

𝑃(𝐿) is the prior probability of class.

𝑃(𝑓𝑒𝑎𝑡𝑢𝑟𝑒𝑠|𝐿) is the likelihood which is the probability of predictor given class.

𝑃(𝑓𝑒𝑎𝑡𝑢𝑟𝑒𝑠) is the prior probability of predictor.

## Multinomial Naive Bayes

-Statistical experiment
-n- repeated trials
-Each trail has a discrete number of possible outcomes
-Trails are Independent

**K – Nearest Neighbouring [KNN]Algorithm**

- Supervised Learning Algorithm
- One of Clustering Algorithms
- Used for classification
- Used for Regression Prediction Problem
- Lazy learning Process which uses entire data for classification and has no specialised running process
- Non Parametric Algorithm where it doesn&#39;t assume anything about underlying data
- Uses feature similarity, a new data point will be assigned to a value based on feature similarty

## Working of KNN Algorithm

K-nearest neighbors (KNN) algorithm uses feature similarity to predict the values of new datapoints which further means that the new data point will be assigned a value based on how closely it matches the points in the training set. We can understand its working with the help of following steps −

**Step 1**  − For implementing any algorithm, we need dataset. So during the first step of KNN, we must load the training as well as test data.

**Step 2**  − Next, we need to choose the value of K i.e. the nearest data points. K can be any integer.

**Step 3**  − For each point in the test data do the following −

- **3.1**  − Calculate the distance between test data and each row of training data with the help of any of the method namely: Euclidean, Manhattan or Hamming distance. The most commonly used method to calculate distance is Euclidean.
- **3.2**  − Now, based on the distance value, sort them in ascending order.
- **3.3**  − Next, it will choose the top K rows from the sorted array.
- **3.4**  − Now, it will assign a class to the test point based on most frequent class of these rows.

**Decision Tree Algorithm**

- Decision tree algorithm falls under the category of supervised learning. They can be used to solve both regression and classification problems.
- Decision tree uses the tree representation to solve the problem in which each leaf node corresponds to a class label and attributes are represented on the internal node of the tree.
- We can represent any boolean function on discrete attributes using the decision tree.

**Assumptions made in Decision Tree Algorithm**

- At the beginning, we consider the whole training set as the root.
- Feature values are preferred to be categorical. If the values are continuous then they are discretized prior to building the model.
- On the basis of attribute values records are distributed recursively.
- We use statistical methods for ordering attributes as root or the internal node.

In Decision Tree the major challenge is to identification of the attribute for the root node in each level. This process is attribute selection. We have two popular attribute selection measures:

1. Information Gain
2. Gini Index

**1. Information Gain**
 When we use a node in a decision tree to partition the training instances into smaller subsets the entropy changes. Information gain is a measure of this change in entropy. Suppose S is a set of instances, A is an attribute, Sv is the subset of S with A = v, and Values (A) is the set of all possible values of A, then ![](RackMultipart20200701-4-u1de09_html_90f25eb6f58a2e3f.png)

**Entropy**
 Entropy is the measure of uncertainty of a random variable, it characterizes the impurity of an arbitrary collection of examples. The higher the entropy more the information content. Suppose S is a set of instances, A is an attribute, Sv is the subset of S with A = v, and Values (A) is the set of all possible values of A, then




**Performance Metrics**
 There are various metrics which we can use to evaluate the performance of ML algorithms, classification as well as regression algorithms. We must carefully choose the metrics for evaluating ML performance because −

- How the performance of ML algorithms is measured and compared will be dependent entirely on the metric you choose.
- How you weight the importance of various characteristics in the result will be influenced completely by the metric you choose.

### **Confusion Matrix**

It is the easiest way to measure the performance of a classification problem where the output can be of two or more type of classes. A confusion matrix is nothing but a table with two dimensions viz. &quot;Actual&quot; and &quot;Predicted&quot; and furthermore, both the dimensions have &quot;True Positives (TP)&quot;, &quot;True Negatives (TN)&quot;, &quot;False Positives (FP)&quot;, &quot;False Negatives (FN)&quot; as shown below −



Explanation of the terms associated with confusion matrix are as follows −

- **True Positives (TP)** − It is the case when both actual class and predicted class of data point is 1.
- **True Negatives (TN)** − It is the case when both actual class and predicted class of data point is 0.
- **False Positives (FP)** − It is the case when actual class of data point is 0 and predicted class of data point is 1.
- **False Negatives (FN)** − It is the case when actual class of data point is 1 and predicted class of data point is 0.

We can use _confusion\_matrix_ function of _sklearn.metrics_ to compute Confusion Matrix of our classification model.

###

### **Classification Accuracy**

It is most common performance metric for classification algorithms. It may be defined as the number of correct predictions made as a ratio of all predictions made. We can easily calculate it by confusion matrix with the help of following formula −

Accuracy=TP+TNTP+FP+FN+TN

We can use _accuracy\_score_ function of _sklearn.metrics_ to compute accuracy of our classification model.

### **Classification Report**

This report consists of the scores of Precisions, Recall, F1 and Support. They are explained as follows −

### **Precision**

Precision, used in document retrievals, may be defined as the number of correct documents returned by our ML model. We can easily calculate it by confusion matrix with the help of following formula −

Precision=TPTP+FN

### **Recall or Sensitivity**

Recall may be defined as the number of positives returned by our ML model. We can easily calculate it by confusion matrix with the help of following formula.

Recall=TPTP+FN

### **Specificity**

Specificity, in contrast to recall, may be defined as the number of negatives returned by our ML model. We can easily calculate it by confusion matrix with the help of following formula −

Specificity=TNTN+FP

### **Support**

Support may be defined as the number of samples of the true response that lies in each class of target values.

### **F1 Score**

This score will give us the harmonic mean of precision and recall. Mathematically, F1 score is the weighted average of the precision and recall. The best value of F1 would be 1 and worst would be 0. We can calculate F1 score with the help of following formula −

F1=2∗(precision∗recall)/(precision+recall)

F1 score is having equal relative contribution of precision and recall.

We can use classification\_report function of sklearn.metrics to get the classification report of our classification model


## Final Result Based on Accuaracy
Decision Tree : 95.39%
KNN classifier : 90.43%
Multinomial Naive Bayes:95.09%
