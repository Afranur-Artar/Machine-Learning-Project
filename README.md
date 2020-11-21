# Machine Learning 

Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.

* In machine learning, a target is called a label.
* In statistics, a target is called a dependent variable.
* A variable in statistics is called a feature in machine learning.
* A transformation in statistics is called feature creation in machine learning.

Two of the most widely adopted machine learning methods are "supervised learning" and "unsupervised learning" and the rest of them is "Semisupervised learning", "Reinforcement learning"

## Supervised Learning Algorithms
They are trained using labeled examples, such as an input where the desired output is known. For example, a piece of equipment could have data points labeled either “F” (failed) or “R” (runs). The learning algorithm receives a set of inputs along with the corresponding correct outputs, and the algorithm learns by comparing its actual output with correct outputs to find errors. It contains regression problems and classification problems. If the target variable is continuous, it is called regression problems; if the target labeled is categorical, is called classification problems

#### 1) Regression  Problems
Regression problems have a continuous outcome variable. For example, you may want to build a model that predicts the next day closing price of Apple’s stock on the Nasdaq. Obviously, Apple’s stock price is not a categorical variable. The first step is to choose an initial set of features that you believe have a relationship with the target variable.

##### y=α+βx+θz

Assumptions of linear regression:
* The relationship between feature(s) and target(s) is linear.
* The errors of the model should be equal to zero on average.
* The model’s errors are consistently distributed, which is known as heteroscedasticity.
* Features are at most only weakly correlated. Put differently there is not strong multicollinearity.
* The model’s errors should be uncorrelated with each other.
* The features and model errors are independent of one another.

#### 2) Classification Problems
While the aim of the regression is to estimate the value of the target variable, the main purpose of the classification is to estimate the class label of the observation and to determine the class in which the observation will take place in accordance with the predetermined class labels. As classification can be made between two options, it can be made among more than two options.

Types of classification:
1. Binary Classficcation
2. Multi-Class Classification

Classification Algorithms:
* Logistic Regression
* Naive Bayes
* Support Vector Machines (SVM)
* K-Nearest Neighbor (KNN)
* Decision Tree
* Random Forest
* Boosting Methods

