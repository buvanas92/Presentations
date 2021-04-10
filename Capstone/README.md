## Executive Summary


We will be exploring the public dataset, on Stroke drawn from Kaggle. With the help of Python libraries NumPy, Pandas, Matplotlib and Seaborn, this presentation covers the Exploratory Data Analysis (EDA) and fits various Machine Learning techniques and deploys deep learning methods. The acquired dataset contains tabular data on demographics (age, gender and maritial statuses) as well as information on their smoking habits and on health parameters such as BMI and average gluscose levels. We will train the machine and deep learning models to come to a conclusive prediction model.

## Objective


The main objective of this projectis to build and test predictive models to better detect Stroke occurrences fast and accurately. We will be building Machine learning models and Neural Network to predict the Stroke occurrences. In our following dataset, Stroke occurrence is categorised as a binary variable with 0(no stroke) and 1(having stroke).

## Problem statement


To predict stroke occurrences to allow for early detection of the disease and reduce mortality rates.

## Scope


The scope of this project is to run the learning models on two datasets using two different feature selection methods. We will then use accuracy as the performance metrics for evaluation. In this project, we have used what we have learned about deep neural networks and machine learning models namely Logistic Regression,KNN classifier, SCM, Random Forest and AdaBoost techniques to Stroke occurrences accurately and quickly. We will train and validate a model so it can classify if the person is prone to having Stroke or not using the Stroke detection dataset. After the model is trained, we will compare the performance metrics between the Machine and Deep learning models.

# Steps /Project Model







-Load the data set


-Explore,summarise and visualise the dataset (Extensive EDA)


-Encoding of categorical variables with dummy variables and normalisation of data set to pre process the dataset


-Feature selection methords (Forward Feature Selection and PCA)


-Design, train and test the model architecture


-Deploy machine learning techniques of Logistic Regression, SVM, KNN, Random Forest and Ada Boost


-Build Artificial Neural Network of 15 nodes and 1 hidden layer


-Perform hyper parameter tuning uusing GridSearchCV


-Summarize results and identify the best model to achieve desirable data and busness outcomes for Medical institutes and relevant stakeholders

## Parameters used for Machine Learning models across both the datasets with two different feature selection methods were kept uniform

Train/Test split of 80%/20%
Cross validation of 10 folds
Hyper parameter tuning by GridSearch CV

## Parameters used for ANN model across both the datasets with identified predictors were also kept uniform

Keras sequential model,ANN model with 15 Neurons in L1 and 1 hidden layer

Drop out rate :0.2


Optimizer: Adam


Loss: Binary cross entropy


Metrics: Accuracy


100 epoch runs

## Results
Across both the datasets,Machine learning models performed way better than ANN even with parameter tuning in place.For detailed accuracy score  and recall metrics, please refer to the jupyter notebook and presentation slides.

## Summary

Based on the data collected, we can use Machine Learning model — Random Forest to help with the prediction of Stroke occurrences. It provided the best accuracy at the rate of 99.9% across both datasets.
						
AUC Score of 1.00 for and train and 0.97 for test indicates a good classifier and overfitting is unlikely.The recall and precision are high above 90% indicating the exactness and proper classification of data points, which is crucial in detecting diseases.
						
Medical institutes can detect Stroke in accurate and timely manner to reduce mortality rates further.







