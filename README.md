Introduction
In recent years, machine learning has been increasingly used in the financial sector, especially in credit card fraud detection. Although fraudulent transactions account for a very low proportion, the economic losses they cause are very huge. Therefore, building an efficient fraud detection model has important practical significance. However, one of the main challenges facing fraud detection is the class imbalance problem in the data, that is, the number of fraudulent transactions is far less than that of normal transactions. This makes traditional evaluation indicators (such as accuracy) no longer applicable, and the Precision-Recall curve is more suitable for measuring model performance. Therefore, in this assignment, I used the automated machine learning (AutoML) tool to build a credit card fraud detection model. AutoML can automatically select models and optimize hyperparameters, which is very suitable for handling complex classification tasks. My goal is to explore the application of AutoML in fraud detection and evaluate the model performance through the Precision-Recall curve. I chose the credit card fraud detection dataset on Kaggle, which contains 284,807 transaction records.
My process includes: training the model with AutoML, evaluating the performance through the Precision-Recall curve, and setting the threshold to balance the precision and recall. In addition, I also recorded the decision-making process in the assignment, including the selection of datasets, the economic feasibility analysis of AI solutions, and the lessons learned from the experiments.
