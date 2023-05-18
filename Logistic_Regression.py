from sklearn import metrics
from sklearn.linear_model import LogisticRegression


class logistic_regression:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

    def Accuracy(self):
        Model = LogisticRegression()
        Model.fit(self.X_train, self.Y_train)
        Prediction = Model.predict(self.X_test)
        print('Logistic Accuracy is : ', metrics.accuracy_score(Prediction, self.Y_test))
        print('Logistic mean squared error is : ', metrics.mean_squared_error(Prediction, self.Y_test))
        print('Logistic confusion matrix is : ', metrics.confusion_matrix(Prediction, self.Y_test))
        print('Logistic classification report is : ', metrics.classification_report(Prediction, self.Y_test))
        print("--------------------------------------------------")
