from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
class knn:

    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test


    def Accuracy(self):
        knn = KNeighborsClassifier()
        knn.fit(self.X_train, self.Y_train)
        Prediction = knn.predict(self.X_test)
        print('KNN Accuracy is : ', metrics.accuracy_score(Prediction, self.Y_test))
        print('KNN mean squared error is : ', metrics.mean_squared_error(Prediction, self.Y_test))
        print('KNN confusion matrix is : ', metrics.confusion_matrix(Prediction, self.Y_test))
        print('KNN classification report is : ', metrics.classification_report(Prediction, self.Y_test))
        print("--------------------------------------------------")

