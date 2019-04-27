from sklearn.neighbors import KNeighborsClassifier

class KNN:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def classify(self):    
        data_model = KNeighborsClassifier()
        data_model.fit(self.X, self.y)

        return data_model
