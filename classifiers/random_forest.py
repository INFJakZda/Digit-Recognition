from sklearn.ensemble import RandomForestClassifier

class RandomForest:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def classify(self):    
        data_model = RandomForestClassifier(random_state=1)
        data_model.fit(self.X, self.y)

        return data_model
