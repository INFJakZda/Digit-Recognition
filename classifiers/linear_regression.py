from sklearn.linear_model import LinearRegression

class LinearRegressor:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def classify(self):    
        data_model = LinearRegression()
        data_model.fit(self.X, self.y)

        return data_model
