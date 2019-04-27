import pandas as pd
from classifiers import regressor, random_forest

train_file_path = 'data/train.csv'
test_file_path = 'data/test.csv'

# read data from files
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# print a summary
# print(train_data.describe())
# print columns
# print(train_data.columns)

# set label - prediction target
y = train_data.label
# set pixels - features
X = train_data.iloc[:, 1:]

# ****************** CLASSIFY ****************** 

# data_model = regressor.Regressor(X, y).classify()
data_model = random_forest.RandomForest(X, y).classify()
# data_model = regressor.Regressor(X, y).classify()
# data_model = regressor.Regressor(X, y).classify()


# ****************** END CLASSIFY ******************

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(data_model.predict(X.head()))
print(y.head())

# predicted_values = data_model.predict(test_data)
predicted_values = data_model.predict(test_data)

file=open('submission.csv','w')
header="ImageId,Label"
header=header+'\n'
file.write(header)
for i, id in enumerate(predicted_values):
    # print(id)
    # print(type(id))
    str="{},{}".format(i + 1, int(id))
    str=str+'\n'
    file.write(str)
