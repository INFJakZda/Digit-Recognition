import pandas as pd

train_file_path = 'data/train.csv'
test_file_path = 'data/test.csv'

# read data from files
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# print a summary
print(train_data.describe())
