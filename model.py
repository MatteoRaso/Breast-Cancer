import pandas as pd
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("data.csv")
del data["id"]
train = data.sample(frac=0.8)
validate = data.drop(train.index)
train_Y = train["diagnosis"]
validate_Y = validate["diagnosis"]
del train["diagnosis"]
del validate["diagnosis"]

model = LogisticRegression(max_iter=5000).fit(train, train_Y)
print(model.score(validate, validate_Y))
