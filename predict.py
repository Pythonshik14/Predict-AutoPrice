import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

path = "updated_pakwheels.csv"
data = pd.read_csv(path)

X = data.drop("Price", axis=1).columns
Y = data['Price']

Input = [('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]

pipe = Pipeline(Input)
pipe.fit(data[X], Y)

while True:
    inp = [[int(i)] for i in input(f"Введите {list(X)} машины, разделяя всё через запятую: ").split(",")]

    df = pd.DataFrame(dict(zip(X, inp)))

    prediction = pipe.predict(df)
    print("цена - ", prediction[0] * 2.24)
