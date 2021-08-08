# -*- coding: utf-8 -*-
"""Review_dataset_auto.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oCAPTkRUL5ADq18RNWrht0I3I6Mf1vBT

# **1. Data wrangling**

* identify missing values
* deal with missing data
    1. drop data 
    2. replace (by mean, by frequency)

* correct data format


---



* standartization (transforming data into a common format, f.e. *mpg into L/100km *)

* normalization (transforming values of several values into a simillar range, f.e. *columns height, width transforming to 0-1*)

* binning (transforming continuos numerical variables into discrete categorical f.e. *column curb-weight transforming into categorical: easy, easy-medium, medium, heavy-medium, heavy*)



'curb-weight' | 'curb-weight-binning' 
:---:|:---:
    1200      |      easy-medium   
    950       |      easy         
    1850      |      heavy-medium        
    1900      |      heavy-medium        
    2500      |      heavy         


* indicator variable (is a numerical variable used to label categories, add new columns for each unique value f.e. column fuel-type transforming into 2 columns 'fuel-type-gas', 'fuel-type-diesel')



'fuel-type' | 'fuel-type-gas' | 'fuel-type-diesel'
:---:|:---:|:---:
    gas     |       1         |       0
    gas     |       1         |       0
    diesel  |       0         |       1
    gas     |       1         |       0
    diesel  |       0         |       1
"""

from google.colab import drive
drive.mount("/content/gdrive", force_remount=True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
path = "/content/gdrive/My Drive/Home_works/auto_dataset.csv"
data = pd.read_csv(path)

path = "/content/gdrive/My Drive/Home_works/clean_auto.csv"
data = pd.read_csv(path)



data.drop("Unnamed: 0", axis=1, inplace=True)

data['horsepower-binned']

plt.hist(data['normalized-losses'], bins=3)

for i in data.columns:
  print(i, " - ", len(set(data[i])))

data['normalized-losses-binning'] = pd.cut(data['normalized-losses'], [ 65, 128.66666667, 192.33333333, 256], labels=['low', 'medium', 'high'], include_lowest=True)











"""# ** 1.1 Module Seaborn **"""

import seaborn as sns

data.corr()

vers = data[data['name'] == "versicolor"]
virg = data[data['name'] == "virginica"]

plt.plot(vers['petal_length'], virg['sepal_width'], "rs")

"""**NOTE :**

normalization | standartization | binned | indicator | correlation
:---:|:---:|:---:|:---:|:---:
width| city-mpg | horsepower | aspiration | city-lkm
height | highway-mpg | | fuel-type | fuel-type-diesel
length



normalization, standartization, binned, indicator columns to *remove*

---



---



* city_lkm remove
* fuel-type-diesel remove


"""





"""**2. Categorical variables, remove some columns**
for predict price depend columns

**Task 1**
- seperate price for 5 groops
- find column which is better for predict group of price
- try to predict price_group with used the column
- calculate metric (TRUETH/TOTAL)

"""

def binning(data, col):
  lin = np.linspace(pd.to_numeric(data[col]).min(), pd.to_numeric(data[col]).max(), 6)

  low = lin[1]; lm = lin[2]; mid = lin[3]; mh = lin[4]; high = lin[5]
  
  data[col] = data[col].astype(float)

  comp = []
  for i in data[col]:

    if i < low:
      comp.append("low")

    if i >= low and i < lm:
      comp.append('below-medium')

    elif i > lm and i < mid:
      comp.append('medium')

    elif i > mid and i < mh:
      comp.append('above-medium')
    
    elif i > mh  and i <= high:
      comp.append('high')
   
    return comp





"""**2.1  find border in column for predict price_group**

| column | value | between price group |
|:---:|:---:|:---:|
|city-mpg | 18 |medium   above-medium |
|engine-size|260|above-medium    expensive|
|horsepower|100|cheap   below-medium|
|width|69|below-medium   medium

"""

sp = [0] * 201
for i, row in df.iterrows():
  if row['horsepower'] < 100:
    if sp[i] == 0:
      sp[i] = "cheap"

print(sp)
for i, row in df.iterrows():
  if row['engine-size'] > 260:
    if sp[i] == 0:
      sp[i] = "expensive"
  
print(sp)

TRUETH = 0
for i, row in df.iterrows():
  if row['petal_length'] < 2.5:
    pred = "setosa"
  elif row['petal_width'] > 1.8 or row['sepal_length'] > 7.1 or row['petal_length'] > 5.1 or row['sepal_width'] > 3.5:
    pred = "virginica"
  elif (row['petal_width'] > 1.7 and row['sepal_length'] > 5.9):
    pred = "virginica"
  elif (row['sepal_length'] >= 5.9 and row['petal_length'] >= 5 and row['sepal_width'] <= 3 and row['petal_width'] < 1.6):
    pred = "virginica"
  elif row['sepal_width'] >= 2.5 and row['sepal_length'] < 5:
    pred = "virginica"
  elif row['petal_length'] >= 5 and row['petal_width'] >= 1.8:
    pred = "virginica"
  else:
    pred = "versicolor"
  # TRUETH = TRUETH + 1 if row['name'] == pred else TRUETH
  if row['name'] == pred:
    TRUETH += 1
  else:
    print(i, row)

TRUETH / len(df)







"""#**3. Correlation and causation**

**3.1 module stats**

---

* p-value < 0.001 - is **strong evidence** that the correlation is significant
* p-value > 0.001 and p-value < 0.005 - is **moderate evidence** that the correlation is significant
* p-value > 0.005 and p-value < 0.01 - is **weak evidence** that the correlation is significant
* p-value > 0.01 - is no evidence that the correlation is significant
"""

# P-value
# Pearson correlationfrom
from scipy import stats

coeffPearson, p_value = stats.pearsonr(data['wheel-base'], data['price'])
print(p_value)

for c in df.corr().columns:
  coef, p_value = stats.pearsonr(df[c], df['price'])
  print(c, p_value, sep="\t")



"""**Important variables**,  columns for predict price:
* length
* width
* curb-weight
- engine-size
* horsepower
* city-mpg
* highway-mpg
* bore
* wheel-base

* drive-wheels
* engine-location
* num-of-cylinders


"""



"""# **5. Model Development**

Linear regression - is a method to help us understanding the relationship between two variables

* parameter - highway-mpg
* coefficient - b slope of the regression line
* intercept - a of the regression line

"""

# classification [cheap, below-medium, medium, above-medium, expensive], iris, naiveBayes
# regression [0, 1, 2, 10 .... 100, 101, 102.... ] real numbers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = "/content/gdrive/My Drive/Home_works/auto_important_column.csv"
data = pd.read_csv(path)

from sklearn.linear_model import LinearRegression
ln = LinearRegression()

X = data[list(data.columns)]
Y = data['price']

data

"""**formula : Y = a + b * X**

a - intercept
b - slope
"""

ln.fit(X, Y)

print("intercept = ", ln.intercept_)
print("coeff = ", ln.coef_)

y = 38423.3058581574 + -821.73337832 * 22

y_pred = ln.predict(X)

print(y_pred[200])

from sklearn.metrics import mean_squared_error
# mean_squared_error(Y, y_pred)

data

len(data.corr().columns)



"""# **7. Polynomial Regression**"""

plt.plot(sorted(data['price']), data['horsepower'])



from sklearn.preprocessing import PolynomialFeatures as PF
pr = PF(degree=2)

z = data[['horsepower']]

z_pr = pr.fit_transform(z)

z_pr

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def indicate(data, col):
  dicty = {}
  for un in data[col].unique():
    dicty[str(un)] = []

  for i in data[col]:
    dicty[str(i)].append("1")
    for j in dicty:
      if j != str(i):
        dicty[str(j)].append("0")

  for res in dicty:
    data[f"{col}-{res}"] = dicty[res]
  
  data.drop(col, axis=1, inplace=True)

def binning(df, col):
  borders = plt.hist(df[col], bins=50)[1]
  labels = [f"{i}" for i in range(len(borders)-1)]
  df[f'{col}_binned'] = pd.cut(df[col], borders, labels=labels, include_lowest=True)
  # del df['curb-weight']
  indicate(df, "curb-weight_binned")
  indicate(df, "city-mpg")

binning(data, "curb-weight")

def PolynomialRegression(data, cols, y, DEGREE=2):

  pr = PolynomialFeatures(degree=DEGREE)
  params = data[cols]

  params_pr = pr.fit_transform(params)
  Input = [('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
  pipe = Pipeline(Input)

  pipe.fit(params, y)
  prediction = pipe.predict(params)
  print(prediction)

  print('The R-square is: ', pipe.score(params, y))
  return f"metric MSE - {mean_squared_error(y, prediction)}"

params = list(data.columns)
params.remove("price")
PolynomialRegression(data, params)





"""# **8. Split train-test**
- An important step in testing your model is to split your data into training and testing data. We will place the target data price in a separate dataframe y

"""

from sklearn.model_selection import train_test_split
params = list(data.columns)
params.remove("price")

x_train, x_test, y_train, y_test = train_test_split(data[params], data['price'], test_size=0.15, random_state=1)
print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

x_train

PolynomialRegression(x_train, params, y_train)

PolynomialRegression(x_test, params, y_test)







import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

params = ['length',
 'width',
 'curb-weight',
 'engine-size',
 'horsepower',
 'city-mpg',
 'highway-mpg']

x_data = data[params]
y_data = data['price']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=0)
pr = PolynomialFeatures(degree=2)

x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)

poly = LinearRegression()
poly.fit(x_train_pr, y_train)

yhat = poly.predict(x_test_pr)

mean_squared_error(yhat, y_test)
print(poly.intercept_)
poly.coef_

import itertools
print(len(list(itertools.combinations(params, 2))))
list(itertools.combinations(params, 2))









"""# **10. GridSearch**"""

def func(names, digit):
  MSE = {}
  for i in list(itertools.combinations(names, digit)):
    pr = PolynomialFeatures(degree=2)
    x_train, x_test, y_train, y_test = train_test_split(data[list(i)], data['price'], test_size=0.2, random_state=1)

    # params_pr = pr.fit_transform(x_train)

    Input = [('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
    
    pipe = Pipeline(Input)
    pipe.fit(x_train, y_train)

    prediction = pipe.predict(x_test)

    MSE[i] = mean_squared_error(prediction, y_test)

  return MSE

from sklearn.model_selection import train_test_split
names = data.drop("price", axis=1).columns

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.1)
X = df[['length', 'width', 'curb-weight', 'horsepower', 'highway-mpg', 'num-of-cylinders-four', 'num-of-cylinders-six', 'num-of-cylinders-two']]
ridge.fit(X, df['price'])
y_pred = ridge.predict(X)
mean_squared_error(y_pred, df['price'])
ridge.score(X, df['price'])
parameters1= [{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000, 100000]}] # linspace(0, 1000000000, 10)
grid = GridSearchCV(RR, parameters1, cv=12)
grid.fit(X, df['price'])
