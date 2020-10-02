# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Supervised Learning

# ## Regression
# In this Notebook we will examine bitcoin prices and see if we can predict the value.
#
# https://www.kaggle.com/mczielinski/bitcoin-historical-data/data
#
# Data under CC BY-SA 4.0 License
#
# https://www.kaggle.com/mczielinski/bitcoin-historical-data

# +
# %matplotlib inline
    
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import ensemble, linear_model, model_selection, preprocessing, svm
from sklearn import metrics 
from yellowbrick import regressor 
# -

 %%time
# Resampling data from minute interval to day
bit_df = pd.read_csv('../data/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv')
# Convert unix time to datetime
bit_df['date'] = pd.to_datetime(bit_df.Timestamp, unit='s')
# Reset index
bit_df = bit_df.set_index('date')
# Rename columns so easier to code
bit_df = bit_df.rename(columns={'Open':'open', 'High': 'hi', 'Low': 'lo', 
                       'Close': 'close', 'Volume_(BTC)': 'vol_btc',
                       'Volume_(Currency)': 'vol_cur', 
                       'Weighted_Price': 'wp', 'Timestamp': 'ts'})
# Resample and only use recent samples that aren't missing
bit_df = bit_df.resample('d').agg({'open': 'first', 'hi': 'max', 
    'lo': 'min', 'close': 'last', 'vol_btc': 'sum',
    'vol_cur': 'sum', 'wp': 'mean', 'ts': 'min'}).iloc[-1000:]
# drop last row as it is not complete
bit_df = bit_df.iloc[:-1]

bit_df

bit_df.dtypes

bit_df.describe()

bit_df.plot(figsize=(6,4))

bit_df.close.plot(figsize=(6,4))

# ### Can we predict tomorrow's close based on today's info?
# We will use a row of data for input. 
#
# We will call the input X and the prediction y. This is called "supervised learning" as we will feed in both X and y to train the model.
#
# Let's use a model called Linear Regression. This performs better if we standardize the data (0 mean, 1 std).
#
# For 2 dimensions this takes the form of 
#
# \begin{align}y = m*x + b\end{align}
#
# M is the slope (or coefficient) and b is the intercept.
#
# Let's see if we can predict the open price from the ts component.

bit_df.plot(kind='scatter', x='ts', y='open', figsize=(6,4)) 

# Create our input (X) and our labelled data (y) to train our model
X = bit_df[['ts']].iloc[:-1]  # drop last row because we don't know what future is
y = bit_df.close.shift(-1).iloc[:-1]

# Train a model and predict output if it were given X
lr_model = linear_model.LinearRegression()
lr_model.fit(X, y)
pred = lr_model.predict(X)

# Plot the real data, our prediction (blue), and the model from the coeffictient (green shifted)
ax = bit_df.plot(kind='scatter', x='ts', y='open', color='black', figsize=(6,4))
ax.plot(X, pred, color='blue')  # matplotlib plot
ax.plot(X, X*lr_model.coef_ + lr_model.intercept_+ 100, linestyle='--', color='green')

# Vertical distance between line and point is the error. *Ordinary Least Squares* 
# regression tries to minimize the square of the distance.
metrics.mean_squared_error(y, pred) 

# Vertical distance between line and point is the error. *Ordinary Least Squares* 
# regression tries to minimize the square of the distance.
metrics.mean_squared_error(y, pred) 



# ### Lab Data
#  
# This exercise looks at predicting the size of forest fires based on meteorological data 
# https://archive.ics.uci.edu/ml/datasets/Forest+Fires
#
# The file is in ../data/forestfires.csv
#
# * Read the data into a DataFrame
# * Examine the types
# * Describe the data
#
# Attribute information:   For more information, read [Cortez and Morais, 2007].
#
# 1. X - x-axis spatial coordinate within the Montesinho park map: 1 to 9
# 2. Y - y-axis spatial coordinate within the Montesinho park map: 2 to 9
# 3. month - month of the year: "jan" to "dec" 
# 4. day - day of the week: "mon" to "sun"
# 5. FFMC - FFMC index from the FWI system: 18.7 to 96.20
# 6. DMC - DMC index from the FWI system: 1.1 to 291.3 
# 7. DC - DC index from the FWI system: 7.9 to 860.6 
# 8. ISI - ISI index from the FWI system: 0.0 to 56.10
# 9. temp - temperature in Celsius degrees: 2.2 to 33.30
# 10. RH - relative humidity in %: 15.0 to 100
# 11. wind - wind speed in km/h: 0.40 to 9.40 
# 12. rain - outside rain in mm/m2 : 0.0 to 6.4 
# 13. area - the burned area of the forest (in ha): 0.00 to 1090.84 (this output variable is very skewed towards 0.0, thus it may make
# sense to model with the logarithm transform).

ff = pd.read_csv('../data/forestfires.csv') 

ff.dtypes 

ff.describe()

ff.area.hist() 



# ## Regression Exercise
#
# * Use linear regression to predict area from the other columns. (If you have object data columns, you can create dummy columns using `pd.get_dummies`, `pd.concat`, and `pd.drop`)
# * What is your score?

ff.dtypes 

yff = ff.area
ff2 = (
    pd.concat([ff, pd.get_dummies(ff.month),
              pd.get_dummies(ff.day)], axis=1)
    .drop(['month', 'day', 'area'], axis=1)
)
Xff = ff2

ff_lr = linear_model.LinearRegression()
ff_lr.fit(Xff, yff)
ff_lr.score(Xff, yff)


 # try log area import numpy as np
def log(x):
    return np.log(x+1)
ff_lr2 = linear_model.LinearRegression()
ff_lr2.fit(Xff, log(yff))
ff_lr2.score(Xff, log(yff))



# ### Regression Evaluation

# +
# Use more columns
# drop last row because we don't know what future is

X = (bit_df
         .drop(['close'], axis=1)
         .iloc[:-1])
y = bit_df.close.shift(-1).iloc[:-1]
cols = X.columns

# We are going to scale the data so that volume and ts don't get more
# weight that other values
ss = preprocessing.StandardScaler()
ss.fit(X)
X = ss.transform(X)
X = pd.DataFrame(X, columns=cols)

X_train, X_test, y_train, y_test = model_selection.\
    train_test_split(X, y, test_size=.3, random_state=42)
# -

# We can now see that the data has a mean close to 0
# and a std of 1
X.describe()

lr_model2 = linear_model.LinearRegression()
lr_model2.fit(X_train, y_train)
lr_model2.score(X_test, y_test)

metrics.r2_score(y_test, lr_model2.predict(X_test))

metrics.mean_squared_error(y_test, lr_model2.predict(X_test))

metrics.mean_absolute_error(y_test, lr_model2.predict(X_test))

regressor.residuals_plot(lr_model2, X_train, y_train,X_test, y_test )



# ## Regression Evaluation Exercise
#
# Using the forest fire data set
#
# * Evaluate the performance of your model with the R2 and MSE score (remember to split your data)
# * Plot a residuals plot

# +
Xff_train, Xff_test, yff_train, yff_test = model_selection.\
    train_test_split(Xff, yff, test_size=.3, random_state=42)

ff_lr = linear_model.LinearRegression()
ff_lr.fit(Xff_train, yff_train)
ff_lr.score(Xff_test, yff_test)
# -

metrics.r2_score(yff_test, ff_lr.predict(Xff_test))

metrics.mean_squared_error(yff_test, ff_lr.predict(Xff_test))

metrics.mean_absolute_error(yff_test, ff_lr.predict(Xff_test))

regressor.residuals_plot(ff_lr, Xff_train, yff_train, Xff_test, yff_test )



# ## Classification
#
#  
# In this Notebook we will examine bitcoin data and see if we can predict a buy or sell.
#
# https://www.kaggle.com/mczielinski/bitcoin-historical-data/data
#
# Data under CC BY-SA 4.0 License
#
# https://www.kaggle.com/mczielinski/bitcoin-historical-data

# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import ensemble, model_selection, preprocessing, tree
from yellowbrick import classifier


# %%time
# Resampling data from minute interval to day
bit_df = pd.read_csv('../data/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv')
# Convert unix time to datetime
bit_df['date'] = pd.to_datetime(bit_df.Timestamp, unit='s')
# Reset index
bit_df = bit_df.set_index('date')
# Rename columns so easier to code
bit_df = bit_df.rename(columns={'Open':'open', 'High': 'hi', 'Low': 'lo', 
                       'Close': 'close', 'Volume_(BTC)': 'vol_btc',
                       'Volume_(Currency)': 'vol_cur', 
                       'Weighted_Price': 'wp', 'Timestamp': 'ts'})
# Resample and only use recent samples that aren't missing
bit_df = bit_df.resample('d').agg({'open': 'first', 'hi': 'max', 
    'lo': 'min', 'close': 'last', 'vol_btc': 'sum',
    'vol_cur': 'sum', 'wp': 'mean', 'ts': 'min'}).iloc[-1000:]
bit_df['buy'] = (bit_df.close.shift(-1) > bit_df.close).astype(int)
# drop last row as it is not complete
bit_df = bit_df.iloc[:-1]

bit_df

bit_df.dtypes

bit_df.describe()



# ## Decision Tree
#
# * The process of training classifier is to get X and y and call `.fit(X, y)`
# * To predict values of y (y hat), call `.predict(X)`
# * To get the accuracy call `.score(X, y)`

ignore = {'buy'}
cols = [c for c in bit_df.columns if c not in ignore]
X = bit_df[cols]
y = bit_df.buy
X_train, X_test, y_train, y_test = model_selection.\
    train_test_split(X, y, test_size=.3, random_state=42)

dt_model = tree.DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_model.score(X_test, y_test)

dt_model.predict(X)

fig, ax = plt.subplots(figsize=(10,4))
_ = tree.plot_tree(dt_model, ax=ax, feature_names=X.columns,
               class_names=['Sell', 'Buy'],
               filled=True)


# note that this goes to a Unix path
tree.export_graphviz(dt_model, out_file='/tmp/tree1.dot', 
                     feature_names=X.columns, class_names=['Sell', 'Buy'],
                    filled=True
                    )

%%bash
# This doesn't run on Windows. Also requires that you have graphviz installed (not a Python module)
dot -Tpng -otree1.png /tmp/tree1.dot

# <img src="tree1.png">



# ### Lab Data
#
# Mushroom data https://archive.ics.uci.edu/ml/datasets/Mushroom
#
# data in ../data/agaricus-lepiota.data.txt
#
# First column is class: edible=e, poisonous=p
#
# Attribute Information:
# * cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s 
# * cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s 
# * cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y 
# * bruises?: bruises=t,no=f 
# * odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s 
# * gill-attachment: attached=a,descending=d,free=f,notched=n 
# * gill-spacing: close=c,crowded=w,distant=d 
# * gill-size: broad=b,narrow=n 
# * gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g,green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y 
# * stalk-shape: enlarging=e,tapering=t 
# * stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=? 
# * stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s 
# * stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s 
# * stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y 
# * stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y 
# * veil-type: partial=p,universal=u 
# * veil-color: brown=n,orange=o,white=w,yellow=y 
# * ring-number: none=n,one=o,two=t 
# * ring-type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z 
# * spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y 
# * population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y 
# * habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d

mush_df = pd.read_csv('../data/agaricus-lepiota.data.txt',
     names='class,cap_shape,cap_surface,cap_color,bruises,'
     'odor,g_attachment,g_spacing,g_size,g_color,s_shape,'
     's_root,s_surface_a,s_surface_b,s_color_a,s_color_b,'
     'v_type,v_color,ring_num,ring_type,spore_color,pop,hab'.split(','))
mush_df = pd.get_dummies(mush_df, columns=mush_df.columns).drop(['class_e'],axis=1)



# ## Predict Exercise
#
# * Create a decision tree to model whether a mushroom is poisonous. 
# * What is the score?

mush_X = mush_df.iloc[:,1:]
mush_y = mush_df.class_p
mush_X_train, mush_X_test, mush_y_train, mush_y_test = model_selection.\
    train_test_split(mush_X, mush_y, test_size=.3, random_state=42)
mush_dt = tree.DecisionTreeClassifier()
mush_dt.fit(mush_X_train, mush_y_train)
mush_dt.score(mush_X_test, mush_y_test)



# ## Classification Evaluation

dt_model = tree.DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_model.score(X_test, y_test)

metrics.accuracy_score(y_test, dt_model.predict(X_test))

metrics.precision_score(y_test, dt_model.predict(X_test))

metrics.recall_score(y_test, dt_model.predict(X_test))

classifier.confusion_matrix(dt_model, X_train, y_train, X_test, y_test)

classifier.confusion_matrix(dt_model, X_train, y_train, X_test, y_test,
                   classes=['sell', 'buy'])

classifier.roc_auc(dt_model, X_train, y_train, X_test, y_test,
                   classes=['sell', 'buy'], micro=False, macro=False)

classifier.precision_recall_curve(dt_model, X_train, y_train, X_test, y_test,
                   classes=['sell', 'buy'], micro=False, macro=False)



# ## Classification Evaluation Exercise
# With the mushroom dataset
# * Evaluate the accuracy
# * Plot a confusion matrix
# * Plot an ROC 

mush_X = mush_df.iloc[:,1:]
mush_y = mush_df.class_p
mush_X_train, mush_X_test, mush_y_train, mush_y_test = model_selection.\
    train_test_split(mush_X, mush_y, test_size=.3, random_state=42)
mush_dt = tree.DecisionTreeClassifier()
mush_dt.fit(mush_X_train, mush_y_train)
mush_dt.score(mush_X_test, mush_y_test)

metrics.accuracy_score(mush_y_test, mush_dt.predict(mush_X_test))

classifier.confusion_matrix(mush_dt, mush_X_train, mush_y_train, mush_X_test, mush_y_test,
                   classes=['poisonous', 'edible'])

classifier.roc_auc(mush_dt, mush_X_train, mush_y_train, mush_X_test, mush_y_test,
                   classes=['poisonous', 'edible'], micro=False, macro=False)







