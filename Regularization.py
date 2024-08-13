import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import warnings
warnings.filterwarnings("ignore")
sns.set()

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

all_data = pd.concat((train_data.loc[:,"MSSubClass":"SaleCondition"],
                     test_data.loc[:,"MSSubClass":"SaleCondition"]))


"""Outliers - >
   Data points that are significantly different from the rest.
   Removing outliers is often risky because it involves discarding data.
   68-95-99 rule:
   68 -> Within 1 standard deviation
   95 -> Within 2 standard deviations
   99 -> Within 3 standard deviations
   We will use 3 here.
   Find the mean value of GrLivArea: mean = 1515
   std = 525.48
   3 standard deviations = 525.48*3=1576.44
   3 standard deviations on both sides of the mean.
   -60.98 < ... > 3091.6
   Therefore, we can consider anything above 3100 as an outlier.
   Use the describe function to find std and mean.
"""
print(train_data["GrLivArea"].describe())

# Set figure size
rcParams["figure.figsize"] = (6.0,6.0)  # This sets the figure size for all plots.
sns.scatterplot(x="GrLivArea", y="SalePrice", data=train_data)
plt.show()

train_data = train_data.drop(train_data[(train_data["GrLivArea"]>3200)].index).reset_index(drop=True)

# Combine the data after outlier removal
all_data = pd.concat((train_data.loc[:,"MSSubClass":"SaleCondition"],
                     test_data.loc[:,"MSSubClass":"SaleCondition"]))

# Convert numerical columns to strings (e.g., for dates)
print(all_data["YrSold"])  # year

all_data["MSSubClass"] = all_data["MSSubClass"].apply(str)    # type of house (e.g., 1-story, 2-story)
all_data["YrSold"] = all_data["YrSold"].apply(str)            # the years the houses were sold
all_data["OverallCond"] = all_data["OverallCond"].apply(str)  # overall condition of the house
all_data["MoSold"] = all_data["MoSold"].apply(str)            # the months the houses were sold (1-12)

# Encode Categorical Data
categorical_columns = all_data.select_dtypes(include=["object"]).columns  # Get the number and assign to a variable

from sklearn.preprocessing import LabelEncoder
cols = ["FireplaceQu","BsmtQual","BsmtCond","GarageQual","GarageCond","ExterQual",
        "ExterCond","HeatingQC","PoolQC","KitchenQual","BsmtFinType1","BsmtFinType2",
        "Functional","Fence","BsmtExposure","GarageFinish","LandSlope","LotShape","PavedDrive",
        "Street","Alley","CentralAir","MSSubClass","OverallCond","YrSold","MoSold"]

for c in cols:
    lbl = LabelEncoder()
    all_data[c] = lbl.fit_transform(all_data[c])

# One hot encoding
all_data = pd.get_dummies(all_data)


# Normalization
# The target variable's distribution is skewed to the right

# Scipy's skew function gives us the skewness value
from scipy.stats import skew

rcParams["figure.figsize"] = (12.0,6.0)
g = sns.distplot(train_data["SalePrice"],label=f"Skewness: {train_data['SalePrice'].skew()}")
g = g.legend(loc="best")
plt.show()

""" Regression models generally perform better on normally distributed data.
    If we use the target variable (y) as it is, we might not get healthy results.
    This is especially true for regression.
    Let's normalize the data by applying a log transformation.
    For right-skewed data, applying a log transform helps make the data more normally distributed.
    However, this does not work for left-skewed data.
"""

normalizedSalesPrice = np.log1p(train_data["SalePrice"])
rcParams["figure.figsize"] = (12.0,6.0)
g = sns.distplot(normalizedSalesPrice, label=f"Skewness: {train_data['SalePrice'].skew()}")
g = g.legend(loc="best")
plt.show()

train_data["SalePrice"] = np.log1p(train_data["SalePrice"])

# MISSING VALUES
# Check if there are missing values
t = all_data.isnull().any().any()
all_data = all_data.fillna(all_data.mean())
tk = all_data.isnull().any().any()

x_train = all_data[:train_data.shape[0]]
x_test = all_data[train_data.shape[0]:]

y = train_data.SalePrice


# Linear Regression
from sklearn.model_selection import cross_val_score

def rmse_cv(model,cv=5):
    rmse = np.sqrt(-cross_val_score(model, x_train,y,scoring="neg_mean_squared_error",cv=cv))
    return rmse

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

rmse = rmse_cv(lr)
print(f"RMSE Mean: {rmse.mean()}, std: {rmse.std()}")

lr.fit(x_train,y)

# coefficients 
weights = lr.coef_

# Get the top 10 coefficients (absolute values)
coef = pd.Series(weights,index=x_train.columns)

imp_coef = pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])

imp_coef.plot(kind="barh")
plt.title("Top Significant Coefficients")
plt.show()

lr_predict = lr.predict(x_test)
lr_predict = np.expm1(lr_predict)
"""
test_data['SalePrice'] = lr_predict
submission = test_data[['Id', 'SalePrice']]
submission.to_csv('1.csv', index=False)
"""
# Ridge Regression
""" Let's now apply Ridge Regression, which uses L2 regularization, on the same data.
    Traditionally, we set alpha to 0.1.
"""
from sklearn.linear_model import Ridge
ridgeModel = Ridge(alpha=0.1)

rmse2 = rmse_cv(ridgeModel)
print(f"RMSE Mean: {rmse2.mean()}, std: {rmse2.std()}")

ridgeModel.fit(x_train,y)
ridge_predict = ridgeModel.predict(x_test)
ridge_predict = np.expm1(ridge_predict)

"""
test_data['SalePrice'] = ridge_predict
submission = test_data[['Id', 'SalePrice']]
submission.to_csv('12.csv', index=False)
"""
# If coefficients are not sufficiently reduced, it means alpha wasn't set correctly.
alphas = [0.05,0.1,0.3,1,3,5,10,15,30,50,75]

cv_ridge = [rmse_cv(Ridge(alpha=alpha)).mean() for alpha in alphas]

cv_ridge = pd.Series(cv_ridge,index=alphas)
# The best alpha was found to be 10

ridgeModel2 = Ridge(alpha=10)
rmse = rmse_cv(ridgeModel2)
ridgeModel2.fit(x_train,y)

ridge_predict2 = ridgeModel.predict(x_test)
ridge_predict2 = np.expm1(ridge_predict2)

"""
test_data['SalePrice'] = ridge_predict2
submission = test_data[['Id', 'SalePrice']]
submission.to_csv('52.csv', index=False)
"""

# Let's also try Lasso (L1)
from sklearn.linear_model import LassoCV

lasso = LassoCV(alphas=np.linspace(0.0002,0.0022,21),cv=5).fit(x_train,y)
# The Lasso alpha values I considered may not give the desired results, so we use cross-validation.
print(lasso.alpha_)

from sklearn.linear_model import Lasso

lasso2 = Lasso(alpha=0.0005)
lasso2.fit(x_train,y)
rmse = rmse_cv(lasso2)

lasso_predict = lasso2.predict(x_test)
lasso_predict = np.expm1(lasso_predict)


from xgboost import XGBRegressor  # We will use XGBRegressor to make regression predictions.

xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=0)  # Reduced learning_rate
xgb.fit(x_train, y)
xgb_predict = xgb.predict(x_test)  # Make predictions on the test data
xgb_predict = np.expm1(xgb_predict)

# Save the predictions to a CSV file
submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': xgb_predict})
submission.to_csv('31.csv', index=False)
