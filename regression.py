# %% read data
import pandas as pd

train = pd.read_csv(
    "house-prices-advanced-regression-techniques/train.csv"
)
test = pd.read_csv(
    "house-prices-advanced-regression-techniques/test.csv"
)


# %% checkout out first few rows
import pandas as pd
train = pd.read_csv("house-prices-advanced-regression-techniques/train.csv")
train.head()


# %% checkout out dataframe info
import pandas as pd
train = pd.read_csv("house-prices-advanced-regression-techniques/train.csv")
train.info()


# %% describe the dataframe
import pandas as pd
train = pd.read_csv("house-prices-advanced-regression-techniques/train.csv")
train.describe(include="all")


# %% SalePrice distribution
import seaborn as sns
import pandas as pd
train = pd.read_csv("house-prices-advanced-regression-techniques/train.csv")

sns.distplot(train["SalePrice"])


# %% SalePrice distribution w.r.t CentralAir / OverallQual / BldgType / etc


# %% SalePrice distribution w.r.t YearBuilt / Neighborhood
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
train = pd.read_csv("house-prices-advanced-regression-techniques/train.csv")

house_prices = sns.load_dataset('house_prices')

# Plot SalePrice distribution with respect to YearBuilt and Neighborhood
plt.figure(figsize=(12, 8))
sns.scatterplot(x='YearBuilt', y='salePrice', hue='Neighborhood', data=house_prices, alpha=0.7, palette='viridis')

# Customize the plot
plt.title('SalePrice Distribution with Respect to YearBuilt and Neighborhood')
plt.xlabel('YearBuilt')
plt.ylabel('SalePrice')
plt.legend(title='Neighborhood', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.show()

# %%
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_log_error
import numpy as np


def evaluate(reg, x, y):
    pred = reg.predict(x)
    result = np.sqrt(mean_squared_log_error(y, pred))
    return f"RMSLE score: {result:.3f}"


dummy_reg = DummyRegressor()

dummy_selected_columns = ["MSSubClass"]
dummy_train_x = train[dummy_selected_columns]
dummy_train_y = train["SalePrice"]

dummy_reg.fit(dummy_train_x, dummy_train_y)
print("Training Set Performance")
print(evaluate(dummy_reg, dummy_train_x, dummy_train_y))

truth = pd.read_csv("truth_house_prices.csv")
dummy_test_x = test[dummy_selected_columns]
dummy_test_y = truth["SalePrice"]

print("Test Set Performance")
print(evaluate(dummy_reg, dummy_test_x, dummy_test_y))

print("Can you do better than a dummy regressor?")


# %% your solution to the regression problem
