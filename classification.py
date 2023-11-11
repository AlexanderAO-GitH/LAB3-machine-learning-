# %% read data
import pandas as pd

train = pd.read_csv("titanic/train.csv")
test = pd.read_csv("titanic/test.csv")


# %% checkout out first few rows
import pandas as pd
train = pd.read_csv("titanic/train.csv")
train.head()


# %% checkout out dataframe info
import pandas as pd
train = pd.read_csv("titanic/train.csv")
train.info()


# %% describe the dataframe
import pandas as pd
train = pd.read_csv("titanic/train.csv")
train.describe(include="all")


# %% visualize the dataset, starting with the Survied distribution
import seaborn as sns
import pandas as pd
train = pd.read_csv("titanic/train.csv")
sns.countplot(x="Survived", data=train)


# %% Survived w.r.t Pclass / Sex / Embarked ?
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

# Set up the figure with subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# Plot survival distribution with respect to Pclass
sns.countplot(x='pclass', hue='survived', data=titanic, ax=axes[0])
axes[0].set_title('Survival Distribution with Respect to Pclass')

# Plot survival distribution with respect to Sex
sns.countplot(x='sex', hue='survived', data=titanic, ax=axes[1])
axes[1].set_title('Survival Distribution with Respect to Sex')

# Plot survival distribution with respect to Embarked
sns.countplot(x='embarked', hue='survived', data=titanic, ax=axes[2])
axes[2].set_title('Survival Distribution with Respect to Embarked')

# Customize the overall layout
plt.tight_layout()

# Show the plot
plt.show()


# %% Age distribution ?
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
train = pd.read_csv("titanic/train.csv")


titanic = sns.load_dataset('titanic')

# Plot the age distribution
sns.histplot(titanic['age'].dropna(), kde=True, bins=20)

# Customize the plot
plt.title('Age Distribution on Titanic')
plt.xlabel('Age')
plt.ylabel('Count')

# Show the plot
plt.show()

# %% Survived w.r.t Age distribution ?
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
train = pd.read_csv("titanic/train.csv")

titanic = sns.load_dataset('titanic')

sns.histplot(data=titanic, x='age', hue='survived', multiple='stack', kde=True)

plt.title('Survival Distribution with Respect to Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()


# %% SibSp / Parch distribution ?


# %% Survived w.r.t SibSp / Parch  ?


# %% Dummy Classifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score


def evaluate(clf, x, y):
    pred = clf.predict(x)
    result = f1_score(y, pred)
    return f"F1 score: {result:.3f}"


dummy_clf = DummyClassifier(random_state=2020)

dummy_selected_columns = ["Pclass"]
dummy_train_x = train[dummy_selected_columns]
dummy_train_y = train["Survived"]

dummy_clf.fit(dummy_train_x, dummy_train_y)
print("Training Set Performance")
print(evaluate(dummy_clf, dummy_train_x, dummy_train_y))

truth = pd.read_csv("truth_titanic.csv")
dummy_test_x = test[dummy_selected_columns]
dummy_test_y = truth["Survived"]

print("Test Set Performance")
print(evaluate(dummy_clf, dummy_test_x, dummy_test_y))

print("Can you do better than a dummy classifier?")


# %% Your solution to this classification problem

