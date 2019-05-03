import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

# For continuous integration with Travis CI
build_dir = os.environ.get('TRAVIS_BUILD_DIR', './')

# Load the data
df = pd.read_csv(os.path.join(build_dir, 'data/processed/mpg.csv'),
                 index_col='name')

df_train, df_validate = train_test_split(df, test_size=0.2)

# Train a decision tree classifier model
model = tree.DecisionTreeClassifier()
response = df_train['cylinders']
predictors = df_train[['mpg', 'displacement']]

model.fit(predictors, response)

# Validate the model on the validation data frame
validate_pred = df_validate[['mpg', 'displacement']]
predicted = model.predict(validate_pred)
print('Accuracy: ', accuracy_score(df_validate['cylinders'], predicted))

# Package model into a serialized pickle file
with open(os.path.join(build_dir, 'models/cylinder_model.pkl'), 'wb') as fd:
    pickle.dump(model, fd)
