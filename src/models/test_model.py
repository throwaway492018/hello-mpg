import os
import pickle
import pandas as pd

from sklearn.metrics import accuracy_score

build_dir = os.environ.get('TRAVIS_BUILD_DIR', './')

#Load data for testing
df_test = pd.read_csv(os.path.join(build_dir, 'test/data/test_mpg.csv'),
                      index_col='name')

with open(os.path.join(build_dir, 'models/cylinder_model.pkl'), 'rb') as fd:
    model = pickle.load(fd)

    predicted = model.predict(df_test[['mpg', 'displacement']])

    print('Accuracy: ', accuracy_score(df_test['cylinders'], predicted))
