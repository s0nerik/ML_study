
# coding: utf-8

# In[ ]:

# Import stuff
import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.contrib.learn import SVM
from tensorflow.contrib.layers import *
# from IPython.core.interactiveshell import InteractiveShell

# InteractiveShell.ast_node_interactivity = "all"


# In[2]:

# Constants
ID_COLUMN_NAME = "id"

FEATURE_COLUMN_NAMES = ["satisfaction_level", "average_monthly_hours", "salary"]
FEATURE_COLUMNS = [real_valued_column(i) for i in FEATURE_COLUMN_NAMES]

LABEL_COLUMN_NAME = "salary"
LABEL_COLUMN = real_valued_column(LABEL_COLUMN_NAME)


# In[3]:

# Get input data
data = pd.read_csv('input/HR_comma_sep.csv', header=0, usecols=FEATURE_COLUMN_NAMES)
train_data = data[::2]
test_data = data[1::2]
# train_data.head()


# In[4]:

# Training data provider
def input_fn(df):
    features = {
        col: tf.constant(df[col].values)
        for col in df.columns
    }
    features[ID_COLUMN_NAME] = tf.constant([str(i) for i in df.index])
    labels = tf.constant(df[LABEL_COLUMN_NAME].values)
    return features, labels


def input_train():
    return input_fn(train_data)


def input_test():
    return input_fn(test_data)

# In[ ]:

# Create estimator
estimator = SVM(
    example_id_column=ID_COLUMN_NAME,
    feature_columns=FEATURE_COLUMNS
)


# In[ ]:

# Learn!
estimator.fit(input_fn=input_train, steps=200)
results = estimator.evaluate(input_fn=input_test, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))


# # Predict
# def input_test():
#     features = {
#         k: tf.constant([train_data[k].values[0], train_data[k].values[1]])
#         for k in FEATURE_COLUMN_NAMES
#     }
#     return features
#
# for x in estimator.predict(input_fn=input_test):
#     print(x)
