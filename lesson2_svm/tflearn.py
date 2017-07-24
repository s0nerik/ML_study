import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.contrib.learn import SVM
from tensorflow.contrib.layers import *

tf.logging.set_verbosity(tf.logging.INFO)

# sess = tf.Session()

# Load the data
# iris = [(ID, Sepal Length, Sepal Width, Petal Length, Petal Width, Species)]
# iris = np.genfromtxt('input/Iris.csv', delimiter=',', comments='#')

iris = np.genfromtxt('input/Iris_normalized_2.csv', delimiter=',', skip_header=True, comments='#')
# # iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
# # iris = datasets.load_iris()
# x_vals = np.array([i[:-1] for i in iris])
# # x_vals = np.array([[x[1], x[4]] for x in iris])
# y_vals = np.array([i[5] for i in iris])

df_train = pd.DataFrame(
    iris,
    columns=["id", "sepal_length", "sepal_width", "petal_length", "petal_width", "label"]
)

df_test = pd.DataFrame(
    iris,
    columns=["id", "sepal_length", "sepal_width", "petal_length", "petal_width", "label"]
)

ID_COLUMN_NAME = "id"
id_column = real_valued_column("id")

FEATURE_COLUMN_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
feature_columns = [real_valued_column(i) for i in FEATURE_COLUMN_NAMES]

LABEL_COLUMN_NAME = "label"


def input_fn(df):
    features = {
        k: tf.constant(df[k].values)
        for k in FEATURE_COLUMN_NAMES
    }

    features[ID_COLUMN_NAME] = tf.constant([str(i) for i in df[ID_COLUMN_NAME].values])
    # ids = {ID_COLUMN_NAME: tf.constant(df[ID_COLUMN_NAME].values)}

    label = tf.constant(df[LABEL_COLUMN_NAME].values)

    return features, label


def input_train():
    return input_fn(df_train)


def input_eval():
    return input_fn(df_test)


# model_dir = 'svm_model_dir'

estimator = SVM(
    example_id_column=ID_COLUMN_NAME,
    feature_columns=feature_columns,
    # model_dir=model_dir
)
estimator.fit(input_fn=input_train, steps=100)
results = estimator.evaluate(input_fn=input_eval, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))


def test_fn():
    features = {
        k: tf.constant([df_test[k].values[0], df_test[k].values[100]])
        for k in FEATURE_COLUMN_NAMES
    }
    return features

for x in estimator.predict(input_fn=test_fn):
    print(x)
