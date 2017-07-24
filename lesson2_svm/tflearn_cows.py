import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.contrib.learn import SVM
from tensorflow.contrib.layers import *

tf.logging.set_verbosity(tf.logging.INFO)

# data = np.genfromtxt('input/cowhealth_normalized.csv', delimiter=',', skip_header=True, comments='#')
data = np.genfromtxt('input/cowhealth_test_1.csv', delimiter=',')

ID_COLUMN_NAME = "id"
FEATURE_COLUMN_NAMES = ["d%s" % i for i in range(1, 8)]
LABEL_COLUMN_NAME = "label"

df_train = pd.DataFrame(
    data,
    columns=[FEATURE_COLUMN_NAMES + [LABEL_COLUMN_NAME]]
)

feature_columns = [real_valued_column(i) for i in FEATURE_COLUMN_NAMES]


def input_fn(df):
    features = {
        k: tf.constant(df[k].values)
        for k in FEATURE_COLUMN_NAMES
    }

    features[ID_COLUMN_NAME] = tf.constant([str(i) for i in range(df.shape[0])])

    label = tf.constant(df[LABEL_COLUMN_NAME].values)

    return features, label


def input_train():
    return input_fn(df_train)


def input_eval():
    return input_fn(df_train)


# model_dir = 'svm_model_dir'

estimator = SVM(
    example_id_column=ID_COLUMN_NAME,
    feature_columns=feature_columns,
    # l2_regularization=1.
    # model_dir=model_dir
)
estimator.fit(input_fn=input_train, steps=1000)
results = estimator.evaluate(input_fn=input_eval, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))


def test_fn():
    features = {
        k: tf.constant([df_train[k].values[0], df_train[k].values[-1]])
        for k in FEATURE_COLUMN_NAMES
    }
    return features

for x in estimator.predict(input_fn=test_fn):
    print(x)