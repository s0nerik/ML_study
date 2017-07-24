import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.contrib.learn import SVM
from tensorflow.contrib.layers import *

tf.logging.set_verbosity(tf.logging.INFO)

# data = np.genfromtxt('input/cowhealth_normalized.csv', delimiter=',', skip_header=True, comments='#')
data_x = np.genfromtxt('input/HR_comma_sep.csv', delimiter=',', skip_header=True)

# data = np.array([np.array(list(i[:-2]) + list([i[-1]])) for i in data_x])
data = np.array([np.array(list([i[0]]) + list([i[3]]) + list([i[-1]])) for i in data_x])

ID_COLUMN_NAME = "id"
FEATURE_COLUMN_NAMES = [
    "satisfaction_level", "average_monthly_hours"
    # "satisfaction_level", "last_evaluation", "number_project", "average_monthly_hours",
    # "time_spend_company", "Work_accident", "left", "promotion_last_5years"
]
LABEL_COLUMN_NAME = "salary"

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
    l2_regularization=10.
    # model_dir=model_dir
)
estimator.fit(input_fn=input_train, steps=200)
results = estimator.evaluate(input_fn=input_eval, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))


def test_fn():
    features = {
        k: tf.constant([df_train[k].values[0], df_train[k].values[1]])
        for k in FEATURE_COLUMN_NAMES
    }
    return features

for x in estimator.predict(input_fn=test_fn):
    print(x)
