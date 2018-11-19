from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
import six
import tensorflow as tf
from google.cloud import storage

storage_client = storage.Client()

TRAIN_FILE = 'gs://airline-tf/train-0.01m.csv'
EVAL_FILE  = 'gs://airline-tf/test.csv'
CSV_COLUMNS = ['Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'UniqueCarrier',
               'Origin', 'Dest', 'Distance', 'dep_delayed_15min']
CSV_COLUMN_DEFAULTS = [[''], [''], [''], [0], [''], [''], [''], [0], ['']]
LABEL_COLUMN = 'dep_delayed_15min'
LABELS = ['Y', 'N']

INPUT_COLUMNS = [
    tf.feature_column.categorical_column_with_vocabulary_list(
        'Month',
        ['1', '2', '3', '4', '5', '6',
         '7', '8', '9', '10', '11', '12']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'DayofMonth',
        ['1','2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',  '16', 
         '17', '18', '19', '20', '21', '22', '23','24', '25', '26', '27', '28', '29', '30', '31']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'DayOfWeek',
        ['1', '2', '3', '4', '5', '6','7']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'UniqueCarrier',
        ['AA', 'AQ', 'AS', 'B6', 'CO', 'DH', 'DL', 'EV', 'F9', 'FL', 'HA',
         'HP', 'MQ', 'NW', 'OH', 'OO', 'TZ', 'UA', 'US', 'WN', 'XE', 'YV']),
    
    tf.feature_column.categorical_column_with_hash_bucket(
        'Origin', hash_bucket_size=100),
    tf.feature_column.categorical_column_with_hash_bucket(
        'Dest', hash_bucket_size=100),
    
    tf.feature_column.numeric_column('DepTime'),
    tf.feature_column.numeric_column('Distance')
]

def build_estimator(config, model = "lr"):
    if model == "lr": 
        return tf.estimator.LinearClassifier(feature_columns = INPUT_COLUMNS,
         config=config)

def parse_label_column(label_string_tensor):

  table = tf.contrib.lookup.index_table_from_tensor(tf.constant(LABELS))

  return table.lookup(label_string_tensor)

def parse_csv(rows_string_tensor):
    
    row_columns = tf.expand_dims(rows_string_tensor, -1)
    columns = tf.decode_csv(row_columns, record_defaults=CSV_COLUMN_DEFAULTS)
    features = dict(zip(CSV_COLUMNS, columns))

    return features

def input_fn(filenames,
             num_epochs=None,
             shuffle=True,
             skip_header_lines=1,
             batch_size=1024):

    dataset = tf.data.TextLineDataset(filenames).skip(skip_header_lines).map(parse_csv)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return features, parse_label_column(features.pop(LABEL_COLUMN))

### SERVING FUNCTIONS ###
def csv_serving_input_fn():
  """Build the serving inputs."""
  csv_row = tf.placeholder(
      shape=[None],
      dtype=tf.string
  )
  features = parse_csv(csv_row)
  features.pop(LABEL_COLUMN)
  return tf.estimator.export.ServingInputReceiver(features, {'csv_row': csv_row})


def example_serving_input_fn():
  """Build the serving inputs."""
  example_bytestring = tf.placeholder(
      shape=[None],
      dtype=tf.string,
  )
  feature_scalars = tf.parse_example(
      example_bytestring,
      tf.feature_column.make_parse_example_spec(INPUT_COLUMNS)
  )
  return tf.estimator.export.ServingInputReceiver(
      features,
      {'example_proto': example_bytestring}
  )

# [START serving-function]
def json_serving_input_fn():
  """Build the serving inputs."""
  inputs = {}
  for feat in INPUT_COLUMNS:
    inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)

  return tf.estimator.export.ServingInputReceiver(inputs, inputs)
# [END serving-function]

SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn,
    'EXAMPLE': example_serving_input_fn,
    'CSV': csv_serving_input_fn
}

     
