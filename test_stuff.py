import tensorflow as tf

import input
import data


session = tf.Session()

train_sequence_length = 20
train_batch_size = 16

my_input_fn = data.get_input_fn(input_file_names=input.get_tfrecord_file_names_from_directory("labeled_by_index"),
                               batch_size=train_batch_size,
                               num_epochs=10,
                               shuffle=True,
                               window_size=train_sequence_length,
                               stride=5)


data = my_input_fn()

for i in range(4):
    out = session.run(data)[1]
    print(out)
