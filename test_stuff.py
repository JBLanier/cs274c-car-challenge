import tensorflow as tf
import numpy as np
import rolling_ops
import os
import input
import data


session = tf.Session()

my_input_fn = data.get_input_fn(
    input_file_names=input.get_tfrecord_file_names_from_directory("labeled_by_index"),
    batch_size=4,
    num_epochs=1,
    shuffle=False,
    return_full_size_image=True,
    window_size=20,
    stride=5
)
data = my_input_fn()

for i in range(4000):
    out = session.run(data)[1]
    if len(out) is not 4:
        print(i)
        print(len(out))
        print("WOW")
        print(out)
        exit(1)
