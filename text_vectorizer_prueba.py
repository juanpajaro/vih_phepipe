#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from keras.layers import TextVectorization



max_tokens = 5000  # Maximum vocab size.
max_len = 4  # Sequence length to pad the outputs to.
# Create the layer.
vectorize_layer = TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=max_len)

print(type(vectorize_layer))

vectorize_layer.adapt(["foo bar", "bar baz", "baz bada boom"])

input_data = tf.constant(["foo qux bar", "qux baz"])

output = vectorize_layer(input_data)
print(output.numpy())