import tensorflow as tf
import numpy as np

max_tokens = 5000  # Maximum vocab size.
max_len = 4  # Sequence length to pad the outputs to.
# Create the layer.

vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=max_tokens,    
    output_sequence_length=max_len)

# Now that the vocab layer has been created, call `adapt` on the
# list of strings to create the vocabulary.
vectorize_layer.adapt(["foo bar", "bar baz", "baz bada boom"])

# Now, the layer can map strings to integers -- you can use an
# embedding layer to map these integers to learned embeddings.
input_data = [["foo qux bar"], ["qux baz"]]
vectorize_layer(input_data)
