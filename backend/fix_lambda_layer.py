"""
Fix Lambda layer by recreating the model with a proper layer
"""
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np

# Make tf available globally
globals()['tf'] = tf
globals()['K'] = K

custom_objects = {
    'tf': tf,
    'K': K,
    'reduce_mean': tf.math.reduce_mean,
}

print("Loading model...")
model = keras.models.load_model(
    './model/model.keras',
    custom_objects=custom_objects,
    safe_mode=False
)

print("\nModel architecture:")
model.summary()

# The Lambda layer likely does: tf.reduce_sum(x, axis=1) or similar
# Let's inspect it
print("\n\nLambda layer info:")
for layer in model.layers:
    if 'lambda' in layer.name.lower():
        print(f"Layer name: {layer.name}")
        print(f"Layer type: {type(layer)}")
        
        # Try to get the function
        if hasattr(layer, 'function'):
            print(f"Function: {layer.function}")
        if hasattr(layer, 'arguments'):
            print(f"Arguments: {layer.arguments}")
            
print("\n\nRecreating model with Lambda replaced by custom layer...")

# Create a custom layer to replace the Lambda
class SumLayer(keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

# Get references to specific layers by name
embedding_layer = model.get_layer('embedding_2')
dropout_5 = model.get_layer('dropout_5')
conv1d = model.get_layer('conv1d_1')
bidirectional = model.get_layer('bidirectional_2')
dense_8 = model.get_layer('dense_8')
flatten = model.get_layer('flatten_2')
activation = model.get_layer('activation_2')
repeat_vector = model.get_layer('repeat_vector_2')
permute = model.get_layer('permute_2')
multiply_layer = model.get_layer('multiply_2')
# Skip lambda - we'll replace it
dropout_6 = model.get_layer('dropout_6')
dense_9 = model.get_layer('dense_9')
dense_10 = model.get_layer('dense_10')
dense_11 = model.get_layer('dense_11')

# Build new model
inputs = keras.Input(shape=(50,), name='input_layer')
x = embedding_layer(inputs)
x = dropout_5(x, training=False)
x = conv1d(x)
x = bidirectional(x)

# Attention mechanism
attention_dense = dense_8(x)
attention_flatten = flatten(attention_dense)
attention_softmax = activation(attention_flatten)
attention_repeat = repeat_vector(attention_softmax)
attention_permute = permute(attention_repeat)
x_attended = multiply_layer([x, attention_permute])

# Replace Lambda with custom SumLayer
x = SumLayer(name='sum_layer')(x_attended)

# Continue with rest of model
x = dropout_6(x, training=False)
x = dense_9(x)
x = dense_10(x)
outputs = dense_11(x)

# Create fixed model
fixed_model = keras.Model(inputs=inputs, outputs=outputs)

print("\n\nFixed model summary:")
fixed_model.summary()

# Save fixed model
print("\nSaving fixed model...")
fixed_model.save('./model/model_fixed.keras')
print("✓ Fixed model saved to ./model/model_fixed.keras")

# Test with dummy input
print("\nTesting fixed model...")
dummy_input = np.random.randint(0, 1000, size=(1, 50))
result = fixed_model.predict(dummy_input, verbose=0)
print(f"Test prediction shape: {result.shape}")
print(f"Test prediction value: {result[0]}")

print("\n✅ Done! Use model_fixed.keras in your app")
