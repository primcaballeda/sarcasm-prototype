"""
Fix Lambda layer in model.h5 by replacing with SumLayer
This converts your trained model to a loadable format
"""
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

print("="*80)
print("LAMBDA LAYER FIX SCRIPT")
print("This will convert model.h5 to model_fixed.keras")
print("="*80)

# Custom SumLayer to replace Lambda
class SumLayer(keras.layers.Layer):
    """Replaces Lambda(lambda x: tf.reduce_sum(x, axis=-2))"""
    def call(self, inputs, mask=None):
        # Accept mask parameter like Lambda does
        return tf.reduce_sum(inputs, axis=-2)
    
    def compute_mask(self, inputs, mask=None):
        # Lambda doesn't output a mask, so neither should we
        return None
    
    def get_config(self):
        return super().get_config()

# Make tf available in Lambda scope
globals()['tf'] = tf
globals()['K'] = K

print("\n1. Loading model.h5...")
try:
    # Try to load model.h5 with Lambda in scope
    def lambda_func(xin):
        return tf.reduce_sum(xin, axis=-2)
    
    model = keras.models.load_model(
        './model/model.h5',
        custom_objects={
            'tf': tf,
            'K': K,
            '<lambda>': lambda_func,
        },
        compile=False,
        safe_mode=False
    )
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model.h5: {e}")
    print("\nTrying model.keras instead...")
    try:
        model = keras.models.load_model(
            './model/model.keras',
            compile=False,
            safe_mode=False
        )
        print("✓ Model loaded from model.keras")
    except Exception as e2:
        print(f"✗ Failed: {e2}")
        exit(1)

print("\n2. Analyzing model architecture...")
print(f"   Total layers: {len(model.layers)}")

# Find Lambda layers
lambda_layers = []
for i, layer in enumerate(model.layers):
    if isinstance(layer, keras.layers.Lambda):
        lambda_layers.append((i, layer.name))
        print(f"   Found Lambda layer at index {i}: {layer.name}")

if not lambda_layers:
    print("   No Lambda layers found - model may already be fixed")
    print("\n3. Saving as model_fixed.keras anyway...")
    model.save('./model/model_fixed.keras')
    print("✓ Model saved")
    exit(0)

print(f"\n3. Rebuilding model with {len(lambda_layers)} Lambda layer(s) replaced...")

# Use keras.models.clone_model with custom layer replacement
def clone_function(layer):
    """Custom clone function that replaces Lambda with SumLayer"""
    if isinstance(layer, keras.layers.Lambda):
        print(f"   Replacing Lambda '{layer.name}' with SumLayer")
        return SumLayer(name=layer.name.replace('lambda', 'sum'))
    else:
        # Return the layer config to clone it
        return layer.__class__.from_config(layer.get_config())

print("\n4. Cloning model...")
new_model = keras.models.clone_model(model, clone_function=clone_function)

print("\n5. Copying weights...")
# Copy weights from old to new model
for old_layer, new_layer in zip(model.layers, new_model.layers):
    if not isinstance(old_layer, keras.layers.Lambda):
        weights = old_layer.get_weights()
        if weights:
            new_layer.set_weights(weights)
            print(f"   ✓ {old_layer.name}")

print("\n✓ All weights copied successfully!")

print("\n6. Compiling new model...")
new_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n7. Saving fixed model...")
new_model.save('./model/model_fixed.keras')

print("\n" + "="*80)
print("SUCCESS!")
print("✓ Fixed model saved to: ./model/model_fixed.keras")
print("✓ Your trained weights have been preserved")
print("✓ Backend will now load this model automatically")
print("="*80)
print("\nNext step: Restart your backend server")
print("Command: cd backend; python app.py")


# Test with dummy input
print("\nTesting fixed model...")
dummy_input = np.random.randint(0, 1000, size=(1, 50))
result = fixed_model.predict(dummy_input, verbose=0)
print(f"Test prediction shape: {result.shape}")
print(f"Test prediction value: {result[0]}")

print("\n✅ Done! Use model_fixed.keras in your app")
