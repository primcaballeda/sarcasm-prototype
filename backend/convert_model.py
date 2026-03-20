"""
Convert model.h5 to SavedModel format to fix Lambda layer issues
"""
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

# Make tf and K available globally for Lambda layers during loading
import sys
sys.modules['__main__'].tf = tf
sys.modules['__main__'].K = K
globals()['tf'] = tf
globals()['K'] = K

print("Loading model.h5...")

# Create comprehensive custom_objects for Lambda layers
custom_objects = {
    'tf': tf,
    'K': K,
    'reduce_mean': tf.math.reduce_mean,
    'reduce_sum': tf.math.reduce_sum,
    'reduce_max': tf.math.reduce_max,
    'sqrt': tf.math.sqrt,
    'square': tf.math.square,
    'expand_dims': tf.expand_dims,
    'squeeze': tf.squeeze,
    'concat': tf.concat,
    'stack': tf.stack,
    'transpose': tf.transpose,
}

try:
    # Load the .h5 model
    model = keras.models.load_model(
        './model/model.h5',
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False
    )
    print("Model loaded successfully!")
    print(f"Model summary:")
    model.summary()
    
    # Save as Keras format (.keras) - native Keras 3 format
    print("\n\nSaving as Keras format...")
    model.save('./model/model.keras')
    print(" Keras model saved to ./model/model.keras")
    
    # Export as SavedModel to a directory (Keras 3 style)
    print("\nExporting as TensorFlow SavedModel...")
    model.export('./model/saved_model')
    print("SavedModel exported to ./model/saved_model/")
    
    print("\nConversion complete!")
    print("\nYou can now use:")
    print("  - ./model/model.keras (recommended - native Keras 3)")
    print("  - ./model/saved_model/ (TensorFlow SavedModel format)")
    
except Exception as e:
    print(f"\n Error: {e}")
    print("\nIf the model can't be loaded, you may need to provide:")
    print("  1. The model architecture code")
    print("  2. The training script")
    print("  3. Or save the model in a different format from the source")
