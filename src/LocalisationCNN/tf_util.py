import tensorflow as tf
from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse
import tensorflow.contrib.slim as slim

def print_memory_usage(session, device_name="/device:GPU:0"):
    """
        A way to print the actual usage of memory used by the model.
        https://stackoverflow.com/questions/36123740/is-there-a-way-of-determining-how-much-gpu-memory-is-in-use-by-tensorflow
        device_name '/device:GPU:0', 'device:CPU/0', etc
    """
    with tf.device(device_name):  # Replace with device you are interested in
        bytes_in_use = BytesInUse()
    
    mem_bytes = session.run(bytes_in_use)
    print('\nModel memory usage {:.2f} MB'.format(mem_bytes / 2**20))

def print_model_vars():
    """
        Retrieve the number of variables used by the network and print it.
    """
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)