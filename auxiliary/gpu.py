import os
import tensorflow as tf


def increase_gpu_memory():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]
                )
        except RuntimeError as e:
            print(e)


def set_mixed_precision():
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)


def set_gpu_allocator():
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


def clear_session():
    tf.keras.backend.clear_session()
