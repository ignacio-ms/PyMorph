from mrcnn.config import Config

class CaseConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    #STEPS_PER_EPOCH = 1000

    # Skip detections with < 70% confidence
    DETECTION_MIN_CONFIDENCE = 0.7

    # 3 si son imagenes en RGB
    #IMAGE_CHANNEL_COUNT = 1

    IMAGE_MAX_DIM = 256
    IMAGE_MIN_DIM = 64

    # El numero de objetos maximo de objetos que se pueden encontrar en una imagen
    TRAIN_ROIS_PER_IMAGE = 256

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    MAX_GT_INSTANCES = 256

    POST_NMS_ROIS_INFERENCE = 2048
    POST_NMS_ROIS_TRAINING = 2048
    RPN_NMS_THRESHOLD = 0.8

    # Whether to use image augmentation in training mode
    #AUGMENT = True

    # Whether to use image scaling and rotations in training mode
    #SCALE = True

    # Random crop larger images
    #CROP = True
    #CROP_SHAPE = np.array([256, 256, 3])
