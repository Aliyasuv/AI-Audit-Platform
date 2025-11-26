# Configuration parameters for AI Audit platform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "model.pth"
DISTILLED_MODEL_SAVE_PATH = "distilled_model.pth"
ROBUST_MODEL_SAVE_PATH = "robust_model.pth"
ADVERSARIAL_EPSILON = 0.1
ANOMALY_THRESHOLD_PERCENTILE = 95
NB_CLASSES = 10
INPUT_SHAPE = (1, 28, 28)
