import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from adversarial_attacks import fgsm_attack
from model_training import model, trainloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Vectorized feature extraction
def extract_features(images):
    # images shape: (N, C, H, W) tensor
    np_images = images.cpu().numpy()
    pixel_variance = np.var(np_images, axis=(1, 2, 3))
    mean_pixel_value = np.mean(np_images, axis=(1, 2, 3))
    return np.vstack((pixel_variance, mean_pixel_value)).T


def generate_adversarial_examples(images, epsilon=0.1):
    adv_imgs = []
    for img in images:
        img = img.unsqueeze(0).to(device)
        adv_np = fgsm_attack(model, img, eps=epsilon)  # This is numpy array
        adv_img = torch.from_numpy(adv_np).float().to(device)  # Convert to tensor on device
        adv_img = adv_img.detach().cpu().squeeze(0)  # Now safe to call tensor methods
        adv_imgs.append(adv_img)
    return torch.stack(adv_imgs)


def train_detection_model():
    model.eval()
    normal_images = []
    for images, _ in trainloader:
        normal_images.append(images)
    normal_images = torch.cat(normal_images)

    adv_images = generate_adversarial_examples(normal_images)

    X = np.concatenate([extract_features(normal_images), extract_features(adv_images)])
    y = np.array([0] * len(normal_images) + [1] * len(adv_images))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    detector = RandomForestClassifier()
    detector.fit(X_train, y_train)

    y_pred = detector.predict(X_test)
    print(classification_report(y_test, y_pred))

    return detector


detector_model = train_detection_model()


def classify_input(img):
    features = extract_features(img.unsqueeze(0))
    prediction = detector_model.predict(features)
    return "Adversarial" if prediction[0] == 1 else "Normal"


if __name__ == "__main__":
    sample_img = torch.randn(1, 28, 28)  # Dummy test image with shape (C, H, W)
    result = classify_input(sample_img)
    print(f"Sample Image Classified As: {result}")