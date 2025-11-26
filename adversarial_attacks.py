import torch
import numpy as np
import matplotlib.pyplot as plt
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from model_training import model, criterion, optimizer

# Ensure model is in eval mode and on CPU for ART
model.eval()
model.cpu()

# Define ART classifier wrapper with clip_values for MNIST
classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
    clip_values=(0, 1)
)

def fgsm_attack(model, data, eps=0.1):
    fgsm = FastGradientMethod(estimator=classifier, eps=eps)
    
    # Ensure data is numpy array
    if hasattr(data, 'cpu'):
        data_np = data.cpu().numpy()
    else:
        data_np = data

    adv_example = fgsm.generate(x=data_np)
    return adv_example

def visualize_attack(original, adversarial):
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original.reshape(28, 28), cmap="gray")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Adversarial Image")
    plt.imshow(adversarial.reshape(28, 28), cmap="gray")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create a dummy test image with values in [0,1], shape (1,1,28,28)
    sample_image = np.random.rand(1, 1, 28, 28).astype(np.float32)
    adv_example = fgsm_attack(model, sample_image, eps=0.1)

    # Visualize original and adversarial images (first sample in batch)
    visualize_attack(sample_image[0,0], adv_example[0,0])