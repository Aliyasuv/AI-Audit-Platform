import matplotlib.pyplot as plt


def visualize_images(original, adversarial):
    # Convert torch tensors to numpy if needed
    if hasattr(original, 'cpu'):
        original = original.cpu().numpy()
    if hasattr(adversarial, 'cpu'):
        adversarial = adversarial.cpu().numpy()

    original_img = original.reshape(28, 28)
    adversarial_img = adversarial.reshape(28, 28)

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_img, cmap="gray")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Adversarial Image")
    plt.imshow(adversarial_img, cmap="gray")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
