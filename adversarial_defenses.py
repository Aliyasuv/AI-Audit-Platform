import torch
import torch.nn as nn
import torch.optim as optim
from adversarial_attacks import fgsm_attack
from model_training import model, trainloader, criterion, optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def adversarial_training(epochs=5, epsilon=0.1):
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            # Generate adversarial examples directly as tensors
            adv_images_np = fgsm_attack(model, images.numpy(), eps=epsilon)
            adv_images = torch.from_numpy(adv_images_np).float().to(images.device).detach()

            optimizer.zero_grad()

            # Train on adversarial examples
            outputs = model(adv_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1} (Adversarial Training), Loss: {running_loss/len(trainloader):.4f}")

    torch.save(model.state_dict(), "robust_model.pth")


if __name__ == "__main__":
    adversarial_training()