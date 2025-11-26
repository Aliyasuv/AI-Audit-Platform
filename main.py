import torch
import numpy as np
from model_training import train_model, model
from adversarial_attacks import fgsm_attack
from adversarial_defenses import adversarial_training
from ai_threat_intelligence import detect_future_threats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def main():
    print("Training AI Model...")
    train_model()

    print("Running Adversarial Attacks...")
    sample_image = torch.rand(1, 1, 28, 28).to(device)  # Simulated sample as tensor
    adversarial_example = fgsm_attack(model, sample_image, eps=0.1)

    print("Applying Adversarial Training...")
    adversarial_training()

    print("Analyzing AI Security Threats...")
    attack_history = np.random.rand(100, 2)
    risk_assessment = detect_future_threats(attack_history)
    print(f"Predicted Risk Level → {risk_assessment}")

    print("AI Security Audit Completed Successfully!")


if __name__ == "__main__":
    main()