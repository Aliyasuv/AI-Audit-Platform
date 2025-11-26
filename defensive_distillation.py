import torch
import torch.nn as nn
from model_training import trainloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DistilledNet(nn.Module):
    def __init__(self, temperature=10):
        super(DistilledNet, self).__init__()
        self.temperature = temperature
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1).to(device)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x) / self.temperature
        return x

def train_distilled_model(epochs=5, learning_rate=0.001):
    distilled_model = DistilledNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(distilled_model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        distilled_model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = distilled_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1} (Defensive Distillation), Loss: {running_loss/len(trainloader):.4f}")

    torch.save(distilled_model.state_dict(), "distilled_model.pth")

if __name__ == "__main__":
    train_distilled_model()