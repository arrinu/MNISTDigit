import torch
import numpy as np
import torchvision.transforms as tf
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from model import DigitNN

tran = tf.Compose([
    tf.ToTensor(),
    tf.Normalize((0.5,), (0.5,))
])

test_dataset = MNIST(root='./data', train=False, transform=tran, download=True)
test_batch = DataLoader(test_dataset, batch_size=128, shuffle=False)

model = DigitNN()
model.load_state_dict(torch.load('weights.pth'))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

correct = 0
total = 0
all_labels = []
all_preds = []

with torch.no_grad():
    for features, labels in test_batch:
        features, labels = features.to(device), labels.to(device)
        preds = model(features)

        _, predicted = torch.max(preds, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

test_accuracy = correct / total
print(f'Test Accuracy: {test_accuracy:.3f}')

conf_matrix = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
