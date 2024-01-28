import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tf
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from model import DigitNN


torch.manual_seed(4)

totEpoch = 10
lr = 1e-3
betaV = (0.9,0.999)

tran = tf.Compose([
    tf.ToTensor(),
    tf.Normalize((0.5,), (0.5,))
])

dataset = MNIST(root='./data', train=True, transform=tran, download=True)
tsize = int(0.8 * len(dataset))
vsize = len(dataset) - tsize
train_dataset, val_dataset = random_split(dataset, [tsize, vsize])

tbatch = DataLoader(train_dataset, batch_size=128, shuffle=True)
vbatch = DataLoader(val_dataset, batch_size=128, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitNN().to(device)
optimizer = optim.Adam(model.parameters(),lr=lr,betas=betaV)
loss_fn = nn.CrossEntropyLoss()

tloss=[]
best_vacc=0.0

for epoch in range (totEpoch):
    #Train Loop
    correct = 0
    samples = 0
    totLoss = 0
    model.train()
    for features,labels in tbatch:
        
        features, labels = features.to(device), labels.to(device)
        preds = model(features)
        loss = loss_fn(preds,labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        totLoss += loss.item()
        
        _, predicted = torch.max(preds, 1)
        samples += labels.size(0)
        correct += (predicted == labels).sum().item()
    tloss.append(totLoss/len(tbatch))
    tacc = correct/samples
    
    #Val Loop
    correct = 0
    samples = 0    
    model.eval()
    with torch.no_grad():
        for features, labels in vbatch:
            features, labels = features.to(device), labels.to(device)
            preds = model(features)
            
            _, predicted = torch.max(preds, 1)
            samples += labels.size(0)
            correct += (predicted == labels).sum().item()
    vacc = correct/samples
    print(f"Epoch {epoch+1}, T Acc: {tacc:.3f}, V Acc: {vacc:.3f}")
    if vacc > best_vacc:
        best_vacc = vacc
        torch.save(model.state_dict(), 'weights.pth')
        print('Weights Saved')
            
plt.plot(range(1, totEpoch + 1), tloss, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()

