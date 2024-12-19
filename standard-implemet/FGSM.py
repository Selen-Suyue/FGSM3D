import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet152
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
num_epochs = 30
learning_rate = 0.01

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./Datasets', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./Datasets', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

model = resnet152(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 10),
    nn.Sigmoid()
)
model.load_state_dict(torch.load('resnet152_cifar10_best.pth'))
model = model.to(device)
 
judge=model.eval()
test_loss = 0
correct = 0
total = 0

for epsilon in np.linspace(0,1,100):
    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs.requires_grad_(True)
        outputs = model(inputs)
        loss =torch.nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad=inputs.grad.data
        inputs=inputs+epsilon*torch.sign(grad)
        out=judge(inputs)
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    test_acc = correct / total 
    print(test_acc)
