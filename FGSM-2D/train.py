import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet152


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
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)



best_acc = 0.0
best_model_wts = None


for epoch in range(num_epochs):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f'Train Epoch: {epoch + 1} [{i + 1} / {len(trainloader)}], Loss: {running_loss / 100:.4f}, Accuracy: {correct / total * 100:.2f}%')
            running_loss = 0.0
            correct = 0
            total = 0

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    test_acc = correct / total * 100

    print(f'Test Epoch: {epoch + 1}, Loss: {test_loss / len(testloader):.4f}, Accuracy: {test_acc:.2f}%')

    if test_acc > best_acc:
        best_acc = test_acc
        best_model_wts = model.state_dict()

model.load_state_dict(best_model_wts)
print(best_acc)

torch.save(model.state_dict(), 'resnet152_cifar10_best_2.pth')