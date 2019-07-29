import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from simple_cnn import SimpleCNN
from optim_lookahead import Lookahead

ROOT_DIR = './data'
EPOCHS = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root=ROOT_DIR, train=True, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root=ROOT_DIR, train=False, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=4)


net = SimpleCNN()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Lookahead(optim.Adam(net.parameters(), lr=0.001), k=5, alpha=0.5)

for epoch in range(EPOCHS):
    bar = tqdm(trainloader)
    for data in bar:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        bar.set_description('Epoch %d: loss %.4f' % (epoch + 1, loss.item()))

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on test images: %d %%' % (100 * correct / total))
