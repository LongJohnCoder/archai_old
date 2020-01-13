import torch
import torchvision
import torchvision.transforms as transforms
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from basic_net import BasicNet

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
logger = logging.getLogger()
# h = logging.StreamHandler(sys.stdout)
# h.flush = sys.stdout.flush
# logger.addHandler(h)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

datadir = os.path.expanduser('~/torchvision_data_dir')
trainset = torchvision.datasets.CIFAR10(root=datadir, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                        shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root=datadir, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


device = torch.device('cuda')
net = BasicNet().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.016, momentum=0.9)

for epoch in range(1):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, *_ = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 0:    # print every 2000 mini-batches
            logger.info(f'[{epoch}, {i}] loss: {running_loss/2000.0}')
            sys.stdout.flush()
            running_loss = 0.0

print('Finished Training')


correct = 0
total = 0
with torch.no_grad():
    for (images, labels) in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs, *_ = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

logger.info(f'Accuracy on test images: {100.0 * correct / total}')

sys.stdout.flush()

exit(0)