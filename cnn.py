import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import time
import copy

#torch.manual_seed(10)

# Define the transform for the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the dataset
trainset = MNIST(root='./data', train=True, download=True, transform=transform)
testset = MNIST(root='./data', train=False, download=True, transform=transform)

# Define the dataloaders
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=True)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 10) 

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.avgpool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.avgpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return out

modelCNN = CNN()

modelToPrune = copy.deepcopy(modelCNN)

# Train the model
def trainModel(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.train()
    trainStart = time.time()
    for epoch in range(5):
        for images, labels in trainloader:
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}: training loss: {loss.item()}')
            
    trainEnd = time.time()
    trainTotal = trainEnd - trainStart
    minutes = round(trainTotal // 60)
    seconds = round(trainTotal % 60)
    print(f'Time training: {minutes}min {seconds}sec')

def evaluateModel(model):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct / total} %')

def get_total_parameters_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_pruned_parameters_count(pruned_model, total_params_count):
    params = 0
    for param in pruned_model.parameters():
        if param is not None:
            params += torch.nonzero(param).size(0)
    print('Pruned Model parameter count:', params)
    print(f'Compressed Percentage: {(100 - (params / total_params_count) * 100)}%')

def L1Structured(modelCNN, percentage, total_params_count):
    model = copy.deepcopy(modelCNN)
    print(f'\nPruning L1 Structured {percentage * 100}%')
    prune.ln_structured(model.conv1, 'weight', amount=percentage, n=1, dim=0)
    prune.ln_structured(model.conv2, 'weight', amount=percentage, n=1, dim=0)
    prune.ln_structured(model.fc1, 'weight', amount=percentage, n=1, dim=0)

    trainModel(model)

    prune.remove(model.conv1, 'weight')
    prune.remove(model.conv2, 'weight')
    prune.remove(model.fc1, 'weight')

    evaluateModel(model)

    get_pruned_parameters_count(model, total_params_count)

def L2Structured(modelCNN, percentage, total_params_count):
    model = copy.deepcopy(modelCNN)
    print(f'\nPruning L2 Structured {percentage * 100}%')
    prune.ln_structured(model.conv1, 'weight', amount=percentage, n=2, dim=0)
    prune.ln_structured(model.conv2, 'weight', amount=percentage, n=2, dim=0)
    prune.ln_structured(model.fc1, 'weight', amount=percentage, n=2, dim=0)

    trainModel(model)

    prune.remove(model.conv1, 'weight')
    prune.remove(model.conv2, 'weight')

    evaluateModel(model)

    get_pruned_parameters_count(model, total_params_count)
    
def globalL1Unstructured(modelCNN, percentage, total_params_count):
    model = copy.deepcopy(modelCNN)
    print(f'\nPruning L1 Unstructured {percentage * 100}%')
    parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1, 'weight')
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=percentage,
    )

    trainModel(model)

    prune.remove(model.conv1, 'weight')
    prune.remove(model.conv2, 'weight')
    prune.remove(model.fc1, 'weight')

    evaluateModel(model)

    get_pruned_parameters_count(model, total_params_count)
    
def globalRandomUnstructured(modelCNN, percentage, total_params_count):
    model = copy.deepcopy(modelCNN)
    print(f'\nPruning Random Unstructured {percentage * 100}%')
    parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1, 'weight')
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=percentage,
    )

    trainModel(model)

    prune.remove(model.conv1, 'weight')
    prune.remove(model.conv2, 'weight')
    prune.remove(model.fc1, 'weight')

    evaluateModel(model)

    get_pruned_parameters_count(model, total_params_count)


trainModel(modelCNN)
evaluateModel(modelCNN)
total_params_count = get_total_parameters_count(modelCNN)
print('Original Model paramete count:', total_params_count)

L1Structured(modelToPrune, 0.1, total_params_count)
L1Structured(modelToPrune, 0.3, total_params_count)
L1Structured(modelToPrune, 0.5, total_params_count)
L1Structured(modelToPrune, 0.9, total_params_count)

L2Structured(modelToPrune, 0.1, total_params_count)
L2Structured(modelToPrune, 0.3, total_params_count)
L2Structured(modelToPrune, 0.5, total_params_count)
L2Structured(modelToPrune, 0.9, total_params_count)

globalL1Unstructured(modelToPrune, 0.1, total_params_count)
globalL1Unstructured(modelToPrune, 0.3, total_params_count)
globalL1Unstructured(modelToPrune, 0.5, total_params_count)
globalL1Unstructured(modelToPrune, 0.9, total_params_count)

globalRandomUnstructured(modelToPrune, 0.1, total_params_count)
globalRandomUnstructured(modelToPrune, 0.3, total_params_count)
globalRandomUnstructured(modelToPrune, 0.5, total_params_count)
globalRandomUnstructured(modelToPrune, 0.9, total_params_count)
