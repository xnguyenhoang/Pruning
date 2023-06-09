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

# Define the FNN model
class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

modelFNN = FNN(28*28, 100, 10)


modelToPrune = copy.deepcopy(modelFNN)

# Train the model
def trainModel(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.train()
    trainStart = time.time()
    for epoch in range(5):
        for images, labels in trainloader:
            images = images.view(-1, 28*28).requires_grad_()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
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
    for images, labels in testloader:
        images = images.view(-1, 28*28).requires_grad_()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        
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
    prune.ln_structured(model.fc1, 'weight', amount=percentage, n=1, dim=0)
    prune.ln_structured(model.fc2, 'weight', amount=percentage, n=1, dim=0)
    prune.ln_structured(model.fc3, 'weight', amount=percentage, n=1, dim=0)

    trainModel(model)

    prune.remove(model.fc1, 'weight')
    prune.remove(model.fc2, 'weight')
    prune.remove(model.fc3, 'weight')

    evaluateModel(model)

    get_pruned_parameters_count(model, total_params_count)

def L2Structured(modelCNN, percentage, total_params_count):
    model = copy.deepcopy(modelCNN)
    print(f'\nPruning L2 Structured {percentage * 100}%')
    prune.ln_structured(model.fc1, 'weight', amount=percentage, n=2, dim=0)
    prune.ln_structured(model.fc2, 'weight', amount=percentage, n=2, dim=0)
    prune.ln_structured(model.fc3, 'weight', amount=percentage, n=2, dim=0)

    trainModel(model)

    prune.remove(model.fc1, 'weight')
    prune.remove(model.fc2, 'weight')
    prune.remove(model.fc3, 'weight')

    evaluateModel(model)

    get_pruned_parameters_count(model, total_params_count)
    
def globalL1Unstructured(modelCNN, percentage, total_params_count):
    model = copy.deepcopy(modelCNN)
    print(f'\nPruning L1 Unstructured {percentage * 100}%')
    parameters_to_prune = (
    (model.fc1, 'weight'),
    (model.fc2, 'weight'),
    (model.fc3, 'weight')
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=percentage,
    )

    trainModel(model)

    prune.remove(model.fc1, 'weight')
    prune.remove(model.fc2, 'weight')
    prune.remove(model.fc3, 'weight')

    evaluateModel(model)

    get_pruned_parameters_count(model, total_params_count)
    
def globalRandomUnstructured(modelCNN, percentage, total_params_count):
    model = copy.deepcopy(modelCNN)
    print(f'\nPruning Random Unstructured {percentage * 100}%')
    parameters_to_prune = (
    (model.fc1, 'weight'),
    (model.fc2, 'weight'),
    (model.fc3, 'weight')
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=percentage,
    )

    trainModel(model)

    prune.remove(model.fc1, 'weight')
    prune.remove(model.fc2, 'weight')
    prune.remove(model.fc3, 'weight')

    evaluateModel(model)

    get_pruned_parameters_count(model, total_params_count)


trainModel(modelFNN)
evaluateModel(modelFNN)
total_params_count = get_total_parameters_count(modelFNN)
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
