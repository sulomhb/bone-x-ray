import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

class LeNet(nn.Module):
    """Custom implementation of LeNet architecture for binary classification."""
    def __init__(self, numClasses=2):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)  
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  
        self.fc1 = nn.Linear(16 * 53 * 53, 120) 
        self.fc2 = nn.Linear(120, 84)  
        self.fc3 = nn.Linear(84, numClasses) 

    def forward(self, x):
        """Defines the forward pass for LeNet."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ModelTrainer:
    """Trainer class to handle model training, evaluation, and visualization."""
    def __init__(self, datasetPath, batchSize=32, numClasses=2, device=None):
        self.datasetPath = datasetPath
        self.batchSize = batchSize
        self.numClasses = numClasses
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {
            'lenet': LeNet(numClasses=numClasses),
            'alexnet': models.alexnet(pretrained=True),
            'vgg16': models.vgg16(pretrained=True),
            'googlenet': models.googlenet(pretrained=True),
            'resnet18': models.resnet18(pretrained=True)
        }
        self.results = {}
        self.hyperparamResults = {modelName: [] for modelName in self.models.keys()}

    def preprocessData(self, inputSize):
        """Preprocess the dataset by resizing, normalizing, and preparing DataLoaders."""
        transform = transforms.Compose([
            transforms.Resize(inputSize),
            transforms.RandomHorizontalFlip(),  # Data augmentation
            transforms.RandomRotation(10),     # Data augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.trainLoader = self._createLoader('train', transform)
        self.valLoader = self._createLoader('val', transform)
        self.testLoader = self._createLoader('test', transform)

    def _createLoader(self, subdir, transform):
        """Helper method to create a DataLoader for a given dataset subset."""
        path = os.path.join(self.datasetPath, subdir)
        dataset = datasets.ImageFolder(path, transform=transform)
        return DataLoader(dataset, batch_size=self.batchSize, shuffle=(subdir == 'train'))

    def modifyFinalLayer(self, model, modelName):
        """Modify the final layer of the model for binary classification."""
        if modelName == 'lenet':
            return model
        if hasattr(model, 'fc'):
            model.fc = nn.Linear(model.fc.in_features, self.numClasses)
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, self.numClasses)
        return model

    def trainModel(self, modelName, epochs, learningRate):
        """Train the specified model and track accuracy and loss."""
        model = self.models[modelName]
        model = self.modifyFinalLayer(model, modelName).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learningRate)

        trainAcc, valAcc = [], []

        for epoch in range(epochs):
            model.train()
            correct, total = 0, 0
            for images, labels in self.trainLoader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            trainAcc.append(100 * correct / total)

            # Validation
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in self.valLoader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            valAcc.append(100 * correct / total)

        self.hyperparamResults[modelName].append({
            "lr": learningRate,
            "batch_size": self.batchSize,
            "epochs": epochs,
            "accuracy": valAcc[-1]
        })

    def evaluateModel(self, modelName):
        """Evaluate the trained model on the test dataset."""
        model = self.results[modelName]['model']
        model.eval()

        allLabels, allPreds = [], []
        with torch.no_grad():
            for images, labels in self.testLoader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                allPreds.extend(preds.cpu().numpy())
                allLabels.extend(labels.cpu().numpy())

        report = classification_report(allLabels, allPreds, output_dict=True)
        cm = confusion_matrix(allLabels, allPreds)
        return cm, report

    def plotHyperparameterResults(self):
        """Plot hyperparameter tuning results."""
        fig, ax = plt.subplots(figsize=(15, 10))
        for model, configs in self.hyperparamResults.items():
            x = [f"LR: {conf['lr']}, BS: {conf['batch_size']}, E: {conf['epochs']}" for conf in configs]
            y = [conf["accuracy"] for conf in configs]
            ax.plot(x, y, marker="o", label=model)
        ax.set_title("Hyperparameter Tuning Results Across All Configurations")
        ax.set_xlabel("Configurations (LR, Batch Size, Epochs)")
        ax.set_ylabel("Accuracy (%)")
        ax.legend()
        plt.xticks(rotation=90, fontsize=8)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    datasetPath = "./data"
    modelNames = ['lenet', 'alexnet', 'vgg16', 'googlenet', 'resnet18']
    inputSize = (224, 224)
    learningRates = [0.001, 0.0001, 0.01]
    batchSizes = [16, 32, 64]
    epochOptions = [20, 30, 40]

    for modelName in modelNames:
        for lr in learningRates:
            for batchSize in batchSizes:
                for epochs in epochOptions:
                    trainer = ModelTrainer(datasetPath, batchSize=batchSize)
                    trainer.preprocessData(inputSize)
                    trainer.trainModel(modelName, epochs=epochs, learningRate=lr)

    trainer.plotHyperparameterResults()