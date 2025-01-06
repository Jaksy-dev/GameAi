# MK2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Define dataset class
class ClassificationDataset(Dataset):
    def __init__(self, dataframe):
        self.features = dataframe.iloc[:, :-1].values.astype(np.float32)
        self.labels = dataframe.iloc[:, -1].values.astype(np.int64)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Define neural network
class ClassificationModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassificationModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

# Training function
def train_model(n,model, train_loader, val_loader, criterion, optimizer, scheduler, patience, num_epochs):
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item() * features.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        # Learning rate scheduler step
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'bid_2_{n}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

for n in range(2,14):
    # Prepare dataset
    data = pd.read_csv(f"bid_data_{n}.csv",)

    train_data, test_data = train_test_split(data, test_size=0.2)
    train_dataset = ClassificationDataset(pd.DataFrame(train_data))
    test_dataset = ClassificationDataset(pd.DataFrame(test_data))

    # Split train into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClassificationModel(input_size=52, num_classes=14).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print(f"Training model for hand size {n}...")
    # Train
    train_model(
        n,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        patience=5,
        num_epochs=50
    )

    # Load best model and evaluate
    model.load_state_dict(torch.load(f'bid_2_{n}.pth', weights_only=False))
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_loss /= len(test_loader.dataset)
    accuracy = correct / total
    print(f"For hand size {n}: Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
