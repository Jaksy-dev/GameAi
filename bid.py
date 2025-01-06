import pandas as pd
import time
# ranks = "23456789TJQKA"
# suits = "shdc"  # Spades, Hearts, Diamonds, Clubs



# # Example: take a look at how many times a hand has been played, and see the average win amount for that hand
# hand=[1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,0] # only 52 long
# hand_percent = 0.23076923076923078
# # Get the rows where the first 52 values match the list above.

# # Compare only the first 52 columns with the hand
# print((df.iloc[:, :52] == hand).all(axis=1).sum())  # Count matching rows

# matching_rows = df.iloc[:, -1][(df.iloc[:, :52] == hand).all(axis=1)]

# # Output the matching rows
# print("Played ", matching_rows.count(), "times with an average score of ", matching_rows.mean())

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR


# Define the dataset class
class BinaryFeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=128, hidden_size3=64, hidden_size4=64, output_size=1):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size3, hidden_size4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size4, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

# Train the model
def train_model(train_loader, model, criterion, optimizer, scheduler, epochs):
    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X).squeeze()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

# Evaluate the model
def evaluate_model(test_loader, model):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            outputs = model(X).squeeze()
            y_true.extend(y.tolist())
            y_pred.extend(outputs.tolist())
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"Test MSE: {mse:.4f}, R2 Score: {r2:.4f}")
    return y_true, y_pred

# Main workflow
def main(data, epochs=10, batch_size=32, learning_rate=0.001, test_size=0.2):
    # Split the data
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    # Create datasets and dataloaders
    train_dataset = BinaryFeatureDataset(X_train, y_train)
    test_dataset = BinaryFeatureDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, criterion, and optimizer
    model = SimpleNN(input_size=X.shape[1])
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce LR every 10 epochs
    
    # Train the model
    print("Training the model...")
    train_model(train_loader, model, criterion, optimizer, scheduler, epochs)
    
    # Test the model
    print("Evaluating the model...")
    y_true, y_pred = evaluate_model(test_loader, model)

    return model, y_true, y_pred


for n in range(2,14):
    df = pd.read_csv(f"bid_data_{n}.csv")
    trained_model, y_true, y_pred = main(df)
    torch.save(trained_model, f"bid_{n}.pt")


# Improvements:
# Learning rate adjustment
# Early stopping





# # Convert to PyTorch tensor
# input_tensor = torch.tensor(hand, dtype=torch.float32)

# # Put model in evaluation mode
# trained_model.eval()

# # Get prediction
# with torch.no_grad():
#     prediction = trained_model(input_tensor).item()

# print(f"Predicted Output for already existing hand: {prediction} and it's actual value: {hand_percent}")

# # Change the 2 of spades into a 3 of spades
# example_input = [0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,0]

# input_tensor = torch.tensor(example_input, dtype=torch.float32)

# # Get prediction
# with torch.no_grad():
#     prediction = trained_model(input_tensor).item()
    
# print(f"Predicted Output for non existing hand: {prediction}")
