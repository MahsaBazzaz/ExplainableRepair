
from generic_classifier import Model
import torch
import torch.nn as nn
import torch.optim as optim

from data_utility import get_dataset
import argparse
import numpy as np
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from constants import CAVE_CHANNELS, CAVE_COLS, CAVE_ROWS, MARIO_CHANNELS, MARIO_COLS, MARIO_ROWS, SUPERCAT_CHANNELS, SUPERCAT_COLS, SUPERCAT_ROWS

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    # Train the model for the specified number of epochs
    for epoch in range(num_epochs):
        print("epoch: ", epoch)
        # Set the model to train mode
        model.train()

        # Initialize the running loss and accuracy
        running_loss = 0.0
        running_corrects = 0

        # Iterate over the batches of the train loader
        for inputs, labels in train_loader:
            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the optimizer gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            # Update the running loss and accuracy
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # Calculate the train loss and accuracy
        train_loss = running_loss / len(train_dataset)
        train_acc = running_corrects.double() / len(train_dataset)

        # Set the model to evaluation mode
        model.eval()

        # Initialize the running loss and accuracy
        running_loss = 0.0
        running_corrects = 0

        # Iterate over the batches of the validation loader
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Move the inputs and labels to the device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Update the running loss and accuracy
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        # Calculate the validation loss and accuracy
        val_loss = running_loss / len(val_dataset)
        val_acc = running_corrects.double() / len(val_dataset)

        # Print the epoch results
        print('Epoch [{}/{}], train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}'
              .format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))
    
    return val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Resnet classifier')

    parser.add_argument('--game', required=True, type=str, help='cave/icarus/mario')

    args = parser.parse_args()
    game = args.game
    if game == "cave":
        cols = CAVE_COLS
        rows = CAVE_ROWS
        channels = CAVE_CHANNELS
    elif game == 'mario':
        cols = MARIO_COLS
        rows = MARIO_ROWS
        channels = MARIO_CHANNELS
    elif game == 'supercat':
        cols = SUPERCAT_COLS
        rows = SUPERCAT_ROWS
        channels = SUPERCAT_CHANNELS

    levels, labels = get_dataset(game)
    x_train, x_test, y_train, y_test = train_test_split(levels, labels, train_size=0.80)
    
    # Define the optimizer and the learning rate
    lr = 0.0001
    model = Model(cols, rows, channels)
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    num_epochs = 100
    batch_size = 32

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        # Iterate over batches of data
        for i in range(0, len(x_train), batch_size):
            inputs = torch.tensor(x_train[i:i+batch_size], dtype=torch.float32)
            labels = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))  # BCELoss expects a column vector

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item() * inputs.size(0)

        # Print epoch statistics
        epoch_loss = running_loss / len(x_train)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print('Finished Training')
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    # Disable gradient computation during evaluation
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            inputs = torch.tensor(x_test[i:i+batch_size], dtype=torch.float32)
            labels = torch.tensor(y_test[i:i+batch_size], dtype=torch.float32)

            # Forward pass
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()  # Convert to binary predictions

            # Count correct predictions
            total += labels.size(0)
            correct += (predicted == labels.view(-1, 1)).sum().item()

    # Compute accuracy
    accuracy = correct / total
    print(f"Accuracy on test set: {accuracy:.2f}")

    print('Test Loss:', loss)
    print('Test Accuracy:', accuracy)
    model_name = "./models/new/" + game + "_" + 'generic_classifier_py' + ".pth"
    torch.save(model.state_dict(), model_name)
    with open('models/metrics_' + game + '.json', "w") as json_file:
        json.dump({
                    "accuracy": accuracy,
                    "num_epochs" : num_epochs,
                    "learning_rate" : lr,
                    "batch_size" : batch_size,
                    "criterion" : "BCE LOSS",
                    "optimizer" : "RMSprop",
                    "timestamp": datetime.now().strftime("%Y%m%d%H%M%S")
                    }, json_file)