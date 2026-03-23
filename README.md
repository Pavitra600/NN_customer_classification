# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="792" height="957" alt="image" src="https://github.com/user-attachments/assets/abadf678-ac2e-4df3-98d8-b6d60e6e17b8" />

## DESIGN STEPS
3 STEP 1:
Import necessary libraries and load the dataset.

# STEP 2:
Encode categorical variables and normalize numerical features.

# STEP 3:
Split the dataset into training and testing subsets.

# STEP 4:
Design a multi-layer neural network with appropriate activation functions.

# STEP 5:
Train the model using an optimizer and loss function.

# STEP 6:
Evaluate the model and generate a confusion matrix.

# STEP 7:
Use the trained model to classify new data samples.

# STEP 8:
Display the confusion matrix, classification report, and predictions.


## PROGRAM

### Name: Pavitra J
### Register Number: 212224110043

```python

# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()

        # Fully Connected Layers
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 2)   # 2 output classes (change if needed)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x

        

        

```
```python
# Initialize the Model, Loss Function, and Optimizer
model =PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(), lr=0.01)



```
```python

# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs):

    for epoch in range(epochs):
        running_loss = 0.0

        for inputs, labels in train_loader:

            # Move to device (if using GPU)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item()

        # Print every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')
```



## Dataset Information
<img width="1191" height="242" alt="image" src="https://github.com/user-attachments/assets/a3be0daf-5816-49b9-8008-a33a176e6bf0" />


## OUTPUT

### Confusion Matrix and Classification report
<img width="843" height="631" alt="image" src="https://github.com/user-attachments/assets/1f4b2308-d208-44b7-a68e-d480d813a2d2" />

<img width="658" height="702" alt="551570529-787b0f51-396c-4088-b7b7-d62116b8f6b9" src="https://github.com/user-attachments/assets/7a2d3612-9ac1-4939-bef2-ca8cf0bfa271" />


## New Sample Data Prediction
<img width="545" height="106" alt="551571122-0a3821a8-92cf-4897-b4ad-6316f9fe561c" src="https://github.com/user-attachments/assets/aef19599-87c7-4d9b-ae37-589ce3215790" />


## RESULT
Thus, a neural network classification model for the given dataset as been created successfully.

