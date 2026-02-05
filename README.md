# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="832" height="834" alt="image" src="https://github.com/user-attachments/assets/eb33c3b3-69b8-437a-82b4-e892642f907b" />

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

### Name: 
### Register Number:

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1=nn.Linear(input_size,16)
        self.fc2=nn.Linear(16,8)
        self.fc3=nn.Linear(8,4)





    def forward(self, x):
      x=F.relu(self.fc1(x))
      x=F.relu(self.fc2(x))
      x=self.fc3(x)
      return x

        

        

```
```python
# Initialize the Model, Loss Function, and Optimizer
model =PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(), lr=0.01)



```
```python
def train_model(model, train_loader, criterion, optimizer, epochs):
   for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
      optimizer.zero_grad()
      outputs = model(X_batch)
      loss = criterion(outputs, y_batch)
      loss.backward ()
      optimizer.step()





    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```



## Dataset Information


## OUTPUT

### Confusion Matrix and Classification report
<img width="910" height="780" alt="image" src="https://github.com/user-attachments/assets/fa04aab4-f622-4dcf-9791-00b4c9fedc6a" />


<img width="350" height="88" alt="image" src="https://github.com/user-attachments/assets/7904ce15-3d99-4814-b592-ea20488e992f" />


## RESULT
Thus, a neural network classification model for the given dataset as been created successfully.

