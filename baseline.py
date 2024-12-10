"""
importance of testing

allow testing multiple times -
sentinels against regressions 
good for documentation for recording what you have checked 
        - expertise / decision making - domain specific requirements
        object building - infrastructure, data preprocessing. -- evaluating - computer), 
        - insight generator think about math and rep building.
        - black box analysis
(other pdb methods are forgotten and are not repeatable )


NOTE: use pytorch dtype to explicitly specify your numpy array dtypes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pdb

from sklearn.model_selection import train_test_split

device = "cuda:0"


class BrainDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = torch.Tensor(self.images[idx])
        label = torch.tensor(self.labels[idx], dtype = torch.long) # How does tensor relate to Tensor? why does dtype long not affect the computatino? labels NEED TO BE INTS for CE loss interface
        if self.transform:
            # Transform is specific to 

            image = self.transform(image)
        return image, label

class BrainCNN(nn.Module):
    def __init__(self):
        super(BrainCNN, self).__init__()
        # Basic CNN architecture
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 1 channel input (grayscale)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Calculate the size after convolutions
        self.fc_input_size = 64 * 12 * 9 
        
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 3)  # 3 classes: healthy, MCI, disease
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):

        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        # Flatten
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

# Training function
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Evaluation function
def evaluate_model(model, test_loader, device):

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return 100 * correct / total



def load_dataset(type_set = 'train'):
    X =  np.load(f'./data/brain_{type_set}_image_final.npy')[:, 1:2, :, :] # I added 1:2 becaause I specificallly want to mainain the channel dimension instead of getting rid of i
    # conv format (1, channel, height, width)
    y = np.load(f'./data/brain_{type_set}_label.npy')
    return X, y


# Example usage
def main():
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 50 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = BrainCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_train, y_train = load_dataset(type_set='train')
    X_test, y_test = load_dataset(type_set='test')

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size = 0.9, random_state=42)

    train_dataset = BrainDataset(X_train, y_train)
    test_dataset = BrainDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size) # in these systems it is good practice to document and be confidentn
    test_final_dataset = BrainDataset(X_test, y_test)
    test_final_loader = DataLoader(test_final_dataset, batch_size=batch_size)

    # TODO: With all the factors, can I parse out how I am getting really good performance on one modification vs another?
    # Here the skill is in finding how the hyperparameters optimize

    for epoch in range(num_epochs):
        train_model(model, train_loader, optimizer, criterion, device = "cuda:0")
    final_result = evaluate_model(model, test_loader=test_loader, device ="cuda:0")


    print(f"Final Accuracy Validation Result: {final_result}")
    final_result2 = evaluate_model(model, test_loader= test_final_loader, device ="cuda:0")    

    print(f"Final Accuracy Validation Result: {final_result2}")
if __name__ == '__main__':
    main()