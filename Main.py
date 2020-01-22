import dataset
import Model
import torch
import torch.nn.functional as F
import torch.optim as optim
import random

# function to count number of parameters
def get_n_params(model):
    np=0
    for p in list(model.parameters()):
        np += p.nelement()
    return np

# =============================================================================
#  Load the data:   
# =============================================================================
path = 'dogs-vs-cats/train/'# replace this by the path containing your image data

mydata = list(dataset.load(img_dir = path).data())

# =============================================================================
# Create  test and training sets
# =============================================================================

random.shuffle(mydata)
n = int(len(mydata)*.3)
test_set = mydata[:n]
train_set = mydata[n:]

# =============================================================================
# Use torch.utils.data.DataLoader to match the requirement of the model
# =============================================================================
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_set,
     batch_size=1000, shuffle=True)


# =============================================================================
# running on gpu
# =============================================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Training and test functions
# =============================================================================

accuracy_list = []

def train(epoch, model, perm=torch.arange(0, 28*28).long()):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        # permute pixels
        data = data.view(-1, 28*28)
        data = data[:, perm]
        data = data.view(-1, 1, 28, 28)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target.long())
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            

def test(model, perm=torch.arange(0, 28*28).long()):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        # permute pixels
        data = data.view(-1, 28*28)
        data = data[:, perm]
        data = data.view(-1, 1, 28, 28)
        output = model(data)
        test_loss += F.nll_loss(output, target.long(), reduction='sum').item() # sum up batch loss                                                               
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability                                                                 
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    accuracy_list.append(accuracy)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    
    
    
# =============================================================================
#  Training settings 
# =============================================================================

input_size  = 28*28   # images are converted to 28x28 pixels in the loading
output_size = 2      # 2 classes : cat and dog
n_features = 4 # number of feature maps

model_cnn = Model.CNN(input_size, n_features, output_size)
optimizer = optim.SGD(model_cnn.parameters(), lr=0.01, momentum=0.5)
print('Number of parameters: {}'.format(get_n_params(model_cnn)))

for epoch in range(0, 1):
    train(epoch, model_cnn)
    test(model_cnn)