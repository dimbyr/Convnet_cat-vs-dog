import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

import dlc_practical_prologue as prologue


# Loading the MNIST dataset with dlc_practical_prolongue
train_input, train_target, test_input, test_target = \
    prologue.load_data(one_hot_labels = True, normalize = True, flatten = False)


# Convlolutional neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x
    
    
# Training the model
def train_model(model, train_input, train_target, mini_batch_size =  100, num_iter = 25):
    train_input, train_target = Variable(train_input), Variable(train_target)
    criterion = nn.MSELoss()
    eta = 1e-1

    for e in range(0, num_iter):
        sum_loss = 0
        # We do this with mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.item()
            model.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.data.sub_(eta * p.grad.data)
        print(e, sum_loss)

        

        
model = Net()
train_model(model,train_input, train_target, mini_batch_size =  100)


# Computing the errors
def compute_nb_error(model, input_, target, mini_batch_size = 100):
    losses = []
    for b in range(0, input_.size(0), mini_batch_size):
        output = model(input_.narrow(0, b, mini_batch_size))
        t = target.narrow(0,b,mini_batch_size)
        pred = torch.tensor([int(torch.argmax(output[i])) for i in range(len(output))])
        true_value = torch.tensor([int(torch.argmax(t[i])) for i in range(len(t))])
        loss = ((pred!=true_value)*1.).mean()*100
        losses.append(loss)
    maxloss = float(max(losses))
    full_loss = float(torch.tensor(losses).mean())
    return f"Maximal error on batches: {maxloss} % ,\t Average error: {full_loss} %"

print(f" Loss:  {compute_nb_error(model, test_input, test_target)} ")