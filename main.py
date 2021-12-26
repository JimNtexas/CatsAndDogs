#https://www.youtube.com/watch?v=9aYuQmMJvjA&t=364s
#https://www.youtube.com/watch?v=1gQR24B3ISE

import os     #operating system utilities
import cv2    #image processing
import numpy as np   #general-purpose array-processing package
from tqdm import tqdm  #makes progress bars
import matplotlib.pyplot as plt   # Matplotlib is a comprehensive library for creating visualizations
import torch  #he torch package contains data structures and operators for tensor computation
import torch.nn as nn  #Datasets, and DataLoaders to help you create and train neural networks
import torch.nn.functional as F  #lamda versions of nn class members
import torch.optim as optim  #package implementing various optimization algorithms

#https://pytorch.org/docs/stable/generated/torch.flatten.html may be useful here

class Net (nn.Module):
    def __init__(self):
        super().__init__()
        #   1 input channel, 32 output channels,  kernel size 5 for five 2d convolution layers
        self.conv1 = nn.Conv2d(1,32,5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)  # flattening.
        self.fc2 = nn.Linear(512, 2)  # 512 in, 2 out bc we're doing 2 classes (dog vs cat).

    def convs(self,x):
        ''' Rectified Linear Activation Function.
            The ReLu function enables us to detect and present the state of the model results.

            Max Pooling reduces the size of our convolutional outputs to reduce computation loading.
            see https://deeplizard.com/learn/video/ZjM_XQa5s6s'''

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)    #see https://sparrow.dev/pytorch-softmax/


class DogsVSCats():
    IMG_SIZE: int = 50
    CATS="PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}

    training_data = []
    catcount = 0
    dogcount = 0

    def make_training(self):
        for label in self.LABELS:
            print("Label: " + label)
            dir = os.listdir(label)
            for f in tqdm(dir):
                if "jpg" in f:
                    try:
                        path = os.path.join(label,f)
                        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE), interpolation = cv2.INTER_AREA)
                        # do something like print(np.eye(2)[1]), just makes one_hot
                        self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                        #verify images are balanced
                        if label == self.CATS:
                            self.catcount += 1
                        elif label == self.DOGS:
                            self.dogcount += 1


                    except Exception as e:
                        print (path + " - " + str(e) + "cat count: " + str(self.catcount))
                        pass

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print('Cats:', self.catcount)
        print('Dogs:', self.dogcount)


if __name__ == '__main__':
    print('welcome to ' + 'Cats v. Dogs')
REBUILD_DATA = False

if REBUILD_DATA:
    critters = DogsVSCats()
    critters.make_training()

training_data = np.load("training_data.npy",allow_pickle=True)
print("Training data size: " + str(len(training_data)))
net = Net()
#plt.imshow(training_data[0][0], cmap="gray")
#plt.show()

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

print(net)

'''UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. 
Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. 
  X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)'''
X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])
VAL_PCT = 0.1  # lets reserve 10% of our data for validation
val_size = int(len(X)*VAL_PCT)
print("val size: " + str(val_size))

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

print("len train_X: " + str(len(train_X)) + " len(test_X: " +  str(len(test_X)))

BATCH_SIZE = 100
EPOCHS = 1

for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)): # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
        #print(f"{i}:{i+BATCH_SIZE}")
        batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
        batch_y = train_y[i:i+BATCH_SIZE]

        net.zero_grad()

        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()    # Does the update

    print(f"Epoch: {epoch}. Loss: {loss}")

correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1, 1, 50, 50))[0]  # returns a list,
        predicted_class = torch.argmax(net_out)

        if predicted_class == real_class:
            correct += 1
        total += 1
print("Accuracy: ", round(correct/total, 3))