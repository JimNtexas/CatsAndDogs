#https://www.youtube.com/watch?v=9aYuQmMJvjA&t=364s
#https://www.youtube.com/watch?v=1gQR24B3ISE

import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


if __name__ == '__main__':
    print('welcome to ' + 'Cats v. Dogs')
REBUILD_DATA = False

class Net (nn.Module):
    def __init__(self):
        super().__init__()
        #   1 input channel, 32 output channels,  kernel size 5 for five 2d convolution layers
        self.conv1 = nn.Conv2d(1,32,5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        #self.fc1 = nn.Linear(???, 512)
        #self.fc2 = nn.Linear(512,2)
    def convs(self,x):
        ''' Rectified Linear Activation Function.
            The ReLu function enables us to detect and present the state of the model results'''
        x = F.max_pool2d((F.relu(self.conv1(x)),(2,2)))
        if self._to_linear is None:
            self._to_linear =

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


if REBUILD_DATA:
    critters = DogsVSCats()
    critters.make_training()

training_data = np.load("training_data.npy",allow_pickle=True)
print("Training data size: " + str(len(training_data)))
plt.imshow(training_data[0][0], cmap="gray")
plt.show()




