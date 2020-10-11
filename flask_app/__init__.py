import flask
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import base64
import io
from PIL import Image
import numpy as np
import logging

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer (sees 32x32x3 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # convolutional layer (sees 16x16x16 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # convolutional layer (sees 8x8x32 tensor)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 * 4 * 4 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(500, 10)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 64 * 4 * 4)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x

def get_model():
    gpu_available = torch.cuda.is_available()

    #Loading model
    # check if CUDA is available
    gpu_available = torch.cuda.is_available()

    # create a complete CNN
    model = Net()
    model.load_state_dict(torch.load('./model_cifar.pt'))
    model.eval()

    # move tensors to GPU if CUDA is available
    if gpu_available:
        model.cuda()    

    return model

def get_image(base64string):
    # reading image
    image = io.BytesIO(base64.b64decode(base64string))
    image = np.array(Image.open(image))

    image = ((image/255) - 0.5)/0.5
    image = np.transpose(image,(2,1,0))
    image = np.expand_dims(image,axis=0)
    image = torch.from_numpy(image)
    image = image.to(torch.float32)

    return image

# Create Flask application
app = flask.Flask(__name__)
app.wsgi_app = Net(app.wsgi_app)

logHandler = logging.FileHandler('./flask_app/logs/app.log')
logHandler.setLevel(logging.INFO)
app.logger.addHandler(logHandler)
app.logger.setLevel(logging.INFO)

