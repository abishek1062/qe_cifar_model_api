import flask
from flask import Flask, request, jsonify
import os
import torch
from torch.nn import Softmax
import numpy as np
from flask_app import app

# Create a URL route in our application for "/"
methods = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'COPY', 'HEAD', 'OPTIONS', 'LINK', 'UNLINK', 'PURGE', 'LOCK', 'UNLOCK', 'PROPFIND', 'VIEW']

@app.route('/',methods=methods)
def recognizeImage():
    gpu_available = torch.cuda.is_available()

    base64string = request.get_json(force=True)['base64']
    image = get_image(base64string)

    if image.shape[1:] != (3,32,32):
        return jsonify( error = "only RGB image 32x32 accepted", message = "failure!" )

    model = get_model()

    output_tensor = model(image.cuda())
    output_tensor = Softmax(dim=1)(output_tensor)
    prob_pred_tensor, pred_tensor = torch.max(output_tensor, 1)

    output = np.squeeze(output_tensor.detach().numpy()) if not gpu_available else np.squeeze(output_tensor.cpu().detach().numpy())

    prob_pred = np.squeeze(prob_pred_tensor.detach().numpy()) if not gpu_available else np.squeeze(prob_pred_tensor.cpu().detach().numpy())    
    pred = np.squeeze(pred_tensor.detach().numpy()) if not gpu_available else np.squeeze(pred_tensor.cpu().detach().numpy())

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

    pred_dict = {'predicted_class' : str(classes[pred]),
                 'prob_predicted_class' : str(prob_pred)}

    for i,prob in enumerate(output):
        pred_dict[classes[i]] = str(prob)


    return jsonify( prediction = pred_dict, message = "success!" )
