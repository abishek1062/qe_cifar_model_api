- The purpose of this project is to deploy and test a CNN model trained on the cifar-10 dataset. This was created for the purpose of testing for the QE team.

- To run the app use the command:
      gunicorn run:flask_app/classify_image -b 127.0.0.1:8000 -w 4 --access-logfile ./flask_app/logs/logs/access.log --error-logfile ./flask_app/logs/error.log

- The port used is 8000 and the log files are present in the directory ./logs

- A wsgi production server is used

- The trained model i.e. state dictionary is at '.model_cifar.pt' 

- The python notebook (cifar10_cnn_solution-ddl.ipynb) contains the code for training the model

- In order to send a request to this REST API (flask), a base64 encoded image needs to be sent. The format ofr the payload is as follows:-
		{
			"base64" : "base64 encoding of the32x32 RGB image"
		}

		command to encode an image to base64 can be found at: https://cloud.google.com/vision/docs/base64

- The payload can be sent through either of these methods : 
	methods = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'COPY', 'HEAD', 'OPTIONS', 'LINK', 'UNLINK', 'PURGE', 'LOCK', 'UNLOCK', 'PROPFIND', 'VIEW']

- Only a 32x32 RGB image can be processed.

- The image sent in the payload will be classified as one of the following :-

      ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

      example of a successful request:
      	{
		    "message": "success!",
		    "prediction": {
		        "airplane": "0.00032329743",
		        "automobile": "0.012829902",
		        "bird": "0.00096437446",
		        "cat": "9.434413e-05",
		        "deer": "4.26209e-07",
		        "dog": "5.861184e-07",
		        "frog": "3.7358205e-07",
		        "horse": "3.733835e-06",
		        "predicted_class": "truck",
		        "prob_predicted_class": "0.9853125",
		        "ship": "0.00047049456",
		        "truck": "0.9853125"
		    }
		}

      example of a failed request:
		{
		    "error": "only RGB image 32x32 accepted",
		    "message": "failure!"
		}

 - Please email: abishek.subramanian@dominodatalab.com for any questions or concerns 

