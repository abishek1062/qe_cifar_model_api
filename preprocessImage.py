import torch
import base64
import io
from PIL import Image
import numpy as np

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