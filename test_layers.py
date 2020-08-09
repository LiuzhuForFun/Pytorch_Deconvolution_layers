import torch
from image_restoration_v3.layer import ISubLayer
from skimage import transform
import torchvision.transforms as transforms
import cv2
import numpy as np
import torchvision

test_k = np.loadtxt('ours.dlm').astype(np.float)
test_k = np.clip(test_k, 0, 1)
test_k = test_k / np.sum(test_k)
test_k = test_k[np.newaxis,np.newaxis,...]
image = cv2.imread('7.png')
image = np.array(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
transform = transforms.Compose([
                                  transforms.ToTensor()])
image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)
print(np.shape(image_tensor))
layer = ISubLayer()
out = layer(image_tensor,image_tensor,torch.Tensor(test_k),torch.Tensor([0.05]))
torchvision.utils.save_image(out[:1], 'test_deblur.png')

