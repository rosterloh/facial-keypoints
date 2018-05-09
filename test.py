import cv2
import torch
from torch.autograd import Variable
from models import Net
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('images/obamas.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# load in a haar cascade classifier for detecting frontal faces
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

# run the detector
# the output here is an array of detections; the corners of each detection box
# if necessary, modify these parameters until you successfully identify every face in a given image
faces = face_cascade.detectMultiScale(image, 1.2, 2)

# make a copy of the original image to plot detections on
image_with_detections = image.copy()

# loop over the detected faces, mark the image where each face is found
for (x,y,w,h) in faces:
    # draw a rectangle around each detected face
    # you may also need to change the width of the rectangle drawn depending on image resolution
    cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3)

net = Net()
if (torch.cuda.is_available()):
    net.cuda()

# load the saved model parameters
net.load_state_dict(torch.load('saved_models/keypoints_model.pt'))

# print out net and prepare it for testing
net.eval()

image_copy = np.copy(image)

face = 1

# loop over the detected faces from haar cascade
for (x,y,w,h) in faces:
    
    # Select the region of interest that is the face in the image 
    roi = image_copy[y:y+h, x:x+w]
    
    # Convert the face region from RGB to grayscale
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    
    # Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
    roi = roi/255.0
    
    # Rescale the detected face to be the expected square size for CNN (224x224, suggested)
    roi = cv2.resize(roi, (96, 96))
    roi = roi.reshape(roi.shape[0], roi.shape[1], 1)
    
    # Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
    roi = roi.transpose((2, 0, 1))
    roi = torch.from_numpy(roi)
    # Since batch size of 1
    roi = roi.unsqueeze(0)
    
    # Make facial keypoint predictions using loaded, trained network 
    # wrap each face region in a Variable and perform a forward pass to get the predicted facial keypoints
    input_image = Variable(roi)
    if (torch.cuda.is_available()):
        input_image = input_image.type(torch.cuda.FloatTensor)
        input_image.cuda()
    else:
        input_image = input_image.type(torch.FloatTensor)
    output_pts = net(input_image)
    output_pts = output_pts.view(output_pts.size()[0], 68, -1)
    # un-transform the predicted key_pts data
    predicted_key_pts = output_pts[0].data
    if (torch.cuda.is_available()):
        predicted_key_pts = predicted_key_pts.cpu()
    predicted_key_pts = predicted_key_pts.numpy()
    # undo normalization of keypoints
    predicted_key_pts = predicted_key_pts*50.0+100
    
    # Display each detected face and the corresponding keypoints
    plt.subplot(1, 2, face)
    plt.imshow(np.squeeze(input_image), cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    
    face += 1

plt.show()
