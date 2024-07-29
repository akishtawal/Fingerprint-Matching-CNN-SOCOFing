#FETCHING THE IMAGE DATA (PREPROCESSING)

#Importing required libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from skimage.transform import resize

#Path to the image data
#Make sure that you change this path to the path to the data folder on your system
fingerprint_path = 'path\to\your\data_folder'

#Function to extract the label attribute for the image from its file path

def extract_attr(img_path):

  file_name, _ = os.path.splitext(os.path.basename(img_path))

  #Getting the Subject ID number
  subject_id = file_name.split('__')[0]

  #Getting the rest of the info
  rest_info = file_name.split('__')[1]

  #Getting the gender of the subject
  gender = rest_info.split('_')[0]
  
  #Encoding the gender as a binary value (as sex)
  if gender=='M':
    sex=0 
  else:
    sex=1
  
  #Getting the side of the hand 
  left_right = rest_info.split('_')[1]
  
  #Encoding the side as a binary value
  if left_right=='Left':
    side=0
  else:
    side=1
  
  #Getting the type of the finger
  finger_type = rest_info.split('_')[2]

  #Encoding the finger type as numeric values
  if finger_type == 'thumb':
      finger = 0
  elif finger_type == 'index':
      finger = 1
  elif finger_type == 'middle':
      finger = 2
  elif finger_type == 'ring':
      finger = 3
  elif finger_type == 'little':
      finger = 4
  
  #Returning a numpy array of all the information collected (Image Attributes)
  img_attr = np.array([subject_id, sex, side, finger], dtype=np.uint16)

  return img_attr


def process_images(img_paths, extract_func):
    imgs = []
    attr = []

    for img_path in img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (96, 96))  # Resize the image to (96, 96)
        img = resize(img, (90, 90), anti_aliasing=True)  # Resize the image to (90, 90)
        imgs.append(img)
        attr.append(extract_func(img_path))

    imgs = np.array(imgs)
    attr = np.array(attr)

    return imgs, attr

#Ensuring that a 'dataset' directory exists. If not, we create it
if not os.path.exists('dataset'):
    os.makedirs('dataset')

#Loading and Processing different image types in our dataset
#Real images
img_list = sorted(glob.glob(f'{fingerprint_path}/Real/*.BMP'))
print('Total Real Images: ', len(img_list))

imgs, attr = process_images(img_list, extract_attr)

np.save('dataset/x_real.npy', imgs)
np.save('dataset/y_real.npy', attr)

#Altered-Easy images
img_list = sorted(glob.glob(f'{fingerprint_path}/Altered/Altered-Easy/*.BMP'))
print('Total Altered-Easy Images: ', len(img_list))

imgs, attr = process_images(img_list, extract_attr)

np.save('dataset/x_easy.npy', imgs)
np.save('dataset/y_easy.npy', attr)

#Altered-Medium images
img_list = sorted(glob.glob(f'{fingerprint_path}/Altered/Altered-Medium/*.BMP'))
print('Total Altered-Medium Images: ', len(img_list))

imgs, attr = process_images(img_list, extract_attr)

np.save('dataset/x_medium.npy', imgs)
np.save('dataset/y_medium.npy', attr)

#Altered-Hard images
img_list = sorted(glob.glob(f'{fingerprint_path}/Altered/Altered-Hard/*.BMP'))
print('Total Altered-Hard Images: ', len(img_list))

imgs, attr = process_images(img_list, extract_attr)

np.save('dataset/x_hard.npy', imgs)
np.save('dataset/y_hard.npy', attr)

