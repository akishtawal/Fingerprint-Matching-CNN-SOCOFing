#Considering the preprocessing is done and the datasets are created 
#(Fingerprint_CNN_Preprocess.py)
#TRAINING THE DATA

#Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers, metrics
from keras.models import Model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa #For image augmentation
import random 
from skimage.transform import resize
from tqdm import tqdm


#Saving the file path
data_file_path = 'F:\TeamEpic_Work\LiveProjects_March-24\Computer Vision\Fingerprint_Matching_CNN\dataset'

#Loading the datasets creating during the pre-processing
#Here x -> image files, and y -> image attributes
x_real = np.load(fr'{data_file_path}\x_real.npy').reshape(-1, 90, 90, 1)
y_real = np.load(fr'{data_file_path}\y_real.npy')
x_easy = np.load(fr'{data_file_path}\x_easy.npy')
y_easy = np.load(fr'{data_file_path}\y_easy.npy')
x_medium = np.load(fr'{data_file_path}\x_medium.npy')
y_medium = np.load(fr'{data_file_path}\y_medium.npy')
x_hard = np.load(fr'{data_file_path}\x_hard.npy')
y_hard = np.load(fr'{data_file_path}\y_hard.npy')

#SAMPLING THE DATA
# Sample the datasets (adjust the fraction as needed)
# sample_fraction = 0.2  # Use 20% of the data
sample_fraction = 0.5  # Use 50% of the data
x_real = x_real[:int(len(x_real) * sample_fraction)]
y_real = y_real[:int(len(y_real) * sample_fraction)]
x_easy = x_easy[:int(len(x_easy) * sample_fraction)]
y_easy = y_easy[:int(len(y_easy) * sample_fraction)]
x_medium = x_medium[:int(len(x_medium) * sample_fraction)]
y_medium = y_medium[:int(len(y_medium) * sample_fraction)]
x_hard = x_hard[:int(len(x_hard) * sample_fraction)]
y_hard = y_hard[:int(len(y_hard) * sample_fraction)]

#Printing the shapes of the real images dataset
print('x_real shape: ', x_real.shape)
print('y_real shape: ', y_real.shape)

#Displaying a random image from the dataset,
#Along with its associated augmentations (Easy, Medium, and Hard)
rand_idx_1 = random.randint(0, len(x_real))
plt.figure(figsize=(15, 10))
plt.subplot(1, 4, 1)
plt.title(str(y_real[rand_idx_1]))
plt.imshow(x_real[rand_idx_1].squeeze(), cmap='gray')
plt.subplot(1, 4, 2)
plt.title(str(y_easy[rand_idx_1]))
plt.imshow(x_easy[rand_idx_1].squeeze(), cmap='gray')
plt.subplot(1, 4, 3)
plt.title(str(y_medium[rand_idx_1]))
plt.imshow(x_medium[rand_idx_1].squeeze(), cmap='gray')
plt.subplot(1, 4, 4)
plt.title(str(y_hard[rand_idx_1]))
plt.imshow(x_hard[rand_idx_1].squeeze(), cmap='gray')


#TRAIN-TEST-SPLIT

#Creating a combined dataset of all image data
x_data = np.concatenate([x_easy, x_medium, x_hard], axis=0)

#Creating a combined dataset of all (image) attribute data
attr_data = np.concatenate([y_easy, y_medium, y_hard], axis=0)

#Printing the shapes of the combined datasets
print('Combined Image Data Shape: ', x_data.shape)
print('Combined Attribute Data Shape: ', attr_data.shape)

#Performing a train-test split
x_train, x_val, attr_train, attr_val = train_test_split(x_data, attr_data, test_size=0.1)

#Displaying the shapes of the produced datasets
print('Train Image Data Shape: ', x_train.shape)
print('Train Attribute Data Shape: ', attr_train.shape)
print('Validation Image Data Shape: ', x_val.shape)
print('Validation Attribute Data Shape: ', attr_val.shape)


#IMAGE AUGMENTATION (Using ImgAug)

#Defining an image augmentation sequence
aug_seq = iaa.Sequential([

          #Applying a gaussian blur
          iaa.GaussianBlur(sigma=(0, 0.5)),  
          
          #Applying transformations
          iaa.Affine(
              scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, #Scaling the image between 90% to 110% of its original size
              translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, #Translating the image by -10% to +10% along both x and y axes
              rotate=(-30, 30), #Rotating the image between -30 to +30 degrees
              order=[0, 1], #Interpolation order, using either nearest neighbor or bilinear interpolation.
              cval=255 #Constant value used for filling in new pixels during transformations
            )
          ])

#Trying out a sample image to apply the augmentations
rand_idx_2 = random.randint(0, len(x_data))

augs = [x_data[rand_idx_2]] * 9 #Creating a list of 9 duplicate images

#Applying the augmentation to the created list
augs = aug_seq.augment_images(augs)

#Visualising the generated augmentations
plt.figure(figsize=(16, 6))

#Plotting the original image in the first subplot
plt.subplot(2, 5, 1)
plt.title('Original')
plt.imshow(x_data[rand_idx_2].squeeze(), cmap='gray')

#Looping over the augmented images and plot them in subsequent subplots
for i, aug in enumerate(augs):
    plt.subplot(2, 5, i+2)
    plt.title('Aug %02d' % int(i+1))
    plt.imshow(aug.squeeze(), cmap='gray')


#CREATING AN ATTRIBUTE DICTIONARY LOOKUP TABLE
#Initializing an empty dictionary to store labels as keys and indices as values
attr_real_dict = {}

#Looping over the attributes in y_real and their corresponding indices
for i, y in enumerate(y_real):
    # Convert the label to a string
    key = y.astype(str)
    
    # Join the string representation of the label and fill it with zeros to make it 6 characters long
    key = ''.join(key).zfill(6)

    # Add the label (as key) and its index (as value) to the dictionary
    attr_real_dict[key] = i 


#DEFINING A DATA GENERATOR
class DataGenerator(keras.utils.Sequence):
    def __init__(self, x, label, x_real, attr_real_dict, batch_size=32, shuffle=True):
        'Initialization'
        self.x = x  #Augmented image data
        self.label = label  #Attribute labels for augmented images
        self.x_real = x_real  #Real image data
        self.attr_real_dict = attr_real_dict  # Dictionary to map label keys to indices
        
        self.batch_size = batch_size  #Batch size for training
        self.shuffle = shuffle  #Whether to shuffle data
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x) / self.batch_size))

    # def __getitem__(self, index):
    #     'Generate one batch of data'
        
    #     #Generate indexes of the batch
    #     x1_batch = self.x[index*self.batch_size:(index+1)*self.batch_size]
    #     label_batch = self.label[index*self.batch_size:(index+1)*self.batch_size]
        
    #     x2_batch = np.empty((self.batch_size, 90, 90, 1), dtype=np.float32)
    #     y_batch = np.zeros((self.batch_size, 1), dtype=np.float32)
        
    #     #Augmentation
    #     if self.shuffle:
    #         seq = iaa.Sequential([
    #             iaa.GaussianBlur(sigma=(0, 0.5)),
    #             iaa.Affine(
    #                 scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
    #                 translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
    #                 rotate=(-30, 30),
    #                 order=[0, 1],
    #                 cval=255
    #             )
    #         ], random_order=True)

    #         x1_batch = seq.augment_images(x1_batch)
        
    #     #Pick matched images (label 1.0) and unmatched images (label 0.0) and put together in batch
    #     #Matched images must be all same, [subject_id(3), gender(1), left_right(1), finger(1)], e.g) 034010
    #     for i, l in enumerate(label_batch):
    #         match_key = l.astype(str)
    #         match_key = ''.join(match_key).zfill(6)

    #         if random.random() > 0.5:
    #             #Put matched image
    #             x2_batch[i] = self.x_real[self.attr_real_dict[match_key]]
    #             y_batch[i] = 1.
    #         else:
    #             #Put unmatched image
    #             while True:
    #                 unmatch_key, unmatch_idx = random.choice(list(self.attr_real_dict.items()))

    #                 if unmatch_key != match_key:
    #                     break

    #             #Resize the image to 90x90
    #             resized_image = resize(self.x_real[unmatch_idx], (90, 90))

    #             # Add channel dimension
    #             x2_batch[i] = resized_image[:, :, np.newaxis]
    #             y_batch[i] = 0.

    #     return [x1_batch.astype(np.float32) / 255., x2_batch.astype(np.float32) / 255.], y_batch

    def __getitem__(self, index):
        'Generate one batch of data'
        
        # Generate indexes of the batch
        x1_batch = self.x[index*self.batch_size:(index+1)*self.batch_size]
        label_batch = self.label[index*self.batch_size:(index+1)*self.batch_size]
        
        x2_batch = np.empty((self.batch_size, 90, 90, 1), dtype=np.float32)
        y_batch = np.zeros((self.batch_size, 1), dtype=np.float32)
        
        # Augmentation
        if self.shuffle:
            seq = iaa.Sequential([
                iaa.GaussianBlur(sigma=(0, 0.5)),
                iaa.Affine(
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    rotate=(-30, 30),
                    order=[0, 1],
                    cval=255
                )
            ], random_order=True)

            x1_batch = seq.augment_images(x1_batch)
        
        # Pick matched images (label 1.0) and unmatched images (label 0.0) and put together in batch
        # Matched images must be all same, [subject_id(3), gender(1), left_right(1), finger(1)], e.g) 034010
        for i, l in enumerate(label_batch):
            match_key = l.astype(str)
            match_key = ''.join(match_key).zfill(6)

            if random.random() > 0.5:
                # Put matched image
                if match_key in self.attr_real_dict:
                    x2_batch[i] = self.x_real[self.attr_real_dict[match_key]]
                    y_batch[i] = 1.
                else:
                    # If match_key is not found, randomly select an image from x_real
                    rand_idx = random.randint(0, len(self.x_real) - 1)
                    x2_batch[i] = self.x_real[rand_idx]
                    y_batch[i] = 0.  # Assign a "no match" label since the match key was not found
            else:
                # Put unmatched image
                while True:
                    unmatch_key, unmatch_idx = random.choice(list(self.attr_real_dict.items()))

                    if unmatch_key != match_key:
                        break

                # Resize the image to 90x90
                resized_image = resize(self.x_real[unmatch_idx], (90, 90))

                # Remove extra dimension and add channel dimension
                x2_batch[i] = resized_image.squeeze()[:, :, np.newaxis]
                y_batch[i] = 0.

        return [x1_batch.astype(np.float32) / 255., x2_batch.astype(np.float32) / 255.], y_batch
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            self.x, self.label = shuffle(self.x, self.label)

#Creating data generators for both training and validation data
train_gen = DataGenerator(x_train, attr_train, x_real, attr_real_dict, shuffle=True)
val_gen = DataGenerator(x_val, attr_val, x_real, attr_real_dict, shuffle=False)


#CREATING THE MODEL
# Overall Architecture:
# This CNN consists of two identical feature extraction branches sharing weights.  
# After extracting features from both branches, their features are subtracted.
# The resulting feature is then passed through additional convolutional and dense layers to perform classification with a binary output.
# The output (0 or 1), decides if there is a match

# Define input layers for two images
x1 = layers.Input(shape=(90, 90, 1))
x2 = layers.Input(shape=(90, 90, 1))

# Shared input layer for the feature extraction branch
inputs = layers.Input(shape=(90, 90, 1))

# Feature extraction layers
feature = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(inputs)
feature = layers.MaxPooling2D(pool_size=2)(feature)

feature = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(feature)
feature = layers.MaxPooling2D(pool_size=2)(feature)

# Define feature extraction model
feature_model = Model(inputs=inputs, outputs=feature)

# Apply feature extraction model to both inputs
x1_net = feature_model(x1)
x2_net = feature_model(x2)

# Subtract features from both inputs
net = layers.Subtract()([x1_net, x2_net])

# Additional convolutional layers
net = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(net)
net = layers.MaxPooling2D(pool_size=2)(net)

# Flatten the features
net = layers.Flatten()(net)

# Dense layers for classification
net = layers.Dense(64, activation='relu')(net)

# Output layer with sigmoid activation for binary classification
net = layers.Dense(1, activation='sigmoid')(net)

# Define the final model with two input branches and one output
model = Model(inputs=[x1, x2], outputs=net)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# Display the model summary
print("#"*30)
print("MODEL SUMMARY")
print("#"*30)
print(model.summary())


#TRAINING THE MODEL 
model_train = model.fit(train_gen, epochs=15, validation_data=val_gen)


#EVALUATING THE MODEL

#Generating a random index to select a validation image
rand_idx_3 = random.randint(0, len(x_val))

# Retrieve a random image and its label from the validation set
random_img = x_val[rand_idx_3]
random_label = attr_val[rand_idx_3]

#Apply augmentation to the random image
random_img = aug_seq.augment_image(random_img).reshape((1, 90, 90, 1)).astype(np.float32) / 255.

#Create a matched image by looking up the label in the real images dataset
match_key = random_label.astype(str)
match_key = ''.join(match_key).zfill(6)
rx = x_real[attr_real_dict[match_key]].reshape((1, 90, 90, 1)).astype(np.float32) / 255.
ry = y_real[attr_real_dict[match_key]]

# Predict the similarity score for the matched image
pred_rx = model.predict([random_img, rx])

# Create an unmatched image by randomly selecting a label from the dictionary
unmatch_key, unmatch_idx = random.choice(list(attr_real_dict.items()))
ux = x_real[unmatch_idx].reshape((1, 90, 90, 1)).astype(np.float32) / 255.
uy = y_real[unmatch_idx]

# Predict the similarity score for the unmatched image
pred_ux = model.predict([random_img, ux])

# After the model evaluation
y_true = np.concatenate([attr_val[:, np.newaxis], 1 - attr_val[:, np.newaxis]], axis=1)
y_pred = np.concatenate([1 - model.predict([x_val[:, :, :, np.newaxis], x_val[:, :, :, np.newaxis]]), model.predict([x_val[:, :, :, np.newaxis], x_val[:, :, :, np.newaxis]])], axis=1)

precision = metrics.Precision()
recall = metrics.Recall()
f1_score = metrics.F1Score()

precision_value = precision(y_true, y_pred).numpy()
recall_value = recall(y_true, y_pred).numpy()
f1_score_value = f1_score(y_true, y_pred).numpy()

print(f"Precision: {precision_value}")
print(f"Recall: {recall_value}")
print(f"F1 Score: {f1_score_value}")

# Display the original random image, matched image, and unmatched image with their predictions
plt.figure(figsize=(8, 4))
plt.subplot(1, 3, 1)
plt.title('Input: %s' % random_label)
plt.imshow(random_img.squeeze(), cmap='gray')
plt.subplot(1, 3, 2)
# plt.title('O: %.02f, %s' % (pred_rx, ry))
plt.title(f'O: {pred_rx[0][0]:.2f}, {ry[0]}')
plt.imshow(rx.squeeze(), cmap='gray')
plt.subplot(1, 3, 3)
# plt.title('X: %.02f, %s' % (pred_ux, uy))
plt.title(f'X: {pred_ux[0][0]:.2f}, {uy[0]}')
plt.imshow(ux.squeeze(), cmap='gray')
