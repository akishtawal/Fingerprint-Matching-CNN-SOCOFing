#Considering the preprocessing is done and the datasets are created 
#(Fingerprint_CNN_Preprocess.py)
#TRAINING THE DATA

#Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers, metrics
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import Precision, Recall, F1Score
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa #For image augmentation
import random 
from skimage.transform import resize
from tqdm import tqdm


#Saving the file path
#Make sure that you change this path to the path to the dataset fplder on your system
#(As created in the preprocessing script)
data_file_path = 'path\to\your\dataset_folder'

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
# sample_fraction = 0.75  # Use 75% of the data
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
plt.savefig('Rand_Img.png')

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
              cval=0 #Constant value used for filling in new pixels during transformations
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
    # plt.imshow(aug.squeeze(), cmap='gray')
    plt.imshow(aug, cmap='gray')

plt.savefig('Aug_Img.png')

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
                    y_batch[i, 0] = 1.  # Assign y_batch as a 2D tensor with shape (batch_size, 1)
                else:
                    # If match_key is not found, randomly select an image from x_real
                    rand_idx = random.randint(0, len(self.x_real) - 1)
                    x2_batch[i] = self.x_real[rand_idx]
                    y_batch[i, 0] = 0.  # Assign y_batch as a 2D tensor with shape (batch_size, 1)
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
                y_batch[i, 0] = 0.  # Assign y_batch as a 2D tensor with shape (batch_size, 1)

        return [x1_batch.astype(np.float32) / 255., x2_batch.astype(np.float32) / 255.], y_batch
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            self.x, self.label = shuffle(self.x, self.label)

#Creating data generators for both training and validation data
train_gen = DataGenerator(x_train, attr_train, x_real, attr_real_dict, shuffle=True)
val_gen = DataGenerator(x_val, attr_val, x_real, attr_real_dict, shuffle=False)


# CREATING THE MODEL
# Define input layers for two images
x1 = layers.Input(shape=(90, 90, 1))
x2 = layers.Input(shape=(90, 90, 1))

# Feature extraction branch
def feature_extraction_branch(inputs):
    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)  # Add dropout after max pooling

    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)  # Add dropout after max pooling

    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)  # Add dropout after max pooling

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)  # Add L2 regularization
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    return x

# Apply feature extraction branch to both inputs
x1_net = feature_extraction_branch(x1)
x2_net = feature_extraction_branch(x2)

# Subtract features from both inputs
net = layers.Subtract()([x1_net, x2_net])

# Classification head
net = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(net)  # Add L2 regularization
net = layers.BatchNormalization()(net)
net = layers.Dropout(0.5)(net)
net = layers.Dense(1, activation='sigmoid')(net)

# Define the final model with two input branches and one output
model = Model(inputs=[x1, x2], outputs=net)

# Compile the model
optimizer = Adam(learning_rate=0.001)
loss_fn = BinaryCrossentropy()
model.compile(optimizer=optimizer, loss=loss_fn, metrics=[Precision(), Recall(), metrics.BinaryAccuracy()])

# Print the model summary
print(model.summary())

# Train the model
model.fit(train_gen, epochs=2, validation_data=val_gen)

# # Evaluate the model
# y_pred = model.predict([x_val[:, :, :, np.newaxis], x_val[:, :, :, np.newaxis]])
# y_true = attr_val[:, np.newaxis]  # Convert y_true to 2D tensor with shape (batch_size, 1)


# Evaluate the model
y_pred = model.predict([x_val[:, :, :, np.newaxis], x_val[:, :, :, np.newaxis]])
y_true = y_true.reshape(-1, 1)  # Reshape y_true to have shape (None, 1)

loss, precision, recall, accuracy = model.evaluate([x_val[:, :, :, np.newaxis], x_val[:, :, :, np.newaxis]], y_true)

print(f"Loss: {loss}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Accuracy: {accuracy}")

y_pred_binary = (y_pred > 0.5).astype(int).ravel()
y_true_binary = y_true.ravel()  # Convert y_true back to 1D for computing F1 score

f1_score_value = f1_score(y_true_binary, y_pred_binary)
print(f"F1 Score: {f1_score_value}")
