import tensorflow as tf
import os
import cv2
import imghdr
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

import pdb
# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

data_dir = 'data'
print(os.listdir(data_dir))
image_exts = ['jpeg', 'jpg', 'png', 'bmp']

#Remove bad images

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))

#print(img.shape)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#Load data using tensorflow's nice keras data pipeline building

data = tf.keras.utils.image_dataset_from_directory('data')

#allow us to access the loaded data. Images represented as numpy arrays
data_iterator =  data.as_numpy_iterator()
batch = data_iterator.next()
#print(batch[1])

# fig, ax = plt.subplots(ncols=4, figsize=(20,20))
# for idx, img in enumerate(batch[0][:4]):
#     ax[idx].imshow(img.astype(int))
#     ax[idx].title.set_text(batch[1][idx])
#     plt.show()
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



#Scaling data using map function inside the pipeline

#x is our images, y is our target variable

data = data.map(lambda x,y: (x/255, y))

#Split data into train and test

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

#Building the DNN model

model = Sequential()

#Conv block 1 | takes the input | has 16 filters of 3*3 pixels | stride of 1
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

#Conv block 2
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

#Conv block 3
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

#Flatten layer
model.add(Flatten())

#Fully connected layer (256 values as the output)
model.add(Dense(256, activation='relu'))

#Activation function (1 value as the output)
model.add(Dense(1, activation='sigmoid'))


#Compile the model | Adam optimizer | Cross entropy loss | Accuracy as metric
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

print(model.summary())
pdb.set_trace()

#Train the model

logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

#Plot model performance during training

#Plot loss changes with each batch

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

#Plot accuracy changes with each batch

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

#Evaluation


pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(pre.result(), re.result(), acc.result())

#Testing

yhat = model.predict(np.expand_dims(resize/255, 0))


#Save the model

model.save(os.path.join('models','imageclassifier.h5'))

#Test Loading the model

new_model = load_model('imageclassifier.h5')
new_model.predict(np.expand_dims(resize/255, 0))
