from tensorflow.contrib.keras.api.keras.layers import Dropout
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Conv2D
from tensorflow.contrib.keras.api.keras.layers import MaxPooling2D
from tensorflow.contrib.keras.api.keras.layers import Flatten
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras.api.keras.callbacks import Callback
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras import backend
import os
import sys
import tensorflow as tf
 
 
class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_id = 0
        self.losses = ''
 
    def on_epoch_end(self, epoch, logs={}):
        self.losses += "Epoch {}: accuracy -> {:.4f}, val_accuracy -> {:.4f}\n"\
            .format(str(self.epoch_id), logs.get('acc'), logs.get('val_acc'))
        self.epoch_id += 1
 
    def on_train_begin(self, logs={}):
        self.losses += 'Training begins...\n'
 
#script_dir = os.path.dirname(__file__)
script_dir = os.path.dirname(sys.argv[0])
training_set_path = os.path.join(script_dir, 'dataset/training_set')
test_set_path = os.path.join(script_dir, 'dataset/test_set')



# Initialising the CNN
classifier = Sequential()
 
# Step 1 - Convolution
input_size = (128, 128)
classifier.add(Conv2D(32, (3, 3), input_shape=(*input_size, 3), activation='relu'))
 
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))  # 2x2 is optimal
 
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
 
# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
 
# Step 3 - Flattening
classifier.add(Flatten())
 
# Step 4 - Full connection
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=1, activation='sigmoid'))
 
# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
 
# Part 2 - Fitting the CNN to the images
batch_size = 32
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
 
test_datagen = ImageDataGenerator(rescale=1. / 255)
 
training_set = train_datagen.flow_from_directory(training_set_path,
                                                 target_size=input_size,
                                                 batch_size=batch_size,
                                                 class_mode='binary')
 
test_set = test_datagen.flow_from_directory(test_set_path,
                                            target_size=input_size,
                                            batch_size=batch_size,
                                            class_mode='binary')
 
# Create a loss history
history = LossHistory()
 
classifier.fit_generator(training_set,
                         steps_per_epoch=8000/batch_size,
                         epochs=90,
                         validation_data=test_set,
                         validation_steps=2000/batch_size,
                         workers=12,
                         max_queue_size=100,
                         callbacks=[history])
 
 
# Save model
from keras.models import model_from_json

#Complete saved model
model_backup_path = os.path.join(script_dir, 'dataset/cat_or_dogs_classifier.h5')
classifier.save(model_backup_path)
print("Model saved to", model_backup_path)


# serialize model to JSON
model_json = classifier.to_json()
with open("model/classifier_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("model/classifier_weights.h5")
print("Saved model to disk")
# serialize model to Yaml
yaml_string = classifier.to_yaml()
with open("model/classifier_model.yaml", "w") as yaml_file:
    yaml_file.write(yaml_string)
 
# Save loss history to file
loss_history_path = os.path.join(script_dir, 'model/loss_history.log')
myFile = open(loss_history_path, 'w+')
myFile.write(history.losses)
myFile.close()
 
backend.clear_session()
print("The model class indices are:", training_set.class_indices)




import numpy as np
from keras.preprocessing import image
import tensorflow as tf

global graph,model
graph = tf.get_default_graph()

with graph.as_default():
    test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = input_size)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
   
result = classifier.predict(test_image)
#training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
prediction




graph = tf.get_default_graph()