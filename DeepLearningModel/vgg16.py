
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
IMAGE_SIZE = [224,224]

training_path = "/content/drive/MyDrive/Training"
testing_path = "/content/drive/MyDrive/Testing"

vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet',include_top = False)
for layer in vgg.layers:
  layer.trainable = False

x = Flatten()(vgg.output)
prediction = Dense(1,activation = 'sigmoid')(x)

model = Model(inputs = vgg.input, outputs = prediction)
model.summary()
model.compile(loss = 'binary_crossentropy',
optimizer = 'adam',
metrics = ['accuracy']
)

from keras.preprocessing.image import ImageDataGenerator
train_datagen =  ImageDataGenerator(
    rescale = 1./255,
)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_dataset = train_datagen.flow_from_directory(training_path,target_size = (224,224),batch_size = 32,)
test_dataset = test_datagen.flow_from_directory(testing_path,target_size = (224,224),batch_size = 32,)

""".fit is used when the entire training dataset can fit into the memory and no data augmentation is applied.

.fit_generator is used when either we have a huge dataset to fit into our memory or when data augmentation needs to be applied.
"""

history = model.fit(
    train_dataset,
    validation_data = test_dataset,
    epochs = 5,
    steps_per_epoch = len(train_dataset),
    validation_steps = len(test_dataset)

)

history_dict = history.history
print(history_dict.keys())
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['accuracy']
epochs = range(1,len(acc)+1)
plt.plot(epochs, loss_values, label = "Training loss")
plt.plot(epochs, val_loss_values,label = 'Validation loss')
plt.title("Training and validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("vggtLossvsvLoss.jpeg")
plt.show()
 
accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']
plt.plot(epochs, accuracy, label = 'Training accuracy')
plt.plot(epochs, val_accuracy, label = 'Validation accuracy')
plt.xlabel("Epochs")
plt.ylabel('Loss')
plt.legend()
plt.savefig("vggTaccuraVsVaccuracy.jpeg")
plt.show()

import pandas as pd
df = pd.DataFrame(model.history.history)
df.to_csv('vgg16.csv')
model.save('vgg16.h5')