
import tensorflow as tf
from tensorflow.keras.models import Model,Sequential
import numpy as np
from tensorflow.keras.models import  load_model
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns


Class_names = ['Baila','Boumach','Hilsa','Kholse','Koral','Loitta',
 'Mini','Pabda','Poa','Puti','Rupchanda','Tailla']

model = load_model('..\Database\M2_C4_k3_2_p2_2.h5')
# print(model.metrics_names)

from tensorflow.keras.preprocessing import image
predictlist = list()
Datadir = '../Database/Validation_set/Loitta/dis_agu_57_9747024.jpeg'
my_image = image.load_img(Datadir, target_size=(224, 224))
plt.imshow(my_image)
plt.show()
my_img_arr = image.img_to_array(my_image)
my_img_arr = np.expand_dims(my_img_arr,axis=0)

classValue = model.predict(my_img_arr)[0]
print(classValue)
# for i in classValue:
#     predictlist.append(round(i))
# print(predictlist)
# RightIndex = predictlist.index(1)
# print(RightIndex)


