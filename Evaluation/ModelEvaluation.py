import cv2
import tensorflow as tf
from tensorflow.keras.models import Model,Sequential
import numpy as np
from tensorflow.keras.models import  load_model
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
Class_names = ['Borer','Choanephora','Sound']

model = load_model('..\Database\\vgg16.h5')
# print(model.metrics_names)

from tensorflow.keras.preprocessing import image
realIndex = list()
predictionIndex = list()
Datadir = '../Database/Validation_set'
Categories=['Borer','Choanephora','Sound']
for category in Categories:
    path = os.path.join(Datadir, category)
    class_num = Categories.index(category)  # make classification value
    for img in os.listdir(path):
        try:
            img = cv2.imread(os.path.join(path, img))
            img = cv2.resize(img, (224, 224))
            my_img_arr = image.img_to_array(img)
            my_img_arr = np.expand_dims(my_img_arr,axis=0)
            classValue = model.predict(my_img_arr)[0]

            predictlist=list()
            for i in classValue:
                predictlist.append(round(i))
            RightIndex = predictlist.index(1)
            predictionIndex.append(RightIndex)
            realIndex.append(class_num)

        except:
            pass


predf = pd.DataFrame({'real': realIndex,
                   'predicted': predictionIndex,
                   })
print(predf)
predicted0 = predf.loc[(predf['real'] == 0) & (predf['predicted'] == 0) ] #(), & are  very very important
predicted1 = predf.loc[(predf['real'] == 1) & (predf['predicted'] == 1) ]
predicted2 = predf.loc[(predf['real'] == 2) & (predf['predicted'] == 2) ]


predicted0Len = len(predicted0)
predicted1Len = len(predicted1)
predicted2Len = len(predicted2)

original0Len = len(predf.loc[(predf['real'] == 0)])
original1Len = len(predf.loc[(predf['real'] == 1)])
original2Len = len(predf.loc[(predf['real'] == 2)])




real = [original0Len,original1Len,original2Len]
predicted = [predicted0Len,predicted1Len,predicted2Len]
# Create the pandas DataFrame
index = ['Borer','Choanephora','Sound']
df = pd.DataFrame({'real': real,
                   'predicted': predicted,
                   }, index=index)

df.plot.bar(rot=0,width=.3)
plt.ylabel('Numbers')
plt.yticks([2,4,6,8,10,12,14,16,18,20,])
plt.grid()
plt.xlabel('Categories')
plt.savefig('realVsPredicted.png')
plt.show()


######################## confusion matrix #######################
real = predf['real'].values
predicted = predf['predicted'].values
cf_matrix= confusion_matrix(real,predicted)
group_names = ['TN','FP','FN','TP']
group_counts = ['{0:0.0f}'.format(value) for value in
                cf_matrix.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_names,group_counts)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=True, fmt='',) # annot True is vvi for multiclass

plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig("Confusion Matrix.png")
plt.show()