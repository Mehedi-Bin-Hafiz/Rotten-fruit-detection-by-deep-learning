import pandas as pd
from matplotlib import pyplot as plt
# plt.rcParams["figure.figsize"] = [12, 6]
plt.rcParams.update({'font.size': 11})
plt.rcParams["font.family"] = "Times New Roman"


############################ Model 1 ####################3
resnet50 = pd.read_csv(r'Dataset/resnet50.csv')
resnet50VAccu = resnet50['val_accuracy'].values.tolist()

vgg16 = pd.read_csv(r'Dataset/vgg16.csv')
vgg16VAccu = vgg16['val_accuracy'].values.tolist()
################### Comparision of 3 model ###################

axes = plt.axes()
plt.plot(range(1,len(resnet50VAccu)+1),resnet50VAccu,color='green',linewidth=2)
plt.plot(range(1,len(resnet50VAccu)+1),vgg16VAccu,color='red',linewidth=2)
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
# plt.ylabel('Loss')
plt.legend(['Resnet50', 'vgg16'])
plt.savefig('2model comparision.png')
plt.show()