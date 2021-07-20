import pandas as pd
from matplotlib import pyplot as plt
# plt.rcParams["figure.figsize"] = [12, 6]
plt.rcParams.update({'font.size': 11})
plt.rcParams["font.family"] = "Times New Roman"


############################ Model 1 ####################3
inceptionV3 = pd.read_csv(r'Dataset/inceptionV3.csv')

inceptionV3TAccu = inceptionV3['accuracy'].values.tolist()
inceptionV3VAccu = inceptionV3['val_accuracy'].values.tolist()
inceptionV3TLoss = inceptionV3['loss'].values.tolist()
inceptionV3VLoss =inceptionV3['val_loss'].values.tolist()


axes = plt.axes()
plt.plot(range(1,len(inceptionV3TAccu)+1),inceptionV3TAccu,color='green',linewidth=2)
plt.plot(range(1,len(inceptionV3TAccu)+1),inceptionV3VAccu,color='red',linewidth=2)
# axes.set_yticks([0.8,0.85,0.9,0.95,1.00,1.050,1.100])
plt.grid()
axes.set_yticks([0.7,0.8,0.9,1,1.1,1.2,1.3])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
# plt.ylabel('Loss')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.savefig('inceptionv3  tain vs validation accuracy.png')
plt.show()

axes = plt.axes()
plt.plot(range(1,len(inceptionV3TLoss)+1),inceptionV3TLoss,color='green',linewidth=2)
plt.plot(range(1,len(inceptionV3TLoss)+1),inceptionV3VLoss,color='red',linewidth=2)
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
# plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.savefig('inceptionv3 tain vs validation loss.png')
plt.show()


##################### Model 2 #####################################
resnet50 = pd.read_csv(r'Dataset/resnet50.csv')

resnet50TAccu = resnet50['accuracy'].values.tolist()
resnet50VAccu = resnet50['val_accuracy'].values.tolist()
resnet50TLoss = resnet50['loss'].values.tolist()
resnet50VLoss = resnet50['val_loss'].values.tolist()


axes = plt.axes()
plt.plot(range(1,len(resnet50TAccu)+1),resnet50TAccu,color='green',linewidth=2)
plt.plot(range(1,len(resnet50TAccu)+1),resnet50VAccu,color='red',linewidth=2)
# axes.set_yticks([0.8,0.85,0.9,0.95,1.00,1.050,1.100])
plt.grid()
axes.set_yticks([0.7,0.8,0.9,1,1.1,1.2,1.3])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
# plt.ylabel('Loss')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.savefig('resnet50 tain vs validation accuracy.png')
plt.show()

axes = plt.axes()
plt.plot(range(1,len(resnet50TLoss)+1),resnet50TLoss,color='green',linewidth=2)
plt.plot(range(1,len(resnet50TLoss)+1),resnet50VLoss,color='red',linewidth=2)
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
# plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.savefig('resnet50 tain vs validation loss.png')
plt.show()

 ##################### Model 3 #####################################
vgg16 = pd.read_csv(r'Dataset/vgg16.csv')

vgg16TAccu = vgg16['accuracy'].values.tolist()
vgg16VAccu = vgg16['val_accuracy'].values.tolist()
vgg16TLoss = vgg16['loss'].values.tolist()
vgg16VLoss = vgg16['val_loss'].values.tolist()

axes = plt.axes()
plt.plot(range(1,len(vgg16TAccu)+1),vgg16TAccu,color='green',linewidth=2)
plt.plot(range(1,len(vgg16TAccu)+1),vgg16VAccu,color='red',linewidth=2)
# axes.set_yticks([0.8,0.85,0.9,0.95,1.00,1.050,1.100])
plt.grid()
axes.set_yticks([0.7,0.8,0.9,1,1.1,1.2,1.3])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
# plt.ylabel('Loss')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.savefig('vgg16 tain vs validation accuray.png')
plt.show()

axes = plt.axes()
plt.plot(range(1,len(vgg16TLoss)+1),vgg16TLoss,color='green',linewidth=2)
plt.plot(range(1,len(vgg16TLoss)+1),vgg16VLoss,color='red',linewidth=2)
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.savefig('vgg16 tain vs validation loss.png')
plt.show()


 ################### Comparision of 3 model ###################

axes = plt.axes()
plt.plot(range(1,len(resnet50VAccu)+1),inceptionV3VAccu,color='red',linewidth=2)
plt.plot(range(1,len(resnet50VAccu)+1),resnet50VAccu,color='green',linewidth=2)
plt.plot(range(1,len(resnet50VAccu)+1),vgg16VAccu,color='blue',linewidth=2)
plt.grid()
axes.set_yticks([0.7,0.8,0.9,1,1.1,1.2,1.3])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
# plt.ylabel('Loss')
plt.legend(['InceptionV3', 'Resnet50', 'vgg16'])
plt.savefig('3model comparision.png')
plt.show()