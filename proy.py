#%%
#ANCHOR  Librerias que nos apoyan 
import tensorflow as tf

import numpy as np #esta es la encargada de operaciones numericas
import matplotlib.pyplot as plt #para sacar graficas
import _pickle as elPickle #para serializar binarios
import random
import os

# Use PrettyTensor to simplify Neural Network construction.
import prettytensor as pt


#%%
# ANCHOR deserializa
# ** ayuda a deserializar los archivos que vienen las imagen
def deserializar(arch):
    archivoCargado = open(arch, 'rb') #abrimos los archivos
    archivoPickleado = elPickle.load(archivoCargado, encoding='latin1') # cargamos los archivos pickleado
    data = archivoPickleado['data']
    filas = archivoPickleado['labels']
    archivos = archivoPickleado['filenames']
    imagenes = np.transpose(np.reshape(data,(-1,32,32,3), order='F'),axes=(0,2,1,3))
    losLabel = np.asarray(archivoPickleado['labels'], dtype='uint8')
    return losLabel, imagenes,data,archivos, filas


#%%
#ANCHOR labels que estan dentro de los archivos
labelsGeneral = ['avion','carro','pajaro','gato','venado','perro','sapo','caballo','barco','camion']

#%%
#ANCHOR dibujar imagenes
def dibujar(data, labels,archivos,img):
    X = np.asarray(data.T).astype("uint8") 
    labelPos = labelsGeneral[labels[img]]
    Y = np.zeros((10,10000))
    for i in range(10000):
        Y[labels[i],i] = 1
    plotear(X,Y,archivos,img,labelPos)

def plotear(X,Y,archivos,id,posLabel):
	rgb = X[:,id]
	img = rgb.reshape(3,32,32).transpose([1, 2, 0])
	plt.imshow(img)
	plt.title(archivos[id]+'-'+posLabel)

def plotearImagenesEntrenadas(images, cls_true, cls_pred=None, smooth=True):

    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'
        ax.imshow(images[i, :, :, :],
                  interpolation=interpolation)
        cls_true_name = labelsGeneral[cls_true[i]] #ploteamos el nombre verdaderos de la imagen
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name) 
        else:
            cls_pred_name = labelsGeneral[cls_pred[i]]#ploteamos el nombre que la maquina cree quees
            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
	
#%%
#ANCHOR detalle de batch
def detalleBatch(data,labels,nombrearch):
    item, conteo = np.unique(labels,return_counts=True)
    arreglo = dict(zip(labelsGeneral ,conteo))
    print(arreglo)
   

#%%
#ANCHOR para los usuarios de jupyter notebook
%matplotlib inline

# %%

labelsentreno1,dataEntreno1,data1,arch1,y1 = deserializar('cifar-10-batches-py\data_batch_1')
labelsentreno2,dataEntreno2,data2,arch2,y2 = deserializar('cifar-10-batches-py\data_batch_2')
labelsentreno3,dataEntreno3,data3,arch3,y3 = deserializar('cifar-10-batches-py\data_batch_3')
labelsentreno4,dataEntreno4,data4,arch4,y4 = deserializar('cifar-10-batches-py\data_batch_4')
labelsentreno5,dataEntreno5,data5,arch5,y5 = deserializar('cifar-10-batches-py\data_batch_5')
labelsentrenoPrueba,dataEntrenoPrueba,dataPrueba,archPrueba,yprueba = deserializar('cifar-10-batches-py/test_batch')

# %%
# dibuja aleatorio
dibujar(data5,labelsentreno5,arch5,random.randint(1,10000))


# %%
# ANCHOR detalle del batch

detalleBatch(data5,labelsentreno5,arch5)

