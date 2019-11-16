#%%
#ANCHOR  Librerias que nos apoyan 
import tensorflow as tf
import numpy as np #esta es la encargada de operaciones numericas
import matplotlib.pyplot as plt #para sacar graficas
import _pickle as elPickle #para serializar binarios
import random


#%%
# ANCHOR deserializa
# ** ayuda a deserializar los archivos que vienen las imagen
def deserializar(arch):
    archivoCargado = open(arch, 'rb') #abrimos los archivos
    archivoPickleado = elPickle.load(archivoCargado, encoding='latin1') # cargamos los archivos pickleado
    data = archivoPickleado['data']
    archivos = archivoPickleado['filenames']
    imagenes = np.transpose(np.reshape(data,(-1,32,32,3), order='F'),axes=(0,2,1,3))
    losLabel = np.asarray(archivoPickleado['labels'], dtype='uint8')
    return losLabel, imagenes,data,archivos


#%%
#ANCHOR labels que estan dentro de los archivos
labelsGeneral = ['avion','carro','pajaro','gato','venado','perro','sapo','caballo','barco','camion']

#%%
#ANCHOR dibujar imagenes
def dibujar(data, labels,archivos,img):
    print(labels)
    X = np.asarray(data.T).astype("uint8") 
    labelPos = labelsGeneral[labels[img]]
    Y = np.zeros((10,10000))
    for i in range(10000):
        Y[labels[i],i] = 1
    plotear(X,Y,archivos,img,labelPos)

def plotear(X,Y,archivos,id,posLabel):
	rgb = X[:,id]
	print(X,rgb.shape,id)
	img = rgb.reshape(3,32,32).transpose([1, 2, 0])
	# labelPos = Y[id]
	plt.imshow(img)
	plt.title(archivos[id]+'-'+posLabel)
	

#%%
#ANCHOR para los usuarios de jupyter notebook
%matplotlib inline

# %%

labelsentreno1,dataEntreno1,data1,arch1 = deserializar('cifar-10-batches-py\data_batch_1')
labelsentreno2,dataEntreno2,data2,arch2 = deserializar('cifar-10-batches-py\data_batch_2')
labelsentreno3,dataEntreno3,data3,arch3 = deserializar('cifar-10-batches-py\data_batch_3')
labelsentreno4,dataEntreno4,data4,arch4 = deserializar('cifar-10-batches-py\data_batch_4')
labelsentreno5,dataEntreno5,data5,arch5 = deserializar('cifar-10-batches-py\data_batch_5')
labelsentrenoPrueba,dataEntrenoPrueba,dataPrueba,archPrueba = deserializar('cifar-10-batches-py/test_batch')

# %%
dibujar(data5,labelsentreno5,arch5,random.randint(1,10000))


# %%
