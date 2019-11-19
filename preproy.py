#%%

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os



#%%
#nos sirve para dibujar los datos
import matplotlib.pyplot as plt
#framework para hacer las funciones del grafo
import tensorflow as tf
#Matetmaticos
import numpy as np
#tiempo
import time
import random
from datetime import timedelta
#operaciones matematicas
import math
#operciones de IO
import os
# para la lectura de los archivos
import pickle



#ANCHOR OPERACIONES DE LAS IMAGENES
# %%
#ANCHOR Filtros
def filtrarImagen(img, tam, estaEntrenando):    
    if estaEntrenando:
        # delimitamos la imgane usando el tensor flow
        imgTransformado = tf.random_crop(img, size=[tam, tam, 3])
        # rotamos la imagen para la aumentar la eficiencia
        imgTransformado = tf.image.random_flip_left_right(img)
        #Agregamos filtros aleatorios
        imgTransformado = tf.image.random_hue(img, max_delta=0.05)
        imgTransformado = tf.image.random_contrast(img, lower=0.3, upper=1.0)
        imgTransformado = tf.image.random_brightness(img, max_delta=0.2)
        imgTransformado = tf.image.random_saturation(img, lower=0.0, upper=2.0)
        #obtenemos los pixeles que necesitamos
        imgTransformado = tf.minimum(img, 1.0)
        imgTransformado = tf.maximum(img, 0.0)
    
    else:
        #Solo lo cambiamos o cortamos
        imgTransformado = tf.image.resize_image_with_crop_or_pad(img,
                                                       target_height=tam,
                                                       target_width=tam)
    
    return imgTransformado
#* aplicamos los filtros a cada llamada
def aplicarFiltros(imagenes,tam,estaEntrenando):   
    imgTransformados = tf.map_fn(lambda img: filtrarImagen(img, tam,estaEntrenando), imagenes)
    return imgTransformados

#ANCHOR Pasar a vector
#* reajustamos imagenes para dividirlo en vectores
def reajustarImagenes(imagenes, tamaño):
   #Cambiamos las imagenes para que sean un arreglo de numpy
    inicio = imagenes.shape[0]
    #Inicializamos nueva vector con las imagenes
    nuevaImagen = np.zeros((inicio, tamaño, tamaño, 3))
    j = 0
    #por cada imagen
    for img in imagenes:
        #obtemos la capa de los colores
        rojo = img[0:tamaño*tamaño]/255
        verde = img[tamaño*tamaño:tamaño*tamaño*2]/255
        azul = img[tamaño*tamaño*2:tamaño*tamaño*3]/255
        #inicializamos la nueva matriz de cada imagenes
        resultado = np.zeros((32,32,3))
        for fila in range(0, tamaño): 
            for col in range(0, tamaño): 
                point = np.zeros(3)
                point[0] = rojo[fila*32+col]
                point[1] = verde[fila*32+col]
                point[2] = azul[fila*32+col]
                resultado[fila][col] = point
        nuevaImagen[j] = resultado
        j += 1
    return nuevaImagen

#%%
#ANCHOR Clases que vienen en el cifar 10
labelsOriginales = ['Avion',
    'Carro',
    'Pajaro',
    'Gato',
    'Venado',
    'Perro',
    'Sapo',
    'Caballo',
    'Barco',
    'Camion']

#%%
#ANCHOR dibujar imagenes
def dibujar(data, labels,archivos,img):
    X = np.asarray(data.T).astype("uint8") 
    labelPos = labelsOriginales[labels[img]]
    Y = np.zeros((10,10000))
    for i in range(10000):
        Y[labels[i],i] = 1
    plotear(X,Y,archivos,img,labelPos)

def plotear(X,Y,archivos,id,posLabel):
    rgb = X[:,id]
    img = rgb.reshape(3,32,32).transpose([1, 2, 0])
    plt.imshow(img) 
    print('Clase: ' + posLabel)
    print('Archivo: ' , archivos[id])
    #plt.title(archivos[id]+'-'+posLabel)

# %%
#ANCHOR matplotib
%matplotlib inline
import matplotlib.pyplot as plt

# %% ANCHOR control del archivo de cifar, obtenemos archivo y lo devolvemos
# *lee el archivo y devuelvo los dato
def deserializar(arch):
    with open(arch, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# *Produce un codigo convolucional de un mensaje.
def convEncoding(input):
    result = []
    for code in input:
        inner = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        inner[code] = 1.0
        result.append(inner)
    return result



# ANCHOR cargar archivo
# %%
direccionCifar = "cifar-10-batches-py/data_batch_"
direccionCifarPrueba = "cifar-10-batches-py/"

# * Funcion que carga los datos
def cargarArchivo():
    #Construimos un vector con los 5 archivos 
    archEntreno = [direccionCifar + str(i) for i in range(1, 6)]
    print('Archivos con datasets: ')
    print(archEntreno)
    archPrueba = direccionCifarPrueba + 'test_batch'
    #inicializmaos los datos que nos van a utilizar
    datosEntrena = deserializar(archEntreno[0])
    datosPrueba = deserializar(archPrueba)
    #Obtenemos los datos de la imagen de la columan data
    datosImagen = datosEntrena[b'data']
    datosArchivos = datosEntrena[b'filenames']
    datosImagenPrueba = datosPrueba[b'data']
    #Obtenemos los nombres de las clases
    labelEntrenamiento = datosEntrena[b'labels']
    labelPrueba = datosPrueba[b'labels']
    #obtenemso los datos de los siguientes archivos
    for _arch in archEntreno[1:5]:
        _data = deserializar(_arch)
        datosImagen = np.concatenate((datosImagen, _data[b'data']), axis=0)
        labelEntrenamiento += _data[b'labels']
        datosArchivos += _data[b'filenames']

    labelEntrenamiento = np.asarray(labelEntrenamiento)
    labelPrueba = np.asarray(labelPrueba)

    return datosImagen, datosImagenPrueba, labelEntrenamiento, labelPrueba, datosArchivos



#%%
imagenesEntrena, imagenesPrueba, etiquetaEntrena, etiquetaPrueba, nombreImagenes = cargarArchivo()
print("a entrenar:", imagenesEntrena.shape)
print("a probar:", imagenesPrueba.shape)
dibujar(imagenesEntrena,etiquetaEntrena,nombreImagenes,random.randint(1,10000))




#%%
data_dir = "cifar-10-batches-py/data_batch_"
data_dir_test = "cifar-10-batches-py/"
# labels are not one-hot-encoding!
#images_train, images_test, labels_train, labels_test, class_names = load_data()
images_train, images_test, labels_train, labels_test, nombreImagenes = cargarArchivo()
#imagenesEntrena, imagenesPrueba, etiquetaEntrena, etiquetaPrueba, nombreImagenes = cargarArchivo()
print("Data loaded: ")
print("==>training data shape:", images_train.shape)
print("==>test data shape:", images_test.shape)


# %%

# %%
def reshape(images, width):
    
    '''reshape the input into 32x32x3 np.ndarray'''
    
    # input images should be a 2-dimensional np.array 
    # e.g: [[1,2,3,...]] for one image only
    first = images.shape[0]
    result = np.zeros((first, width, width, 3))
    index = 0
    for image in images:
#         assert len(image) == width*width*3
        # Get color out of original array
        redPixel = image[0:width*width]/255
        greenPixel = image[width*width:width*width*2]/255
        bluePixel = image[width*width*2:width*width*3]/255
        reshaped = np.zeros((32, 32, 3))
        for i in range(0, width): #row
            for j in range(0, width): #column
                point = np.zeros(3)
                point[0] = redPixel[i*32+j]
                point[1] = greenPixel[i*32+j]
                point[2] = bluePixel[i*32+j]
                # add to result
                reshaped[i][j] = point
        result[index] = reshaped
        index += 1
            
    return result

# %%

images_train_reshaped = reshape(imagenesEntrena, 32)
images_test_reshaped = reshape(imagenesPrueba, 32)
print("Data reshaped: ")
print("==>training data reshaped:", images_train_reshaped.shape)
print("==>test data reshaped:", images_test_reshaped.shape)


# %%
def distorted_image(image, cropped_size, training):    
    '''This function takes a single image from training set as input'''
    
    if training:
        # Randomly crop the input image.
        image = tf.random_crop(image, size=[cropped_size, cropped_size, 3])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    
    else:
        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
    
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=cropped_size,
                                                       target_width=cropped_size)
    
    return image



def preprocess(images,cropped_size,training):   
    '''This function takes multiple images as input,
    will call distorted_image()'''

    images = tf.map_fn(lambda image: distorted_image(image, cropped_size,training), images)
    
    return images

# %%
def prediction (logits):
    predicted_class = tf.argmax(logits, 1, name='pred_class')
    return predicted_class

def compute_accuracy(logits, y):
    prediction = tf.argmax(logits, 1, name='pred_class')
    true_label = tf.argmax(y, 1, name='true_class')
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, true_label), tf.float64))
    return accuracy

def compute_cross_entropy(logits, y):
    # Compute the average cross-entropy across all the examples.
    sm_ce = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits, name='cross_ent_terms')
    cross_ent = tf.reduce_mean(sm_ce, name='cross_ent')
    return cross_ent

def conv(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def fractional_maxpool(x):
    return tf.nn.fractional_max_pool(x, pooling_ratio=[1, 1.44, 1.44, 1])

def cp_cp_fc_fc_logits(x):

    x_image = tf.cast(x, tf.float32)
    n_conv1 = 64
    n_conv2 = 64


    W_conv1 = tf.get_variable("W_conv1", shape = [5,5,3,n_conv1])
    b_conv1 = tf.get_variable("b_conv1", shape = [n_conv1])
    h_conv1 = tf.nn.relu(tf.add(conv(x_image, W_conv1), b_conv1))
    p_1 = maxpool(h_conv1)
    
    W_conv2 = tf.get_variable("W_conv2", shape=[5,5,n_conv1, n_conv2])
    b_conv2 = tf.get_variable("b_conv2", shape=[n_conv2])
    h_conv2 = tf.nn.relu(tf.add(conv(p_1, W_conv2), b_conv2))
    p_2 = maxpool(h_conv2)
    p_2_drop = tf.nn.dropout(p_2, 0.8)
    
    p_2_flatten = tf.reshape(p_2, [-1, 6*6*n_conv2])
    W_fc1 = tf.get_variable('W_fc1', shape=[6*6*n_conv2, 256])
    b_fc1 = tf.get_variable('b_fc1', shape=[256])
    h_fc1 = tf.nn.relu(tf.add(tf.matmul(p_2_flatten, W_fc1), b_fc1))

    W_fc2 = tf.get_variable("W_fc2", shape=[256, 128])
    b_fc2 = tf.get_variable("b_fc2", shape=[128])
    h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_fc1, W_fc2), b_fc2))

    W_fc3 = tf.get_variable('W_fc3', shape=[128, 10])
    b_fc3 = tf.get_variable('b_fc3', shape=[10])
    logits = tf.add(tf.matmul(h_fc2, W_fc3), b_fc3, name='cp_cp_fc_fc_logits')

    print('Check shape before flatten layer:', p_2.get_shape())
    return(logits)

# %%
labels_train = convEncoding(labels_train)
labels_test = convEncoding(etiquetaPrueba)
labels_train = np.asarray(labels_train)
labels_test = np.asarray(etiquetaPrueba)

cropped_size = 24
learning_rate = 0.05
max_steps = 500
batch_size = int(len(labels_train)/max_steps)
print("Images cropped size is {:g}".format(cropped_size))
print("Batch size is {:g}".format(batch_size))
print("Max steps in each batch is {:g}".format(max_steps))
print("Learning rate is {:g}".format(learning_rate))

# %%
tf.reset_default_graph()
# change here: define a graph
with tf.Graph().as_default():
    start_time = time.time()
    print('starting time', start_time)
    
    x = tf.placeholder(tf.float32, [None, 32,32,3], name="x")
    y = tf.placeholder(tf.float32, [None, 10], name = "y")
#     prob_keep_1 = tf.placeholder(tf.float32, name="prob_keep_1")
#     prob_keep_2 = tf.placeholder(tf.float32, name="prob_keep_2")
#     prob_keep_3 = tf.placeholder(tf.float32, name="prob_keep_3")
#     prob_keep_4 = tf.placeholder(tf.float32, name="prob_keep_4")
#     prob_keep_5 = tf.placeholder(tf.float32, name="prob_keep_5")
    
    training = True
    
    with tf.variable_scope("preprocess", reuse=not training):
        preprocessed_data = preprocess(x,cropped_size,training) #tensor with distorted images
    with tf.variable_scope("model", reuse=not training):
        logits = cp_cp_fc_fc_logits(preprocessed_data)
    with tf.variable_scope("loss",reuse=not training):
        loss = compute_cross_entropy(logits=logits, y=y)
    with tf.variable_scope("accuracy", reuse=not training):
        accuracy = compute_accuracy(logits, y)
    with tf.variable_scope("prediction", reuse=not training):
        pred_class = prediction(logits)
    with tf.variable_scope("training"):
        train_step = tf.train.AdamOptimizer(0.00001).minimize(loss)
    
    saver = tf.train.Saver()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
        
    # save loss and accuracies here
    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []
    
    with tf.device("/device:GPU:0"):        
        for index in range(0, 10):
            loss_list = []
            acc_list = []
            print("This is the {:g}th epochs!!!".format(index))
            for i in range(max_steps):
                training = True
                images_batch = imagenesEntrenaReajustado[i * batch_size: (i + 1) * batch_size]
                labels_batch = etiquetaEntrena[i * batch_size: (i + 1) * batch_size]

                _ , loss_val, acc_val = sess.run((train_step,loss,accuracy), feed_dict={x:images_batch, y:labels_batch})
                
                loss_list.append(loss_val)
                acc_list.append(acc_val)
                
                if i%200==0:
                    print("> it is the {:g}th iteration".format(i))
                    
            epoch_loss_train = sum(loss_list)/ float(len(loss_list))
            train_loss.append(epoch_loss_train)
            epoch_acc_train =  sum(acc_list)/ float(len(acc_list))
            train_accuracy.append(epoch_acc_train)
            
            print("--> Train accuracy for {:g}th epoch: ".format(index), epoch_acc_train)
            
            if index%100 == 0:
                saver.save(sess, '../output/scratch/my-model', global_step=index)
            
            # output accuracy, loss, predicted labels
            acc_list = []
            loss_list = []
            pred_labels =[]
            for i in range(0, 100):
                training =False
                loss_val, acc_val, pred_val = sess.run((loss,accuracy, pred_class),
                          feed_dict={x: images_test_reshaped[i*100:i*100+100], 
                                     y: labels_test[i*100:i*100+100]})
                loss_list.append(loss_val)
                acc_list.append(acc_val)
                pred_labels.append(pred_val)
                
            epoch_loss_test = sum(loss_list)/ float(len(loss_list))
            test_loss.append(epoch_loss_test)
            epoch_acc_test =  sum(acc_list)/ float(len(acc_list))
            test_accuracy.append(epoch_acc_test)
            
            print("--> Test accuracy for {:g}th epoch: ".format(index), epoch_acc_test)
            print("=========================================")
        end_time = time.time()
        process_timg = end_time - start_time

# %%
