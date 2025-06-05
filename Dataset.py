#DATASET PROYECTO CLASIFICADOR DE FRUTAS 
import cv2 
import os 
import glob 
from sklearn.utils import shuffle 
import numpy as np 

def load_train(train_path, image_size, classes): 
    images, labels, img_names, cls = [], [], [], []
    print('Leyendo imágenes de entrenamiento...')
    
    for field in classes:    
        index = classes.index(field) 
        print(f'Leyendo archivos de {field} (Índice: {index})')
        
        path = os.path.join(train_path, field, '*.*')  # Buscar archivos JPG y PNG
        files = glob.glob(os.path.join(train_path, field, '*.*'))
        print(f"Imágenes encontradas en {train_path}/{field}: {files}")
        
        for fl in files: 
            image = cv2.imread(fl)
            if image is None:
                print(f"Advertencia: No se pudo leer la imagen {fl}")
                continue
                
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            image = image.astype(np.float32) / 255.0  # Normalización
            
            images.append(image) 
            label = np.zeros(len(classes)) 
            label[index] = 1.0 
            labels.append(label) 
            
            img_names.append(os.path.basename(fl)) 
            cls.append(field) 

    if not images:
        raise ValueError("Error: No se encontraron imágenes en la ruta proporcionada.")
    
    return np.array(images), np.array(labels), np.array(img_names), np.array(cls)

class DataSet:
    def __init__(self, images, labels, img_names, cls): 
        self._num_examples = images.shape[0]
        self._images = images 
        self._labels = labels 
        self._img_names = img_names 
        self._cls = cls 
        self._epochs_done = 0 
        self._index_in_epoch = 0 

    def next_batch(self, batch_size): 
        start = self._index_in_epoch 
        self._index_in_epoch += batch_size 
        
        if self._index_in_epoch > self._num_examples: 
            self._epochs_done += 1 
            start = 0 
            self._index_in_epoch = batch_size 
            np.random.shuffle(self._images)
        
        end = self._index_in_epoch 
        return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end] 


def read_train_sets(train_path, image_size, classes, validation_size): 
    class DataSets:
        pass 
    data_sets = DataSets()
    
    images, labels, img_names, cls = load_train(train_path, image_size, classes)
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls)   
    
    if isinstance(validation_size, float): 
        validation_size = int(validation_size * images.shape[0]) 
    
    if validation_size >= images.shape[0]:
        raise ValueError("Error: El tamaño de validación es mayor o igual al conjunto de datos.")
    
    data_sets.train = DataSet(images[validation_size:], labels[validation_size:], img_names[validation_size:], cls[validation_size:])
    data_sets.valid = DataSet(images[:validation_size], labels[:validation_size], img_names[:validation_size], cls[:validation_size])
    
    return data_sets
