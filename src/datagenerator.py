import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, atkOrNorm , list_IDs, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, progress=None, shuffle=True):
        'Initialization'
        self.atk = atkOrNorm
        self.dim = dim
        self.progress = progress
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim[0], self.dim[1]))
        y = np.empty((self.batch_size, self.dim[0], self.dim[1]), dtype=int)
        
                
        if self.atk : 
            directory = 'attack'
        else :
            directory = 'normal'   
            
        progress = self.progress
        
        dataX = np.load(progress + '/' + 'artefact' + '/' + directory + 'URITraining.npy')
        dataY = np.load(progress + '/' + 'artefact' + '/' + directory + 'URILabel.npy')
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = dataX[ID]

            # Store class
            y[i,] = dataY[ID]

        return [X], [keras.utils.to_categorical(y, num_classes=self.n_classes[0] + 1)]