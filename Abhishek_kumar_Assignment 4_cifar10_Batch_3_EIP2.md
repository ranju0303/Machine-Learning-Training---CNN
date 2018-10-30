

```python
# https://keras.io/
# !pip install -q keras 
# import keras 
# print(keras.__version__)
```


```python
import keras
import time
from keras.datasets import cifar10
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, AveragePooling2D, merge, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Concatenate
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
```

    Using TensorFlow backend.



```python
# this part will prevent tensorflow to allocate all the avaliable GPU Memory
# backend
import tensorflow as tf
from keras import backend as k

# Don't pre-allocate memory; allocate as-needed
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))
```


```python
class SGDRScheduler(Callback):
    
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay,
                 cycle_length,
                 mult_factor):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.set_weights(self.best_weights)
```


```python
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
```


```python
# Hyperparameters
# batch_size = 32
# num_classes =  10
# epochs = 50
# l = 40
# num_filter = 10
# compression = 0.5
# dropout_rate = 0

batch_size = 64
num_classes =  10
epochs = 50
l = 16
num_filter = 12
compression = 1
dropout_rate = 0.2

```


```python
# Load CIFAR10 Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
img_height, img_width, channel = x_train.shape[1],x_train.shape[2],x_train.shape[3]

# convert to one hot encoding 
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

```


```python
# Dense Block
def add_denseblock(input, num_filter, dropout_rate):
    global compression
    temp = input
    for _ in range(l):
        BatchNorm = BatchNormalization()(temp)
        relu = Activation('relu')(BatchNorm)
        Conv2D_3_3 = Conv2D(int(num_filter*compression), (3,3), use_bias=False ,padding='same')(relu)
        if dropout_rate>0:
          Conv2D_3_3 = Dropout(dropout_rate)(Conv2D_3_3)
        concat = Concatenate(axis=-1)([temp,Conv2D_3_3])
        
        temp = concat
        
    return temp
```


```python
def add_transition(input, num_filter, dropout_rate):
    global compression
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    Conv2D_BottleNeck = Conv2D(int(num_filter*compression), (1,1), use_bias=False ,padding='same')(relu)
    if dropout_rate>0:
      Conv2D_BottleNeck = Dropout(dropout_rate)(Conv2D_BottleNeck)
    avg = AveragePooling2D(pool_size=(2,2))(Conv2D_BottleNeck)
    
    return avg
```


```python
def output_layer(input):
    global compression
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    AvgPooling = AveragePooling2D(pool_size=(2,2))(relu)
    flat = Flatten()(AvgPooling)
    output = Dense(num_classes, activation='softmax')(flat)
    
    return output
```


```python
input = Input(shape=(img_height, img_width, channel,))
First_Conv2D = Conv2D(num_filter, (3,3), use_bias=False ,padding='same')(input)

First_Block = add_denseblock(First_Conv2D, num_filter, dropout_rate)
First_Transition = add_transition(First_Block, num_filter, dropout_rate)

Second_Block = add_denseblock(First_Transition, num_filter, dropout_rate)
Second_Transition = add_transition(Second_Block, num_filter, dropout_rate)

Third_Block = add_denseblock(Second_Transition, num_filter, dropout_rate)
Third_Transition = add_transition(Third_Block, num_filter, dropout_rate)

Last_Block = add_denseblock(Third_Transition,  num_filter, dropout_rate=0.2)
output = output_layer(Last_Block)

```


```python
model = Model(inputs=[input], outputs=[output])
model.summary()
```

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            (None, 32, 32, 3)    0                                            
    __________________________________________________________________________________________________
    conv2d_1 (Conv2D)               (None, 32, 32, 12)   324         input_1[0][0]                    
    __________________________________________________________________________________________________
    batch_normalization_1 (BatchNor (None, 32, 32, 12)   48          conv2d_1[0][0]                   
    __________________________________________________________________________________________________
    activation_1 (Activation)       (None, 32, 32, 12)   0           batch_normalization_1[0][0]      
    __________________________________________________________________________________________________
    conv2d_2 (Conv2D)               (None, 32, 32, 12)   1296        activation_1[0][0]               
    __________________________________________________________________________________________________
    dropout_1 (Dropout)             (None, 32, 32, 12)   0           conv2d_2[0][0]                   
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, 32, 32, 24)   0           conv2d_1[0][0]                   
                                                                     dropout_1[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_2 (BatchNor (None, 32, 32, 24)   96          concatenate_1[0][0]              
    __________________________________________________________________________________________________
    activation_2 (Activation)       (None, 32, 32, 24)   0           batch_normalization_2[0][0]      
    __________________________________________________________________________________________________
    conv2d_3 (Conv2D)               (None, 32, 32, 12)   2592        activation_2[0][0]               
    __________________________________________________________________________________________________
    dropout_2 (Dropout)             (None, 32, 32, 12)   0           conv2d_3[0][0]                   
    __________________________________________________________________________________________________
    concatenate_2 (Concatenate)     (None, 32, 32, 36)   0           concatenate_1[0][0]              
                                                                     dropout_2[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_3 (BatchNor (None, 32, 32, 36)   144         concatenate_2[0][0]              
    __________________________________________________________________________________________________
    activation_3 (Activation)       (None, 32, 32, 36)   0           batch_normalization_3[0][0]      
    __________________________________________________________________________________________________
    conv2d_4 (Conv2D)               (None, 32, 32, 12)   3888        activation_3[0][0]               
    __________________________________________________________________________________________________
    dropout_3 (Dropout)             (None, 32, 32, 12)   0           conv2d_4[0][0]                   
    __________________________________________________________________________________________________
    concatenate_3 (Concatenate)     (None, 32, 32, 48)   0           concatenate_2[0][0]              
                                                                     dropout_3[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_4 (BatchNor (None, 32, 32, 48)   192         concatenate_3[0][0]              
    __________________________________________________________________________________________________
    activation_4 (Activation)       (None, 32, 32, 48)   0           batch_normalization_4[0][0]      
    __________________________________________________________________________________________________
    conv2d_5 (Conv2D)               (None, 32, 32, 12)   5184        activation_4[0][0]               
    __________________________________________________________________________________________________
    dropout_4 (Dropout)             (None, 32, 32, 12)   0           conv2d_5[0][0]                   
    __________________________________________________________________________________________________
    concatenate_4 (Concatenate)     (None, 32, 32, 60)   0           concatenate_3[0][0]              
                                                                     dropout_4[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_5 (BatchNor (None, 32, 32, 60)   240         concatenate_4[0][0]              
    __________________________________________________________________________________________________
    activation_5 (Activation)       (None, 32, 32, 60)   0           batch_normalization_5[0][0]      
    __________________________________________________________________________________________________
    conv2d_6 (Conv2D)               (None, 32, 32, 12)   6480        activation_5[0][0]               
    __________________________________________________________________________________________________
    dropout_5 (Dropout)             (None, 32, 32, 12)   0           conv2d_6[0][0]                   
    __________________________________________________________________________________________________
    concatenate_5 (Concatenate)     (None, 32, 32, 72)   0           concatenate_4[0][0]              
                                                                     dropout_5[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_6 (BatchNor (None, 32, 32, 72)   288         concatenate_5[0][0]              
    __________________________________________________________________________________________________
    activation_6 (Activation)       (None, 32, 32, 72)   0           batch_normalization_6[0][0]      
    __________________________________________________________________________________________________
    conv2d_7 (Conv2D)               (None, 32, 32, 12)   7776        activation_6[0][0]               
    __________________________________________________________________________________________________
    dropout_6 (Dropout)             (None, 32, 32, 12)   0           conv2d_7[0][0]                   
    __________________________________________________________________________________________________
    concatenate_6 (Concatenate)     (None, 32, 32, 84)   0           concatenate_5[0][0]              
                                                                     dropout_6[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_7 (BatchNor (None, 32, 32, 84)   336         concatenate_6[0][0]              
    __________________________________________________________________________________________________
    activation_7 (Activation)       (None, 32, 32, 84)   0           batch_normalization_7[0][0]      
    __________________________________________________________________________________________________
    conv2d_8 (Conv2D)               (None, 32, 32, 12)   9072        activation_7[0][0]               
    __________________________________________________________________________________________________
    dropout_7 (Dropout)             (None, 32, 32, 12)   0           conv2d_8[0][0]                   
    __________________________________________________________________________________________________
    concatenate_7 (Concatenate)     (None, 32, 32, 96)   0           concatenate_6[0][0]              
                                                                     dropout_7[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_8 (BatchNor (None, 32, 32, 96)   384         concatenate_7[0][0]              
    __________________________________________________________________________________________________
    activation_8 (Activation)       (None, 32, 32, 96)   0           batch_normalization_8[0][0]      
    __________________________________________________________________________________________________
    conv2d_9 (Conv2D)               (None, 32, 32, 12)   10368       activation_8[0][0]               
    __________________________________________________________________________________________________
    dropout_8 (Dropout)             (None, 32, 32, 12)   0           conv2d_9[0][0]                   
    __________________________________________________________________________________________________
    concatenate_8 (Concatenate)     (None, 32, 32, 108)  0           concatenate_7[0][0]              
                                                                     dropout_8[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_9 (BatchNor (None, 32, 32, 108)  432         concatenate_8[0][0]              
    __________________________________________________________________________________________________
    activation_9 (Activation)       (None, 32, 32, 108)  0           batch_normalization_9[0][0]      
    __________________________________________________________________________________________________
    conv2d_10 (Conv2D)              (None, 32, 32, 12)   11664       activation_9[0][0]               
    __________________________________________________________________________________________________
    dropout_9 (Dropout)             (None, 32, 32, 12)   0           conv2d_10[0][0]                  
    __________________________________________________________________________________________________
    concatenate_9 (Concatenate)     (None, 32, 32, 120)  0           concatenate_8[0][0]              
                                                                     dropout_9[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_10 (BatchNo (None, 32, 32, 120)  480         concatenate_9[0][0]              
    __________________________________________________________________________________________________
    activation_10 (Activation)      (None, 32, 32, 120)  0           batch_normalization_10[0][0]     
    __________________________________________________________________________________________________
    conv2d_11 (Conv2D)              (None, 32, 32, 12)   12960       activation_10[0][0]              
    __________________________________________________________________________________________________
    dropout_10 (Dropout)            (None, 32, 32, 12)   0           conv2d_11[0][0]                  
    __________________________________________________________________________________________________
    concatenate_10 (Concatenate)    (None, 32, 32, 132)  0           concatenate_9[0][0]              
                                                                     dropout_10[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_11 (BatchNo (None, 32, 32, 132)  528         concatenate_10[0][0]             
    __________________________________________________________________________________________________
    activation_11 (Activation)      (None, 32, 32, 132)  0           batch_normalization_11[0][0]     
    __________________________________________________________________________________________________
    conv2d_12 (Conv2D)              (None, 32, 32, 12)   14256       activation_11[0][0]              
    __________________________________________________________________________________________________
    dropout_11 (Dropout)            (None, 32, 32, 12)   0           conv2d_12[0][0]                  
    __________________________________________________________________________________________________
    concatenate_11 (Concatenate)    (None, 32, 32, 144)  0           concatenate_10[0][0]             
                                                                     dropout_11[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_12 (BatchNo (None, 32, 32, 144)  576         concatenate_11[0][0]             
    __________________________________________________________________________________________________
    activation_12 (Activation)      (None, 32, 32, 144)  0           batch_normalization_12[0][0]     
    __________________________________________________________________________________________________
    conv2d_13 (Conv2D)              (None, 32, 32, 12)   15552       activation_12[0][0]              
    __________________________________________________________________________________________________
    dropout_12 (Dropout)            (None, 32, 32, 12)   0           conv2d_13[0][0]                  
    __________________________________________________________________________________________________
    concatenate_12 (Concatenate)    (None, 32, 32, 156)  0           concatenate_11[0][0]             
                                                                     dropout_12[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_13 (BatchNo (None, 32, 32, 156)  624         concatenate_12[0][0]             
    __________________________________________________________________________________________________
    activation_13 (Activation)      (None, 32, 32, 156)  0           batch_normalization_13[0][0]     
    __________________________________________________________________________________________________
    conv2d_14 (Conv2D)              (None, 32, 32, 12)   16848       activation_13[0][0]              
    __________________________________________________________________________________________________
    dropout_13 (Dropout)            (None, 32, 32, 12)   0           conv2d_14[0][0]                  
    __________________________________________________________________________________________________
    concatenate_13 (Concatenate)    (None, 32, 32, 168)  0           concatenate_12[0][0]             
                                                                     dropout_13[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_14 (BatchNo (None, 32, 32, 168)  672         concatenate_13[0][0]             
    __________________________________________________________________________________________________
    activation_14 (Activation)      (None, 32, 32, 168)  0           batch_normalization_14[0][0]     
    __________________________________________________________________________________________________
    conv2d_15 (Conv2D)              (None, 32, 32, 12)   18144       activation_14[0][0]              
    __________________________________________________________________________________________________
    dropout_14 (Dropout)            (None, 32, 32, 12)   0           conv2d_15[0][0]                  
    __________________________________________________________________________________________________
    concatenate_14 (Concatenate)    (None, 32, 32, 180)  0           concatenate_13[0][0]             
                                                                     dropout_14[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_15 (BatchNo (None, 32, 32, 180)  720         concatenate_14[0][0]             
    __________________________________________________________________________________________________
    activation_15 (Activation)      (None, 32, 32, 180)  0           batch_normalization_15[0][0]     
    __________________________________________________________________________________________________
    conv2d_16 (Conv2D)              (None, 32, 32, 12)   19440       activation_15[0][0]              
    __________________________________________________________________________________________________
    dropout_15 (Dropout)            (None, 32, 32, 12)   0           conv2d_16[0][0]                  
    __________________________________________________________________________________________________
    concatenate_15 (Concatenate)    (None, 32, 32, 192)  0           concatenate_14[0][0]             
                                                                     dropout_15[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_16 (BatchNo (None, 32, 32, 192)  768         concatenate_15[0][0]             
    __________________________________________________________________________________________________
    activation_16 (Activation)      (None, 32, 32, 192)  0           batch_normalization_16[0][0]     
    __________________________________________________________________________________________________
    conv2d_17 (Conv2D)              (None, 32, 32, 12)   20736       activation_16[0][0]              
    __________________________________________________________________________________________________
    dropout_16 (Dropout)            (None, 32, 32, 12)   0           conv2d_17[0][0]                  
    __________________________________________________________________________________________________
    concatenate_16 (Concatenate)    (None, 32, 32, 204)  0           concatenate_15[0][0]             
                                                                     dropout_16[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_17 (BatchNo (None, 32, 32, 204)  816         concatenate_16[0][0]             
    __________________________________________________________________________________________________
    activation_17 (Activation)      (None, 32, 32, 204)  0           batch_normalization_17[0][0]     
    __________________________________________________________________________________________________
    conv2d_18 (Conv2D)              (None, 32, 32, 12)   2448        activation_17[0][0]              
    __________________________________________________________________________________________________
    dropout_17 (Dropout)            (None, 32, 32, 12)   0           conv2d_18[0][0]                  
    __________________________________________________________________________________________________
    average_pooling2d_1 (AveragePoo (None, 16, 16, 12)   0           dropout_17[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_18 (BatchNo (None, 16, 16, 12)   48          average_pooling2d_1[0][0]        
    __________________________________________________________________________________________________
    activation_18 (Activation)      (None, 16, 16, 12)   0           batch_normalization_18[0][0]     
    __________________________________________________________________________________________________
    conv2d_19 (Conv2D)              (None, 16, 16, 12)   1296        activation_18[0][0]              
    __________________________________________________________________________________________________
    dropout_18 (Dropout)            (None, 16, 16, 12)   0           conv2d_19[0][0]                  
    __________________________________________________________________________________________________
    concatenate_17 (Concatenate)    (None, 16, 16, 24)   0           average_pooling2d_1[0][0]        
                                                                     dropout_18[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_19 (BatchNo (None, 16, 16, 24)   96          concatenate_17[0][0]             
    __________________________________________________________________________________________________
    activation_19 (Activation)      (None, 16, 16, 24)   0           batch_normalization_19[0][0]     
    __________________________________________________________________________________________________
    conv2d_20 (Conv2D)              (None, 16, 16, 12)   2592        activation_19[0][0]              
    __________________________________________________________________________________________________
    dropout_19 (Dropout)            (None, 16, 16, 12)   0           conv2d_20[0][0]                  
    __________________________________________________________________________________________________
    concatenate_18 (Concatenate)    (None, 16, 16, 36)   0           concatenate_17[0][0]             
                                                                     dropout_19[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_20 (BatchNo (None, 16, 16, 36)   144         concatenate_18[0][0]             
    __________________________________________________________________________________________________
    activation_20 (Activation)      (None, 16, 16, 36)   0           batch_normalization_20[0][0]     
    __________________________________________________________________________________________________
    conv2d_21 (Conv2D)              (None, 16, 16, 12)   3888        activation_20[0][0]              
    __________________________________________________________________________________________________
    dropout_20 (Dropout)            (None, 16, 16, 12)   0           conv2d_21[0][0]                  
    __________________________________________________________________________________________________
    concatenate_19 (Concatenate)    (None, 16, 16, 48)   0           concatenate_18[0][0]             
                                                                     dropout_20[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_21 (BatchNo (None, 16, 16, 48)   192         concatenate_19[0][0]             
    __________________________________________________________________________________________________
    activation_21 (Activation)      (None, 16, 16, 48)   0           batch_normalization_21[0][0]     
    __________________________________________________________________________________________________
    conv2d_22 (Conv2D)              (None, 16, 16, 12)   5184        activation_21[0][0]              
    __________________________________________________________________________________________________
    dropout_21 (Dropout)            (None, 16, 16, 12)   0           conv2d_22[0][0]                  
    __________________________________________________________________________________________________
    concatenate_20 (Concatenate)    (None, 16, 16, 60)   0           concatenate_19[0][0]             
                                                                     dropout_21[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_22 (BatchNo (None, 16, 16, 60)   240         concatenate_20[0][0]             
    __________________________________________________________________________________________________
    activation_22 (Activation)      (None, 16, 16, 60)   0           batch_normalization_22[0][0]     
    __________________________________________________________________________________________________
    conv2d_23 (Conv2D)              (None, 16, 16, 12)   6480        activation_22[0][0]              
    __________________________________________________________________________________________________
    dropout_22 (Dropout)            (None, 16, 16, 12)   0           conv2d_23[0][0]                  
    __________________________________________________________________________________________________
    concatenate_21 (Concatenate)    (None, 16, 16, 72)   0           concatenate_20[0][0]             
                                                                     dropout_22[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_23 (BatchNo (None, 16, 16, 72)   288         concatenate_21[0][0]             
    __________________________________________________________________________________________________
    activation_23 (Activation)      (None, 16, 16, 72)   0           batch_normalization_23[0][0]     
    __________________________________________________________________________________________________
    conv2d_24 (Conv2D)              (None, 16, 16, 12)   7776        activation_23[0][0]              
    __________________________________________________________________________________________________
    dropout_23 (Dropout)            (None, 16, 16, 12)   0           conv2d_24[0][0]                  
    __________________________________________________________________________________________________
    concatenate_22 (Concatenate)    (None, 16, 16, 84)   0           concatenate_21[0][0]             
                                                                     dropout_23[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_24 (BatchNo (None, 16, 16, 84)   336         concatenate_22[0][0]             
    __________________________________________________________________________________________________
    activation_24 (Activation)      (None, 16, 16, 84)   0           batch_normalization_24[0][0]     
    __________________________________________________________________________________________________
    conv2d_25 (Conv2D)              (None, 16, 16, 12)   9072        activation_24[0][0]              
    __________________________________________________________________________________________________
    dropout_24 (Dropout)            (None, 16, 16, 12)   0           conv2d_25[0][0]                  
    __________________________________________________________________________________________________
    concatenate_23 (Concatenate)    (None, 16, 16, 96)   0           concatenate_22[0][0]             
                                                                     dropout_24[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_25 (BatchNo (None, 16, 16, 96)   384         concatenate_23[0][0]             
    __________________________________________________________________________________________________
    activation_25 (Activation)      (None, 16, 16, 96)   0           batch_normalization_25[0][0]     
    __________________________________________________________________________________________________
    conv2d_26 (Conv2D)              (None, 16, 16, 12)   10368       activation_25[0][0]              
    __________________________________________________________________________________________________
    dropout_25 (Dropout)            (None, 16, 16, 12)   0           conv2d_26[0][0]                  
    __________________________________________________________________________________________________
    concatenate_24 (Concatenate)    (None, 16, 16, 108)  0           concatenate_23[0][0]             
                                                                     dropout_25[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_26 (BatchNo (None, 16, 16, 108)  432         concatenate_24[0][0]             
    __________________________________________________________________________________________________
    activation_26 (Activation)      (None, 16, 16, 108)  0           batch_normalization_26[0][0]     
    __________________________________________________________________________________________________
    conv2d_27 (Conv2D)              (None, 16, 16, 12)   11664       activation_26[0][0]              
    __________________________________________________________________________________________________
    dropout_26 (Dropout)            (None, 16, 16, 12)   0           conv2d_27[0][0]                  
    __________________________________________________________________________________________________
    concatenate_25 (Concatenate)    (None, 16, 16, 120)  0           concatenate_24[0][0]             
                                                                     dropout_26[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_27 (BatchNo (None, 16, 16, 120)  480         concatenate_25[0][0]             
    __________________________________________________________________________________________________
    activation_27 (Activation)      (None, 16, 16, 120)  0           batch_normalization_27[0][0]     
    __________________________________________________________________________________________________
    conv2d_28 (Conv2D)              (None, 16, 16, 12)   12960       activation_27[0][0]              
    __________________________________________________________________________________________________
    dropout_27 (Dropout)            (None, 16, 16, 12)   0           conv2d_28[0][0]                  
    __________________________________________________________________________________________________
    concatenate_26 (Concatenate)    (None, 16, 16, 132)  0           concatenate_25[0][0]             
                                                                     dropout_27[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_28 (BatchNo (None, 16, 16, 132)  528         concatenate_26[0][0]             
    __________________________________________________________________________________________________
    activation_28 (Activation)      (None, 16, 16, 132)  0           batch_normalization_28[0][0]     
    __________________________________________________________________________________________________
    conv2d_29 (Conv2D)              (None, 16, 16, 12)   14256       activation_28[0][0]              
    __________________________________________________________________________________________________
    dropout_28 (Dropout)            (None, 16, 16, 12)   0           conv2d_29[0][0]                  
    __________________________________________________________________________________________________
    concatenate_27 (Concatenate)    (None, 16, 16, 144)  0           concatenate_26[0][0]             
                                                                     dropout_28[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_29 (BatchNo (None, 16, 16, 144)  576         concatenate_27[0][0]             
    __________________________________________________________________________________________________
    activation_29 (Activation)      (None, 16, 16, 144)  0           batch_normalization_29[0][0]     
    __________________________________________________________________________________________________
    conv2d_30 (Conv2D)              (None, 16, 16, 12)   15552       activation_29[0][0]              
    __________________________________________________________________________________________________
    dropout_29 (Dropout)            (None, 16, 16, 12)   0           conv2d_30[0][0]                  
    __________________________________________________________________________________________________
    concatenate_28 (Concatenate)    (None, 16, 16, 156)  0           concatenate_27[0][0]             
                                                                     dropout_29[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_30 (BatchNo (None, 16, 16, 156)  624         concatenate_28[0][0]             
    __________________________________________________________________________________________________
    activation_30 (Activation)      (None, 16, 16, 156)  0           batch_normalization_30[0][0]     
    __________________________________________________________________________________________________
    conv2d_31 (Conv2D)              (None, 16, 16, 12)   16848       activation_30[0][0]              
    __________________________________________________________________________________________________
    dropout_30 (Dropout)            (None, 16, 16, 12)   0           conv2d_31[0][0]                  
    __________________________________________________________________________________________________
    concatenate_29 (Concatenate)    (None, 16, 16, 168)  0           concatenate_28[0][0]             
                                                                     dropout_30[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_31 (BatchNo (None, 16, 16, 168)  672         concatenate_29[0][0]             
    __________________________________________________________________________________________________
    activation_31 (Activation)      (None, 16, 16, 168)  0           batch_normalization_31[0][0]     
    __________________________________________________________________________________________________
    conv2d_32 (Conv2D)              (None, 16, 16, 12)   18144       activation_31[0][0]              
    __________________________________________________________________________________________________
    dropout_31 (Dropout)            (None, 16, 16, 12)   0           conv2d_32[0][0]                  
    __________________________________________________________________________________________________
    concatenate_30 (Concatenate)    (None, 16, 16, 180)  0           concatenate_29[0][0]             
                                                                     dropout_31[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_32 (BatchNo (None, 16, 16, 180)  720         concatenate_30[0][0]             
    __________________________________________________________________________________________________
    activation_32 (Activation)      (None, 16, 16, 180)  0           batch_normalization_32[0][0]     
    __________________________________________________________________________________________________
    conv2d_33 (Conv2D)              (None, 16, 16, 12)   19440       activation_32[0][0]              
    __________________________________________________________________________________________________
    dropout_32 (Dropout)            (None, 16, 16, 12)   0           conv2d_33[0][0]                  
    __________________________________________________________________________________________________
    concatenate_31 (Concatenate)    (None, 16, 16, 192)  0           concatenate_30[0][0]             
                                                                     dropout_32[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_33 (BatchNo (None, 16, 16, 192)  768         concatenate_31[0][0]             
    __________________________________________________________________________________________________
    activation_33 (Activation)      (None, 16, 16, 192)  0           batch_normalization_33[0][0]     
    __________________________________________________________________________________________________
    conv2d_34 (Conv2D)              (None, 16, 16, 12)   20736       activation_33[0][0]              
    __________________________________________________________________________________________________
    dropout_33 (Dropout)            (None, 16, 16, 12)   0           conv2d_34[0][0]                  
    __________________________________________________________________________________________________
    concatenate_32 (Concatenate)    (None, 16, 16, 204)  0           concatenate_31[0][0]             
                                                                     dropout_33[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_34 (BatchNo (None, 16, 16, 204)  816         concatenate_32[0][0]             
    __________________________________________________________________________________________________
    activation_34 (Activation)      (None, 16, 16, 204)  0           batch_normalization_34[0][0]     
    __________________________________________________________________________________________________
    conv2d_35 (Conv2D)              (None, 16, 16, 12)   2448        activation_34[0][0]              
    __________________________________________________________________________________________________
    dropout_34 (Dropout)            (None, 16, 16, 12)   0           conv2d_35[0][0]                  
    __________________________________________________________________________________________________
    average_pooling2d_2 (AveragePoo (None, 8, 8, 12)     0           dropout_34[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_35 (BatchNo (None, 8, 8, 12)     48          average_pooling2d_2[0][0]        
    __________________________________________________________________________________________________
    activation_35 (Activation)      (None, 8, 8, 12)     0           batch_normalization_35[0][0]     
    __________________________________________________________________________________________________
    conv2d_36 (Conv2D)              (None, 8, 8, 12)     1296        activation_35[0][0]              
    __________________________________________________________________________________________________
    dropout_35 (Dropout)            (None, 8, 8, 12)     0           conv2d_36[0][0]                  
    __________________________________________________________________________________________________
    concatenate_33 (Concatenate)    (None, 8, 8, 24)     0           average_pooling2d_2[0][0]        
                                                                     dropout_35[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_36 (BatchNo (None, 8, 8, 24)     96          concatenate_33[0][0]             
    __________________________________________________________________________________________________
    activation_36 (Activation)      (None, 8, 8, 24)     0           batch_normalization_36[0][0]     
    __________________________________________________________________________________________________
    conv2d_37 (Conv2D)              (None, 8, 8, 12)     2592        activation_36[0][0]              
    __________________________________________________________________________________________________
    dropout_36 (Dropout)            (None, 8, 8, 12)     0           conv2d_37[0][0]                  
    __________________________________________________________________________________________________
    concatenate_34 (Concatenate)    (None, 8, 8, 36)     0           concatenate_33[0][0]             
                                                                     dropout_36[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_37 (BatchNo (None, 8, 8, 36)     144         concatenate_34[0][0]             
    __________________________________________________________________________________________________
    activation_37 (Activation)      (None, 8, 8, 36)     0           batch_normalization_37[0][0]     
    __________________________________________________________________________________________________
    conv2d_38 (Conv2D)              (None, 8, 8, 12)     3888        activation_37[0][0]              
    __________________________________________________________________________________________________
    dropout_37 (Dropout)            (None, 8, 8, 12)     0           conv2d_38[0][0]                  
    __________________________________________________________________________________________________
    concatenate_35 (Concatenate)    (None, 8, 8, 48)     0           concatenate_34[0][0]             
                                                                     dropout_37[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_38 (BatchNo (None, 8, 8, 48)     192         concatenate_35[0][0]             
    __________________________________________________________________________________________________
    activation_38 (Activation)      (None, 8, 8, 48)     0           batch_normalization_38[0][0]     
    __________________________________________________________________________________________________
    conv2d_39 (Conv2D)              (None, 8, 8, 12)     5184        activation_38[0][0]              
    __________________________________________________________________________________________________
    dropout_38 (Dropout)            (None, 8, 8, 12)     0           conv2d_39[0][0]                  
    __________________________________________________________________________________________________
    concatenate_36 (Concatenate)    (None, 8, 8, 60)     0           concatenate_35[0][0]             
                                                                     dropout_38[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_39 (BatchNo (None, 8, 8, 60)     240         concatenate_36[0][0]             
    __________________________________________________________________________________________________
    activation_39 (Activation)      (None, 8, 8, 60)     0           batch_normalization_39[0][0]     
    __________________________________________________________________________________________________
    conv2d_40 (Conv2D)              (None, 8, 8, 12)     6480        activation_39[0][0]              
    __________________________________________________________________________________________________
    dropout_39 (Dropout)            (None, 8, 8, 12)     0           conv2d_40[0][0]                  
    __________________________________________________________________________________________________
    concatenate_37 (Concatenate)    (None, 8, 8, 72)     0           concatenate_36[0][0]             
                                                                     dropout_39[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_40 (BatchNo (None, 8, 8, 72)     288         concatenate_37[0][0]             
    __________________________________________________________________________________________________
    activation_40 (Activation)      (None, 8, 8, 72)     0           batch_normalization_40[0][0]     
    __________________________________________________________________________________________________
    conv2d_41 (Conv2D)              (None, 8, 8, 12)     7776        activation_40[0][0]              
    __________________________________________________________________________________________________
    dropout_40 (Dropout)            (None, 8, 8, 12)     0           conv2d_41[0][0]                  
    __________________________________________________________________________________________________
    concatenate_38 (Concatenate)    (None, 8, 8, 84)     0           concatenate_37[0][0]             
                                                                     dropout_40[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_41 (BatchNo (None, 8, 8, 84)     336         concatenate_38[0][0]             
    __________________________________________________________________________________________________
    activation_41 (Activation)      (None, 8, 8, 84)     0           batch_normalization_41[0][0]     
    __________________________________________________________________________________________________
    conv2d_42 (Conv2D)              (None, 8, 8, 12)     9072        activation_41[0][0]              
    __________________________________________________________________________________________________
    dropout_41 (Dropout)            (None, 8, 8, 12)     0           conv2d_42[0][0]                  
    __________________________________________________________________________________________________
    concatenate_39 (Concatenate)    (None, 8, 8, 96)     0           concatenate_38[0][0]             
                                                                     dropout_41[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_42 (BatchNo (None, 8, 8, 96)     384         concatenate_39[0][0]             
    __________________________________________________________________________________________________
    activation_42 (Activation)      (None, 8, 8, 96)     0           batch_normalization_42[0][0]     
    __________________________________________________________________________________________________
    conv2d_43 (Conv2D)              (None, 8, 8, 12)     10368       activation_42[0][0]              
    __________________________________________________________________________________________________
    dropout_42 (Dropout)            (None, 8, 8, 12)     0           conv2d_43[0][0]                  
    __________________________________________________________________________________________________
    concatenate_40 (Concatenate)    (None, 8, 8, 108)    0           concatenate_39[0][0]             
                                                                     dropout_42[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_43 (BatchNo (None, 8, 8, 108)    432         concatenate_40[0][0]             
    __________________________________________________________________________________________________
    activation_43 (Activation)      (None, 8, 8, 108)    0           batch_normalization_43[0][0]     
    __________________________________________________________________________________________________
    conv2d_44 (Conv2D)              (None, 8, 8, 12)     11664       activation_43[0][0]              
    __________________________________________________________________________________________________
    dropout_43 (Dropout)            (None, 8, 8, 12)     0           conv2d_44[0][0]                  
    __________________________________________________________________________________________________
    concatenate_41 (Concatenate)    (None, 8, 8, 120)    0           concatenate_40[0][0]             
                                                                     dropout_43[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_44 (BatchNo (None, 8, 8, 120)    480         concatenate_41[0][0]             
    __________________________________________________________________________________________________
    activation_44 (Activation)      (None, 8, 8, 120)    0           batch_normalization_44[0][0]     
    __________________________________________________________________________________________________
    conv2d_45 (Conv2D)              (None, 8, 8, 12)     12960       activation_44[0][0]              
    __________________________________________________________________________________________________
    dropout_44 (Dropout)            (None, 8, 8, 12)     0           conv2d_45[0][0]                  
    __________________________________________________________________________________________________
    concatenate_42 (Concatenate)    (None, 8, 8, 132)    0           concatenate_41[0][0]             
                                                                     dropout_44[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_45 (BatchNo (None, 8, 8, 132)    528         concatenate_42[0][0]             
    __________________________________________________________________________________________________
    activation_45 (Activation)      (None, 8, 8, 132)    0           batch_normalization_45[0][0]     
    __________________________________________________________________________________________________
    conv2d_46 (Conv2D)              (None, 8, 8, 12)     14256       activation_45[0][0]              
    __________________________________________________________________________________________________
    dropout_45 (Dropout)            (None, 8, 8, 12)     0           conv2d_46[0][0]                  
    __________________________________________________________________________________________________
    concatenate_43 (Concatenate)    (None, 8, 8, 144)    0           concatenate_42[0][0]             
                                                                     dropout_45[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_46 (BatchNo (None, 8, 8, 144)    576         concatenate_43[0][0]             
    __________________________________________________________________________________________________
    activation_46 (Activation)      (None, 8, 8, 144)    0           batch_normalization_46[0][0]     
    __________________________________________________________________________________________________
    conv2d_47 (Conv2D)              (None, 8, 8, 12)     15552       activation_46[0][0]              
    __________________________________________________________________________________________________
    dropout_46 (Dropout)            (None, 8, 8, 12)     0           conv2d_47[0][0]                  
    __________________________________________________________________________________________________
    concatenate_44 (Concatenate)    (None, 8, 8, 156)    0           concatenate_43[0][0]             
                                                                     dropout_46[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_47 (BatchNo (None, 8, 8, 156)    624         concatenate_44[0][0]             
    __________________________________________________________________________________________________
    activation_47 (Activation)      (None, 8, 8, 156)    0           batch_normalization_47[0][0]     
    __________________________________________________________________________________________________
    conv2d_48 (Conv2D)              (None, 8, 8, 12)     16848       activation_47[0][0]              
    __________________________________________________________________________________________________
    dropout_47 (Dropout)            (None, 8, 8, 12)     0           conv2d_48[0][0]                  
    __________________________________________________________________________________________________
    concatenate_45 (Concatenate)    (None, 8, 8, 168)    0           concatenate_44[0][0]             
                                                                     dropout_47[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_48 (BatchNo (None, 8, 8, 168)    672         concatenate_45[0][0]             
    __________________________________________________________________________________________________
    activation_48 (Activation)      (None, 8, 8, 168)    0           batch_normalization_48[0][0]     
    __________________________________________________________________________________________________
    conv2d_49 (Conv2D)              (None, 8, 8, 12)     18144       activation_48[0][0]              
    __________________________________________________________________________________________________
    dropout_48 (Dropout)            (None, 8, 8, 12)     0           conv2d_49[0][0]                  
    __________________________________________________________________________________________________
    concatenate_46 (Concatenate)    (None, 8, 8, 180)    0           concatenate_45[0][0]             
                                                                     dropout_48[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_49 (BatchNo (None, 8, 8, 180)    720         concatenate_46[0][0]             
    __________________________________________________________________________________________________
    activation_49 (Activation)      (None, 8, 8, 180)    0           batch_normalization_49[0][0]     
    __________________________________________________________________________________________________
    conv2d_50 (Conv2D)              (None, 8, 8, 12)     19440       activation_49[0][0]              
    __________________________________________________________________________________________________
    dropout_49 (Dropout)            (None, 8, 8, 12)     0           conv2d_50[0][0]                  
    __________________________________________________________________________________________________
    concatenate_47 (Concatenate)    (None, 8, 8, 192)    0           concatenate_46[0][0]             
                                                                     dropout_49[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_50 (BatchNo (None, 8, 8, 192)    768         concatenate_47[0][0]             
    __________________________________________________________________________________________________
    activation_50 (Activation)      (None, 8, 8, 192)    0           batch_normalization_50[0][0]     
    __________________________________________________________________________________________________
    conv2d_51 (Conv2D)              (None, 8, 8, 12)     20736       activation_50[0][0]              
    __________________________________________________________________________________________________
    dropout_50 (Dropout)            (None, 8, 8, 12)     0           conv2d_51[0][0]                  
    __________________________________________________________________________________________________
    concatenate_48 (Concatenate)    (None, 8, 8, 204)    0           concatenate_47[0][0]             
                                                                     dropout_50[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_51 (BatchNo (None, 8, 8, 204)    816         concatenate_48[0][0]             
    __________________________________________________________________________________________________
    activation_51 (Activation)      (None, 8, 8, 204)    0           batch_normalization_51[0][0]     
    __________________________________________________________________________________________________
    conv2d_52 (Conv2D)              (None, 8, 8, 12)     2448        activation_51[0][0]              
    __________________________________________________________________________________________________
    dropout_51 (Dropout)            (None, 8, 8, 12)     0           conv2d_52[0][0]                  
    __________________________________________________________________________________________________
    average_pooling2d_3 (AveragePoo (None, 4, 4, 12)     0           dropout_51[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_52 (BatchNo (None, 4, 4, 12)     48          average_pooling2d_3[0][0]        
    __________________________________________________________________________________________________
    activation_52 (Activation)      (None, 4, 4, 12)     0           batch_normalization_52[0][0]     
    __________________________________________________________________________________________________
    conv2d_53 (Conv2D)              (None, 4, 4, 12)     1296        activation_52[0][0]              
    __________________________________________________________________________________________________
    dropout_52 (Dropout)            (None, 4, 4, 12)     0           conv2d_53[0][0]                  
    __________________________________________________________________________________________________
    concatenate_49 (Concatenate)    (None, 4, 4, 24)     0           average_pooling2d_3[0][0]        
                                                                     dropout_52[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_53 (BatchNo (None, 4, 4, 24)     96          concatenate_49[0][0]             
    __________________________________________________________________________________________________
    activation_53 (Activation)      (None, 4, 4, 24)     0           batch_normalization_53[0][0]     
    __________________________________________________________________________________________________
    conv2d_54 (Conv2D)              (None, 4, 4, 12)     2592        activation_53[0][0]              
    __________________________________________________________________________________________________
    dropout_53 (Dropout)            (None, 4, 4, 12)     0           conv2d_54[0][0]                  
    __________________________________________________________________________________________________
    concatenate_50 (Concatenate)    (None, 4, 4, 36)     0           concatenate_49[0][0]             
                                                                     dropout_53[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_54 (BatchNo (None, 4, 4, 36)     144         concatenate_50[0][0]             
    __________________________________________________________________________________________________
    activation_54 (Activation)      (None, 4, 4, 36)     0           batch_normalization_54[0][0]     
    __________________________________________________________________________________________________
    conv2d_55 (Conv2D)              (None, 4, 4, 12)     3888        activation_54[0][0]              
    __________________________________________________________________________________________________
    dropout_54 (Dropout)            (None, 4, 4, 12)     0           conv2d_55[0][0]                  
    __________________________________________________________________________________________________
    concatenate_51 (Concatenate)    (None, 4, 4, 48)     0           concatenate_50[0][0]             
                                                                     dropout_54[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_55 (BatchNo (None, 4, 4, 48)     192         concatenate_51[0][0]             
    __________________________________________________________________________________________________
    activation_55 (Activation)      (None, 4, 4, 48)     0           batch_normalization_55[0][0]     
    __________________________________________________________________________________________________
    conv2d_56 (Conv2D)              (None, 4, 4, 12)     5184        activation_55[0][0]              
    __________________________________________________________________________________________________
    dropout_55 (Dropout)            (None, 4, 4, 12)     0           conv2d_56[0][0]                  
    __________________________________________________________________________________________________
    concatenate_52 (Concatenate)    (None, 4, 4, 60)     0           concatenate_51[0][0]             
                                                                     dropout_55[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_56 (BatchNo (None, 4, 4, 60)     240         concatenate_52[0][0]             
    __________________________________________________________________________________________________
    activation_56 (Activation)      (None, 4, 4, 60)     0           batch_normalization_56[0][0]     
    __________________________________________________________________________________________________
    conv2d_57 (Conv2D)              (None, 4, 4, 12)     6480        activation_56[0][0]              
    __________________________________________________________________________________________________
    dropout_56 (Dropout)            (None, 4, 4, 12)     0           conv2d_57[0][0]                  
    __________________________________________________________________________________________________
    concatenate_53 (Concatenate)    (None, 4, 4, 72)     0           concatenate_52[0][0]             
                                                                     dropout_56[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_57 (BatchNo (None, 4, 4, 72)     288         concatenate_53[0][0]             
    __________________________________________________________________________________________________
    activation_57 (Activation)      (None, 4, 4, 72)     0           batch_normalization_57[0][0]     
    __________________________________________________________________________________________________
    conv2d_58 (Conv2D)              (None, 4, 4, 12)     7776        activation_57[0][0]              
    __________________________________________________________________________________________________
    dropout_57 (Dropout)            (None, 4, 4, 12)     0           conv2d_58[0][0]                  
    __________________________________________________________________________________________________
    concatenate_54 (Concatenate)    (None, 4, 4, 84)     0           concatenate_53[0][0]             
                                                                     dropout_57[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_58 (BatchNo (None, 4, 4, 84)     336         concatenate_54[0][0]             
    __________________________________________________________________________________________________
    activation_58 (Activation)      (None, 4, 4, 84)     0           batch_normalization_58[0][0]     
    __________________________________________________________________________________________________
    conv2d_59 (Conv2D)              (None, 4, 4, 12)     9072        activation_58[0][0]              
    __________________________________________________________________________________________________
    dropout_58 (Dropout)            (None, 4, 4, 12)     0           conv2d_59[0][0]                  
    __________________________________________________________________________________________________
    concatenate_55 (Concatenate)    (None, 4, 4, 96)     0           concatenate_54[0][0]             
                                                                     dropout_58[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_59 (BatchNo (None, 4, 4, 96)     384         concatenate_55[0][0]             
    __________________________________________________________________________________________________
    activation_59 (Activation)      (None, 4, 4, 96)     0           batch_normalization_59[0][0]     
    __________________________________________________________________________________________________
    conv2d_60 (Conv2D)              (None, 4, 4, 12)     10368       activation_59[0][0]              
    __________________________________________________________________________________________________
    dropout_59 (Dropout)            (None, 4, 4, 12)     0           conv2d_60[0][0]                  
    __________________________________________________________________________________________________
    concatenate_56 (Concatenate)    (None, 4, 4, 108)    0           concatenate_55[0][0]             
                                                                     dropout_59[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_60 (BatchNo (None, 4, 4, 108)    432         concatenate_56[0][0]             
    __________________________________________________________________________________________________
    activation_60 (Activation)      (None, 4, 4, 108)    0           batch_normalization_60[0][0]     
    __________________________________________________________________________________________________
    conv2d_61 (Conv2D)              (None, 4, 4, 12)     11664       activation_60[0][0]              
    __________________________________________________________________________________________________
    dropout_60 (Dropout)            (None, 4, 4, 12)     0           conv2d_61[0][0]                  
    __________________________________________________________________________________________________
    concatenate_57 (Concatenate)    (None, 4, 4, 120)    0           concatenate_56[0][0]             
                                                                     dropout_60[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_61 (BatchNo (None, 4, 4, 120)    480         concatenate_57[0][0]             
    __________________________________________________________________________________________________
    activation_61 (Activation)      (None, 4, 4, 120)    0           batch_normalization_61[0][0]     
    __________________________________________________________________________________________________
    conv2d_62 (Conv2D)              (None, 4, 4, 12)     12960       activation_61[0][0]              
    __________________________________________________________________________________________________
    dropout_61 (Dropout)            (None, 4, 4, 12)     0           conv2d_62[0][0]                  
    __________________________________________________________________________________________________
    concatenate_58 (Concatenate)    (None, 4, 4, 132)    0           concatenate_57[0][0]             
                                                                     dropout_61[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_62 (BatchNo (None, 4, 4, 132)    528         concatenate_58[0][0]             
    __________________________________________________________________________________________________
    activation_62 (Activation)      (None, 4, 4, 132)    0           batch_normalization_62[0][0]     
    __________________________________________________________________________________________________
    conv2d_63 (Conv2D)              (None, 4, 4, 12)     14256       activation_62[0][0]              
    __________________________________________________________________________________________________
    dropout_62 (Dropout)            (None, 4, 4, 12)     0           conv2d_63[0][0]                  
    __________________________________________________________________________________________________
    concatenate_59 (Concatenate)    (None, 4, 4, 144)    0           concatenate_58[0][0]             
                                                                     dropout_62[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_63 (BatchNo (None, 4, 4, 144)    576         concatenate_59[0][0]             
    __________________________________________________________________________________________________
    activation_63 (Activation)      (None, 4, 4, 144)    0           batch_normalization_63[0][0]     
    __________________________________________________________________________________________________
    conv2d_64 (Conv2D)              (None, 4, 4, 12)     15552       activation_63[0][0]              
    __________________________________________________________________________________________________
    dropout_63 (Dropout)            (None, 4, 4, 12)     0           conv2d_64[0][0]                  
    __________________________________________________________________________________________________
    concatenate_60 (Concatenate)    (None, 4, 4, 156)    0           concatenate_59[0][0]             
                                                                     dropout_63[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_64 (BatchNo (None, 4, 4, 156)    624         concatenate_60[0][0]             
    __________________________________________________________________________________________________
    activation_64 (Activation)      (None, 4, 4, 156)    0           batch_normalization_64[0][0]     
    __________________________________________________________________________________________________
    conv2d_65 (Conv2D)              (None, 4, 4, 12)     16848       activation_64[0][0]              
    __________________________________________________________________________________________________
    dropout_64 (Dropout)            (None, 4, 4, 12)     0           conv2d_65[0][0]                  
    __________________________________________________________________________________________________
    concatenate_61 (Concatenate)    (None, 4, 4, 168)    0           concatenate_60[0][0]             
                                                                     dropout_64[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_65 (BatchNo (None, 4, 4, 168)    672         concatenate_61[0][0]             
    __________________________________________________________________________________________________
    activation_65 (Activation)      (None, 4, 4, 168)    0           batch_normalization_65[0][0]     
    __________________________________________________________________________________________________
    conv2d_66 (Conv2D)              (None, 4, 4, 12)     18144       activation_65[0][0]              
    __________________________________________________________________________________________________
    dropout_65 (Dropout)            (None, 4, 4, 12)     0           conv2d_66[0][0]                  
    __________________________________________________________________________________________________
    concatenate_62 (Concatenate)    (None, 4, 4, 180)    0           concatenate_61[0][0]             
                                                                     dropout_65[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_66 (BatchNo (None, 4, 4, 180)    720         concatenate_62[0][0]             
    __________________________________________________________________________________________________
    activation_66 (Activation)      (None, 4, 4, 180)    0           batch_normalization_66[0][0]     
    __________________________________________________________________________________________________
    conv2d_67 (Conv2D)              (None, 4, 4, 12)     19440       activation_66[0][0]              
    __________________________________________________________________________________________________
    dropout_66 (Dropout)            (None, 4, 4, 12)     0           conv2d_67[0][0]                  
    __________________________________________________________________________________________________
    concatenate_63 (Concatenate)    (None, 4, 4, 192)    0           concatenate_62[0][0]             
                                                                     dropout_66[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_67 (BatchNo (None, 4, 4, 192)    768         concatenate_63[0][0]             
    __________________________________________________________________________________________________
    activation_67 (Activation)      (None, 4, 4, 192)    0           batch_normalization_67[0][0]     
    __________________________________________________________________________________________________
    conv2d_68 (Conv2D)              (None, 4, 4, 12)     20736       activation_67[0][0]              
    __________________________________________________________________________________________________
    dropout_67 (Dropout)            (None, 4, 4, 12)     0           conv2d_68[0][0]                  
    __________________________________________________________________________________________________
    concatenate_64 (Concatenate)    (None, 4, 4, 204)    0           concatenate_63[0][0]             
                                                                     dropout_67[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_68 (BatchNo (None, 4, 4, 204)    816         concatenate_64[0][0]             
    __________________________________________________________________________________________________
    activation_68 (Activation)      (None, 4, 4, 204)    0           batch_normalization_68[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_4 (AveragePoo (None, 2, 2, 204)    0           activation_68[0][0]              
    __________________________________________________________________________________________________
    flatten_1 (Flatten)             (None, 816)          0           average_pooling2d_4[0][0]        
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 10)           8170        flatten_1[0][0]                  
    ==================================================================================================
    Total params: 750,238
    Trainable params: 735,550
    Non-trainable params: 14,688
    __________________________________________________________________________________________________



```python
# ak - block 3 SGDRScheduler

schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-1,
                                     steps_per_epoch=2*(x_train.shape[0]//batch_size),
                                     lr_decay=1,
                                     cycle_length=5,
                                     mult_factor=1.5)

# checkpointer = ModelCheckpoint(filepath='v6_2_weights.{epoch:02d}-{val_loss:.2f}--{val_acc:.2f}.hdf5',monitor='val_acc', mode=max, verbose=1, save_best_only=True, save_weights_only=True)



# keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto'
#                               , baseline=None, restore_best_weights=False)

earlystopper = EarlyStopping(monitor='val_loss', min_delta=1, patience=20, verbose=1,mode='min')

```


```python
# determine Loss function and Optimizer
# decay=10e-4,momentum=0.9
sgd = SGD(momentum=1)

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```


```python
# ak - block 4 - image aug

# we can compare the performance with or without data augmentation
data_augmentation = True
callbacks_list=[schedule,earlystopper]
start = time.time()
if not data_augmentation:
    print('Not using data augmentation.')
    model_info = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=callbacks_list
        )
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by dataset std
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in 0 to 180 degrees
        width_shift_range=0.1,  # randomly shift images horizontally
        height_shift_range=0.1,  # randomly shift images vertically
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    
    model_info = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=3*(x_train.shape[0]//batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks_list
                       )

end = time.time()
print ("Model took %0.2f seconds to train"%(end - start)/3600)

plot_model_history(model_info)
```

    Using real-time data augmentation.
    Epoch 1/50
    2343/2343 [==============================] - 264s 113ms/step - loss: 0.9702 - acc: 0.6521 - val_loss: 1.0551 - val_acc: 0.6679
    Epoch 2/50
    2343/2343 [==============================] - 266s 114ms/step - loss: 0.8324 - acc: 0.7026 - val_loss: 1.0436 - val_acc: 0.6869
    Epoch 3/50
    2343/2343 [==============================] - 253s 108ms/step - loss: 0.8050 - acc: 0.7130 - val_loss: 1.0672 - val_acc: 0.6769
    Epoch 4/50
    2343/2343 [==============================] - 248s 106ms/step - loss: 0.8204 - acc: 0.7070 - val_loss: 1.2928 - val_acc: 0.6318
    Epoch 5/50
    2343/2343 [==============================] - 263s 112ms/step - loss: 0.7962 - acc: 0.7190 - val_loss: 1.3054 - val_acc: 0.6430
    Epoch 6/50
    2343/2343 [==============================] - 262s 112ms/step - loss: 0.7163 - acc: 0.7482 - val_loss: 0.9423 - val_acc: 0.7293
    Epoch 7/50
    2343/2343 [==============================] - 263s 112ms/step - loss: 0.6229 - acc: 0.7823 - val_loss: 0.8103 - val_acc: 0.7630
    Epoch 8/50
    2343/2343 [==============================] - 261s 112ms/step - loss: 0.5579 - acc: 0.8062 - val_loss: 0.7346 - val_acc: 0.7906
    Epoch 9/50
    2343/2343 [==============================] - 262s 112ms/step - loss: 0.5047 - acc: 0.8244 - val_loss: 0.6351 - val_acc: 0.8173
    Epoch 10/50
    2343/2343 [==============================] - 267s 114ms/step - loss: 0.4687 - acc: 0.8374 - val_loss: 0.5978 - val_acc: 0.8288
    Epoch 11/50
    2343/2343 [==============================] - 261s 111ms/step - loss: 0.4605 - acc: 0.8408 - val_loss: 0.6110 - val_acc: 0.8254
    Epoch 12/50
    2343/2343 [==============================] - 264s 112ms/step - loss: 0.4660 - acc: 0.8376 - val_loss: 0.6072 - val_acc: 0.8247
    Epoch 13/50
    2343/2343 [==============================] - 261s 111ms/step - loss: 0.4842 - acc: 0.8319 - val_loss: 0.7957 - val_acc: 0.7872
    Epoch 14/50
    2343/2343 [==============================] - 251s 107ms/step - loss: 0.5370 - acc: 0.8121 - val_loss: 1.0214 - val_acc: 0.7425
    Epoch 15/50
    2343/2343 [==============================] - 260s 111ms/step - loss: 0.4962 - acc: 0.8281 - val_loss: 0.7569 - val_acc: 0.7912
    Epoch 16/50
    2343/2343 [==============================] - 264s 113ms/step - loss: 0.4587 - acc: 0.8404 - val_loss: 0.6630 - val_acc: 0.8150
    Epoch 17/50
    2343/2343 [==============================] - 253s 108ms/step - loss: 0.4232 - acc: 0.8530 - val_loss: 0.6265 - val_acc: 0.8252
    Epoch 18/50
    2343/2343 [==============================] - 250s 107ms/step - loss: 0.3904 - acc: 0.8646 - val_loss: 0.4716 - val_acc: 0.8631
    Epoch 19/50
    2343/2343 [==============================] - 244s 104ms/step - loss: 0.3673 - acc: 0.8721 - val_loss: 0.4608 - val_acc: 0.8680
    Epoch 20/50
    2343/2343 [==============================] - 255s 109ms/step - loss: 0.3482 - acc: 0.8794 - val_loss: 0.4510 - val_acc: 0.8706
    Epoch 21/50
    2343/2343 [==============================] - 260s 111ms/step - loss: 0.3379 - acc: 0.8824 - val_loss: 0.4509 - val_acc: 0.8710
    Epoch 00021: early stopping



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-16-7d97847719a9> in <module>
         59 
         60 end = time.time()
    ---> 61 print ("Model took %0.2f seconds to train"%(end - start)/3600)
         62 
         63 plot_model_history(model_info)


    TypeError: unsupported operand type(s) for /: 'str' and 'int'



```python
print ("Model took %0.2f hours to train"%((end - start)/3600))

plot_model_history(model_info)
```

    Model took 1.51 hours to train



![png](output_15_1.png)



```python
# lr_reducer      = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
#                                    cooldown=0, patience=10, min_lr=0.5e-6)
# early_stopper   = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=20)

# model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=50,
#                     verbose=1,
#                     validation_data=(x_test, y_test),
#                     callbacks=[lr_reducer,early_stopper])
```


```python
# Test the model
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

    10000/10000 [==============================] - 4s 409us/step
    Test loss: 0.7957158389568328
    Test accuracy: 0.7872



```python
# Save the trained weights in to .h5 format
model.save_weights("DNST_model_weights_v6_3.h5")
print("Saved model to disk")
```

    Saved model to disk



```python

```




```python
# https://keras.io/
# !pip install -q keras 
# import keras 
# print(keras.__version__)
```


```python
import keras
import time
from keras.datasets import cifar10
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, AveragePooling2D, merge, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Concatenate
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
```

    Using TensorFlow backend.



```python
# this part will prevent tensorflow to allocate all the avaliable GPU Memory
# backend
import tensorflow as tf
from keras import backend as k

# Don't pre-allocate memory; allocate as-needed
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))
```


```python
class SGDRScheduler(Callback):

    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay,
                 cycle_length,
                 mult_factor):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.set_weights(self.best_weights)
```


```python
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
```


```python

batch_size = 64
num_classes =  10
epochs = 50
l = 16
num_filter = 12
compression = 1
dropout_rate = 0.2

```


```python
# Load CIFAR10 Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
img_height, img_width, channel = x_train.shape[1],x_train.shape[2],x_train.shape[3]

# convert to one hot encoding 
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

```


```python
# Dense Block
def add_denseblock(input, num_filter, dropout_rate):
    global compression
    temp = input
    for _ in range(l):
        BatchNorm = BatchNormalization()(temp)
        relu = Activation('relu')(BatchNorm)
        Conv2D_3_3 = Conv2D(int(num_filter*compression), (3,3), use_bias=False ,padding='same')(relu)
        if dropout_rate>0:
          Conv2D_3_3 = Dropout(dropout_rate)(Conv2D_3_3)
        concat = Concatenate(axis=-1)([temp,Conv2D_3_3])
        
        temp = concat
        
    return temp
```


```python
def add_transition(input, num_filter, dropout_rate):
    global compression
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    Conv2D_BottleNeck = Conv2D(int(num_filter*compression), (1,1), use_bias=False ,padding='same')(relu)
    if dropout_rate>0:
      Conv2D_BottleNeck = Dropout(dropout_rate)(Conv2D_BottleNeck)
    avg = AveragePooling2D(pool_size=(2,2))(Conv2D_BottleNeck)
    
    return avg
```


```python
def output_layer(input):
    global compression
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    AvgPooling = AveragePooling2D(pool_size=(2,2))(relu)
    flat = Flatten()(AvgPooling)
    output = Dense(num_classes, activation='softmax')(flat)
    
    return output
```


```python
input = Input(shape=(img_height, img_width, channel,))
First_Conv2D = Conv2D(num_filter, (3,3), use_bias=False ,padding='same')(input)

First_Block = add_denseblock(First_Conv2D, num_filter, dropout_rate)
First_Transition = add_transition(First_Block, num_filter, dropout_rate)

Second_Block = add_denseblock(First_Transition, num_filter, dropout_rate)
Second_Transition = add_transition(Second_Block, num_filter, dropout_rate)

Third_Block = add_denseblock(Second_Transition, num_filter, dropout_rate)
Third_Transition = add_transition(Third_Block, num_filter, dropout_rate)

Last_Block = add_denseblock(Third_Transition,  num_filter, dropout_rate=0.2)
output = output_layer(Last_Block)

```


```python
model = Model(inputs=[input], outputs=[output])
model.load_weights('DNST_model_weights_v6_3.h5')
model.summary()
```

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            (None, 32, 32, 3)    0                                            
    __________________________________________________________________________________________________
    conv2d_1 (Conv2D)               (None, 32, 32, 12)   324         input_1[0][0]                    
    __________________________________________________________________________________________________
    batch_normalization_1 (BatchNor (None, 32, 32, 12)   48          conv2d_1[0][0]                   
    __________________________________________________________________________________________________
    activation_1 (Activation)       (None, 32, 32, 12)   0           batch_normalization_1[0][0]      
    __________________________________________________________________________________________________
    conv2d_2 (Conv2D)               (None, 32, 32, 12)   1296        activation_1[0][0]               
    __________________________________________________________________________________________________
    dropout_1 (Dropout)             (None, 32, 32, 12)   0           conv2d_2[0][0]                   
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, 32, 32, 24)   0           conv2d_1[0][0]                   
                                                                     dropout_1[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_2 (BatchNor (None, 32, 32, 24)   96          concatenate_1[0][0]              
    __________________________________________________________________________________________________
    activation_2 (Activation)       (None, 32, 32, 24)   0           batch_normalization_2[0][0]      
    __________________________________________________________________________________________________
    conv2d_3 (Conv2D)               (None, 32, 32, 12)   2592        activation_2[0][0]               
    __________________________________________________________________________________________________
    dropout_2 (Dropout)             (None, 32, 32, 12)   0           conv2d_3[0][0]                   
    __________________________________________________________________________________________________
    concatenate_2 (Concatenate)     (None, 32, 32, 36)   0           concatenate_1[0][0]              
                                                                     dropout_2[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_3 (BatchNor (None, 32, 32, 36)   144         concatenate_2[0][0]              
    __________________________________________________________________________________________________
    activation_3 (Activation)       (None, 32, 32, 36)   0           batch_normalization_3[0][0]      
    __________________________________________________________________________________________________
    conv2d_4 (Conv2D)               (None, 32, 32, 12)   3888        activation_3[0][0]               
    __________________________________________________________________________________________________
    dropout_3 (Dropout)             (None, 32, 32, 12)   0           conv2d_4[0][0]                   
    __________________________________________________________________________________________________
    concatenate_3 (Concatenate)     (None, 32, 32, 48)   0           concatenate_2[0][0]              
                                                                     dropout_3[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_4 (BatchNor (None, 32, 32, 48)   192         concatenate_3[0][0]              
    __________________________________________________________________________________________________
    activation_4 (Activation)       (None, 32, 32, 48)   0           batch_normalization_4[0][0]      
    __________________________________________________________________________________________________
    conv2d_5 (Conv2D)               (None, 32, 32, 12)   5184        activation_4[0][0]               
    __________________________________________________________________________________________________
    dropout_4 (Dropout)             (None, 32, 32, 12)   0           conv2d_5[0][0]                   
    __________________________________________________________________________________________________
    concatenate_4 (Concatenate)     (None, 32, 32, 60)   0           concatenate_3[0][0]              
                                                                     dropout_4[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_5 (BatchNor (None, 32, 32, 60)   240         concatenate_4[0][0]              
    __________________________________________________________________________________________________
    activation_5 (Activation)       (None, 32, 32, 60)   0           batch_normalization_5[0][0]      
    __________________________________________________________________________________________________
    conv2d_6 (Conv2D)               (None, 32, 32, 12)   6480        activation_5[0][0]               
    __________________________________________________________________________________________________
    dropout_5 (Dropout)             (None, 32, 32, 12)   0           conv2d_6[0][0]                   
    __________________________________________________________________________________________________
    concatenate_5 (Concatenate)     (None, 32, 32, 72)   0           concatenate_4[0][0]              
                                                                     dropout_5[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_6 (BatchNor (None, 32, 32, 72)   288         concatenate_5[0][0]              
    __________________________________________________________________________________________________
    activation_6 (Activation)       (None, 32, 32, 72)   0           batch_normalization_6[0][0]      
    __________________________________________________________________________________________________
    conv2d_7 (Conv2D)               (None, 32, 32, 12)   7776        activation_6[0][0]               
    __________________________________________________________________________________________________
    dropout_6 (Dropout)             (None, 32, 32, 12)   0           conv2d_7[0][0]                   
    __________________________________________________________________________________________________
    concatenate_6 (Concatenate)     (None, 32, 32, 84)   0           concatenate_5[0][0]              
                                                                     dropout_6[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_7 (BatchNor (None, 32, 32, 84)   336         concatenate_6[0][0]              
    __________________________________________________________________________________________________
    activation_7 (Activation)       (None, 32, 32, 84)   0           batch_normalization_7[0][0]      
    __________________________________________________________________________________________________
    conv2d_8 (Conv2D)               (None, 32, 32, 12)   9072        activation_7[0][0]               
    __________________________________________________________________________________________________
    dropout_7 (Dropout)             (None, 32, 32, 12)   0           conv2d_8[0][0]                   
    __________________________________________________________________________________________________
    concatenate_7 (Concatenate)     (None, 32, 32, 96)   0           concatenate_6[0][0]              
                                                                     dropout_7[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_8 (BatchNor (None, 32, 32, 96)   384         concatenate_7[0][0]              
    __________________________________________________________________________________________________
    activation_8 (Activation)       (None, 32, 32, 96)   0           batch_normalization_8[0][0]      
    __________________________________________________________________________________________________
    conv2d_9 (Conv2D)               (None, 32, 32, 12)   10368       activation_8[0][0]               
    __________________________________________________________________________________________________
    dropout_8 (Dropout)             (None, 32, 32, 12)   0           conv2d_9[0][0]                   
    __________________________________________________________________________________________________
    concatenate_8 (Concatenate)     (None, 32, 32, 108)  0           concatenate_7[0][0]              
                                                                     dropout_8[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_9 (BatchNor (None, 32, 32, 108)  432         concatenate_8[0][0]              
    __________________________________________________________________________________________________
    activation_9 (Activation)       (None, 32, 32, 108)  0           batch_normalization_9[0][0]      
    __________________________________________________________________________________________________
    conv2d_10 (Conv2D)              (None, 32, 32, 12)   11664       activation_9[0][0]               
    __________________________________________________________________________________________________
    dropout_9 (Dropout)             (None, 32, 32, 12)   0           conv2d_10[0][0]                  
    __________________________________________________________________________________________________
    concatenate_9 (Concatenate)     (None, 32, 32, 120)  0           concatenate_8[0][0]              
                                                                     dropout_9[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_10 (BatchNo (None, 32, 32, 120)  480         concatenate_9[0][0]              
    __________________________________________________________________________________________________
    activation_10 (Activation)      (None, 32, 32, 120)  0           batch_normalization_10[0][0]     
    __________________________________________________________________________________________________
    conv2d_11 (Conv2D)              (None, 32, 32, 12)   12960       activation_10[0][0]              
    __________________________________________________________________________________________________
    dropout_10 (Dropout)            (None, 32, 32, 12)   0           conv2d_11[0][0]                  
    __________________________________________________________________________________________________
    concatenate_10 (Concatenate)    (None, 32, 32, 132)  0           concatenate_9[0][0]              
                                                                     dropout_10[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_11 (BatchNo (None, 32, 32, 132)  528         concatenate_10[0][0]             
    __________________________________________________________________________________________________
    activation_11 (Activation)      (None, 32, 32, 132)  0           batch_normalization_11[0][0]     
    __________________________________________________________________________________________________
    conv2d_12 (Conv2D)              (None, 32, 32, 12)   14256       activation_11[0][0]              
    __________________________________________________________________________________________________
    dropout_11 (Dropout)            (None, 32, 32, 12)   0           conv2d_12[0][0]                  
    __________________________________________________________________________________________________
    concatenate_11 (Concatenate)    (None, 32, 32, 144)  0           concatenate_10[0][0]             
                                                                     dropout_11[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_12 (BatchNo (None, 32, 32, 144)  576         concatenate_11[0][0]             
    __________________________________________________________________________________________________
    activation_12 (Activation)      (None, 32, 32, 144)  0           batch_normalization_12[0][0]     
    __________________________________________________________________________________________________
    conv2d_13 (Conv2D)              (None, 32, 32, 12)   15552       activation_12[0][0]              
    __________________________________________________________________________________________________
    dropout_12 (Dropout)            (None, 32, 32, 12)   0           conv2d_13[0][0]                  
    __________________________________________________________________________________________________
    concatenate_12 (Concatenate)    (None, 32, 32, 156)  0           concatenate_11[0][0]             
                                                                     dropout_12[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_13 (BatchNo (None, 32, 32, 156)  624         concatenate_12[0][0]             
    __________________________________________________________________________________________________
    activation_13 (Activation)      (None, 32, 32, 156)  0           batch_normalization_13[0][0]     
    __________________________________________________________________________________________________
    conv2d_14 (Conv2D)              (None, 32, 32, 12)   16848       activation_13[0][0]              
    __________________________________________________________________________________________________
    dropout_13 (Dropout)            (None, 32, 32, 12)   0           conv2d_14[0][0]                  
    __________________________________________________________________________________________________
    concatenate_13 (Concatenate)    (None, 32, 32, 168)  0           concatenate_12[0][0]             
                                                                     dropout_13[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_14 (BatchNo (None, 32, 32, 168)  672         concatenate_13[0][0]             
    __________________________________________________________________________________________________
    activation_14 (Activation)      (None, 32, 32, 168)  0           batch_normalization_14[0][0]     
    __________________________________________________________________________________________________
    conv2d_15 (Conv2D)              (None, 32, 32, 12)   18144       activation_14[0][0]              
    __________________________________________________________________________________________________
    dropout_14 (Dropout)            (None, 32, 32, 12)   0           conv2d_15[0][0]                  
    __________________________________________________________________________________________________
    concatenate_14 (Concatenate)    (None, 32, 32, 180)  0           concatenate_13[0][0]             
                                                                     dropout_14[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_15 (BatchNo (None, 32, 32, 180)  720         concatenate_14[0][0]             
    __________________________________________________________________________________________________
    activation_15 (Activation)      (None, 32, 32, 180)  0           batch_normalization_15[0][0]     
    __________________________________________________________________________________________________
    conv2d_16 (Conv2D)              (None, 32, 32, 12)   19440       activation_15[0][0]              
    __________________________________________________________________________________________________
    dropout_15 (Dropout)            (None, 32, 32, 12)   0           conv2d_16[0][0]                  
    __________________________________________________________________________________________________
    concatenate_15 (Concatenate)    (None, 32, 32, 192)  0           concatenate_14[0][0]             
                                                                     dropout_15[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_16 (BatchNo (None, 32, 32, 192)  768         concatenate_15[0][0]             
    __________________________________________________________________________________________________
    activation_16 (Activation)      (None, 32, 32, 192)  0           batch_normalization_16[0][0]     
    __________________________________________________________________________________________________
    conv2d_17 (Conv2D)              (None, 32, 32, 12)   20736       activation_16[0][0]              
    __________________________________________________________________________________________________
    dropout_16 (Dropout)            (None, 32, 32, 12)   0           conv2d_17[0][0]                  
    __________________________________________________________________________________________________
    concatenate_16 (Concatenate)    (None, 32, 32, 204)  0           concatenate_15[0][0]             
                                                                     dropout_16[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_17 (BatchNo (None, 32, 32, 204)  816         concatenate_16[0][0]             
    __________________________________________________________________________________________________
    activation_17 (Activation)      (None, 32, 32, 204)  0           batch_normalization_17[0][0]     
    __________________________________________________________________________________________________
    conv2d_18 (Conv2D)              (None, 32, 32, 12)   2448        activation_17[0][0]              
    __________________________________________________________________________________________________
    dropout_17 (Dropout)            (None, 32, 32, 12)   0           conv2d_18[0][0]                  
    __________________________________________________________________________________________________
    average_pooling2d_1 (AveragePoo (None, 16, 16, 12)   0           dropout_17[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_18 (BatchNo (None, 16, 16, 12)   48          average_pooling2d_1[0][0]        
    __________________________________________________________________________________________________
    activation_18 (Activation)      (None, 16, 16, 12)   0           batch_normalization_18[0][0]     
    __________________________________________________________________________________________________
    conv2d_19 (Conv2D)              (None, 16, 16, 12)   1296        activation_18[0][0]              
    __________________________________________________________________________________________________
    dropout_18 (Dropout)            (None, 16, 16, 12)   0           conv2d_19[0][0]                  
    __________________________________________________________________________________________________
    concatenate_17 (Concatenate)    (None, 16, 16, 24)   0           average_pooling2d_1[0][0]        
                                                                     dropout_18[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_19 (BatchNo (None, 16, 16, 24)   96          concatenate_17[0][0]             
    __________________________________________________________________________________________________
    activation_19 (Activation)      (None, 16, 16, 24)   0           batch_normalization_19[0][0]     
    __________________________________________________________________________________________________
    conv2d_20 (Conv2D)              (None, 16, 16, 12)   2592        activation_19[0][0]              
    __________________________________________________________________________________________________
    dropout_19 (Dropout)            (None, 16, 16, 12)   0           conv2d_20[0][0]                  
    __________________________________________________________________________________________________
    concatenate_18 (Concatenate)    (None, 16, 16, 36)   0           concatenate_17[0][0]             
                                                                     dropout_19[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_20 (BatchNo (None, 16, 16, 36)   144         concatenate_18[0][0]             
    __________________________________________________________________________________________________
    activation_20 (Activation)      (None, 16, 16, 36)   0           batch_normalization_20[0][0]     
    __________________________________________________________________________________________________
    conv2d_21 (Conv2D)              (None, 16, 16, 12)   3888        activation_20[0][0]              
    __________________________________________________________________________________________________
    dropout_20 (Dropout)            (None, 16, 16, 12)   0           conv2d_21[0][0]                  
    __________________________________________________________________________________________________
    concatenate_19 (Concatenate)    (None, 16, 16, 48)   0           concatenate_18[0][0]             
                                                                     dropout_20[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_21 (BatchNo (None, 16, 16, 48)   192         concatenate_19[0][0]             
    __________________________________________________________________________________________________
    activation_21 (Activation)      (None, 16, 16, 48)   0           batch_normalization_21[0][0]     
    __________________________________________________________________________________________________
    conv2d_22 (Conv2D)              (None, 16, 16, 12)   5184        activation_21[0][0]              
    __________________________________________________________________________________________________
    dropout_21 (Dropout)            (None, 16, 16, 12)   0           conv2d_22[0][0]                  
    __________________________________________________________________________________________________
    concatenate_20 (Concatenate)    (None, 16, 16, 60)   0           concatenate_19[0][0]             
                                                                     dropout_21[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_22 (BatchNo (None, 16, 16, 60)   240         concatenate_20[0][0]             
    __________________________________________________________________________________________________
    activation_22 (Activation)      (None, 16, 16, 60)   0           batch_normalization_22[0][0]     
    __________________________________________________________________________________________________
    conv2d_23 (Conv2D)              (None, 16, 16, 12)   6480        activation_22[0][0]              
    __________________________________________________________________________________________________
    dropout_22 (Dropout)            (None, 16, 16, 12)   0           conv2d_23[0][0]                  
    __________________________________________________________________________________________________
    concatenate_21 (Concatenate)    (None, 16, 16, 72)   0           concatenate_20[0][0]             
                                                                     dropout_22[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_23 (BatchNo (None, 16, 16, 72)   288         concatenate_21[0][0]             
    __________________________________________________________________________________________________
    activation_23 (Activation)      (None, 16, 16, 72)   0           batch_normalization_23[0][0]     
    __________________________________________________________________________________________________
    conv2d_24 (Conv2D)              (None, 16, 16, 12)   7776        activation_23[0][0]              
    __________________________________________________________________________________________________
    dropout_23 (Dropout)            (None, 16, 16, 12)   0           conv2d_24[0][0]                  
    __________________________________________________________________________________________________
    concatenate_22 (Concatenate)    (None, 16, 16, 84)   0           concatenate_21[0][0]             
                                                                     dropout_23[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_24 (BatchNo (None, 16, 16, 84)   336         concatenate_22[0][0]             
    __________________________________________________________________________________________________
    activation_24 (Activation)      (None, 16, 16, 84)   0           batch_normalization_24[0][0]     
    __________________________________________________________________________________________________
    conv2d_25 (Conv2D)              (None, 16, 16, 12)   9072        activation_24[0][0]              
    __________________________________________________________________________________________________
    dropout_24 (Dropout)            (None, 16, 16, 12)   0           conv2d_25[0][0]                  
    __________________________________________________________________________________________________
    concatenate_23 (Concatenate)    (None, 16, 16, 96)   0           concatenate_22[0][0]             
                                                                     dropout_24[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_25 (BatchNo (None, 16, 16, 96)   384         concatenate_23[0][0]             
    __________________________________________________________________________________________________
    activation_25 (Activation)      (None, 16, 16, 96)   0           batch_normalization_25[0][0]     
    __________________________________________________________________________________________________
    conv2d_26 (Conv2D)              (None, 16, 16, 12)   10368       activation_25[0][0]              
    __________________________________________________________________________________________________
    dropout_25 (Dropout)            (None, 16, 16, 12)   0           conv2d_26[0][0]                  
    __________________________________________________________________________________________________
    concatenate_24 (Concatenate)    (None, 16, 16, 108)  0           concatenate_23[0][0]             
                                                                     dropout_25[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_26 (BatchNo (None, 16, 16, 108)  432         concatenate_24[0][0]             
    __________________________________________________________________________________________________
    activation_26 (Activation)      (None, 16, 16, 108)  0           batch_normalization_26[0][0]     
    __________________________________________________________________________________________________
    conv2d_27 (Conv2D)              (None, 16, 16, 12)   11664       activation_26[0][0]              
    __________________________________________________________________________________________________
    dropout_26 (Dropout)            (None, 16, 16, 12)   0           conv2d_27[0][0]                  
    __________________________________________________________________________________________________
    concatenate_25 (Concatenate)    (None, 16, 16, 120)  0           concatenate_24[0][0]             
                                                                     dropout_26[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_27 (BatchNo (None, 16, 16, 120)  480         concatenate_25[0][0]             
    __________________________________________________________________________________________________
    activation_27 (Activation)      (None, 16, 16, 120)  0           batch_normalization_27[0][0]     
    __________________________________________________________________________________________________
    conv2d_28 (Conv2D)              (None, 16, 16, 12)   12960       activation_27[0][0]              
    __________________________________________________________________________________________________
    dropout_27 (Dropout)            (None, 16, 16, 12)   0           conv2d_28[0][0]                  
    __________________________________________________________________________________________________
    concatenate_26 (Concatenate)    (None, 16, 16, 132)  0           concatenate_25[0][0]             
                                                                     dropout_27[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_28 (BatchNo (None, 16, 16, 132)  528         concatenate_26[0][0]             
    __________________________________________________________________________________________________
    activation_28 (Activation)      (None, 16, 16, 132)  0           batch_normalization_28[0][0]     
    __________________________________________________________________________________________________
    conv2d_29 (Conv2D)              (None, 16, 16, 12)   14256       activation_28[0][0]              
    __________________________________________________________________________________________________
    dropout_28 (Dropout)            (None, 16, 16, 12)   0           conv2d_29[0][0]                  
    __________________________________________________________________________________________________
    concatenate_27 (Concatenate)    (None, 16, 16, 144)  0           concatenate_26[0][0]             
                                                                     dropout_28[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_29 (BatchNo (None, 16, 16, 144)  576         concatenate_27[0][0]             
    __________________________________________________________________________________________________
    activation_29 (Activation)      (None, 16, 16, 144)  0           batch_normalization_29[0][0]     
    __________________________________________________________________________________________________
    conv2d_30 (Conv2D)              (None, 16, 16, 12)   15552       activation_29[0][0]              
    __________________________________________________________________________________________________
    dropout_29 (Dropout)            (None, 16, 16, 12)   0           conv2d_30[0][0]                  
    __________________________________________________________________________________________________
    concatenate_28 (Concatenate)    (None, 16, 16, 156)  0           concatenate_27[0][0]             
                                                                     dropout_29[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_30 (BatchNo (None, 16, 16, 156)  624         concatenate_28[0][0]             
    __________________________________________________________________________________________________
    activation_30 (Activation)      (None, 16, 16, 156)  0           batch_normalization_30[0][0]     
    __________________________________________________________________________________________________
    conv2d_31 (Conv2D)              (None, 16, 16, 12)   16848       activation_30[0][0]              
    __________________________________________________________________________________________________
    dropout_30 (Dropout)            (None, 16, 16, 12)   0           conv2d_31[0][0]                  
    __________________________________________________________________________________________________
    concatenate_29 (Concatenate)    (None, 16, 16, 168)  0           concatenate_28[0][0]             
                                                                     dropout_30[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_31 (BatchNo (None, 16, 16, 168)  672         concatenate_29[0][0]             
    __________________________________________________________________________________________________
    activation_31 (Activation)      (None, 16, 16, 168)  0           batch_normalization_31[0][0]     
    __________________________________________________________________________________________________
    conv2d_32 (Conv2D)              (None, 16, 16, 12)   18144       activation_31[0][0]              
    __________________________________________________________________________________________________
    dropout_31 (Dropout)            (None, 16, 16, 12)   0           conv2d_32[0][0]                  
    __________________________________________________________________________________________________
    concatenate_30 (Concatenate)    (None, 16, 16, 180)  0           concatenate_29[0][0]             
                                                                     dropout_31[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_32 (BatchNo (None, 16, 16, 180)  720         concatenate_30[0][0]             
    __________________________________________________________________________________________________
    activation_32 (Activation)      (None, 16, 16, 180)  0           batch_normalization_32[0][0]     
    __________________________________________________________________________________________________
    conv2d_33 (Conv2D)              (None, 16, 16, 12)   19440       activation_32[0][0]              
    __________________________________________________________________________________________________
    dropout_32 (Dropout)            (None, 16, 16, 12)   0           conv2d_33[0][0]                  
    __________________________________________________________________________________________________
    concatenate_31 (Concatenate)    (None, 16, 16, 192)  0           concatenate_30[0][0]             
                                                                     dropout_32[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_33 (BatchNo (None, 16, 16, 192)  768         concatenate_31[0][0]             
    __________________________________________________________________________________________________
    activation_33 (Activation)      (None, 16, 16, 192)  0           batch_normalization_33[0][0]     
    __________________________________________________________________________________________________
    conv2d_34 (Conv2D)              (None, 16, 16, 12)   20736       activation_33[0][0]              
    __________________________________________________________________________________________________
    dropout_33 (Dropout)            (None, 16, 16, 12)   0           conv2d_34[0][0]                  
    __________________________________________________________________________________________________
    concatenate_32 (Concatenate)    (None, 16, 16, 204)  0           concatenate_31[0][0]             
                                                                     dropout_33[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_34 (BatchNo (None, 16, 16, 204)  816         concatenate_32[0][0]             
    __________________________________________________________________________________________________
    activation_34 (Activation)      (None, 16, 16, 204)  0           batch_normalization_34[0][0]     
    __________________________________________________________________________________________________
    conv2d_35 (Conv2D)              (None, 16, 16, 12)   2448        activation_34[0][0]              
    __________________________________________________________________________________________________
    dropout_34 (Dropout)            (None, 16, 16, 12)   0           conv2d_35[0][0]                  
    __________________________________________________________________________________________________
    average_pooling2d_2 (AveragePoo (None, 8, 8, 12)     0           dropout_34[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_35 (BatchNo (None, 8, 8, 12)     48          average_pooling2d_2[0][0]        
    __________________________________________________________________________________________________
    activation_35 (Activation)      (None, 8, 8, 12)     0           batch_normalization_35[0][0]     
    __________________________________________________________________________________________________
    conv2d_36 (Conv2D)              (None, 8, 8, 12)     1296        activation_35[0][0]              
    __________________________________________________________________________________________________
    dropout_35 (Dropout)            (None, 8, 8, 12)     0           conv2d_36[0][0]                  
    __________________________________________________________________________________________________
    concatenate_33 (Concatenate)    (None, 8, 8, 24)     0           average_pooling2d_2[0][0]        
                                                                     dropout_35[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_36 (BatchNo (None, 8, 8, 24)     96          concatenate_33[0][0]             
    __________________________________________________________________________________________________
    activation_36 (Activation)      (None, 8, 8, 24)     0           batch_normalization_36[0][0]     
    __________________________________________________________________________________________________
    conv2d_37 (Conv2D)              (None, 8, 8, 12)     2592        activation_36[0][0]              
    __________________________________________________________________________________________________
    dropout_36 (Dropout)            (None, 8, 8, 12)     0           conv2d_37[0][0]                  
    __________________________________________________________________________________________________
    concatenate_34 (Concatenate)    (None, 8, 8, 36)     0           concatenate_33[0][0]             
                                                                     dropout_36[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_37 (BatchNo (None, 8, 8, 36)     144         concatenate_34[0][0]             
    __________________________________________________________________________________________________
    activation_37 (Activation)      (None, 8, 8, 36)     0           batch_normalization_37[0][0]     
    __________________________________________________________________________________________________
    conv2d_38 (Conv2D)              (None, 8, 8, 12)     3888        activation_37[0][0]              
    __________________________________________________________________________________________________
    dropout_37 (Dropout)            (None, 8, 8, 12)     0           conv2d_38[0][0]                  
    __________________________________________________________________________________________________
    concatenate_35 (Concatenate)    (None, 8, 8, 48)     0           concatenate_34[0][0]             
                                                                     dropout_37[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_38 (BatchNo (None, 8, 8, 48)     192         concatenate_35[0][0]             
    __________________________________________________________________________________________________
    activation_38 (Activation)      (None, 8, 8, 48)     0           batch_normalization_38[0][0]     
    __________________________________________________________________________________________________
    conv2d_39 (Conv2D)              (None, 8, 8, 12)     5184        activation_38[0][0]              
    __________________________________________________________________________________________________
    dropout_38 (Dropout)            (None, 8, 8, 12)     0           conv2d_39[0][0]                  
    __________________________________________________________________________________________________
    concatenate_36 (Concatenate)    (None, 8, 8, 60)     0           concatenate_35[0][0]             
                                                                     dropout_38[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_39 (BatchNo (None, 8, 8, 60)     240         concatenate_36[0][0]             
    __________________________________________________________________________________________________
    activation_39 (Activation)      (None, 8, 8, 60)     0           batch_normalization_39[0][0]     
    __________________________________________________________________________________________________
    conv2d_40 (Conv2D)              (None, 8, 8, 12)     6480        activation_39[0][0]              
    __________________________________________________________________________________________________
    dropout_39 (Dropout)            (None, 8, 8, 12)     0           conv2d_40[0][0]                  
    __________________________________________________________________________________________________
    concatenate_37 (Concatenate)    (None, 8, 8, 72)     0           concatenate_36[0][0]             
                                                                     dropout_39[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_40 (BatchNo (None, 8, 8, 72)     288         concatenate_37[0][0]             
    __________________________________________________________________________________________________
    activation_40 (Activation)      (None, 8, 8, 72)     0           batch_normalization_40[0][0]     
    __________________________________________________________________________________________________
    conv2d_41 (Conv2D)              (None, 8, 8, 12)     7776        activation_40[0][0]              
    __________________________________________________________________________________________________
    dropout_40 (Dropout)            (None, 8, 8, 12)     0           conv2d_41[0][0]                  
    __________________________________________________________________________________________________
    concatenate_38 (Concatenate)    (None, 8, 8, 84)     0           concatenate_37[0][0]             
                                                                     dropout_40[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_41 (BatchNo (None, 8, 8, 84)     336         concatenate_38[0][0]             
    __________________________________________________________________________________________________
    activation_41 (Activation)      (None, 8, 8, 84)     0           batch_normalization_41[0][0]     
    __________________________________________________________________________________________________
    conv2d_42 (Conv2D)              (None, 8, 8, 12)     9072        activation_41[0][0]              
    __________________________________________________________________________________________________
    dropout_41 (Dropout)            (None, 8, 8, 12)     0           conv2d_42[0][0]                  
    __________________________________________________________________________________________________
    concatenate_39 (Concatenate)    (None, 8, 8, 96)     0           concatenate_38[0][0]             
                                                                     dropout_41[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_42 (BatchNo (None, 8, 8, 96)     384         concatenate_39[0][0]             
    __________________________________________________________________________________________________
    activation_42 (Activation)      (None, 8, 8, 96)     0           batch_normalization_42[0][0]     
    __________________________________________________________________________________________________
    conv2d_43 (Conv2D)              (None, 8, 8, 12)     10368       activation_42[0][0]              
    __________________________________________________________________________________________________
    dropout_42 (Dropout)            (None, 8, 8, 12)     0           conv2d_43[0][0]                  
    __________________________________________________________________________________________________
    concatenate_40 (Concatenate)    (None, 8, 8, 108)    0           concatenate_39[0][0]             
                                                                     dropout_42[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_43 (BatchNo (None, 8, 8, 108)    432         concatenate_40[0][0]             
    __________________________________________________________________________________________________
    activation_43 (Activation)      (None, 8, 8, 108)    0           batch_normalization_43[0][0]     
    __________________________________________________________________________________________________
    conv2d_44 (Conv2D)              (None, 8, 8, 12)     11664       activation_43[0][0]              
    __________________________________________________________________________________________________
    dropout_43 (Dropout)            (None, 8, 8, 12)     0           conv2d_44[0][0]                  
    __________________________________________________________________________________________________
    concatenate_41 (Concatenate)    (None, 8, 8, 120)    0           concatenate_40[0][0]             
                                                                     dropout_43[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_44 (BatchNo (None, 8, 8, 120)    480         concatenate_41[0][0]             
    __________________________________________________________________________________________________
    activation_44 (Activation)      (None, 8, 8, 120)    0           batch_normalization_44[0][0]     
    __________________________________________________________________________________________________
    conv2d_45 (Conv2D)              (None, 8, 8, 12)     12960       activation_44[0][0]              
    __________________________________________________________________________________________________
    dropout_44 (Dropout)            (None, 8, 8, 12)     0           conv2d_45[0][0]                  
    __________________________________________________________________________________________________
    concatenate_42 (Concatenate)    (None, 8, 8, 132)    0           concatenate_41[0][0]             
                                                                     dropout_44[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_45 (BatchNo (None, 8, 8, 132)    528         concatenate_42[0][0]             
    __________________________________________________________________________________________________
    activation_45 (Activation)      (None, 8, 8, 132)    0           batch_normalization_45[0][0]     
    __________________________________________________________________________________________________
    conv2d_46 (Conv2D)              (None, 8, 8, 12)     14256       activation_45[0][0]              
    __________________________________________________________________________________________________
    dropout_45 (Dropout)            (None, 8, 8, 12)     0           conv2d_46[0][0]                  
    __________________________________________________________________________________________________
    concatenate_43 (Concatenate)    (None, 8, 8, 144)    0           concatenate_42[0][0]             
                                                                     dropout_45[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_46 (BatchNo (None, 8, 8, 144)    576         concatenate_43[0][0]             
    __________________________________________________________________________________________________
    activation_46 (Activation)      (None, 8, 8, 144)    0           batch_normalization_46[0][0]     
    __________________________________________________________________________________________________
    conv2d_47 (Conv2D)              (None, 8, 8, 12)     15552       activation_46[0][0]              
    __________________________________________________________________________________________________
    dropout_46 (Dropout)            (None, 8, 8, 12)     0           conv2d_47[0][0]                  
    __________________________________________________________________________________________________
    concatenate_44 (Concatenate)    (None, 8, 8, 156)    0           concatenate_43[0][0]             
                                                                     dropout_46[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_47 (BatchNo (None, 8, 8, 156)    624         concatenate_44[0][0]             
    __________________________________________________________________________________________________
    activation_47 (Activation)      (None, 8, 8, 156)    0           batch_normalization_47[0][0]     
    __________________________________________________________________________________________________
    conv2d_48 (Conv2D)              (None, 8, 8, 12)     16848       activation_47[0][0]              
    __________________________________________________________________________________________________
    dropout_47 (Dropout)            (None, 8, 8, 12)     0           conv2d_48[0][0]                  
    __________________________________________________________________________________________________
    concatenate_45 (Concatenate)    (None, 8, 8, 168)    0           concatenate_44[0][0]             
                                                                     dropout_47[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_48 (BatchNo (None, 8, 8, 168)    672         concatenate_45[0][0]             
    __________________________________________________________________________________________________
    activation_48 (Activation)      (None, 8, 8, 168)    0           batch_normalization_48[0][0]     
    __________________________________________________________________________________________________
    conv2d_49 (Conv2D)              (None, 8, 8, 12)     18144       activation_48[0][0]              
    __________________________________________________________________________________________________
    dropout_48 (Dropout)            (None, 8, 8, 12)     0           conv2d_49[0][0]                  
    __________________________________________________________________________________________________
    concatenate_46 (Concatenate)    (None, 8, 8, 180)    0           concatenate_45[0][0]             
                                                                     dropout_48[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_49 (BatchNo (None, 8, 8, 180)    720         concatenate_46[0][0]             
    __________________________________________________________________________________________________
    activation_49 (Activation)      (None, 8, 8, 180)    0           batch_normalization_49[0][0]     
    __________________________________________________________________________________________________
    conv2d_50 (Conv2D)              (None, 8, 8, 12)     19440       activation_49[0][0]              
    __________________________________________________________________________________________________
    dropout_49 (Dropout)            (None, 8, 8, 12)     0           conv2d_50[0][0]                  
    __________________________________________________________________________________________________
    concatenate_47 (Concatenate)    (None, 8, 8, 192)    0           concatenate_46[0][0]             
                                                                     dropout_49[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_50 (BatchNo (None, 8, 8, 192)    768         concatenate_47[0][0]             
    __________________________________________________________________________________________________
    activation_50 (Activation)      (None, 8, 8, 192)    0           batch_normalization_50[0][0]     
    __________________________________________________________________________________________________
    conv2d_51 (Conv2D)              (None, 8, 8, 12)     20736       activation_50[0][0]              
    __________________________________________________________________________________________________
    dropout_50 (Dropout)            (None, 8, 8, 12)     0           conv2d_51[0][0]                  
    __________________________________________________________________________________________________
    concatenate_48 (Concatenate)    (None, 8, 8, 204)    0           concatenate_47[0][0]             
                                                                     dropout_50[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_51 (BatchNo (None, 8, 8, 204)    816         concatenate_48[0][0]             
    __________________________________________________________________________________________________
    activation_51 (Activation)      (None, 8, 8, 204)    0           batch_normalization_51[0][0]     
    __________________________________________________________________________________________________
    conv2d_52 (Conv2D)              (None, 8, 8, 12)     2448        activation_51[0][0]              
    __________________________________________________________________________________________________
    dropout_51 (Dropout)            (None, 8, 8, 12)     0           conv2d_52[0][0]                  
    __________________________________________________________________________________________________
    average_pooling2d_3 (AveragePoo (None, 4, 4, 12)     0           dropout_51[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_52 (BatchNo (None, 4, 4, 12)     48          average_pooling2d_3[0][0]        
    __________________________________________________________________________________________________
    activation_52 (Activation)      (None, 4, 4, 12)     0           batch_normalization_52[0][0]     
    __________________________________________________________________________________________________
    conv2d_53 (Conv2D)              (None, 4, 4, 12)     1296        activation_52[0][0]              
    __________________________________________________________________________________________________
    dropout_52 (Dropout)            (None, 4, 4, 12)     0           conv2d_53[0][0]                  
    __________________________________________________________________________________________________
    concatenate_49 (Concatenate)    (None, 4, 4, 24)     0           average_pooling2d_3[0][0]        
                                                                     dropout_52[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_53 (BatchNo (None, 4, 4, 24)     96          concatenate_49[0][0]             
    __________________________________________________________________________________________________
    activation_53 (Activation)      (None, 4, 4, 24)     0           batch_normalization_53[0][0]     
    __________________________________________________________________________________________________
    conv2d_54 (Conv2D)              (None, 4, 4, 12)     2592        activation_53[0][0]              
    __________________________________________________________________________________________________
    dropout_53 (Dropout)            (None, 4, 4, 12)     0           conv2d_54[0][0]                  
    __________________________________________________________________________________________________
    concatenate_50 (Concatenate)    (None, 4, 4, 36)     0           concatenate_49[0][0]             
                                                                     dropout_53[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_54 (BatchNo (None, 4, 4, 36)     144         concatenate_50[0][0]             
    __________________________________________________________________________________________________
    activation_54 (Activation)      (None, 4, 4, 36)     0           batch_normalization_54[0][0]     
    __________________________________________________________________________________________________
    conv2d_55 (Conv2D)              (None, 4, 4, 12)     3888        activation_54[0][0]              
    __________________________________________________________________________________________________
    dropout_54 (Dropout)            (None, 4, 4, 12)     0           conv2d_55[0][0]                  
    __________________________________________________________________________________________________
    concatenate_51 (Concatenate)    (None, 4, 4, 48)     0           concatenate_50[0][0]             
                                                                     dropout_54[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_55 (BatchNo (None, 4, 4, 48)     192         concatenate_51[0][0]             
    __________________________________________________________________________________________________
    activation_55 (Activation)      (None, 4, 4, 48)     0           batch_normalization_55[0][0]     
    __________________________________________________________________________________________________
    conv2d_56 (Conv2D)              (None, 4, 4, 12)     5184        activation_55[0][0]              
    __________________________________________________________________________________________________
    dropout_55 (Dropout)            (None, 4, 4, 12)     0           conv2d_56[0][0]                  
    __________________________________________________________________________________________________
    concatenate_52 (Concatenate)    (None, 4, 4, 60)     0           concatenate_51[0][0]             
                                                                     dropout_55[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_56 (BatchNo (None, 4, 4, 60)     240         concatenate_52[0][0]             
    __________________________________________________________________________________________________
    activation_56 (Activation)      (None, 4, 4, 60)     0           batch_normalization_56[0][0]     
    __________________________________________________________________________________________________
    conv2d_57 (Conv2D)              (None, 4, 4, 12)     6480        activation_56[0][0]              
    __________________________________________________________________________________________________
    dropout_56 (Dropout)            (None, 4, 4, 12)     0           conv2d_57[0][0]                  
    __________________________________________________________________________________________________
    concatenate_53 (Concatenate)    (None, 4, 4, 72)     0           concatenate_52[0][0]             
                                                                     dropout_56[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_57 (BatchNo (None, 4, 4, 72)     288         concatenate_53[0][0]             
    __________________________________________________________________________________________________
    activation_57 (Activation)      (None, 4, 4, 72)     0           batch_normalization_57[0][0]     
    __________________________________________________________________________________________________
    conv2d_58 (Conv2D)              (None, 4, 4, 12)     7776        activation_57[0][0]              
    __________________________________________________________________________________________________
    dropout_57 (Dropout)            (None, 4, 4, 12)     0           conv2d_58[0][0]                  
    __________________________________________________________________________________________________
    concatenate_54 (Concatenate)    (None, 4, 4, 84)     0           concatenate_53[0][0]             
                                                                     dropout_57[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_58 (BatchNo (None, 4, 4, 84)     336         concatenate_54[0][0]             
    __________________________________________________________________________________________________
    activation_58 (Activation)      (None, 4, 4, 84)     0           batch_normalization_58[0][0]     
    __________________________________________________________________________________________________
    conv2d_59 (Conv2D)              (None, 4, 4, 12)     9072        activation_58[0][0]              
    __________________________________________________________________________________________________
    dropout_58 (Dropout)            (None, 4, 4, 12)     0           conv2d_59[0][0]                  
    __________________________________________________________________________________________________
    concatenate_55 (Concatenate)    (None, 4, 4, 96)     0           concatenate_54[0][0]             
                                                                     dropout_58[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_59 (BatchNo (None, 4, 4, 96)     384         concatenate_55[0][0]             
    __________________________________________________________________________________________________
    activation_59 (Activation)      (None, 4, 4, 96)     0           batch_normalization_59[0][0]     
    __________________________________________________________________________________________________
    conv2d_60 (Conv2D)              (None, 4, 4, 12)     10368       activation_59[0][0]              
    __________________________________________________________________________________________________
    dropout_59 (Dropout)            (None, 4, 4, 12)     0           conv2d_60[0][0]                  
    __________________________________________________________________________________________________
    concatenate_56 (Concatenate)    (None, 4, 4, 108)    0           concatenate_55[0][0]             
                                                                     dropout_59[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_60 (BatchNo (None, 4, 4, 108)    432         concatenate_56[0][0]             
    __________________________________________________________________________________________________
    activation_60 (Activation)      (None, 4, 4, 108)    0           batch_normalization_60[0][0]     
    __________________________________________________________________________________________________
    conv2d_61 (Conv2D)              (None, 4, 4, 12)     11664       activation_60[0][0]              
    __________________________________________________________________________________________________
    dropout_60 (Dropout)            (None, 4, 4, 12)     0           conv2d_61[0][0]                  
    __________________________________________________________________________________________________
    concatenate_57 (Concatenate)    (None, 4, 4, 120)    0           concatenate_56[0][0]             
                                                                     dropout_60[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_61 (BatchNo (None, 4, 4, 120)    480         concatenate_57[0][0]             
    __________________________________________________________________________________________________
    activation_61 (Activation)      (None, 4, 4, 120)    0           batch_normalization_61[0][0]     
    __________________________________________________________________________________________________
    conv2d_62 (Conv2D)              (None, 4, 4, 12)     12960       activation_61[0][0]              
    __________________________________________________________________________________________________
    dropout_61 (Dropout)            (None, 4, 4, 12)     0           conv2d_62[0][0]                  
    __________________________________________________________________________________________________
    concatenate_58 (Concatenate)    (None, 4, 4, 132)    0           concatenate_57[0][0]             
                                                                     dropout_61[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_62 (BatchNo (None, 4, 4, 132)    528         concatenate_58[0][0]             
    __________________________________________________________________________________________________
    activation_62 (Activation)      (None, 4, 4, 132)    0           batch_normalization_62[0][0]     
    __________________________________________________________________________________________________
    conv2d_63 (Conv2D)              (None, 4, 4, 12)     14256       activation_62[0][0]              
    __________________________________________________________________________________________________
    dropout_62 (Dropout)            (None, 4, 4, 12)     0           conv2d_63[0][0]                  
    __________________________________________________________________________________________________
    concatenate_59 (Concatenate)    (None, 4, 4, 144)    0           concatenate_58[0][0]             
                                                                     dropout_62[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_63 (BatchNo (None, 4, 4, 144)    576         concatenate_59[0][0]             
    __________________________________________________________________________________________________
    activation_63 (Activation)      (None, 4, 4, 144)    0           batch_normalization_63[0][0]     
    __________________________________________________________________________________________________
    conv2d_64 (Conv2D)              (None, 4, 4, 12)     15552       activation_63[0][0]              
    __________________________________________________________________________________________________
    dropout_63 (Dropout)            (None, 4, 4, 12)     0           conv2d_64[0][0]                  
    __________________________________________________________________________________________________
    concatenate_60 (Concatenate)    (None, 4, 4, 156)    0           concatenate_59[0][0]             
                                                                     dropout_63[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_64 (BatchNo (None, 4, 4, 156)    624         concatenate_60[0][0]             
    __________________________________________________________________________________________________
    activation_64 (Activation)      (None, 4, 4, 156)    0           batch_normalization_64[0][0]     
    __________________________________________________________________________________________________
    conv2d_65 (Conv2D)              (None, 4, 4, 12)     16848       activation_64[0][0]              
    __________________________________________________________________________________________________
    dropout_64 (Dropout)            (None, 4, 4, 12)     0           conv2d_65[0][0]                  
    __________________________________________________________________________________________________
    concatenate_61 (Concatenate)    (None, 4, 4, 168)    0           concatenate_60[0][0]             
                                                                     dropout_64[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_65 (BatchNo (None, 4, 4, 168)    672         concatenate_61[0][0]             
    __________________________________________________________________________________________________
    activation_65 (Activation)      (None, 4, 4, 168)    0           batch_normalization_65[0][0]     
    __________________________________________________________________________________________________
    conv2d_66 (Conv2D)              (None, 4, 4, 12)     18144       activation_65[0][0]              
    __________________________________________________________________________________________________
    dropout_65 (Dropout)            (None, 4, 4, 12)     0           conv2d_66[0][0]                  
    __________________________________________________________________________________________________
    concatenate_62 (Concatenate)    (None, 4, 4, 180)    0           concatenate_61[0][0]             
                                                                     dropout_65[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_66 (BatchNo (None, 4, 4, 180)    720         concatenate_62[0][0]             
    __________________________________________________________________________________________________
    activation_66 (Activation)      (None, 4, 4, 180)    0           batch_normalization_66[0][0]     
    __________________________________________________________________________________________________
    conv2d_67 (Conv2D)              (None, 4, 4, 12)     19440       activation_66[0][0]              
    __________________________________________________________________________________________________
    dropout_66 (Dropout)            (None, 4, 4, 12)     0           conv2d_67[0][0]                  
    __________________________________________________________________________________________________
    concatenate_63 (Concatenate)    (None, 4, 4, 192)    0           concatenate_62[0][0]             
                                                                     dropout_66[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_67 (BatchNo (None, 4, 4, 192)    768         concatenate_63[0][0]             
    __________________________________________________________________________________________________
    activation_67 (Activation)      (None, 4, 4, 192)    0           batch_normalization_67[0][0]     
    __________________________________________________________________________________________________
    conv2d_68 (Conv2D)              (None, 4, 4, 12)     20736       activation_67[0][0]              
    __________________________________________________________________________________________________
    dropout_67 (Dropout)            (None, 4, 4, 12)     0           conv2d_68[0][0]                  
    __________________________________________________________________________________________________
    concatenate_64 (Concatenate)    (None, 4, 4, 204)    0           concatenate_63[0][0]             
                                                                     dropout_67[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_68 (BatchNo (None, 4, 4, 204)    816         concatenate_64[0][0]             
    __________________________________________________________________________________________________
    activation_68 (Activation)      (None, 4, 4, 204)    0           batch_normalization_68[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_4 (AveragePoo (None, 2, 2, 204)    0           activation_68[0][0]              
    __________________________________________________________________________________________________
    flatten_1 (Flatten)             (None, 816)          0           average_pooling2d_4[0][0]        
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 10)           8170        flatten_1[0][0]                  
    ==================================================================================================
    Total params: 750,238
    Trainable params: 735,550
    Non-trainable params: 14,688
    __________________________________________________________________________________________________



```python
# ak - block 3 SGDRScheduler

schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-1,
                                     steps_per_epoch=2*(x_train.shape[0]//batch_size),
                                     lr_decay=1,
                                     cycle_length=5,
                                     mult_factor=1.5)

earlystopper = EarlyStopping(monitor='val_loss', min_delta=1, patience=20, verbose=1,mode='min')

```


```python
# determine Loss function and Optimizer
# decay=10e-4,momentum=0.9
sgd = SGD(momentum=1)

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```


```python
# ak - block 4 - image aug

# we can compare the performance with or without data augmentation
data_augmentation = True
callbacks_list=[schedule,earlystopper]
start = time.time()
if not data_augmentation:
    print('Not using data augmentation.')
    model_info = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=callbacks_list
        )
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by dataset std
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in 0 to 180 degrees
        width_shift_range=0.1,  # randomly shift images horizontally
        height_shift_range=0.1,  # randomly shift images vertically
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    
    model_info = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=3*(x_train.shape[0]//batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks_list
                       )

end = time.time()

```

    Using real-time data augmentation.
    Epoch 1/50
    2343/2343 [==============================] - 254s 108ms/step - loss: 0.5309 - acc: 0.8149 - val_loss: 0.5867 - val_acc: 0.8167
    Epoch 2/50
    2343/2343 [==============================] - 238s 102ms/step - loss: 0.4685 - acc: 0.8369 - val_loss: 0.5701 - val_acc: 0.8345
    Epoch 3/50
    2343/2343 [==============================] - 236s 101ms/step - loss: 0.4142 - acc: 0.8558 - val_loss: 0.5279 - val_acc: 0.8498
    Epoch 4/50
    2343/2343 [==============================] - 233s 99ms/step - loss: 0.3940 - acc: 0.8629 - val_loss: 0.5297 - val_acc: 0.8478
    Epoch 5/50
    2343/2343 [==============================] - 235s 100ms/step - loss: 0.4137 - acc: 0.8567 - val_loss: 0.5443 - val_acc: 0.8392
    Epoch 6/50
    2343/2343 [==============================] - 233s 99ms/step - loss: 0.4730 - acc: 0.8353 - val_loss: 0.6962 - val_acc: 0.8053
    Epoch 7/50
    2343/2343 [==============================] - 234s 100ms/step - loss: 0.4353 - acc: 0.8492 - val_loss: 0.6060 - val_acc: 0.8287
    Epoch 8/50
    2343/2343 [==============================] - 233s 99ms/step - loss: 0.3986 - acc: 0.8624 - val_loss: 0.4786 - val_acc: 0.8637
    Epoch 9/50
    2343/2343 [==============================] - 235s 100ms/step - loss: 0.3611 - acc: 0.8754 - val_loss: 0.4620 - val_acc: 0.8662
    Epoch 10/50
    2343/2343 [==============================] - 234s 100ms/step - loss: 0.3369 - acc: 0.8826 - val_loss: 0.4636 - val_acc: 0.8695
    Epoch 11/50
    2343/2343 [==============================] - 232s 99ms/step - loss: 0.3287 - acc: 0.8860 - val_loss: 0.4743 - val_acc: 0.8681
    Epoch 12/50
    2343/2343 [==============================] - 232s 99ms/step - loss: 0.3367 - acc: 0.8836 - val_loss: 0.4802 - val_acc: 0.8633
    Epoch 13/50
    2343/2343 [==============================] - 231s 99ms/step - loss: 0.3526 - acc: 0.8770 - val_loss: 0.7248 - val_acc: 0.8155
    Epoch 14/50
    2343/2343 [==============================] - 231s 99ms/step - loss: 0.4104 - acc: 0.8581 - val_loss: 0.5850 - val_acc: 0.8386
    Epoch 15/50
    2343/2343 [==============================] - 232s 99ms/step - loss: 0.3899 - acc: 0.8644 - val_loss: 0.6099 - val_acc: 0.8287
    Epoch 16/50
    2343/2343 [==============================] - 231s 98ms/step - loss: 0.3636 - acc: 0.8737 - val_loss: 0.4882 - val_acc: 0.8629
    Epoch 17/50
    2343/2343 [==============================] - 233s 100ms/step - loss: 0.3417 - acc: 0.8810 - val_loss: 0.5108 - val_acc: 0.8605
    Epoch 18/50
    2343/2343 [==============================] - 232s 99ms/step - loss: 0.3148 - acc: 0.8899 - val_loss: 0.4866 - val_acc: 0.8666
    Epoch 19/50
    2343/2343 [==============================] - 233s 99ms/step - loss: 0.2943 - acc: 0.8980 - val_loss: 0.4628 - val_acc: 0.8750
    Epoch 20/50
    2343/2343 [==============================] - 230s 98ms/step - loss: 0.2768 - acc: 0.9033 - val_loss: 0.4160 - val_acc: 0.8876
    Epoch 21/50
    2343/2343 [==============================] - 232s 99ms/step - loss: 0.2701 - acc: 0.9062 - val_loss: 0.4204 - val_acc: 0.8872
    Epoch 22/50
    2343/2343 [==============================] - 238s 102ms/step - loss: 0.2700 - acc: 0.9058 - val_loss: 0.4226 - val_acc: 0.8871
    Epoch 23/50
    2343/2343 [==============================] - 255s 109ms/step - loss: 0.2731 - acc: 0.9045 - val_loss: 0.4263 - val_acc: 0.8850
    Epoch 24/50
    2343/2343 [==============================] - 252s 108ms/step - loss: 0.2823 - acc: 0.9013 - val_loss: 0.5276 - val_acc: 0.8658
    Epoch 25/50
    2343/2343 [==============================] - 238s 102ms/step - loss: 0.2964 - acc: 0.8965 - val_loss: 0.5256 - val_acc: 0.8613
    Epoch 26/50
    2343/2343 [==============================] - 236s 101ms/step - loss: 0.3504 - acc: 0.8767 - val_loss: 0.5724 - val_acc: 0.8462
    Epoch 27/50
    2343/2343 [==============================] - 236s 101ms/step - loss: 0.3374 - acc: 0.8825 - val_loss: 0.7110 - val_acc: 0.8250
    Epoch 28/50
    2343/2343 [==============================] - 235s 100ms/step - loss: 0.3218 - acc: 0.8888 - val_loss: 0.4533 - val_acc: 0.8737
    Epoch 29/50
    2343/2343 [==============================] - 233s 100ms/step - loss: 0.3034 - acc: 0.8944 - val_loss: 0.5155 - val_acc: 0.8627
    Epoch 30/50
    2343/2343 [==============================] - 234s 100ms/step - loss: 0.2914 - acc: 0.8983 - val_loss: 0.5812 - val_acc: 0.8507
    Epoch 31/50
    2343/2343 [==============================] - 236s 101ms/step - loss: 0.2763 - acc: 0.9035 - val_loss: 0.4465 - val_acc: 0.8790
    Epoch 32/50
    2343/2343 [==============================] - 237s 101ms/step - loss: 0.2610 - acc: 0.9081 - val_loss: 0.4949 - val_acc: 0.8727
    Epoch 33/50
    2343/2343 [==============================] - 235s 100ms/step - loss: 0.2471 - acc: 0.9138 - val_loss: 0.4730 - val_acc: 0.8826
    Epoch 34/50
    2343/2343 [==============================] - 234s 100ms/step - loss: 0.2332 - acc: 0.9187 - val_loss: 0.4102 - val_acc: 0.8924
    Epoch 35/50
    2343/2343 [==============================] - 232s 99ms/step - loss: 0.2239 - acc: 0.9215 - val_loss: 0.4231 - val_acc: 0.8901
    Epoch 36/50
    2343/2343 [==============================] - 239s 102ms/step - loss: 0.2171 - acc: 0.9239 - val_loss: 0.4048 - val_acc: 0.8956
    Epoch 37/50
    2343/2343 [==============================] - 236s 101ms/step - loss: 0.2149 - acc: 0.9248 - val_loss: 0.4112 - val_acc: 0.8954
    Epoch 38/50
    2343/2343 [==============================] - 254s 108ms/step - loss: 0.2145 - acc: 0.9247 - val_loss: 0.4106 - val_acc: 0.8943
    Epoch 39/50
    2343/2343 [==============================] - 256s 109ms/step - loss: 0.2110 - acc: 0.9264 - val_loss: 0.4210 - val_acc: 0.8926
    Epoch 40/50
    2343/2343 [==============================] - 250s 107ms/step - loss: 0.2178 - acc: 0.9242 - val_loss: 0.4427 - val_acc: 0.8911
    Epoch 41/50
    2343/2343 [==============================] - 244s 104ms/step - loss: 0.2235 - acc: 0.9217 - val_loss: 0.3965 - val_acc: 0.8946
    Epoch 42/50
    2343/2343 [==============================] - 257s 110ms/step - loss: 0.2333 - acc: 0.9178 - val_loss: 0.4633 - val_acc: 0.8840
    Epoch 43/50
    2343/2343 [==============================] - 250s 107ms/step - loss: 0.2418 - acc: 0.9158 - val_loss: 0.5136 - val_acc: 0.8755
    Epoch 44/50
    2343/2343 [==============================] - 251s 107ms/step - loss: 0.2931 - acc: 0.8978 - val_loss: 0.6903 - val_acc: 0.8287
    Epoch 45/50
    2343/2343 [==============================] - 248s 106ms/step - loss: 0.2836 - acc: 0.9002 - val_loss: 0.5938 - val_acc: 0.8546
    Epoch 46/50
    2343/2343 [==============================] - 251s 107ms/step - loss: 0.2761 - acc: 0.9039 - val_loss: 0.4685 - val_acc: 0.8716
    Epoch 47/50
    2343/2343 [==============================] - 238s 102ms/step - loss: 0.2680 - acc: 0.9066 - val_loss: 0.4319 - val_acc: 0.8861
    Epoch 48/50
    2343/2343 [==============================] - 236s 101ms/step - loss: 0.2588 - acc: 0.9093 - val_loss: 0.6271 - val_acc: 0.8522
    Epoch 49/50
    2343/2343 [==============================] - 236s 101ms/step - loss: 0.2463 - acc: 0.9141 - val_loss: 0.6127 - val_acc: 0.8591
    Epoch 50/50
    2343/2343 [==============================] - 235s 100ms/step - loss: 0.2380 - acc: 0.9168 - val_loss: 0.4153 - val_acc: 0.8928



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-15-c729612a7691> in <module>
         59 
         60 end = time.time()
    ---> 61 print ("Model took %0.2f seconds to train"%(end - start)/3600)
         62 
         63 plot_model_history(model_info)


    TypeError: unsupported operand type(s) for /: 'str' and 'int'



```python
print ("Model took %0.2f hours to train"%((end - start)/3600))

plot_model_history(model_info)
```

    Model took 3.31 hours to train



![png](output_15_1.png)



```python
# lr_reducer      = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
#                                    cooldown=0, patience=10, min_lr=0.5e-6)
# early_stopper   = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=20)

# model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=50,
#                     verbose=1,
#                     validation_data=(x_test, y_test),
#                     callbacks=[lr_reducer,early_stopper])
```


```python
# Test the model
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

    10000/10000 [==============================] - 4s 424us/step
    Test loss: 0.5136427744150162
    Test accuracy: 0.8755



```python
# Save the trained weights in to .h5 format
model.save_weights("DNST_model_weights_v6_5.h5")
print("Saved model to disk")
```

    Saved model to disk



```python

```
