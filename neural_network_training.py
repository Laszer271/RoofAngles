import utils 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout
from tensorflow import math
from tensorflow import concat
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import ensemble


def random_brightness(arr, mean=0.0, std=0.30):
    brightness_change = np.random.normal(mean, std)
    arr = arr.astype(float)
    arr += brightness_change
    arr = arr.clip(-1.0, 1.0)
    return arr

def residual_block(x, filters):
    x1 = Conv2D(filters, (5, 5), padding="same")(x)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.2)(x1)
    x = x + x1
    return x

def residual_embedded_functions(x, neuron_counts):
    x1 = math.divide(1.0, x + 1e-100)
    x1 = LayerNormalization()(x1)
    x2 = math.multiply(x, x)
    x2 = LayerNormalization()(x2)
    x3 = math.sqrt(math.abs(x))
    x3 = LayerNormalization()(x3)
    x4 = LayerNormalization()(x)
    funs = concat([x1, x2, x3, x4], 1)
    for neuron_count in neuron_counts:
        funs = Dense(neuron_count, activation='relu')(funs)
        #funs = LayerNormalization()(funs)
    return x4 + funs

def build_model(width, height, channels):
     inputShape = (height, width, channels)
     inputs = Input(shape=inputShape)
     
     x = Conv2D(16, (5, 5), padding="same")(inputs)
     x = LeakyReLU(alpha=0.2)(x)
     x = Conv2D(16, (5, 5), strides=2, padding="same")(inputs)
     x = LeakyReLU(alpha=0.2)(x)
     x = Dropout(0.5)(x)
     x = Conv2D(16, (5, 5), strides=2, padding="same")(x)
     x = LeakyReLU(alpha=0.2)(x)
     x = Dropout(0.5)(x)
     x = Conv2D(32, (5, 5), strides=2, padding="same")(x)
     x = LeakyReLU(alpha=0.2)(x)
     x = Dropout(0.5)(x)
     x = Conv2D(64, (5, 5), strides=2, padding="same")(x)
     x = LeakyReLU(alpha=0.2)(x)
     x = Dropout(0.5)(x)
     x = Conv2D(128, (5, 5), strides=2, padding="same")(x)
     x = LeakyReLU(alpha=0.2)(x)
     x = Dropout(0.5)(x)
     '''
     for i in range(20):
         x = residual_block(x, 128)
    '''
     x = Conv2D(128, (5, 5), strides=2, padding="same")(x)
     x = LeakyReLU(alpha=0.2)(x)
     #x = Dropout(0.4)(x)
     x = Flatten()(x)
     encoding = Dense(32, activation='relu')(x)
     
     for i in range(10):
         encoding = residual_embedded_functions(encoding, (128, 32))
     
     x = Dense(1)(encoding)
     model = Model(inputs, x, name="CNN")
     encoder = Model(inputs, encoding, name='encoder')
     return model, encoder

# getting dataset:
dataset = utils.get_dataset('./photos/streetview', 'Prich jobs stats.xlsx',
                            sheet_name='code-address-roofPitch')
y = dataset['front roof angle'].values
X = []
for path in dataset['Paths']:
    array = np.array(Image.open(path))
    X.append(array)
X = np.reshape(X, (len(X), X[0].shape[0], X[0].shape[1], X[0].shape[2]))

# splitting dataset into train and test:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# data augmentation:
generator = ImageDataGenerator(horizontal_flip=True, preprocessing_function=random_brightness)

X_min = X_train.min()
X_train -= X_min
X_max = X_train.max()
X_train = X_train / X_max * 2.0 - 1.0
X_test = (X_test - X_min) / X_max * 2.0 - 1.0
y_mean = y_train.mean()
y_train -= y_mean
y_test -= y_mean

# training model:
EPOCHS = 25
model, encoder = build_model(X[0].shape[1], X[0].shape[0], X[0].shape[2])
model.compile(loss="mae", optimizer='adam')
h = model.fit(generator.flow(X_train, y_train, batch_size=32),
              epochs=EPOCHS, validation_data=(X_test, y_test))

# evaluating:
preds = model.predict(X_test).flatten()
mae = np.abs(preds - y_test).mean()
mse = np.square(preds - y_test).mean()
print("MSE: %.4f" % mse)
print("MAE: %.4f" % mae)
print('min loss: %.4f' % np.min(h.history["val_loss"]))

# basline:
mae = np.abs(y_test).mean()
mse = np.square(y_test).mean()
print('from baseline:')
print("MSE: %.4f" % mse)
print("MAE: %.4f" % mae)

# plotting training:
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, h.history["loss"], label="train_loss")
plt.plot(N, h.history["val_loss"], label="val_loss")
plt.plot(N, [mae]*EPOCHS, label="baseline")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig('training_plot.png')

# tree:
params = {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 20,
      'learning_rate': 0.01, 'loss': 'ls', 'criterion': 'mae'}
clf = ensemble.GradientBoostingRegressor(**params)
temp = encoder.predict(X_train)
clf.fit(temp, y_train)

test_temp = encoder.predict(X_test)
preds = clf.predict(test_temp)

mae = np.abs(preds - y_test).mean()
mse = np.square(preds - y_test).mean()
print('from trees:')
print("MSE: %.4f" % mse)
print("MAE: %.4f" % mae)

preds = model.predict(X_train)

