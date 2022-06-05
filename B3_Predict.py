#%%

import numpy as np
import cv2

from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import datasets

model_architecture = "stored/digit_config.json"
model_weights = "stored/digit_weight.h5"
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)
optim = Adam()
model.compile(loss="categorical_crossentropy",
              optimizer=optim,
              metrics=["accuracy"])
'''
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot
'''
labels = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
    'Sneaker', 'Bag', 'Ankle boot'
]


#%%
def reg(X_Test):
    if (X_Test.ndim == 3):
        X_Test = cv2.cvtColor(X_Test, cv2.COLOR_BGR2GRAY)
    X_Test = cv2.resize(X_Test, (28, 28))
    # reshape
    X_test = X_Test.reshape((28, 28, 1))

    # normalize
    X_test = X_test / 255.0

    # cast
    X_test = X_test.astype('float32')
    # print(X_Test)

    predictions = model.predict(np.array([X_test]))
    # print(predictions)
    predictions = predictions[0]
    return labels[np.argmax(predictions)]


# #%%
# result = cv2.imread('./image/shirt.png')
# print(result)

# print(reg(result))
# cv2.imshow('test', result)
# cv2.waitKey()
# cv2.destroyAllWindows()

# %%
# (X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()
# #%%
# id = 6
# print(reg(X_test[id]))
# print(y_test[id])
# # pic = cv2.resize(X_test[id], (480, 680))
# print(X_test[id])
# cv2.imshow('test', X_test[id])
# cv2.waitKey()
# cv2.destroyAllWindows()
# %%
