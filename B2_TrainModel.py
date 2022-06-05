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

# data: shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()

#%%
X_test_img = X_test

# reshape
X_train = X_train.reshape((60000, 28, 28, 1))
X_test = X_test.reshape((10000, 28, 28, 1))

# normalize
X_train, X_test = X_train / 255.0, X_test / 255.0

# cast
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

for i in range(0, 10):
    index = np.random.randint(0, 9999)
    sample = np.array([X_test[index]])
    predictions = model.predict(sample)
    predictions = predictions[0]
    print(predictions)
    print("Nhan dang:", np.argmax(predictions), type(predictions))
    print(y_test[index])
    image = X_test_img[index]
    m = cv2.moments(image)
    cy = int(m["m10"] / m["m00"])
    cx = int(m["m01"] / m["m00"])
    print(cx, cy)

    cv2.imshow('digit', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#%%