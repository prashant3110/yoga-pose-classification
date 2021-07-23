import tensorflow as tf
import numpy as np
import pickle
import cv2

dict  = {1: 'Downdog', 2: 'Goddess', 3: 'Plank',4: 'Tree',5: 'Warrior2'}

## output conversion function
def out_conversion(out_array):
    for i in range(5):
        if out_array[i] == 1.:
            return dict[i+1]

## reading the image and resizing for the model

image = cv2.imread('DATASET/TEST/tree/00000003.jpg')

image = cv2.resize(image,(150,150))

image = np.array(image)

image = image.reshape(1,150,150,3)



## loading the saved model.h5 file
model = tf.keras.models.load_model('model.h5')


##pickeling the model for deployment

#pickle.dump(model,open('model_pl.sav','wb'))
##  Predicting the one hot encoding as the output
out_arr = model.predict(image)[0]
print(out_arr)
## converting the one hot encoding to the class output
print(f'Predicted class is : {out_conversion(out_arr)}')