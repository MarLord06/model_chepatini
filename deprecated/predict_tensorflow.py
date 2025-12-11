import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("unet_severstal.h5", compile=False)

def predict_mask(image_path):
    img = cv2.imread(image_path)[:,:,::-1]
    img_resized = cv2.resize(img, (512,512)) / 255.0
    img_tensor = np.expand_dims(img_resized, axis=0)

    pred = model.predict(img_tensor)[0]
    pred = (pred > 0.5).astype(np.uint8)

    return pred

mask = predict_mask("test.jpg")
