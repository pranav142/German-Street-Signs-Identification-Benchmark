import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def print_img(img):
    plt.figure(figsize=(10,10))
    plt.tight_layout()
    plt.imshow(img)
    plt.show()

def predict_with_model(model, img_path):

    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels = 3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize(img, [60,60])
    print_img(img)
    img = tf.expand_dims(img, axis=0) 

    prediction = model.predict(img)
    print(prediction)
    prediction = np.argmax(prediction)

    return prediction

if __name__ == "__main__":
    img_path = "C:\\Users\pknad\OneDrive\Documents\Machine_Learning\Signs_Data\Training_Data\\val\\12\\00012_00000_00000.png"

    model = tf.keras.models.load_model("C:\\Users\pknad\OneDrive\Documents\Machine_Learning\Street Signs\SavedModels")
    prediction = predict_with_model(model, img_path)

    print(f"Prediction = {prediction}")