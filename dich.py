import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model('fruit.h5')

class_labels = ['fresh_apple', 'fresh_banana', 'fresh_orange', 'rotten_apple', 'rotten_banana', 'rotten_orange']

def predict_fruit(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    return class_labels[predicted_class], confidence

if __name__ == "__main__":
    image_path = 'img.png'
    label, confidence = predict_fruit(image_path)
    print(f'Predicted: {label} with confidence {confidence:.2f}')