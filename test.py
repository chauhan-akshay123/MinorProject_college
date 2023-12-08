import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load the pre-trained MobileNetV2 model from TensorFlow Hub
model_url = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
model = hub.load(model_url)

# Load an image
image_path = r"C:\Users\Akshay Chauhan\Downloads\birds.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize the image to the input size expected by the model
input_image = tf.image.resize_with_pad(tf.convert_to_tensor(image), 224, 224)
input_image = tf.expand_dims(input_image, axis=0)  # Add batch dimension

# Perform image classification
predictions = model(input_image)

# Get the class label with the highest probability
predicted_label = np.argmax(predictions.numpy()[0])

# Load the ImageNet labels (labels for MobileNetV2)
labels_path = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
labels = tf.keras.utils.get_file("ImageNetLabels.txt", labels_path)
with open(labels) as f:
    class_labels = f.read().splitlines()

# Get the class name for the predicted label
class_name = class_labels[predicted_label]

# Display the result
cv2.putText(image, f"Prediction: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
cv2.imshow('Image Detection', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()