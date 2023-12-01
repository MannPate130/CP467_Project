"""
-----------------------------------------------
CP467 Project
-----------------------------------------------
Author:  Mann Patel, Bilal Asad, Thenukan Velvelicham
ID:      210852760, 
Emails:   pate5276@mylaurier.ca, 
Term:    Fall 2023
updated: "2023-11-28"
-----------------------------------------------
"""
import cv2
import numpy as np

# Load YOLOv3 model and configuration
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load individual scene images (JPG format)
num_scene_images = 10  # Update with the actual number of scene images
individual_scene_images = [cv2.imread(f"scene_image_{i}.jpg") for i in range(1, num_scene_images + 1)]

# Load individual object images (PNG format)
num_object_images = 20  # Update with the actual number of object images
individual_object_images = [cv2.imread(f"object_image_{i}.png", cv2.IMREAD_UNCHANGED) for i in range(1, num_object_images + 1)]

# Function to stitch images using simple image concatenation
def stitch_images(images):
    # Assume all images have the same dimensions
    height, width = images[0].shape[:2]
    stitched_image = np.zeros((height, width * len(images), 3), dtype=np.uint8)
    
    for i, image in enumerate(images):
        stitched_image[:, i * width:(i + 1) * width] = image

    return stitched_image

# Detect objects in scene images using YOLO and stitch them together
detected_objects_list = []
for scene_image in individual_scene_images:
    blob = cv2.dnn.blobFromImage(scene_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    detected_objects = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Consider detections with confidence > 0.5
                # Object detected, get its class label and bounding box
                label = classes[class_id]
                x, y, w, h = list(map(int, detection[0:4] * np.array([scene_image.shape[1], scene_image.shape[0], scene_image.shape[1], scene_image.shape[0]])))
                detected_objects.append((label, (x, y, w, h)))
    
    detected_objects_list.append(detected_objects)

stitched_scene_image = stitch_images(individual_scene_images)

# Identify the location of each object in the stitched scene image
# This would involve placing detected objects accurately on the stitched image
# Additional processing is required to map object coordinates to the stitched image

# You'll need to handle the placement of detected objects accurately onto the stitched image,
# considering potential scaling or transformations due to the stitching process.
# The accuracy of this process highly depends on maintaining consistency in image dimensions, perspectives, and object scales across the different scene images.