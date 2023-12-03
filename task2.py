import cv2
import numpy as np
import os

def stitch_images(images):
    stitcher = cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(images)
    if status == cv2.Stitcher_OK:
        return stitched
    else:
        print('Image stitching failed.')
        return None

def find_objects_in_scene(stitched_image, object_images):
    # Initialize the feature detector
    sift = cv2.SIFT_create()

    # Detect features in the stitched image
    stitched_keypoints, stitched_descriptors = sift.detectAndCompute(stitched_image, None)

    # Initialize the matcher
    matcher = cv2.BFMatcher()

    # Dictionary to store the locations of objects
    object_locations = {}

    # Loop over the object images
    for idx, obj_image in enumerate(object_images):
        obj_keypoints, obj_descriptors = sift.detectAndCompute(obj_image, None)

        # Match descriptors between object image and stitched scene
        matches = matcher.knnMatch(obj_descriptors, stitched_descriptors, k=2)

        # Filter matches using the Lowe's ratio test
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        # If enough matches are found, we proceed to find the homography
        if len(good_matches) > MIN_MATCH_COUNT:
            # Extract location of good matches
            obj_pts = np.float32([obj_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            scene_pts = np.float32([stitched_keypoints[m.trainIdx].pt for m in goodmatches]).reshape(-1, 1, 2)

            # Find homography
            H,  = cv2.findHomography(obj_pts, scenepts, cv2.RANSAC, 5.0)

            # Use homography to get the object's location in the scene
            h, w,  = obj_image.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, H)

            # Store the object location
            objectlocations[f"Object{idx}"] = dst

    return object_locations

folder_path = "scene_images"
folder_path2 = "object_images"

# Load images
scene_images = [cv2.imread(os.path.join(folder_path, f"S{i}.JPG")) for i in range(1, 21)]
object_images = [cv2.imread(os.path.join(folder_path2, f"O{i}.JPG")) for i in range(1, 27)]

#Stitch images together to create a larger scene
stitched_scene = stitch_images(scene_images)

if stitched_scene is not None:
    object_locations = find_objects_in_scene(stitched_scene, object_images)

#Draw the located objects on the stitched image for visualization
    for obj_name, corners in object_locations.items():
        # Draw a bounding box around the detected object
        stitched_scene = cv2.polylines(stitched_scene, [np.int32(corners)], True, (0, 255, 0), 3, cv2.LINE_AA)

#Show or save the stitched image with located objects
    cv2.imwrite('stitched_scene_with_objects.jpg', stitched_scene)