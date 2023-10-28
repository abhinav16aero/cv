# Copyright 2023 Abhinav Kumar. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import cv2
import numpy as np
cv2.ocl.setUseOpenCL(False)

class ObjectDetectorORB(object):
    """
    Object Detection using ORB (Oriented FAST and Rotated BRIEF)
    """
    def __init__(self, original_image, template):
        self.original_image = original_image
        self.template = template

    def detect_objects(self):
        """
        Compares the original image with the template and finds the number of ORB matches.
        """
        img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        template = self.template

        # ORB detector with 1000 keypoints and a scale of 1.2
        orb = cv2.ORB_create(1000, 1.2)

        # Get keypoints and descriptors (k, d) using ORB
        k1, d1 = orb.detectAndCompute(img, None)
        k2, d2 = orb.detectAndCompute(template, None)

        # Define a matcher for ORB
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Get all matches
        matches = matcher.match(d1, d2)

        # Save all matches and sort them in ascending order
        matches = sorted(matches, key=lambda val: val.distance)

        return len(matches)


class ObjectDetectorSIFT(object):
    """
    Object Detection using SIFT (Scale-Invariant Feature Transform)
    """
    def __init(self, original_image, template):
        self.original_image = original_image
        self.template = template

    def detect_objects(self):
        """
        Compares the original image with the template and finds the number of SIFT matches.
        """
        img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        tem = self.template

        # SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()

        # get keypoints and discriptors (k,d) using sift
        k1, d1 = sift.detectAndCompute(img, None)
        k2, d2 = sift.detectAndCompute(tem, None)

        # flann matcher
        FLANN_INDEX_KDTREE = 0
        index_param = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)
        search_param = dict(checks=100)

        flann = cv2.FlannBasedMatcher(index_param, search_param)
        # d1, d2 = None, None

        # getting all matches using kNN
        matches = flann.knnMatch(d1, d2, k=2)

        # Save all matches - Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < (0.7 * n.distance):
                good_matches.append(m)

        return len(good_matches)


def main():
    # Loading the original template image
    template_image_path = 'Presentation/1.png'
    template_image = cv2.imread(template_image_path)
    cv2.imshow("Template Image", template_image)
    cv2.waitKey(0)

    # Initialize video capture
    capture = cv2.VideoCapture(0)

    # Choose the type of object detection algorithm
    # ObjectDetectorORB: ORB (Oriented FAST and Rotated BRIEF)
    # ObjectDetectorSIFT: SIFT (Scale-Invariant Feature Transform)
    use_this = ObjectDetectorORB

    while True:
        # Capture an image from the webcam
        response, frame = capture.read()

        # Get the height and width of webcam images
        height, width = frame.shape[:2]

        # Define box dimensions for the area of interest
        top_x = int(width / 3)
        top_y = int((height / 2) + (height / 4))
        bottom_x = int((width / 3) * 2)
        bottom_y = int((height / 2) - (height / 4))

        # Draw a rectangle around the specified area
        cv2.rectangle(frame, (top_x, top_y), (bottom_x, bottom_y), (0, 255, 0), 3)

        # Extract the area of interest
        area_of_interest = frame[bottom_y:top_y, top_x:bottom_x]

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Create an object for the selected detection class
        object_detector = use_this(area_of_interest, template_image)
        matches = object_detector.detect_objects()

        # Update results and show them on the screen
        text = "We have {} matches".format(str(matches))
        cv2.putText(frame, text, (300, 630), cv2.FONT_ITALIC, 2, (0, 0, 0), 8)

        # Threshold to show object detection
        threshold = 500

        if matches > threshold:
            cv2.rectangle(frame, (top_x, top_y), (bottom_x, bottom_y), (255, 255, 0), 3)
            result = "WOW!! Found!"
            cv2.putText(frame, result, (750, 50), cv2.FONT_ITALIC, 2, (0, 255, 0), 2)

        cv2.imshow("OBJECT DETECTION USING ORB/SIFT", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
