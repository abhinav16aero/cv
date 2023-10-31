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

import numpy as np
import cv2

# Open the camera capture
capture = cv2.VideoCapture(0)

# Parameters for Kanade optical flow
params_kanade = dict(winSize=(10, 10),  # Smaller window size for better tracking of small features
                     maxLevel=5,  # Increased maxLevel for more pyramids
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Parameters for corner detection
params_corner = dict(maxCorners=500,  # Increased maxCorners for more feature points
                     qualityLevel=0.01,  # Lower quality level for detecting more points
                     minDistance=10)

# Select random colors for drawing trails
c = np.random.randint(0, 255, (500, 3))

# Get the first frame
response, first_frame = capture.read()
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Find corners in the first frame
first_frame_corners = cv2.goodFeaturesToTrack(first_gray, mask=None, **params_corner)

# Create an empty kernel for drawing trails
kernel = np.zeros_like(first_frame)

while True:
    # Read a frame from the camera and flip it horizontally
    response, frame = capture.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using the Kanade Lucas-Tomasi algorithm
    corners, status, _ = cv2.calcOpticalFlowPyrLK(first_gray, gray, first_frame_corners, None, **params_kanade)

    # Get good features (old and new)
    new_features = corners[status == 1]
    prev_features = first_frame_corners[status == 1]

    # Draw trails for object tracking
    final = frame.copy()  # Initialize 'final' with a copy of the current frame

    for i, (new, prev) in enumerate(zip(new_features, prev_features)):
        x0, y0 = new.ravel()
        x1, y1 = prev.ravel()

        # Draw lines (trails) for optical flow
        final = cv2.line(final, (x0, y0), (x1, y1), c[i].tolist(), 2)

        # Draw circles at feature points
        final = cv2.circle(final, (x0, y0), 5, c[i].tolist(), -1)

    # Display the result
    cv2.imshow('Kanade Optical Flow', final)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    first_gray = gray.copy()
    first_frame_corners = new_features.reshape(-1, 1, 2)

# Release the camera and close all OpenCV windows
cv2.destroyAllWindows()
capture.release()
