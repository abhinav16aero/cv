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

capture = cv2.VideoCapture(0)

# Get initial frame
response, first_frame = capture.read()
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(first_frame)
hsv[..., 1] = 255

while True:
    response, frame = capture.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculates dense optical flow
    dense_flow = cv2.calcOpticalFlowFarneback(first_gray, frame, None, 0.5, 4, 13, 5, 7, 1.5, 0)

    # Calculate speed and angle(theta) of motion
    speed, theta = cv2.cartToPolar(dense_flow[..., 0], dense_flow[..., 1])
    hsv[..., 0] = theta * (180 / np.pi)  # Adjusted the angle conversion
    hsv[..., 2] = cv2.normalize(speed, None, 0, 255, cv2.NORM_MINMAX)
    final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('Dense Optical Flow', final)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    first_gray = frame

cv2.destroyAllWindows()
capture.release()
