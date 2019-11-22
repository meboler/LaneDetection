import numpy as np
import matplotlib.pyplot as plt
import cv2

import LaneDetector

if __name__ == '__main__':
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    filename = "000000.png"
    img = cv2.imread(filename, 0)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    height = img.shape[0]
    width = img.shape[1]

    # Cropping out upper half b/c doesn't map to road
    cropped_img = img[img.shape[0]/2 : -1, :]

    h = 1.65 # meters
    t = np.asarray([0, 0, -h], np.float32)
    # Map from world frame to camera frame
    R = np.asarray([[0, -1, 0],
                    [0, 0, -1],
                    [1, 0, 0]], np.float32)

    K = np.asarray([[718.856, 0, 607.1928],
                    [0, 718.856, 185.2157],
                    [0, 0, 1]], np.float32)
    D = np.asarray([0, 0, 0, 0, 0], np.float32)
    FOV_h = np.radians(91.2)
    FOV_v = np.radians(65.5)
    params = LaneDetector.CameraParams()
    params.K = K
    params.D = D
    params.FOV_h = FOV_h
    params.FOV_v = FOV_v
    params.height = height
    params.width = width

    det = LaneDetector.LaneDetector(R, t, params)
    warped_img = det.perspective_warp(img)

    plt.figure(1)
    plt.subplot(211)
    plt.imshow(img)

    plt.subplot(212)
    plt.imshow(warped_img)

    plt.show()

