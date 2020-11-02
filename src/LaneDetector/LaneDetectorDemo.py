import numpy as np
import matplotlib.pyplot as plt
import cv2
import LaneDetector 
if __name__ == '__main__':
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    filename = "f1tenth4.jpg" 
    img = cv2.imread(filename)
    print img.size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height = img.shape[0]
    width = img.shape[1]

    # Cropping out upper half b/c doesn't map to road
    cropped_img = img[:, 0 : width//2]

    h = 0.35 # meters
    t = np.asarray([[0], [0], [h]], np.float32)
    # Map from world frame to camera frame
    pitch = 30 # deg
    p = np.deg2rad(pitch)
    R = np.asarray([[0, -1, 0],
                    [-np.sin(p), 0, -1*np.cos(p)],
                    [1*np.cos(p), 0, -np.sin(p)]], np.float32)

    K = np.asarray([[617.2716, 0, 327.2818],
                    [0, 617.1263, 245.0939],
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
    for i in range(1):
        mask_img = det.filter(img)
        blur_img = det.blur_mask(mask_img)
        warped_img = det.perspective_warp(blur_img)
        #warped_img = warped_img[:, 0 : width // 2]

        center = det.fit_center(warped_img)

        waypoints = det.generate_waypoints(img, center)
    

    
    plt.figure(1)
    plt.subplot(321)
    plt.imshow(img)

    plt.subplot(322)
    plt.imshow(det.perspective_warp(img))

    plt.subplot(323)
    plt.imshow(mask_img, cmap='gray')

    plt.subplot(325)
    plt.imshow(warped_img, cmap='gray')

    plt.subplot(326)
    plt.imshow(det.draw_lane(img, center))
    plt.show()
    
    
