import numpy as np # written for 1.11
import cv2 # written for 3.3.1

def coords_to_h_transform(coords):
    cols = coords.shape[1]
    ones = np.ones((1, cols))
    coords_h = np.vstack((coords, ones))
    return coords_h

def coords_h_to_coords_transform(coords_h):
    coords = np.divide(coords_h, coords_h[-1,:])
    return coords[:-1,:]

class CameraParams:
    # Hold K, D, FOV
    def __init__(self):
        pass

class LaneDetector:
    """
    This class detects lanes in an image and returns the polynomials that 
    describe them.
    """
    
    def __init__(self, R = np.eye(3), t = np.zeros((3)), camera_params = CameraParams()):
        self.R = R
        self.t = t
        
        self.T = np.eye(4)
        self.T[0:3,0:3] = self.R
        self.T[0:3,3] = self.t
        
        self.params = camera_params
        
        self.config()

    def config(self):
        # Get ground plane info
        ground_plane_normal_w = np.array([[0],[0],[1]])
        self.normal_c = np.dot(self.R, ground_plane_normal_w)
        origin_w = np.asarray([[0],[0],[0]], np.float32)
        origin_c = self.map_w_to_c(origin_w)
        self.dot_const_c = np.dot(self.normal_c.T, origin_c)

        lane_vec_c = np.dot(self.R, np.asarray([[1],[0],[0]], np.float32))
        lane_vec_pix_c = np.dot(self.params.K, lane_vec_c)
        self.vanishing_point = coords_h_to_coords_transform(lane_vec_pix_c)

        # Get image config info
        height = self.params.height
        width = self.params.width
        self.ipm_top = self.vanishing_point[1] + 20 # experiment

        self.bottom_limit = 415
        self.top_limit = 260 #250
        self.bottom_left_limit = 70
        self.bottom_right_limit = 625
        self.top_left_limit = 295 #310
        self.top_right_limit = 385 #370
        self.src_pixels = np.asarray(
            [[self.top_left_limit, self.top_limit],
             [self.top_right_limit, self.top_limit],
             [self.bottom_left_limit, self.bottom_limit],
             [self.bottom_right_limit, self.bottom_limit]],
            np.float32).T

        # What world points correspond to our warp limits?
        self.world_limits = self.pixel_to_world(self.src_pixels)
        
        # TODO: Set these manually with camera mounted
        self.dst_pixels = np.asarray(
            [[self.bottom_left_limit, 0], #TL
             [self.bottom_right_limit, 0], #TR
             [self.bottom_left_limit, height], #BL
             [self.bottom_right_limit, height]], #BR
            np.float32).T
        (M, M_inv) = self.compute_IPM_matrices(self.src_pixels.T,
                                               self.dst_pixels.T)
        self.M = M
        self.M_inv = M_inv

    def compute_IPM_matrices(self, src, dst):
        M = cv2.getPerspectiveTransform(src, dst)
        M_inv = cv2.getPerspectiveTransform(dst, src)
        return (M, M_inv)

    def process_image(self, img):
        """
        img: Incoming image
        
        Run the entire lane detection process
        """
        filtered_img = self.filter(img) 
        ipm_img = self.perspective_warp(filtered_img)
        (left, center, right) = self.sliding_window(ipm_img)
    
    def perspective_warp(self, img):
        """
        img: Incoming image
        
        This function performs inverse perspective warping to map an
        image from a front-facing camera to a downward view.
        """
        warped_img = cv2.warpPerspective(img,
                                         self.M, 
                                         (self.params.width, self.params.height), 
                                         flags=cv2.INTER_LINEAR)

        return warped_img

    def inverse_perspective_warp(self, img):
        warped_img = cv2.warpPerspective(img,
                                         self.M_inv,
                                         (self.params.width, self.params.height),
                                         flags=cv2.INTER_LINEAR)
        return warped_img
    
    def get_hist(self, img):
        hist = np.sum(img[img.shape[0]//2:, :], axis=0)
        return hist

    def filter(self, img):
        """
        img: Incoming image
        
        This function applies a series of filters and binarization to 
        mask out all pixels not corresponding to road lanes.
        """
        SOBEL_MINIMUM_THRESHOLD = 25
        S_MINIMUM_THRESHOLD = 100

        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]

        sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1)
        sobel_x = np.absolute(sobel_x)
        scaled_sobel_x = np.uint8(255 * sobel_x / np.max(sobel_x))
        sobel_thresholded = np.zeros_like(scaled_sobel_x)
        sobel_thresholded[(scaled_sobel_x >= SOBEL_MINIMUM_THRESHOLD)] = 1

        s_thresholded = np.zeros_like(s_channel)
        s_thresholded[(s_channel >= S_MINIMUM_THRESHOLD)] = 1

        combined_thresholded = np.zeros_like(sobel_thresholded)
        combined_thresholded[(sobel_thresholded == 1) | (s_thresholded == 1)] = 1
        return combined_thresholded
    
    def sliding_window(self, img):
        nwindows = 9
        margin = 75
        minpix = 1

        histogram = self.get_hist(img)
        midpoint = int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        leftx_current = leftx_base
        rightx_current = rightx_base

        window_height = np.int(img.shape[0] / nwindows)
        nonzero = img.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            y_low = img.shape[0] - (window+1)*window_height
            y_high = img.shape[0] - window*window_height
            xleft_low = leftx_current - margin
            xleft_high = leftx_current + margin
            xright_low = rightx_current - margin
            xright_high = rightx_current + margin

            good_left_inds = ((nonzero_y >= y_low) & (nonzero_y < y_high) & (nonzero_x >= xleft_low) & (nonzero_x < xleft_high)).nonzero()[0]
            good_right_inds = ((nonzero_y >= y_low) & (nonzero_y < y_high) & (nonzero_x >= xright_low) & (nonzero_x < xright_high)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        left_x = nonzero_x[left_lane_inds]
        left_y = nonzero_y[left_lane_inds]
        right_x = nonzero_x[right_lane_inds]
        right_y = nonzero_y[right_lane_inds]

        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)

        left_a = []
        left_a.append(left_fit[0])
        left_b = []
        left_b.append(left_fit[1])
        left_c = []
        left_c.append(left_fit[2])

        right_a = []
        right_a.append(right_fit[0])
        right_b = []
        right_b.append(right_fit[1])
        right_c = []
        right_c.append(right_fit[2])

        l_a = np.mean(left_a)
        l_b = np.mean(left_b)
        l_c = np.mean(left_c)

        r_a = np.mean(right_a)
        r_b = np.mean(right_b)
        r_c = np.mean(right_c)

        left_fit_ = np.array([l_a, l_b, l_c])
        right_fit_ = np.array([r_a, r_b, r_c])
        center_fit_ = np.array([(l_a+r_a)/2, (l_b+r_b)/2, (l_c+r_c)/2])
        
        return (left_fit_, center_fit_, right_fit_)

    def draw_lanes(self, unwarped_img, left_fit, right_fit):
        plot_y = np.linspace(0, unwarped_img.shape[0]-1, unwarped_img.shape[0])
        left_fitx = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
        right_fitx = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]

        left = np.array([np.transpose(np.vstack([left_fitx, plot_y]))])
        right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, plot_y])))])
        points = np.hstack((left, right))

        color_img = np.zeros_like(unwarped_img)
        cv2.fillPoly(color_img, np.int_(points), (0,200,255))
        inv_perspective_img = self.inverse_perspective_warp(color_img)
        inv_perspective_img = cv2.addWeighted(unwarped_img, 1, inv_perspective_img, 0.7, 0)
        return inv_perspective_img

    
    def world_grid(self, img):
        """
        img: Incoming image
        
        Find the pixel coords of the corners of the image once transformed
        """

        K = self.params.K
        D = self.params.D
        a_h = self.params.FOV_h/2
        a_v = self.params.FOV_v/2

    def map_w_to_c(self, coords):
        return np.dot(self.R, coords_h_to_coords_transform(
            np.dot(self.T, coords_to_h_transform(coords))))

    def normalize_coords(self, coords):
        K_inv = np.linalg.inv(self.K)
        coords_h = coords_to_h_transform(coords)

    def pixel_to_world(self,pix_coords):
        # Take pixels in camera image and move them to local nav frame
        pix_h = coords_to_h_transform(pix_coords)
        pix_norm = np.dot(np.linalg.inv(self.params.K), pix_h)
        
        # Here we get the 3D location of points in camera frame
        points_c = (self.dot_const_c / np.dot(self.normal_c.T, pix_norm)) * pix_norm
        
        # And then we convert them to the world (local NWU) frame
        xy_w = np.dot(np.linalg.inv(self.R), points_c)[:2,:]
        return xy_w
