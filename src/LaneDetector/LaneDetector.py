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
        self.R = R # R world -> camera
        self.t = t # position of camera in world
        
        self.params = camera_params
        
        self.config()

    def config(self):
        # Get ground plane info
        ground_plane_normal_w = np.array([[0],[0],[1]])
        self.normal_c = np.dot(self.R, ground_plane_normal_w) # ground plane normal rotated into camera frame

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

        self.bottom_limit = 480
        self.top_limit = 150 # 260
        self.bottom_left_limit = 0
        self.bottom_right_limit = 640
        #self.top_left_limit = 180
        self.top_left_limit = 0
        #self.top_right_limit = 460
        self.top_right_limit = 640
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
             [120, height], #BL
             [520, height]], #BR
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
        # Convert to HSL and filter
        filtered_img = self.filter(img) 
        # Warp to overhead
        ipm_img = self.perspective_warp(filtered_img)
        # Detect lines
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
    
    def perspective_transform(self, pts):
        warped_pts = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), self.M)
        return warped_pts
    
    def inverse_perspective_transform(self, pts):
        warped_pts = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), self.M_inv)
        return warped_pts

    def inverse_perspective_warp(self, img):
        warped_img = cv2.warpPerspective(img,
                                         self.M_inv,
                                         (self.params.width, self.params.height),
                                         flags=cv2.INTER_LINEAR)
        return warped_img
    
    def get_hist(self, img):
        hist = np.sum(img[img.shape[0] - 50:img.shape[0], :], axis=0)
        return hist

    def filter(self, img):
        """
        img: Incoming image
        
        This function applies a series of filters and binarization to 
        mask out all pixels not corresponding to road lanes.
        """
        #SOBEL_MINIMUM_THRESHOLD = 25
        #S_MINIMUM_THRESHOLD = 100
        
        # Try to find white and yellow regions for lanes
        hsv_mask = self.color_threshold(img)

        # Because of light, white will not get picked up all the time
        # We fallback to a gradient check and binary or them
        #gradient_mask = self.gradient_threshold(img)

        # Now also perform gradient thresholding
        # ...
        #mask = cv2.bitwise_or(hsv_mask, gradient_mask)
        return hsv_mask

    def gradient_threshold(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return self.abs_sobel(gray_img)

    def abs_sobel(self, img, x_dir=True):
        sobel = cv2.Sobel(img, cv2.CV_64F, x_dir, not x_dir, 15) 
        sobel_abs = np.absolute(sobel)
        sobel_scaled = np.uint8(255 * sobel_abs / np.max(sobel_abs))
        gradient_mask = np.zeros_like(sobel_scaled)
        thresh_min = np.array([5], dtype = np.uint8)
        thresh_max = np.array([120], dtype = np.uint8)
        gradient_mask = cv2.inRange(sobel_scaled, thresh_min, thresh_max)
        return gradient_mask

    def mag_sobel(self, img):
        sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=15)
        sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=15)
        sxy = np.sqrt(np.square(sx) + np.square(sy))
        scaled_sxy = np.uint8(255 * sxy / np.max(sxy))
        thresh_min = np.array([5], dtype = np.uint8)
        thresh_max = np.array([120], dtype = np.uint8)
        sxy_binary = cv2.inRange(sxy, thresh_min, thresh_max)
        return sxy_binary

    def blur_mask(self, mask_img):
        med_img = cv2.medianBlur(mask_img, 5)
        gauss_img = cv2.GaussianBlur(med_img, (5,5), 0)
        gauss_binary = cv2.inRange(gauss_img, 80, 255)
        return gauss_img

    def color_threshold(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        #hsv_mask = cv2.bitwise_or(self.isolate_white(hsv), self.isolate_yellow(hsv))
        hsv_mask = self.isolate_green(hsv)
        return hsv_mask

    def isolate_yellow(self, img):
        # img in HSV
        low_threshold = np.array([20, 100, 100], dtype = np.uint8)
        #low_threshold = np.array([25*255/360/2, 25*255/100, 25*255/100], dtype = np.uint8)
        high_threshold = np.array([45, 255, 255], dtype = np.uint8)
        #high_threshold = np.array([65*255/360/2, 70*255/100, 75*255/100], dtype = np.uint8)
        yellow_mask = cv2.inRange(img, low_threshold, high_threshold)
        return yellow_mask

    def isolate_white(self, img):
        # img in HSV
        low_threshold = np.array([0*255/360/2, 0*255/100, 25*255/100], dtype = np.uint8)
        high_threshold = np.array([360*255/360/2, 40*255/100, 40*255/100], dtype = np.uint8)
        white_mask = cv2.inRange(img, low_threshold, high_threshold)
        return white_mask

    def isolate_green(self, img):
        # Hue : gimp (0 : 360), cv2 (0 : 179)
        # Sat : gimp (0 : 100), cv2 (0 : 255)
        # Val : gimp (0 : 100), cv2 (0 : 255)
        low_threshold = np.array([155 // 2, 30 * 255 // 100, 50 * 255 // 100], dtype = np.uint8)
        high_threshold = np.array([175 // 2, 60 * 255 // 100, 100 * 255 // 100], dtype = np.uint8)
        green_mask = cv2.inRange(img, low_threshold, high_threshold)
        return green_mask
    def sliding_window(self, img, world_coords = False):
        nwindows = 10
        margin = 75
        buffer = 0
        minpix = 1
        
        histogram = self.get_hist(img)
        midpoint = int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint-buffer])
        rightx_base = np.argmax(histogram[midpoint+buffer:]) + midpoint + buffer
        leftx_current = leftx_base
        rightx_current = rightx_base

        window_height = np.int(img.shape[0]) // nwindows

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
        left_pixels_xy = np.vstack([left_x, left_y])
        left_world_xy = self.pixel_to_world(left_pixels_xy)

        right_x = nonzero_x[right_lane_inds]
        right_y = nonzero_y[right_lane_inds]
        right_pixels_xy = np.vstack([right_x, right_y])
        right_world_xy = self.pixel_to_world(right_pixels_xy)

        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)

        if world_coords is True:
            left_fit = left_fit_world
            right_fit = right_fit_world

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
    
    """
    def fit_center(self, warped_img, use_ransac=False):
        # TURBO SPEED BUT PROBABLY NOT V ACCURATE
        nonzero = warped_img.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        if use_ransac:
            center_fit = self.ransac_polyfit(nonzero_y, nonzero_x)
        else:
            center_fit = np.polyfit(nonzero_y, nonzero_x, 2)
        
        return center_fit
    """
    def fit_center(self, img):
        # This fit is in the frame of the input image (usually birds-eye), so transform points after fitting
        nwindows = 10
        margin = 75
        buffer = 0
        minpix = 1
        
        histogram = self.get_hist(img)
        midpoint = int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint-buffer])
        rightx_base = np.argmax(histogram[midpoint+buffer:]) + midpoint + buffer

        x_current = np.argmax(histogram)

        window_height = np.int(img.shape[0]) // nwindows

        nonzero = img.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        fit = np.polyfit(nonzero_y, nonzero_x, 2)
        #fit = self.ransac_polyfit(nonzero_y, nonzero_x)

        return fit

    def draw_lanes(self, unwarped_img, left_fit, right_fit, world_coords = False):
        if world_coords is True:
            return
        else:
            plot_y = np.linspace(0, unwarped_img.shape[0]-1, unwarped_img.shape[0])
            left_fitx = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
            right_fitx = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]

            key_left1 = left_fitx[-1]
            key_right1 = right_fitx[-1]

            key_left2 = left_fitx[0]
            key_right2 = right_fitx[0]

            if (key_left1 > key_right1 or key_left2 > key_right2):
                print "LOLNO"
                raise Exception('lol', 'lol')
            
            left = np.array([np.transpose(np.vstack([left_fitx, plot_y]))])
            right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, plot_y])))])
            points = np.hstack((left, right))
            
            color_img = np.zeros_like(unwarped_img)
            cv2.fillPoly(color_img, np.int_(points), (0,200,255))
            inv_perspective_img = self.inverse_perspective_warp(color_img)
            inv_perspective_img = cv2.addWeighted(unwarped_img, 1, inv_perspective_img, 0.7, 0)
            return inv_perspective_img
    
    def draw_lane(self, img, center_fit):
        plot_y = np.linspace(0, 480, 50)
        plot_y = plot_y[::-1]

        center_fit_x = center_fit[0]*plot_y**2 + center_fit[1]*plot_y + center_fit[2]

        draw_points = (np.asarray([center_fit_x, plot_y]).T).astype(np.int32)

        color_img = np.zeros_like(img)

        for x in draw_points:
            cv2.circle(color_img, tuple(x), 5, (255, 0, 0), -1)
        
        inv_perspective_img = self.inverse_perspective_warp(color_img)
        inv_perspective_img = cv2.addWeighted(img, 1, inv_perspective_img, 0.7, 0)
        return inv_perspective_img


    def generate_waypoints(self, unwarped_img, center_fit):
        # waypoints are of form (x, y, yaw)
        plot_y = np.linspace(0, 480, 50)
        plot_y = plot_y[::-1]
        center_fit_x = center_fit[0]*plot_y**2 + center_fit[1]*plot_y + center_fit[2]

        center_pts = np.vstack([center_fit_x, plot_y]).T
        center_pts_unwarped = self.perspective_transform(center_pts.reshape(-1, 1, 2))

        world_pts = self.pixel_to_world(np.squeeze(center_pts_unwarped.T))

        n_pts = world_pts.shape[1]
        n_out = n_pts - 1
        angles_out = np.zeros((n_out))
        # find angle between current point and next point
        for i in range(n_out):
            x1 = world_pts[0, i]
            y1 = world_pts[1, i]
            # s = 'Point ' + repr(i) + '= x:' + repr(x1) + ' y:' + repr(y1)
            x2 = world_pts[0, i+1]
            y2 = world_pts[1, i+1]
            dx = x2 - x1
            dy = y2 - y1
            theta = np.arctan2(dy, dx)
            angles_out[i] = theta

        waypoints = np.vstack((world_pts[:, 0:-1], angles_out))
        return waypoints
    
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
        return np.dot(self.R, coords - self.t)

    def normalize_coords(self, coords):
        K_inv = np.linalg.inv(self.K)
        coords_h = coords_to_h_transform(coords)

    def pixel_to_world(self,pix_coords):
        
        # Take pixels in camera image and move them to local nav frame
        pix_h = coords_to_h_transform(pix_coords)
        pix_norm = np.dot(np.linalg.inv(self.params.K), pix_h)
        
        # Here we get the 3D location of points in camera frame
        # I have no idea where this came from but it works and it's fast.
        points_c = (self.dot_const_c / np.dot(self.normal_c.T, pix_norm)) * pix_norm
        
        # And then we convert them to the world (local NWU) frame
        xy_w = np.dot(np.linalg.inv(self.R), points_c)[:2,:]
        return xy_w

    def ransac_polyfit(self, x, y, k=50, t=3):
        besterr = np.inf
        bestfit = None
        for kk in range(k):
            maybeinliers = np.random.randint(len(x), size=50)
            maybemodel = np.polyfit(x[maybeinliers], y[maybeinliers], 2)
            alsoinliers = np.abs(np.polyval(maybemodel, x)-y) < t
            bettermodel = np.polyfit(x[alsoinliers], y[alsoinliers], 2)
            thiserr = np.sum(np.abs(np.polyval(bettermodel, x[alsoinliers])-y[alsoinliers]))
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
        return bestfit
