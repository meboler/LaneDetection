import numpy as np # written for 1.11
import cv2 as cv # written for 3.3.1


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
        self.pixel_limits = np.asarray(
            [[0, self.ipm_top],
             [width, self.ipm_top],
             [0, height],
             [width, height]], np.float32).T

        self.world_limits = self.pixel_to_world(self.pixel_limits)

        
    def process_image(self, img):
        """
        img: Incoming image
        
        Run the entire lane detection process
        """
        ipm_img = self.perspective_warp(img)
        filtered_img = self.filter(ipm_img) 
        splines = self.fit_splines(filtered_img)
        return splines
    
    def perspective_warp(self, img):
        """
        img: Incoming image
        
        This function performs inverse perspective warping to map an
        image from a front-facing camera to a downward view.
        """

        # Here we define the ground plane by its normal and the origin
        # under the camera. All points where <normal_c, uv> = dot_const_c 
        # lie on our plane
        return img
    
    def filter(self, img):
        """
        img: Incoming image
        
        This function applies a series of filters and binarization to 
        mask out all pixels not corresponding to road lanes.
        """
        return img
    
    def fit_splines(self, img):
        """
        img: Preprocessed image where hot pixels correspond to lanes and
        all other pixels are black
    
        This function determines the set of polynomials of degree 2 that fit
        the lanes in the image.
        """
        return np.zeros((2, 10))

    def image_coord_limits(img):
        top_left_p = np.asarray([[0],[0]], np.float32)
        top_right_p = np.asarray([[0], [img.shape[1]]], np.float32)
        bot_left_p = np.asarray([[img.shape[0]], [0]], np.float32)
        bot_right_p = np.asarray([[img.shape[0]], [img.shape[1]]], np.float32)

        return np.hstack((top_left_p, top_right_p, bot_left_p, bot_right_p))
    
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
        pix_h = coords_to_h_transform(pix_coords)
        pix_norm = np.dot(np.linalg.inv(self.params.K), pix_h)
        print pix_norm
        points_c = (self.dot_const_c / np.dot(self.normal_c.T, pix_norm)) * pix_norm
        print points_c
        xy_w = np.dot(np.linalg.inv(self.R), points_c)[:2,:]
        print xy_w


    
