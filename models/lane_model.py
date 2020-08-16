import cv2
import numpy as np

class Line():
    def __init__(self, line_type):
        #was the lane line detected successfully
        self.detected = False
        #current polynomial fit for the lane line
        self.current_fit = None
        #x-fit for the lane line
        self.fitx = None
        #lane curvature
        self.curvature = None
        #combined x and y points for the lane line
        self.pts = None
        #distance of lane from the vehicle center
        self.distance_from_center = None
        #type of line (left or right)
        self.type = line_type
        #polynomial coefficients averaged over the last n iterations
        self.prev_fits = []
        #best fit average from previous fits
        self.best_fit = None
        #average x values for the lane lines
        self.prev_fitx = []

    def apply_window_search(self, warped):
        #calculate histogram of the image horizontally summing
        hist = np.sum(warped[350:], axis=0)
        #midpoint of the histogram or midpoint of image in x-direction
        midpoint = warped.shape[1]//2 

        if self.type == 'left':
            #start point for the left line to search in windows
            start = np.argmax(hist[:midpoint])
        else:
            #start point for the right line to search in windows
            start = np.argmax(hist[midpoint:])+midpoint

        #extract all non-zero pixels from the image which will be the lane pixels
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        #width from center of the window i.e. window_size/2
        margin = 100
        #number of windows to apply in lane finding
        n_windows = 9
        #height of each winbdows according to number of windows and image height
        window_height = int(warped.shape[0]/n_windows)
        #minimum pixels which should be detected inside window to be considered as a lane pixel
        min_pix = 50

        #store all the indexes of the left and right lane lines
        lane_inds = []

        for i in range(n_windows):
            bottomleftx = start - margin
            bottomlefty = warped.shape[0] - i * window_height

            topright_x = start + margin
            topright_y = bottomlefty - window_height

            nonzero_inds = (((nonzerox >= bottomleftx) & (nonzerox <= topright_x)) & ((nonzeroy <= bottomlefty) & (nonzeroy >= topright_y))).nonzero()[0]

            lane_inds.append(nonzero_inds)
            
            if len(nonzero_inds) > min_pix:
                start = np.int(np.mean(nonzerox[nonzero_inds]))
            
        lane_inds = np.concatenate(lane_inds)
        
        self.x = nonzerox[lane_inds]
        self.y = nonzeroy[lane_inds]

        if 0 in [len(self.x), len(self.y)]:
            found = False
        else:
            found = True
        return found

    def search_around_poly(self, warped, fit):
        # Choose the width of the margin around the previous polynomial to search
        margin = 100

        # Grab activated pixels
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        lane_inds = ((nonzerox > (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + 
                fit[2] - margin)) & (nonzerox < (fit[0]*(nonzeroy**2) + 
                fit[1]*nonzeroy + fit[2] + margin)))
        
        # Again, extract left and right line pixel positions
        self.x = nonzerox[lane_inds]
        self.y = nonzeroy[lane_inds]

        if 0 in [len(self.x), len(self.y)]:
            found = False
        else:
            found = True
        return found

    def fit_polynomial(self):
        try:
            self.current_fit = np.polyfit(self.y, self.x, 2)
        except:
            pass

    def check_diffs(self):
        if self.best_fit is not None:
            diff = np.abs(self.current_fit - self.best_fit)
        else:
            diff = np.array([0, 0, 0])
        return diff

    def measure_curvature(self, lanewidth, ploty):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 25/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/lanewidth # meters per pixel in x dimension

        X = self.fitx

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        fit_cr = np.polyfit(ploty*ym_per_pix, X*xm_per_pix, 2)

        # Calculation of R_curve (radius of curvature)
        curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])

        return curverad


class Lane():
    def __init__(self):
        self.lanewidth = None
        self.steer_angle = None
        self.left_line = Line('left')
        self.right_line = Line('right')
        #difference in current and previous fits
        self.default_diffs = np.array([9e-05, 0.3, 0.7])
        self.ploty = None
        self.pts = None
        #average over n frames data
        self.n = 5

    def detect(self, image):
        #searching for lane lines
        for line in [self.left_line, self.right_line]:
            if line.best_fit is None:
                found = line.apply_window_search(image) 
            else:
                found = line.search_around_poly(image, line.best_fit)

            line.detected = found
            if line.detected:
                line.fit_polynomial()
                diff = line.check_diffs()
                # if not np.all(diff < self.default_diffs):
                #     print('diff')
                #     line.best_fit = np.mean(np.array(line.prev_fits), axis=0)
                #     line.current_fit = line.best_fit

                if self.ploty is None:
                    self.ploty = np.linspace(0, image.shape[0]-1, image.shape[0])

                #calculate the polynomial fit of the lane line
                line.fitx = line.current_fit[0]*self.ploty**2 + line.current_fit[1]*self.ploty + line.current_fit[2]

                curv = line.measure_curvature(840, self.ploty)
                if not ((curv > 10000) or (curv < 100)):
                    #append to the previous n storate of fits and fitx
                    if len(line.prev_fits) > self.n:
                        line.prev_fits.remove(line.prev_fits[0])

                    line.prev_fits.append(line.current_fit)

                    line.best_fit = np.mean(np.array(line.prev_fits), axis=0)

                    if(len(line.prev_fitx) > self.n):
                        line.prev_fitx.remove(line.prev_fitx[0])
                    line.prev_fitx.append(line.fitx)

            line.fitx = np.mean(np.array(line.prev_fitx), axis=0)

        #stack x and y values of the lane lines to make points for plotting
        self.left_line.pts = np.array([np.transpose(np.vstack([self.left_line.fitx, self.ploty]))])
        self.right_line.pts = np.array([np.flipud(np.transpose(np.vstack([self.right_line.fitx, self.ploty])))])
        #stack both left and right points to one
        self.pts = np.hstack((self.left_line.pts, self.right_line.pts))
        #calculate the width of lane
        self.lanewidth = self.get_lanewidth()
        #measure curvature of both left and right lanes using the lane_width
        self.offset = self.get_offset(image.shape)
        for line in [self.left_line, self.right_line]:
            line.curvature = line.measure_curvature(840, self.ploty)

        img = cv2.cvtColor(np.zeros_like(image, dtype='uint8'), cv2.COLOR_GRAY2BGR)
        lane_mask_image = cv2.fillPoly(img, np.int_([self.pts]), (214, 32, 62)[::-1])

        line_image = np.zeros_like(img).astype(np.uint8)
        pts_left = self.left_line.pts.reshape((-1, 1, 2))
        pts_right = self.right_line.pts.reshape((-1, 1, 2))
        line_image = cv2.polylines(line_image, np.int32([pts_left]), False, (255,255,255), 30)
        line_image = cv2.polylines(line_image, np.int32([pts_right]), False, (255,255,255), 30)

        return lane_mask_image, line_image

    def get_lanewidth(self):
        return np.mean(self.right_line.fitx - self.left_line.fitx) * 3.7/(np.min(self.right_line.fitx) - np.min(self.left_line.fitx))

    def get_offset(self, shape):
        car_center = shape[1]//2
        center_lane = self.left_line.pts[0][:,0][-1] + (self.right_line.pts[0][:,0][0]- self.left_line.pts[0][:,0][-1])/2

        offset = 3.7/840 * (car_center - center_lane)
        return offset