import cv2
import numpy as np
import imutils

class SteerAssist():
	def __init__(self):
		self.current_speed = 30
		self.angle = None

		self.wheel = cv2.imread('imgs/wheel.png')


	def add_wheel(self, background, x, y):
		'''
			Taken from https://stackoverflow.com/a/54058766
		'''
		background_width = background.shape[1]
		background_height = background.shape[0]

		if x >= background_width or y >= background_height:
		    return background

		h, w = self.wheel.shape[0], self.wheel.shape[1]

		if x + w > background_width:
		    w = background_width - x
		    self.wheel = self.wheel[:, :w]

		if y + h > background_height:
		    h = background_height - y
		    self.wheel = self.wheel[:h]

		if self.wheel.shape[2] < 4:
		    self.wheel = np.concatenate(
		        [
		            self.wheel,
		            np.ones((self.wheel.shape[0], self.wheel.shape[1], 1), dtype = self.wheel.dtype) * 255
		        ],
		        axis = 2,
		    )

		self.wheel_image = self.wheel[..., :3]
		mask = self.wheel[..., 3:] / 255.0

		background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * self.wheel_image

		return background


	def rotate_wheel(self, image, left_curvature, right_curvature, offset):

		x, y = 640, 300

		center_lane = image.shape[1]//2, image.shape[0]

		left_theta = 300/left_curvature
		right_theta = 300/right_curvature


		self.angle = (offset/((left_curvature+right_curvature)/2)  + (left_theta+right_theta)/2) * 180/np.pi

		self.wheel = imutils.rotate(self.wheel, self.angle)

		final = cv2.cvtColor(self.add_wheel(image, x, y), cv2.COLOR_BGRA2BGR)
		return final

	def rotate_image(self, image, angle):
		image_center = tuple(np.array(image.shape[1::-1]) / 2)
		rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
		result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
		return result
