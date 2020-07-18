import cv2
import numpy as np

class SteerAssist():
	def __init__(self):
		self.current_speed = 30

		wheel = cv2.imread('imgs/wheel.png')
		wheel = cv2.cvtColor(wheel, cv2.COLOR_BGR2BGRA)
		mask = np.ones_like(wheel[:,:,0])
		mask[:,:][(wheel[...,0]==255) & (wheel[...,1]==255) & ( wheel[...,2]==255)] = 0
		self.wheel_masked = cv2.bitwise_and(wheel, wheel, mask=mask)
		self.init_mask = self.wheel_masked

	def add_wheel(self, image):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

		h, w = self.wheel_masked.shape[:2]
		H, W = image.shape[:2]
		image[:(h), (W//2 - w//2):(w+(W//2 - w//2))] = self.wheel_masked
		# self.wheel_masked = self.init_mask
		return image

	def calculate_offset(self, image, pts_left, pts_right):
		center_x = image.shape[1]//2

		center_lane = (pts_right[:,0][-1]- pts_left[:,0][-1])/2

		offset = 3.7/840 * (center_lane - center_x)
		return offset

	def rotate_wheel(self, image, pts_left, pts_right, left_curvature, right_curvature):
		pts_left = pts_left.reshape(720, 2)
		pts_right = pts_right.reshape(720, 2)

		offset = self.calculate_offset(image, pts_left, pts_right)


		center_lane = image.shape[1]//2, image.shape[0]

		left_theta = 300/left_curvature
		right_theta = 300/right_curvature

		# print(offset/((left_curvature+right_curvature)/2))
		# print((left_theta+right_theta)/2)

		total_angle = (offset/((left_curvature+right_curvature)/2)  + (left_theta+right_theta)/2) * 180/np.pi

		# print(total_angle)

		self.wheel_masked = self.rotate_image(self.wheel_masked, total_angle)

		image = self.add_meanline(image, pts_right, pts_left)

		final = cv2.cvtColor(self.add_wheel(image), cv2.COLOR_BGRA2BGR)
		return final

	def rotate_image(self, image, angle):
		image_center = tuple(np.array(image.shape[1::-1]) / 2)
		# print(angle.shape)
		rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
		result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
		return result

	def add_meanline(self, image, pts_right, pts_left):
		pts_y = np.linspace(0, image.shape[0]-1, image.shape[0])
		pts_x = (pts_right[:,0] + pts_left[:,0])/2

		pts = np.array([np.transpose(np.vstack([pts_x, pts_y]))])

		image = cv2.polylines(image, np.int32([pts]), False, (16, 50, 97)[::-1], 30)
		return image
