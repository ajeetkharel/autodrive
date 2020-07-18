import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import json
import os
import signal
import time

from models import Thresholder, Lane, Calibrator
from helpers import warp, misc

# from applications.yolo import YOLO

class Pipeline():
	def __init__(self, views = ['default'], download_file=False, use_cuda = False):

		self.calibrator = Calibrator('camera_cal', 8, 6)
		self.calibrator.load_camera_matrices('files/cmtx.npy', 'files/dmtx.npy')
		# self.calibrator.calibrate()

		self.thresholder = Thresholder()

		self.views = views

		self.lane = Lane()

		# self.object_detector = YOLO(download_file, use_cuda=use_cuda)

		self.final_height = 720
		self.final_width = 840

	def run_pipeline(self, image, lane_only):
		undistorted = self.calibrator.undistort(image)
		
		self.thresh = self.thresholder.get_thresh(undistorted)
		
		self.warped = warp.transform(self.thresh)
		
		lane_mask_image, lines_image = self.lane.detect(self.warped)
		lane_mask_image = warp.inverse_transform(lane_mask_image)
		lines_image = warp.inverse_transform(lines_image)

		self.lines_image = np.zeros_like(lines_image).astype(np.uint8)
		line_img_color = (245, 247, 247)[::-1]
		self.lines_image[:] = np.uint8(line_img_color)
		
		nonzero = lines_image.nonzero()
		self.lines_image[nonzero] = 1

		self.lane_mask_final = cv2.addWeighted(image, 0.8, lane_mask_image, 0.2, 0)

		if not lane_only:
			self.lane_mask_final = self.object_detector.draw_boxes(image, self.lane_mask_final)

		final_image = self.extract_views()

		return final_image

	def run_video(self, input_video, output_video, lane_only = True):

		#video for performing detection
		cap = cv2.VideoCapture(input_video)

		#for using the fit from initial lane in others wihout performing sliding windows again and again
		if output_video is not None:
			fourcc = cv2.VideoWriter_fourcc(*'XVID')
			out = cv2.VideoWriter(f'outputs/{output_video}', fourcc, cap.get(cv2.CAP_PROP_FPS), (self.final_width, self.final_height))

		while True:
			ret, frame = cap.read()
			if ret:
				start_time = time.time()
				
				final_image = self.run_pipeline(frame, lane_only)
				
				end_time = time.time()
				diff = end_time - start_time

				fps = 1/diff

				cv2.putText(final_image,f"FPS: {int(fps)}", (final_image.shape[1]-60, self.small_view_height+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
				cv2.imshow('Lane', final_image)
				if cv2.waitKey(1) & 0xFF==ord('q'):
					break
				if output_video is not None:
					out.write(final_image)
			else:
				break
		cap.release()
		cv2.destroyAllWindows()

	def run_image(self, lane_only=True):
		frame = cv2.imread('test_images/road.jpg')
		final_image = self.run_pipeline(frame, lane_only)
		cv2.imshow("img", final_image)
		cv2.waitKey()
		cv2.destroyAllWindows()

	def extract_views(self):

		final_image = np.zeros((self.final_height, self.final_width, 3), dtype=np.uint8)

		if len(self.views)>1:
			self.small_view_height = 140
			small_view_width = 280

			main_view_height = self.final_height - self.small_view_height
			main_view_width = self.final_width
			view_images = map(self.get_image, self.views[1:3])

			for i, image in enumerate(view_images):
				try:
					image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
				except:
					pass

				image = cv2.resize(image, (small_view_width, self.small_view_height))

				final_image[0:self.small_view_height, i*small_view_width:i*small_view_width+small_view_width, :] = np.uint8(image/np.max(image)*255)
		else:
			main_view_height = self.final_height
			main_view_width = self.final_width
			self.small_view_height = 0
		lane_mask_final = cv2.resize(self.lines_image, (main_view_width, main_view_height))

		final_image[self.small_view_height:, :] = np.uint8(lane_mask_final)

		if len(self.views[3:]) > 0:
			final_image = self.annotate(final_image)

		return final_image

	def get_image(self, key):
		view_map = {
			'default': self.lines_image,
			'mask': self.lane_mask_final,
			'thresh': self.thresh,
		}
		return view_map.get(key, 'lane_mask_final')

	def annotate(self, image):
		
		cv2.putText(image,f"Left curvature: {self.lane.left_line.curvature}", (15, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
		cv2.putText(image,f"Right curvature: {self.lane.right_line.curvature}", (15, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
		cv2.putText(image,f"Offset: {self.lane.offset}", (15, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
		# cv2.putText(image,f"Steer angle: {self.steer_assist.angle}", (15, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

		return image

if __name__=='__main__':

	input_video = 'test_videos/challenge_video.mp4'
	output_video = None

	ap = argparse.ArgumentParser()
	ap.add_argument('--input', required=False, type=str, help='Input video for the autodrive model')
	ap.add_argument('--output', required=False, type=str, help='Name of output video to store the output of autodrive model')
	ap.add_argument('-g', required=False, action='store_true', help='Add gpu support to perform object detection')
	ap.add_argument('-v', required=False, type=int, default=2, help='Verbosity level for showing output : \
													\n 0: Lane lines with steering assist and FPS counter\
													\n 1: Include lane threshold image, warped image and lane mask of road\
													\n 2(default): Show curvature, offset and curve angle of the road')
	ap.add_argument('-o', required=False, action='store_true', help='Detect objects along with the lane lines')


	args = vars(ap.parse_args())

	#whether to use yolo tiny model or not
	use_cuda = args['g']
	verbosity = args['v']
	lane_only = not args['o']

	if args['input'] is not None:
		input_video = args['input'].strip("'")
	if args['output'] is not None:
		output_video = args['output'].strip("'")

	#if weights present then don't download, otherwise download
	download_file = True
	if os.path.isfile('files/yolov3.weights'):
		download_file = False

	#get all the required views according to the verbosity level
	views = misc.get_views(verbosity)

	pipeline = Pipeline(views=views, download_file=download_file, use_cuda=use_cuda)

	#run the pipeline on a single image
	# pipeline.run_image(lane_only)
	
	#run the pipeline on specified video
	pipeline.run_video(input_video, output_video, lane_only=lane_only)

	#kill the application
	os.kill(os.getpid(), signal.SIGTERM)