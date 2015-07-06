"""
	Andrew Xia playing around with porting c++ code to python
	I want to port singgleeyefitter.cpp and singgleeyefitter.h into this python script here
	June 25 2015

	EDIT June 30 2015: This file will contain only the class EyeModelFitter renamed spherefitter, that was 
	originally in both singleeyefitter.h and singleeyefitter.cpp.

"""

#importing stuff
import numpy as np
import cv2
import scipy

import auxiliary_functions
import geometry
import projection
import intersect
import logging
logging.info('Starting logger for...') 
logger = logging.getLogger(__name__)

#global functions
def toRotatedRect(ellipse):
	toreturn = cv2.RotatedRect(ellipse.center,
		cv2.Size2f(2*ellipse.major_radius),
		cv2.Size2f(2*ellipse.minor_radius),
		ellipse.angle * 180 / scipy.pi)
	return toreturn

def toEllipse(rotatedrect):
	#input is a cv2 rotated rect
	toreturn = geometry.Ellipse(rect.center,
		rect.size.width/2,
		rect.size.height/2,
		rect.angle*scipy.pi/180)
	return toreturn

def circleFromParams_eye(eye,params):
	if (params.radius == None):
		logger.warning("dafaq, gimme params pls")
		return geometry.Circle3D()

	radial = auxiliary_functions.sph2cart(1, params.theta, params.psi)
	return geometry.Circle3D(eye.center + eye.radius*radial,radial, params.radius)

class PupilParams: #was a structure in C
	def __init__(self, theta = 0, psi = 0, radius = 0):
		self.theta = theta
		self.psi = psi
		self.radius = radius

	def __str__(self):
		return "PupilParams Class: Theta" + str(self.theta) + " psi " + str(self.psi) + " r " + str(self.radius)

class Pupil: #data structure for a pupil
	def __init__(self, ellipse = geometry.Ellipse(), focal_length = 879.193, radius = 1):
		self.ellipse = ellipse
		self.circle = geometry.Circle3D() #may delete later
		self.projected_circles = projection.unproject(self.ellipse,circle_radius = radius, focal_length = focal_length)
		self.params = PupilParams()
		self.init_valid = False

	def __str__(self):
		return "Pupil Class: " + str(self.ellipse) + str(self.circle) + " " + str(self.params) + " init_valid: " + str(self.init_valid)


#the class
class Sphere_Fitter():

	def __init__(self, focal_length = 600, region_band_width = 5, region_step_epsilon = 0.5):
		#initialize based on what was done in singleeyefitter.cpp
		self.focal_length = focal_length
		self.camera_center = np.array([0,0,0])
		self.eye = geometry.Sphere() #model of our eye. geometry.sphere()
		self.projected_eye = geometry.Ellipse() #ellipse that is the projected eye sphere
		self.observations = [] #array containing elements in pupil class
		self.model_version = 0

		# self.region_band_width = region_band_width
		# self.region_step_epsilon = region_step_epsilon
		# self.region_scale = 1
		# self.index = []
		self.count = 0 #used for limiting logging purposes

	def add_observation(self,ellipse):
		#ellipse is the ellipse of pupil in camera image
		self.observations.append(Pupil(ellipse = ellipse, focal_length = self.focal_length))

	def reset(self):
		self.observations = []
		eye = geometry.Sphere()
		self.model_version += 1

	def circleFromParams(self, params):
		# currently badly written
		# params = angles + diameter (theta, psi, radius)
		return circleFromParams_eye(self.eye,params)

	def unproject_observations(self,pupil_radius = 1, eye_z = 20): 
		# ransac default to false so I skip for loop (haven't implemented it yet)
		# this function for every ellipse from the image creates corresponding circles 
		# unprojected to the pupil sphere model
		if (len(self.observations) < 2):
			logger.error("Need at least two observations")
			return
		pupil_unprojection_pairs = [] #each element should be [Circle.Cirle3D, Circle.Circle3D]
		pupil_gazelines_proj = [] #it is a vector<line> !!

		for pupil in self.observations:
			""" get pupil circles
				Do a per-image unprojection of the pupil ellipse into the two fixed
				size circles that would project onto it. The size of the circles
				doesn't matter here, only their center and normal does.
			"""
			unprojection_pair = pupil.projected_circles

			""" get projected circles and gaze vectors
				Project the circle centers and gaze vectors down back onto the image plane.
				We're only using them as line parameterizations, so it doesn't matter which of the two centers/gaze
				vectors we use, as the two gazes are parallel and the centers are co-linear
			"""

			#why do I default use the 0th one, not the 1st one???

			#here maybe write some function that determines which line is better

			c = np.reshape(unprojection_pair[0].center, (3,1)) #it is a 3D circle
			v = np.reshape(unprojection_pair[0].normal, (3,1))

			c_proj = projection.project_point(np.reshape(c,(3,1)),self.focal_length)
			c_proj = np.array(c_proj)
			v_proj = projection.project_point(v + c, self.focal_length) - c_proj
			v_proj = v_proj/np.linalg.norm(v_proj) #normalizing

			pupil_unprojection_pairs.append(unprojection_pair)
			line = geometry.Line2D(c_proj, v_proj)
			pupil_gazelines_proj.append(line)

		""" Get eyeball center
			Find a least-squares 'intersection' (point nearest to all lines) of
			the projected 2D gaze vectors. Then, unproject that circle onto a
			point a fixed distance away.
			For robustness, use RANSAC to eliminate stray gaze lines
			(This has to be done here because it's used by the pupil circle disambiguation)
		"""
		eye_center_proj = []
		valid_eye = bool

		# if (use_ransac):
		# 	""" TO BE IMPLEMENTED (or maybe I won't bother since ransac isn't most important part"""
		# 	pass
		# else:
		for pupil in self.observations:
			pupil.init_valid = True
		eye_center_proj = intersect.nearest_intersect_2D(pupil_gazelines_proj)
		eye_center_proj = np.reshape(eye_center_proj,(2,))
		valid_eye = True

		if (valid_eye):
			self.eye.center = [eye_center_proj[0] * eye_z / self.focal_length,
				eye_center_proj[1] * eye_z / self.focal_length, eye_z] #force it to be a 3x1 array
			self.eye.radius = 1
			self.projected_eye = projection.project_sphere(self.eye, self.focal_length)

			for i in xrange(len(self.observations)):
				#disambiguate pupil circles using projected eyeball center
				pupil_pair = pupil_unprojection_pairs[i]
				line = pupil_gazelines_proj[i]
				c_proj = np.reshape(line.origin, (2,))
				v_proj = np.reshape(line.direction, (2,))

				if (np.dot(c_proj - eye_center_proj, v_proj) >= 0):
					#check if v_proj going away from estimated eye center, take the one going away.
					self.observations[i].circle = pupil_pair[0]
				else: 
					self.observations[i].circle = pupil_pair[1]
				#print self.observations[i].circle
		else:
			#no inliers, so no eye
			self.eye = Sphere.Sphere()

			# arbitrarily pick first circle
			for i in xrange(len(self.observations)):
				pupil_pair = pupil_unprojection_pairs[i]
				self.pupil_ellipse_yearray[i].circle = pupil_pair[0]

		self.model_version += 1

		#print every 30
		self.count += 1
		if self.count == 30:
			logger.warning(self.eye)
			self.count = 0

if __name__ == '__main__':

	#testing stuff
	huding = Sphere_Fitter()
	if huding.projected_eye== [0,0]:
		print "huding"
	print "hudong"

	#testing unproject_observation
	ellipse1 = geometry.Ellipse((-141.07,72.6412),46.0443, 34.5685, 0.658744*scipy.pi)
	ellipse2 = geometry.Ellipse((-134.405,98.3423),45.7818, 36.7225, 0.623024*scipy.pi)
	ellipse3 = geometry.Ellipse((75.7523,68.8315),60.8489, 55.8412, 0.132388*scipy.pi)
	ellipse4 = geometry.Ellipse((-76.9547,52.0801),51.8554, 44.3508, 0.753157*scipy.pi)
	ellipse5 = geometry.Ellipse((-73.8259,5.54398),64.1682, 48.5875, 0.810757*scipy.pi)
	ellipse6 = geometry.Ellipse((-62.2873,-60.9237),41.1463, 23.5819, 0.864127*scipy.pi)

	huding.add_observation(ellipse1)
	huding.add_observation(ellipse2)
	huding.add_observation(ellipse3)
	huding.add_observation(ellipse4)
	huding.add_observation(ellipse5)
	huding.add_observation(ellipse6)

	huding.unproject_observations()
	print huding.projected_eye

	print huding.circleFromParams(PupilParams(1,1,50))

	lol = huding.eye
	print circleFromParams_eye(lol, PupilParams(1,1,12))