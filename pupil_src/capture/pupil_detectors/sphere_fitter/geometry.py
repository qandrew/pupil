"""
	Andrew Xia playing around with porting c++ code to python
	This file contains the Circle.py, Conic,py, Ellipse.py, Conicoic.py, and Sphere.py
	Having all 5 geometry classes in one file allows for better maintainability.
	Created July 2 2015

"""

import numpy as np
import math
from solve import solve_four
import scipy
import logging
logging.info('Starting logger for...') 
logger = logging.getLogger(__name__)

class Circle3D:
	def __init__(self,center=[0,0,0],normal=[0,0,0],radius=0):
		self.center = np.array(center)
		self.normal = np.array(normal)
		self.radius = radius

	def __str__(self):
		return "Circle { center: %s  normal: %s radius: %s} "%(self.center,self.normal,self.radius)

	def project_circle_to_ellipse(self,intrinsics):
		#takes in circle and outputs projected ellipse. It is easier to translate ellipse than conic
		"""INSTRUCTIONS
			Construct cone with circle as base and vertex v = (0,0,0).
			
			For the circle,
				|p - c|^2 = r^2 where (p-c).n = 0 (i.e. on the circle plane)
			
			A cone is basically concentric circles, with center on the line c->v.
			For any point p, the corresponding circle center c' is the intersection
			of the line c->v and the plane through p normal to n. So,
			
				d = ((p - v).n)/(c.n)
				c' = d c + v
			
			The radius of these circles decreases linearly as you approach 0, so
			
				|p - c'|^2 = (r*|c' - v|/|c - v|)^2
			
			Since v = (0,0,0), this simplifies to
			
				|p - (p.n/c.n)c|^2 = (r*|(p.n/c.n)c|/|c|)^2
			
				|(c.n)p - (p.n)c|^2         / p.n \^2
				------------------- = r^2 * | --- |
					  (c.n)^2               \ c.n /
			
				|(c.n)p - (p.n)c|^2 - r^2 * (p.n)^2 = 0
			
			Expanding out p, c and n gives
			
				|(c.n)x - (x*n_x + y*n_y + z*n_z)c_x|^2
				|(c.n)y - (x*n_x + y*n_y + z*n_z)c_y|   - r^2 * (x*n_x + y*n_y + z*n_z)^2 = 0
				|(c.n)z - (x*n_x + y*n_y + z*n_z)c_z|
			
				  ((c.n)x - (x*n_x + y*n_y + z*n_z)c_x)^2
				+ ((c.n)y - (x*n_x + y*n_y + z*n_z)c_y)^2
				+ ((c.n)z - (x*n_x + y*n_y + z*n_z)c_z)^2
				- r^2 * (x*n_x + y*n_y + z*n_z)^2 = 0
			
				  (c.n)^2 x^2 - 2*(c.n)*(x*n_x + y*n_y + z*n_z)*x*c_x + (x*n_x + y*n_y + z*n_z)^2 c_x^2
				+ (c.n)^2 y^2 - 2*(c.n)*(x*n_x + y*n_y + z*n_z)*y*c_y + (x*n_x + y*n_y + z*n_z)^2 c_y^2
				+ (c.n)^2 z^2 - 2*(c.n)*(x*n_x + y*n_y + z*n_z)*z*c_z + (x*n_x + y*n_y + z*n_z)^2 c_z^2
				- r^2 * (x*n_x + y*n_y + z*n_z)^2 = 0
			
				  (c.n)^2 x^2 - 2*(c.n)*c_x*(x*n_x + y*n_y + z*n_z)*x
				+ (c.n)^2 y^2 - 2*(c.n)*c_y*(x*n_x + y*n_y + z*n_z)*y
				+ (c.n)^2 z^2 - 2*(c.n)*c_z*(x*n_x + y*n_y + z*n_z)*z
				+ (x*n_x + y*n_y + z*n_z)^2 * (c_x^2 + c_y^2 + c_z^2 - r^2)
			
				  (c.n)^2 x^2 - 2*(c.n)*c_x*(x*n_x + y*n_y + z*n_z)*x
				+ (c.n)^2 y^2 - 2*(c.n)*c_y*(x*n_x + y*n_y + z*n_z)*y
				+ (c.n)^2 z^2 - 2*(c.n)*c_z*(x*n_x + y*n_y + z*n_z)*z
				+ (|c|^2 - r^2) * (n_x^2*x^2 + n_y^2*y^2 + n_z^2*z^2 + 2*n_x*n_y*x*y + 2*n_x*n_z*x*z + 2*n_y*n_z*y*z)
			
			Collecting conicoid terms gives
			
				  [xyz]^2 : (c.n)^2 - 2*(c.n)*c_[xyz]*n_[xyz] + (|c|^2 - r^2)*n_[xyz]^2
			   [yzx][zxy] : - 2*(c.n)*c_[yzx]*n_[zxy] - 2*(c.n)*c_[zxy]*n_[yzx] + (|c|^2 - r^2)*2*n_[yzx]*n_[zxy]
						  : 2*((|c|^2 - r^2)*n_[yzx]*n_[zxy] - (c,n)*(c_[yzx]*n_[zxy] + c_[zxy]*n_[yzx]))
					[xyz] : 0
						1 : 0
		"""
		if intrinsics == None:
			logger.error("please supply circle and/or intrinsics matrix")
			return
		c = self.center
		n = self.normal
		r = self.radius
		focal_length = abs(intrinsics[1,1])
		cn = np.dot(n,c)
		c2r2 = np.dot(c,c) - np.square(r)
		ABC = (np.square(cn) - 2.0*cn*c*n + c2r2*np.square(n))
		F = 2*(c2r2*n[1]*n[2] - cn*(n[1]*c[2] + n[2]*c[1]))
		G = 2*(c2r2*n[2]*n[0] - cn*(n[2]*c[0] + n[0]*c[2]))
		H = 2*(c2r2*n[0]*n[1] - cn*(n[0]*c[1] + n[1]*c[0]))

		conic = Conic(a = ABC[0],b=H,c=ABC[1],d=G*focal_length,e=F*focal_length,f=ABC[2]*math.pow(focal_length,2))
		ellipse = Ellipse(conic = conic)
		ellipse.center = np.array([ellipse.center[0] + intrinsics[0,2], -ellipse.center[1] + intrinsics[1,2]]) #shift ellipse center
		ellipse.angle = -ellipse.angle%np.pi
		return ellipse

class Conic:
	def __init__(self,a=None,b=None,c=None,d=None,e=None,f=None, ellipse=None):
		#a Conic is defined by 6 scalar parameters A,B,C,D,E,F
		if ellipse != None:
			#extracting information from ellipse
			ax = np.cos(ellipse.angle)
			ay = np.sin(ellipse.angle)
			a2 = np.square(ellipse.major_radius)
			b2 = np.square(ellipse.minor_radius)

			#scalars
			self.A = (ax*ax)/a2 + (ay*ay)/b2
			self.B = 2*(ax*ay)/a2 - 2*(ax*ay)/b2
			self.C = (ay*ay)/a2 +(ax*ax)/b2
			self.D = (-2*ax*ay*ellipse.center[1] - 2*ax*ax*ellipse.center[0])/a2 + (2*ax*ay*ellipse.center[1] - 2*ay*ay*ellipse.center[0])/b2
			self.E = (-2*ax*ay*ellipse.center[0] - 2*ay*ay*ellipse.center[1])/a2 + (2*ax*ay*ellipse.center[0] - 2*ax*ax*ellipse.center[1])/b2
			self.F = (2*ax*ay*ellipse.center[0]*ellipse.center[1]+ax*ax*ellipse.center[0]*ellipse.center[0]+ay*ay*ellipse.center[1]*ellipse.center[1])/a2+ (-2*ax*ay*ellipse.center[0]*ellipse.center[1]+ ay*ay*ellipse.center[0]*ellipse.center[0]+ax*ax*ellipse.center[1]*ellipse.center[1])/b2-1

		else:
			self.A = a
			self.B = b
			self.C = c
			self.D = d
			self.E = e
			self.F = f

	def __str__(self):
		return "Conic {  %s x^2  + %s xy +  %s y^2 +  %s x +  %s y  + %s  = 0 }"%(self.A,self.B,self.C,self.D,self.E,self.F)

class Conicoid:
	def __init__(self,A=0.0,B=0.0,C=0.0,F=0.0,G=0.0,H=0.0,U=0.0,V=0.0,W=0.0,D=0.0,conic = None, vertex = None):
		self.A = A
		self.B = B
		self.C = C
		self.D = D
		self.F = F
		self.G = G
		self.H = H
		self.U = U
		self.V = V
		self.W = W

		if (conic != None and vertex != None):
			self._initialize_by_conic(conic,vertex)

	def __str__(self):
		return "Conicoid { %s x^2 + %s y^2 +  %s z^2 + %s yz + 2 %s zx +2 %s xy + %s x + 2 %s y + 2 %s z + %s  = 0 }"%(self.A, self.B, self.C, self.F, self.G, self.H, self.U, self.V, self.W, self.D)

	def _initialize_by_conic(self,conic,vertex):
		#private method
		alpha = vertex[0]
		beta = vertex[1]
		gamma = vertex[2]
		print np.square(gamma)

		self.A = np.square(gamma)*conic.A
		self.B = np.square(gamma)*conic.C
		self.C = np.square(alpha)*conic.A + alpha*beta*conic.B + np.square(beta)*conic.C + conic.D*alpha + conic.E*beta + conic.F
		self.F = -gamma * (conic.C * beta + conic.B / 2 * alpha + conic.E / 2)
		self.G = -gamma * (conic.B / 2 * beta + conic.A * alpha + conic.D / 2)
		self.H = np.square(gamma)*conic.B/2
		self.U = np.square(gamma)*conic.D/2
		self.V = np.square(gamma)*conic.E/2
		self.W = -gamma * (conic.E / 2 * beta + conic.D / 2 * alpha + conic.F)
		self.D = np.square(gamma)*conic.F

class Ellipse:

	def __init__(self,center=[0,0],major_radius=0.0,minor_radius=0.0,angle=0.0, conic = None, pupil_ellipse = None):
		self.center = np.array(center)
		self.major_radius = major_radius
		self.minor_radius = minor_radius
		self.angle = angle #ANGLE SHOULD BE IN RADIANS!
		if conic:
			self._initialize_by_conic(conic)
		elif pupil_ellipse:
			self.initialize_by_pupil_ellipse(pupil_ellipse)

		# enforce angle <pi
		self.angle = self.angle%np.pi

	def initialize_by_pupil_ellipse(self, pupil_ellipse):
		x = pupil_ellipse['center'][0] #- 320 #- width/2 320
		y = pupil_ellipse['center'][1] #- 240 #- height/2 240
		self.center = np.array([x,y]) #shift location
		a,b = pupil_ellipse['axes']
		if a > b:
			self.major_radius = a/2
			self.minor_radius = b/2
			self.angle = pupil_ellipse['angle']*scipy.pi/180 #convert degrees to radians
		else: 
			self.major_radius = b/2
			self.minor_radius = a/2
			self.angle = (pupil_ellipse['angle']+90)*scipy.pi/180

	def _initialize_by_conic(self, conic):
		self.angle = 0.5*np.arctan2(conic.B,conic.A - conic.C)
		cost = np.cos(self.angle)
		sint = np.sin(self.angle)
		cos_squared = np.square(cost)
		sin_squared = np.square(sint)

		Ao = conic.F
		Au = conic.D*cost + conic.E*sint
		Av = -conic.D*sint + conic.E*cost 
		Auu= conic.A*cos_squared +conic.C*sin_squared +conic.B*sint*cost 
		Avv= conic.A*sin_squared +conic.C*cos_squared -conic.B*sint*cost 

		#ROTATED = [Ao Au Av Auu Avv]
		tuCenter = -Au / (2.0*Auu)
		tvCenter = -Av / (2.0*Avv)
		wCenter = Ao - Auu*np.square(tuCenter) - Avv*np.square(tvCenter)

		self.center = [0,0]
		self.center[0] = tuCenter*cost - tvCenter*sint
		self.center[1] = tuCenter*sint + tvCenter*cost
		self.major_radius = np.sqrt(abs(-wCenter/Auu))
		self.minor_radius = np.sqrt(abs(-wCenter/Avv))

		if (self.major_radius < self.minor_radius):
			self.major_radius,self.minor_radius = self.minor_radius,self.major_radius
			self.angle = self.angle + scipy.pi/2

		if (self.angle > scipy.pi):
			self.angle = self.angle - scipy.pi

	def __str__(self):
		return "Ellipse { center: %s  major_radius: %s  minor_radius: %s  angle: %s }"%(self.center,self.major_radius,self.minor_radius,self.angle)

	def scale(self,scale):
		self.center = [self.center[0]*scale,self.center[1]*scale]
		self.major_radius = self.major_radius*scale
		self.minor_radius = self.minor_radius*scale
		self.angle = self.angle*scale
		return self

	def pointAlongEllipse(self, theta):
		#theta is the angle
		xt = self.center[0] + self.major_radius*np.cos(self.angle)*np.cos(theta) - self.minor_radius*np.sin(self.angle)*np.sin(theta)
		yt = self.center[1] + self.major_radius*np.sin(self.angle)*np.cos(theta) + self.major_radius*np.cos(self.angle)*np.sin(theta)
		return np.array([xt,yt])

	def unproject_camera_intrinsics(self,circle_radius,intrinsics):
		# essentially shift ellipse center by the translation cx and cy from camera matrix
		# and feed this back into unproject()
		focal_length = intrinsics[0,0]
		offset_ellipse = Ellipse((self.center * np.array([1,-1]) - intrinsics[0:2,2].T),
			ellipse.major_radius,ellipse.minor_radius, np.sign(intrinsics[1,1])*ellipse.angle)
		return unproject(offset_ellipse,circle_radius,focal_length)


class Line:
	def __init__(self, origin, direction):
		self.origin = np.asarray(origin)
		self.direction = np.asarray(self.direction)
		self.direction /= np.linalg.norm(self.direction)

	def __str__(self):
		return "Line { from %s direction %s }" %(self.origin,self.direction)

""" other functions from the eigen cdef class that exist, but may not be used
	#def distance(self,point):
	#    # the distance of a point p to its projection onto the line
	#    pass
	#def intersection_hyperplane(self,hyperplane):
	#    # the parameter value of intersection between this and given hyperplane
	#    pass
	#def intersection_point(self, hyperplane):
	#    # returns parameter value of intersection between this and given hyperplane
	#    pass
	#def projection(self,point):
	#    # returns projection of a point onto the line
	#    pass
	#def pointAt(self,x):
	#    # point at x along this line
	#    pass
"""

class Line2D(Line):
	pass

class Line3D(Line):
	pass

class Sphere:
	def __init__(self,center=[0,0,0],radius=0):
		self.center = np.array(center)
		self.radius = radius

	def __str__(self):
		return "Sphere center: " + str(self.center) + " ,radius: " + str(self.radius)

	def project_sphere_camera_intrinsics(intrinsics,extrinsics = None):
		center = project_point_camera_intrinsics(self.center,intrinsics,extrinsics)
		radius = abs(self.radius/self.center[2] * intrinsics[1,1]) #scale based on fx in camera intrinsic matrix
		return Ellipse(center,radius,radius,0)

################ AUXILIARY FUNCTIONS ############

def project_point_camera_intrinsics(point = None,intrinsics = None,extrinsics = None):
	#camera intrinsics matrix and extrinsics is rotation-translation matrix. 
	if point == None or intrinsics == None:
		logger.error("please supply point and/or intrinsics matrix")
		return
	if extrinsics == None:
		#set extrinsics matrix as identity matrix appended with 0 [I|0]
		extrinsics = np.matrix('1 0 0 0 ; 0 1 0 0 ; 0 0 1 0')
	point = np.array(point) #force to np array
	point = np.append(point,[1]) #convert point to homogeneous coordinates
	point = point.reshape((4,1))
	projected_pt = intrinsics * extrinsics * point
	projected_pt = (projected_pt/projected_pt[-1])[:-1] #convert back to cartesian
	projected_pt = projected_pt.reshape(2)
	return np.array([projected_pt[0,0],projected_pt[0,1]])

def unproject_point_intrinsics(point,z,intrinsics):
	return [(point[0]-intrinsics[0,2]) * z / intrinsics[0,0],
			(point[1]-intrinsics[1,2]) * z / intrinsics[1,1],
			z]

def unproject(ellipse,circle_radius, focal_length):

	""" TO DO : CASE OF SEEING CIRCLE, DO TRIVIAL CALCULATION (currently wrong result) """
	conic = Conic.from_ellipse(ellipse)
	cam_center_in_ellipse = np.array([[0],[0],[-focal_length]])
	pupil_cone = Conicoid.from_conic(conic = conic, vertex = cam_center_in_ellipse)
	#pupil_cone.initialize_conic(conic,cam_center_in_ellipse) #this step is fine

	a = pupil_cone.A
	b = pupil_cone.B
	c = pupil_cone.C
	f = pupil_cone.F
	g = pupil_cone.G
	h = pupil_cone.H
	u = pupil_cone.U
	v = pupil_cone.V
	w = pupil_cone.W
	d = pupil_cone.D

	""" Get canonical conic form:

		lambda(1) X^2 + lambda(2) Y^2 + lambda(3) Z^2 = mu
		Safaee-Rad 1992 eq (6)
		Done by solving the discriminating cubic (10)
		Lambdas are sorted descending because order of roots doesn't
		matter, and it later eliminates the case of eq (30), where
		lambda(2) > lambda(1)
	"""
	lamb = solve_four(1.,
		-(a + b + c),
		(b*c + c*a + a*b - f*f - g*g - h*h),
		-(a*b*c + 2 * f*g*h - a*f*f - b*g*g - c*h*h) )
	if (lamb[0] < lamb[1]):
		logger.error("Lambda 0 > Lambda 1, die")
		return
	if (lamb[1] <= 0):
		logger.error("Lambda 1 > 0, die")
		return
	if (lamb[2] >= 0):
		logger.error("Lambda 2 < 0, die")
		return

	#Calculate l,m,n of plane
	n = cmath.sqrt((lamb[1] - lamb[2])/(lamb[0]-lamb[2]))
	m = 0.0
	l = cmath.sqrt((lamb[0] - lamb[1])/(lamb[0]-lamb[2]))

	#Safaee-Rad 1992 Eq 12
	t1 = (b - lamb)*g - f*h
	t2 = (a - lamb)*f - g*h
	t3 = -(a - lamb)*(t1/t2)/g - h/g

	#Safaee-Rad 1992 Eq 8
	mi = 1 / np.sqrt(1 + np.square(t1 / t2) + np.square(t3))
	li = (t1 / t2) * mi
	ni = t3 * mi

	#If li,mi,ni follow the left hand rule, flip their signs
	li = np.reshape(li,(3,))
	mi = np.reshape(mi,(3,))
	ni = np.reshape(ni,(3,))

	if (np.dot(np.cross(li,mi),ni) < 0):
		li = -li
		mi = -mi
		ni = -ni

	T1 = np.zeros((3,3))
	T1[:,0] = li
	T1[:,1] = mi
	T1[:,2] = ni
	T1 = np.asmatrix(T1.T)

	#Calculate t2 a translation transformation from the canonical
	#conic frame to the image space in the canonical conic frame
	#Safaee-Rad 1992 eq (14)
	temp = -(u*li + v*mi + w*ni) / lamb
	T2 = [[temp[0]],[temp[1]],[temp[2]]]
	solutions = [] #two solutions for the circles that we will return

	for i in (1,-1):
		l *= i

		gaze = T1 * np.matrix([[l],[m],[n]])

		#calculate t3, rotation from frame where Z is circle normal

		T3 = np.zeros((3,3))
		if (l == 0):
			if (n == 1):
				logger.error("Warning: l == 0")
				break
			T3 = np.matrix([[0,-1,0],
				[1,0,0],
				[0,0,1]])
		else:
			T3 = np.matrix([[0.,-n*np.sign(l),l],
				[np.sign(l),0.,0.],
				[0.,abs(l),n]]) #changed from round down to abs()

		#calculate circle center
		#Safaee-Rad 1992 eq (38), using T3 as defined in (36)
		lamb =  np.reshape(lamb,(3,))
		T30 = np.array([T3[0,0]**2,T3[1,0]**2,T3[2,0]**2 ])
		T31 = np.array([ [T3[0,0]*T3[0,2]], [T3[1,0]*T3[1,2]] , [T3[2,0]*T3[2,2]] ]) #good
		T32 = np.array([ [T3[0,1]*T3[0,2]], [T3[1,1]*T3[1,2]] , [T3[2,1]*T3[2,2]] ]) #good
		T33 = np.array([T3[0,2]**2 ,T3[1,2]**2 ,T3[2,2]**2 ])

		A = np.dot(lamb,T30)
		B = np.dot(lamb,T31) #good
		C = np.dot(lamb,T32) #good
		D = np.dot(lamb,T33)

		# Safaee-Rad 1992 eq 41
		center_in_Xprime = np.zeros((3,1))
		center_in_Xprime[2] = A*circle_radius/ np.sqrt(B**2 + C**2 - A*D)
		center_in_Xprime[0] = -B / A * center_in_Xprime[2]
		center_in_Xprime[1] = -C / A * center_in_Xprime[2]

		# Safaee-Rad 1992 eq 34
		T0 = [[0],[0],[focal_length]]

		# Safaee-Rad 1992 eq 42 using eq 35
		center = T0+T1*(T2+T3*center_in_Xprime)

		if (center[2] < 0):
			center_in_Xprime = -center_in_Xprime
			center = T0+T1*(T2+T3*center_in_Xprime) #make sure z is positive

		gaze = np.reshape(gaze,(3,))

		if (np.dot(gaze,center) > 0):
			gaze = -gaze
		gaze = gaze/np.linalg.norm(gaze) #normalizing
		gaze = np.array([gaze[0,0],gaze[0,1],gaze[0,2]]) #making it 3 instead of 3x1

		center = np.reshape(center,3)
		center = np.array([center[0,0],center[0,1],center[0,2]]) #making it 3 instead of 3x1

		solutions.append(Circle3D(center,gaze,circle_radius))

	return solutions

if __name__ == '__main__':
	#testing if modules here work correctly

	#testing if ellipse -> conic -> ellipse works
	huding = Ellipse((-141.07,72.6412),46.0443, 34.5685, 0.658744*scipy.pi)
	print huding
	hucon = Conic(ellipse = huding)
	print hucon
	huding2 = Ellipse(conic = hucon)
	print huding2
	print huding2.scale(0.5)
	hucon2 = Conic(ellipse = huding2)
	print hucon2