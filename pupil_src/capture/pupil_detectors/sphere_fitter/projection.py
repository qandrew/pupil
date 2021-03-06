"""
	Andrew Xia playing around with porting c++ code to python
	I want to port Projection.h into this python script here
	June 25 2015

	We use the camera intrinsic to determine the focal length for projection/unprojection

	project circle: projects a 3D circle to a conic/ellipse in 2D frame plane
	project sphere: projects a 3D sphere to an ellipse in 2D frame plane
	project point: projects a 3D point to a point in 2D frame plane
	unproject: given ellipse, return two 3D circles of its unprojection
	unproject point: given point and z coordinate, return unprojected point

"""

#THIS FILE IS INCOMPLETE!

import numpy as np
import scipy

import geometry
import solve
import logging
logging.info('Starting logger for...') 
logger = logging.getLogger(__name__)

def project_circle_to_ellipse(circle,intrinsics,extrinsics = None):
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
	if circle == None or intrinsics == None:
		logger.error("please supply circle and/or intrinsics matrix")
		return
	c = circle.center
	n = circle.normal
	r = circle.radius
	focal_length = abs(intrinsics[1,1])
	cn = np.dot(n,c)
	c2r2 = np.dot(c,c) - np.square(r)
	ABC = (np.square(cn) - 2.0*cn*c*n + c2r2*np.square(n))
	F = 2*(c2r2*n[1]*n[2] - cn*(n[1]*c[2] + n[2]*c[1]))
	G = 2*(c2r2*n[2]*n[0] - cn*(n[2]*c[0] + n[0]*c[2]))
	H = 2*(c2r2*n[0]*n[1] - cn*(n[0]*c[1] + n[1]*c[0]))

	conic = geometry.Conic(a = ABC[0],b=H,c=ABC[1],d=G*focal_length,e=F*focal_length,f=ABC[2]*np.square(focal_length))
	ellipse = geometry.Ellipse(conic = conic)
	ellipse.center = np.array([ellipse.center[0] + intrinsics[0,2], -ellipse.center[1] + intrinsics[1,2]]) #shift ellipse center
	ellipse.angle = -ellipse.angle%np.pi
	return ellipse

def project_sphere_camera_intrinsics(sphere,intrinsics,extrinsics = None):
	center = project_point_camera_intrinsics(sphere.center,intrinsics,extrinsics)
	radius = abs(sphere.radius/sphere.center[2] * intrinsics[1,1]) #scale based on fx in camera intrinsic matrix
	return geometry.Ellipse(center,radius,radius,0)

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


def unproject(ellipse,circle_radius,focal_length):

	""" TO DO : CASE OF SEEING CIRCLE, DO TRIVIAL CALCULATION (currently wrong result) """

	circle = geometry.Circle3D()
	Matrix3 = np.zeros((3,3))
	RowArray3 = np.zeros((1,3))
	Translation3 = np.zeros((1,3)) #see T2 for actual implementation

	conic = geometry.Conic(ellipse = ellipse)
	cam_center_in_ellipse = np.array([[0],[0],[-focal_length]])
	pupil_cone = geometry.Conicoid(conic = conic, vertex = cam_center_in_ellipse)
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
	lamb = solve.solve_four(1., 
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
	n = np.sqrt((lamb[1] - lamb[2])/(lamb[0]-lamb[2]))
	m = 0.0
	l = np.sqrt((lamb[0] - lamb[1])/(lamb[0]-lamb[2]))

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

		solutions.append(geometry.Circle3D(center,gaze,circle_radius))

	return solutions

def unproject_camera_intrinsics(ellipse,circle_radius,intrinsics, extrinsics = None):

	"""NEED TO TEST THIS FUNCTION"""

	# essentially shift ellipse center by the translation cx and cy from camera matrix
	# and feed this back into unproject()
	if extrinsics == None:
		#set extrinsics matrix as identity matrix appended with 0 [I|0]
		extrinsics = np.matrix('1 0 0 0 ; 0 1 0 0 ; 0 0 1 0')
	focal_length = intrinsics[0,0]
	offset_ellipse = geometry.Ellipse(np.array([ellipse.center[0] - intrinsics[0,2], -(ellipse.center[1] - intrinsics[1,2])]),
		ellipse.major_radius,ellipse.minor_radius, np.sign(intrinsics[1,1])*ellipse.angle)
	return unproject(offset_ellipse,circle_radius,focal_length)

if __name__ == '__main__':

	k = np.matrix('100 0 10; 0 -100 10; 0 0 1')

	# print k[0,2]
	p3 = unproject_point_intrinsics((0.0,20),100,k)
	p2 = project_point_camera_intrinsics(p3,k)
	# print p3,p2
	#testing uproject
	ellipse = geometry.Ellipse((0.,0.),2.0502,1.0001,2.01)
	circ = unproject_camera_intrinsics(ellipse,1,k)
	print circ[0]
	print circ[1]
	print project_circle_to_ellipse(circ[1],k)
	print project_circle_to_ellipse(circ[0],k)

	# conic = geometry.Ellipse(conic = project_circle(circ[0],200))
	# print conic
	# conic = geometry.Ellipse(conic = project_circle(circ[1],200))
	# print conic
	# ellipse = geometry.Ellipse((92.049,33.9655), 66.3554, 52.3161, 0.752369*scipy.pi)
	# print ellipse
	# huding = unproject(ellipse,1,1000) #879.193
	# print huding[0]
	# print huding[1]
	# huding = unproject_camera_intrinsics(ellipse,1,k)
	# print huding[0]
	# print " "
	# huconic = project_circle(huding[0],879.193)
	# print "first " + str(huconic)
	# huellipse = geometry.Ellipse(conic = huconic)
	# print huellipse
	# print " "
	# huconic = project_circle(huding[1],879.193)
	# print "second " + str(huconic)
	# huellipse = geometry.Ellipse(conic = huconic)
	# print huellipse

	# ellipse = geometry.Ellipse((-152.295,157.418),46.7015,32.4274,0.00458883*scipy.pi)
	# huding = unproject(ellipse,1,1030.3) 
	# print huding[0] 
	#sol0: Circle { center: (-2.23168,2.29378,15.1334), normal: (0.124836,-0.81497,-0.565897), radius: 1 }

	# testing project_circle
	# circle = geometry.Circle3D([1.35664,-0.965954,9.33736],[0.530169,-0.460575,-0.711893],1)
	# tempcon =  project_circle(circle,1000) 
	# tempell = geometry.Ellipse(conic = tempcon)
	# print tempcon
	# print tempell
	# print project_circle_to_ellipse(circle,k)

	#testing project_point
	# point = np.array([[0.493976],[-0.376274],[4.35446]])
	# print point
	# print project_point_camera_intrinsics(point, k, None)
	# print project_point(point,1000)

	#testing project_sphere
	# sphere = geometry.Sphere(center=[-12.3454,5.22129,86.7681],radius=12)
	# print project_sphere(sphere,1000)
	# print project_sphere_camera_intrinsics(sphere,k)
	# sphere = geometry.Sphere(center=(10.06,-6.20644,86.8967),radius=12)
	# print project_sphere(sphere,1030.3) #GOOD