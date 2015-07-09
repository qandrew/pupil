"""
	Andrew Xia working on visualizing data.
	I want to use opengl to display the 3d sphere and lines that connect to it.
	This file is in pupil-labs-andrew/sphere_fitter, so it is the prototype version
	July 6 2015

"""
import logging
from glfw import *
from OpenGL.GL import *
from OpenGL.GLUT import *

# create logger for the context of this function
logger = logging.getLogger(__name__)
from pyglui import ui

from pyglui.cygl.utils import init
from pyglui.cygl.utils import RGBA
from pyglui.cygl.utils import *
from pyglui.cygl import utils as glutils
from pyglui.pyfontstash import fontstash as fs
from trackball import Trackball

import numpy as np
import scipy
import geometry #how do I find this folder?
import cv2

def convert_fov(fov,width):
	fov = fov*scipy.pi/180
	focal_length = (width/2)/np.tan(fov/2)
	return focal_length

class Visualizer():
	def __init__(self,name = "unnamed", run_independently = False, width = 1280, height = 720, focal_length = 554.25625):
		self.sphere = geometry.Sphere([11,14,46],12) #the eyeball, initialized as something random
		self.ellipses = [] #collection of ellipses 
		self.circles = [] #collection of all 3D circles on the sphere
		self.video_frame = (np.linspace(0,1,num=(400*400*4))*255).astype(np.uint8).reshape((400,400,4)) #the randomized image, should be video frame
		# self.screen_points = [] #collection of points

		self.name = name
		self._window = None
		self.width = width
		self.height = height
		self.focal_length = focal_length
		self.input = None
		self.trackball = None
		self.run_independently = run_independently

		self.window_should_close = False

		self.test_ellipse = geometry.Ellipse((0,3),5,3,0)

	############## DRAWING FUNCTIONS ##############################

	def draw_rect(self):
		glBegin(GL_QUADS)
		glColor4f(0.0, 0.0, 0.5,0.2)  #set color to light blue
		glVertex2f(0,0) 
		glVertex2f(0 + 32, 0)
		glVertex2f(0 + 32, 0 + 16)
		glVertex2f(0, 0 + 16)
		glEnd()  

	def draw_frustum(self,f, scale=1):
		# average focal length
		#f = (K[0, 0] + K[1, 1]) / 2
		# compute distances for setting up the camera pyramid
		W = 0.5*self.width
		H = 0.5*self.height
		Z = f
		# scale the pyramid
		W *= scale
		H *= scale
		Z *= scale
		# draw it
		glColor4f( 1, 0.5, 0, 0.5 )
		glBegin( GL_LINE_LOOP )
		glVertex3f( 0, 0, 0 )
		glVertex3f( -W, H, Z )
		glVertex3f( W, H, Z )
		glVertex3f( 0, 0, 0 )
		glVertex3f( W, H, Z )
		glVertex3f( W, -H, Z )
		glVertex3f( 0, 0, 0 )
		glVertex3f( W, -H, Z )
		glVertex3f( -W, -H, Z )
		glVertex3f( 0, 0, 0 )
		glVertex3f( -W, -H, Z )
		glVertex3f( -W, H, Z )
		glEnd( )

	def draw_coordinate_system(self,l=1):
		# Draw x-axis line. RED
		glLineWidth(2)
		glColor3f( 1, 0, 0 )
		glBegin( GL_LINES )
		glVertex3f( 0, 0, 0 )
		glVertex3f( l, 0, 0 )
		glEnd( )

		# Draw z-axis line. BLUE
		glColor3f( 0, 0,1 )
		glBegin( GL_LINES )
		glVertex3f( 0, 0, 0 )
		glVertex3f( 0, 0, l )
		glEnd( )

		# Draw y-axis line. GREEN. #not working... why? 
		glColor3f( 0, 1, 0 )
		glBegin( GL_LINES )
		glVertex3f( 0, 0, 0 )
		glVertex3f( 0, l, 0 )
		glEnd( )

	def draw_sphere(self):
		# this function draws the location of the eye sphere
		glPushMatrix()
		glColor3f(0.0, 0.0, 1.0)  #set color to blue
		glTranslate(self.sphere.center[0], self.sphere.center[1], self.sphere.center[2])
		glutWireSphere(self.sphere.radius,20,20)
		glPopMatrix()

	def draw_all_ellipses(self):
		# draws all ellipses in self.ellipses.
		glPushMatrix()
		for ellipse in self.ellipses:
			glColor3f(0.0, 1.0, 0.0)  #set color to green
			glTranslate(ellipse.center[0], ellipse.center[1], 0) 
			glBegin(GL_LINE_LOOP) #draw ellipse
			for i in xrange(45):
				rad = i*16*scipy.pi/360.
				glVertex2f(np.cos(rad)*ellipse.major_radius,np.sin(rad)*ellipse.minor_radius)	
			glEnd()
			glTranslate(-ellipse.center[0], -ellipse.center[1], 0) #untranslate

			d = np.array([self.sphere.center[0]-ellipse.center[0],self.sphere.center[1]-ellipse.center[1],self.sphere.center[2]]) #direction
			d = d/np.linalg.norm(d)			
			el_center = np.array([ellipse.center[0],ellipse.center[1],0])	
			glutils.draw_polyline3d([self.sphere.center,el_center-d],color=RGBA(0.4,0.5,0.3,1)) #draw line
		glPopMatrix()

	def draw_all_circles(self):
		glPushMatrix()
		glColor3f(0.0, 1.0, 0.0)  #set color to green
		for circle in self.circles:
			glTranslate(circle.center[0], circle.center[1], circle.center[2]) 
			glBegin(GL_LINE_LOOP) #draw ellipse
			for i in xrange(45):
				rad = i*16*scipy.pi/360.
				glVertex2f(np.cos(rad)*circle.radius,np.sin(rad)*ellipse.minor_radius)	
			glEnd()
			glTranslate(-circle.center[0], -circle.center[1], circle.center[2]) #untranslate

		glPopMatrix()

	def draw_ellipse(self,ellipse):
		#draw a single ellipse
		glPushMatrix()  
		glColor3f(0.0, 1.0, 0.0)  #set color to green
		glTranslate(ellipse.center[0], ellipse.center[1], 0)
		glBegin(GL_LINE_LOOP)
		for i in xrange(30): #originally 360, 2. Now 30, 24. Performance gains
			rad = i*24*scipy.pi/360.
			glVertex2f(np.cos(rad)*ellipse.major_radius,np.sin(rad)*ellipse.minor_radius)
		glEnd()
		glPopMatrix()

	def draw_video_screen(self):
		#function to draw self.video_frame
		glPushMatrix()
		tex_id = create_named_texture(self.video_frame.shape)
		update_named_texture(tex_id,self.video_frame) #since image doesn't change, do not need to put in while loop
		draw_named_texture(tex_id)
		glPopMatrix()

	########## Setup functions I don't really understand ############

	def basic_gl_setup(self):
		glEnable(GL_POINT_SPRITE )
		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE) # overwrite pointsize
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		glEnable(GL_BLEND)
		glClearColor(.8,.8,.8,1.)
		glEnable(GL_LINE_SMOOTH)
		# glEnable(GL_POINT_SMOOTH)
		glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
		glEnable(GL_LINE_SMOOTH)
		glEnable(GL_POLYGON_SMOOTH)
		glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)

	def adjust_gl_view(self,w,h):
		"""
		adjust view onto our scene.
		"""
		glViewport(0, 0, w, h)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glOrtho(0, w, h, 0, -1, 1)
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()

	def clear_gl_screen(self):
		glClearColor(.9,.9,0.9,1.)
		glClear(GL_COLOR_BUFFER_BIT)

	########### Open, update, close #####################

	def open_window(self):
		if not self._window:
			self.input = {'down':False, 'mouse':(0,0)}
			self.trackball = Trackball()

			# get glfw started
			if self.run_independently:
				glfwInit()
			window = glfwGetCurrentContext()					
			self._window = glfwCreateWindow(self.width, self.height, self.name, None, window)
			glfwMakeContextCurrent(self._window)

			if not self._window:
				exit()

			glfwSetWindowPos(self._window,2000,0)
			# Register callbacks window
			glfwSetFramebufferSizeCallback(self._window,self.on_resize)
			glfwSetWindowIconifyCallback(self._window,self.on_iconify)
			glfwSetKeyCallback(self._window,self.on_key)
			glfwSetCharCallback(self._window,self.on_char)
			glfwSetMouseButtonCallback(self._window,self.on_button)
			glfwSetCursorPosCallback(self._window,self.on_pos)
			glfwSetScrollCallback(self._window,self.on_scroll)
			glfwSetWindowCloseCallback(self._window,self.on_close)

			# get glfw started
			if self.run_independently:
				init()

			glutInit()
			self.basic_gl_setup()

			# self.gui = ui.UI()
			self.on_resize(self._window,*glfwGetFramebufferSize(self._window))

	def update_window(self):
		if self.window_should_close:
			self.close_window()
		if self._window != None:
			glfwMakeContextCurrent(self._window)
			self.clear_gl_screen()

			self.trackball.push()

			#THINGS I NEED TO DRAW
			self.draw_sphere() #draw the eyeball
			self.draw_all_ellipses()
			self.draw_rect()

			# self.draw_frustum(self.focal_length, scale = .01)
			self.draw_coordinate_system(4)

			self.trackball.pop()
			glfwSwapBuffers(self._window)
			glfwPollEvents()
			return True

	def close_window(self):
		if self.window_should_close == True:
			glfwDestroyWindow(self._window)
			if self.run_independently:
				glfwTerminate()
			self._window = None
			self.window_should_close = False
			logger.debug("Process done")

	############ window callbacks #################
	def on_resize(self,window,w, h):
		h = max(h,1)
		w = max(w,1)
		self.trackball.set_window_size(w,h)

		active_window = glfwGetCurrentContext()
		glfwMakeContextCurrent(window)
		self.adjust_gl_view(w,h)
		glfwMakeContextCurrent(active_window)

	def on_iconify(self,window,x,y): pass

	def on_key(self,window, key, scancode, action, mods): pass
		#self.gui.update_key(key,scancode,action,mods)

	def on_char(window,char): pass
		# self.gui.update_char(char)

	def on_button(self,window,button, action, mods):
		# self.gui.update_button(button,action,mods)
		if action == GLFW_PRESS:
			self.input['down'] = True
			self.input['mouse'] = glfwGetCursorPos(window)
		if action == GLFW_RELEASE:
			self.input['down'] = False

		# pos = normalize(pos,glfwGetWindowSize(window))
		# pos = denormalize(pos,(frame.img.shape[1],frame.img.shape[0]) ) # Position in img pixels

	def on_pos(self,window,x, y):
		hdpi_factor = float(glfwGetFramebufferSize(window)[0]/glfwGetWindowSize(window)[0])
		x,y = x*hdpi_factor,y*hdpi_factor
		# self.gui.update_mouse(x,y)
		if self.input['down']:
			old_x,old_y = self.input['mouse']
			self.trackball.drag_to(x-old_x,y-old_y)
			self.input['mouse'] = x,y

	def on_scroll(self,window,x,y):
		# self.gui.update_scroll(x,y)
		self.trackball.zoom_to(y)

	def on_close(self,window=None):
		self.window_should_close = True

if __name__ == '__main__':
	huding = Visualizer("huding", run_independently = True)

	huding.ellipses.append(geometry.Ellipse((0,4),5,3,0))
	huding.ellipses.append(geometry.Ellipse((2,4),2,3,0))
	huding.ellipses.append(geometry.Ellipse((4,4),2,1,0))

	huding.open_window()
	a = 0
	while huding.update_window():
		a += 1
	huding.close_window()
	print a

	# print convert_fov(60,640)
