"""
	Andrew Xia working on visualizing data.
	I want to use opengl to display the 3d sphere and lines that connect to it.
	July 6 2015

"""

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

window = 0                                             # glut window number
width, height = 500, 400                               # window size

class Visualizer:

	def __init__(self, name = "unnamed", width = 400, height = 500):
		self.name = name
		self.width = width
		self.height = height

	def run(self):
		glutInit()                                             # initialize glut
		glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
		glutInitWindowSize(self.width, self.height)                      # set window size
		glutInitWindowPosition(2000, 0)                           # set window position
		window = glutCreateWindow(self.name)              # create window with title
		glutDisplayFunc(self.draw())                                  # set draw function callback
		glutIdleFunc(self.draw())                                     # draw all the time
		glutMainLoop()                                         # start everything

	def draw(self):
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # clear the screen
		glLoadIdentity()                                   # reset position
		self.refresh2d(width,height)

		glColor3f(0.0, 0.0, 1.0)                           # set color to blue
		self.draw_rect(10, 10, 200, 100)                        # rect

		glutSwapBuffers()                                  # important for double buffering

	def draw_rect(self,x,y,width,height):
		#draws a rectangle.
		glBegin(GL_QUADS)                                  # start drawing a rectangle
		glVertex2f(x, y)                                   # bottom left point
		glVertex2f(x + width, y)                           # bottom right point
		glVertex2f(x + width, y + height)                  # top right point
		glVertex2f(x, y + height)                          # top left point
		glEnd()  

	def refresh2d(self,width, height):
		glViewport(0, 0, width, height)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glOrtho(0.0, width, 0.0, height, 0.0, 1.0)
		glMatrixMode (GL_MODELVIEW)
		glLoadIdentity()

	#### fns to draw surface in separate window
	def gl_display_in_window_3d(self,world_tex_id,camera_intrinsics):
		"""
		here we map a selected surface onto a seperate window.
		"""
		K,dist_coef,img_size = camera_intrinsics

		if self._window and self.detected:
			active_window = glfwGetCurrentContext()
			glfwMakeContextCurrent(self._window)
			glClearColor(.8,.8,.8,1.)

			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
			glClearDepth(1.0)
			glDepthFunc(GL_LESS)
			glEnable(GL_DEPTH_TEST)
			self.trackball.push()

			glMatrixMode(GL_MODELVIEW)

			draw_coordinate_system(l=self.real_world_size['x'])
			glPushMatrix()
			glScalef(self.real_world_size['x'],self.real_world_size['y'],1)
			draw_polyline([[0,0],[0,1],[1,1],[1,0]],color = RGBA(.5,.3,.1,.5),thickness=3)
			glPopMatrix()
			# Draw the world window as projected onto the plane using the homography mapping
			glPushMatrix()
			glScalef(self.real_world_size['x'], self.real_world_size['y'], 1)
			# cv uses 3x3 gl uses 4x4 tranformation matricies
			m = cvmat_to_glmat(self.m_from_screen)
			glMultMatrixf(m)
			glTranslatef(0,0,-.01)
			draw_named_texture(world_tex_id)
			draw_polyline([[0,0],[0,1],[1,1],[1,0]],color = RGBA(.5,.3,.6,.5),thickness=3)
			glPopMatrix()

			# Draw the camera frustum and origin using the 3d tranformation obtained from solvepnp
			glPushMatrix()
			glMultMatrixf(self.camera_pose_3d.T.flatten())
			draw_frustum(self.img_size, K, 150)
			glLineWidth(1)
			draw_frustum(self.img_size, K, .1)
			draw_coordinate_system(l=5)
			glPopMatrix()


			self.trackball.pop()

			glfwSwapBuffers(self._window)
			glfwMakeContextCurrent(active_window)

if __name__ == '__main__':
	print "yay local file"
	huding = Visualizer("huding",400,500)
	huding.run()