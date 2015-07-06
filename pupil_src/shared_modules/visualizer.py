"""
	Andrew Xia working on visualizing data.
	I want to use opengl to display the 3d sphere and lines that connect to it.
	July 6 2015

"""

from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import sys

name = 'ball_glut'

def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(400,400)
    glutCreateWindow(name)

    glClearColor(0.,0.,0.,1.)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    lightZeroPosition = [10.,4.,10.,1.]
    lightZeroColor = [0.8,1.0,0.8,1.0] #green tinged
    glLightfv(GL_LIGHT0, GL_POSITION, lightZeroPosition)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightZeroColor)
    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0.1)
    glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.05)
    glEnable(GL_LIGHT0)
    glutDisplayFunc(display)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(40.,1.,1.,40.)
    glMatrixMode(GL_MODELVIEW)
    gluLookAt(0,0,10,
              0,0,0,
              0,1,0)
    glPushMatrix()
    glutMainLoop()
    return

def display():
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    glPushMatrix()
    color = [1.0,0.,0.,1.]
    glMaterialfv(GL_FRONT,GL_DIFFUSE,color)
    glutSolidSphere(2,20,20)
    glPopMatrix()
    glutSwapBuffers()
    return

    #### fns to draw surface in separate window
# def gl_display_in_window_3d(self,world_tex_id,camera_intrinsics):
#     """
#     here we map a selected surface onto a seperate window.
#     """
#     K,dist_coef,img_size = camera_intrinsics

#     if self._window and self.detected:
#         active_window = glfwGetCurrentContext()
#         glfwMakeContextCurrent(self._window)
#         glClearColor(.8,.8,.8,1.)

#         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#         glClearDepth(1.0)
#         glDepthFunc(GL_LESS)
#         glEnable(GL_DEPTH_TEST)
#         self.trackball.push()

#         glMatrixMode(GL_MODELVIEW)

#         draw_coordinate_system(l=self.real_world_size['x'])
#         glPushMatrix()
#         glScalef(self.real_world_size['x'],self.real_world_size['y'],1)
#         draw_polyline([[0,0],[0,1],[1,1],[1,0]],color = RGBA(.5,.3,.1,.5),thickness=3)
#         glPopMatrix()
#         # Draw the world window as projected onto the plane using the homography mapping
#         glPushMatrix()
#         glScalef(self.real_world_size['x'], self.real_world_size['y'], 1)
#         # cv uses 3x3 gl uses 4x4 tranformation matricies
#         m = cvmat_to_glmat(self.m_from_screen)
#         glMultMatrixf(m)
#         glTranslatef(0,0,-.01)
#         draw_named_texture(world_tex_id)
#         draw_polyline([[0,0],[0,1],[1,1],[1,0]],color = RGBA(.5,.3,.6,.5),thickness=3)
#         glPopMatrix()

#         # Draw the camera frustum and origin using the 3d tranformation obtained from solvepnp
#         glPushMatrix()
#         glMultMatrixf(self.camera_pose_3d.T.flatten())
#         draw_frustum(self.img_size, K, 150)
#         glLineWidth(1)
#         draw_frustum(self.img_size, K, .1)
#         draw_coordinate_system(l=5)
#         glPopMatrix()


#         self.trackball.pop()

#         glfwSwapBuffers(self._window)
#         glfwMakeContextCurrent(active_window)

if __name__ == '__main__': main()