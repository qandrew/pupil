'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import sys, os,platform
import cv2
import numpy as np
from file_methods import Persistent_Dict
from pyglui import ui
from player_methods import transparent_image_overlay
from plugin import Plugin

# helpers/utils
from version_utils import VersionFormat

#capture
from video_capture import autoCreateCapture,EndofVideoFileError,FileSeekError,FakeCapture,FileCaptureError

#mouse
from glfw import glfwGetCursorPos,glfwGetWindowSize,glfwGetCurrentContext
from methods import normalize,denormalize

#logging
import logging
logger = logging.getLogger(__name__)


def get_past_timestamp(idx,timestamps):
    """
    recursive function to find the most recent valid timestamp in the past
    """
    if idx == 0:
        # if at the beginning, we can't go back in time.
        return get_future_timestamp(idx,timestamps)
    if timestamps[idx]:
        res = timestamps[idx][-1]
        return res
    else:
        return get_past_timestamp(idx-1,timestamps)

def get_future_timestamp(idx,timestamps):
    """
    recursive function to find most recent valid timestamp in the future
    """
    if idx == len(timestamps)-1:
        # if at the end, we can't go further into the future.
        return get_past_timestamp(idx,timestamps)
    elif timestamps[idx]:
        return timestamps[idx][0]
    else:
        idx = min(len(timestamps),idx+1)
        return get_future_timestamp(idx,timestamps)

def get_nearest_timestamp(past_timestamp,future_timestamp,world_timestamp):
    dt_past = abs(past_timestamp-world_timestamp)
    dt_future = abs(future_timestamp-world_timestamp) # abs prob not necessary here, but just for sanity
    if dt_past < dt_future:
        return past_timestamp
    else:
        return future_timestamp


def correlate_eye_world(eye_timestamps,world_timestamps):
    """
    This function takes a list of eye timestamps and world timestamps
    and correlates one eye frame per world frame
    Returns a mapping that correlates a single eye frame index with each world frame index.
    Up and downsampling is used to achieve this mapping.
    """
    # return framewise mapping as a list
    e_ts = eye_timestamps
    w_ts = list(world_timestamps)
    eye_frames_by_timestamp = dict(zip(e_ts,range(len(e_ts))))

    eye_timestamps_by_world_index = [[] for i in world_timestamps]

    frame_idx = 0
    try:
        current_e_ts = e_ts.pop(0)
    except:
        logger.warning("No eye timestamps found.")
        return eye_timestamps_by_world_index

    while e_ts:
        # if the current eye timestamp is before the mean of the current world frame timestamp and the next worldframe timestamp
        try:
            t_between_frames = ( w_ts[frame_idx]+w_ts[frame_idx+1] ) / 2.
        except IndexError:
            break
        if current_e_ts <= t_between_frames:
            eye_timestamps_by_world_index[frame_idx].append(current_e_ts)
            current_e_ts = e_ts.pop(0)
        else:
            frame_idx+=1

    idx = 0
    eye_world_frame_map = []
    # some entiries in the `eye_timestamps_by_world_index` might be empty -- no correlated eye timestamp
    # so we will either show the previous frame or next frame - whichever is temporally closest
    for candidate,world_ts in zip(eye_timestamps_by_world_index,w_ts):
        # if there is no candidate, then assign it to the closest timestamp
        if not candidate:
            # get most recent timestamp, either in the past or future
            e_past_ts = get_past_timestamp(idx,eye_timestamps_by_world_index)
            e_future_ts = get_future_timestamp(idx,eye_timestamps_by_world_index)
            eye_world_frame_map.append(eye_frames_by_timestamp[get_nearest_timestamp(e_past_ts,e_future_ts,world_ts)])
        else:
            # TODO - if there is a list of len > 1 - then we should check which is the temporally closest timestamp
            eye_world_frame_map.append(eye_frames_by_timestamp[eye_timestamps_by_world_index[idx][-1]])
        idx += 1

    return eye_world_frame_map


class Eye_Video_Overlay(Plugin):
    """docstring
    """
    def __init__(self,g_pool,opaqueness=0.6,eyesize=1.0,mirror0=1,mirror1=1,flip0 = 0, flip1 = 0,move_around=0,pos=[[640,10],[10,10]]):
        super(Eye_Video_Overlay, self).__init__(g_pool)
        self.order = .6
        self.menu = None

        # user controls
        self.opaqueness = opaqueness #opacity level of eyes
        self.eyesize = eyesize #scale
        self.mirror0 = mirror0
        self.mirror1 = mirror1
        self.flip0 = flip0
        self.flip1 = flip1
        self.showeyes = 'both eye1 and eye2'
        self.move_around = move_around
        self.pos = list(pos)
        self.drag_offset0 = None
        self.drag_offset1 = None
        self.size = [0,0]


        # load eye videos and eye timestamps
        if g_pool.rec_version < VersionFormat('0.4'):
            eye0_video_path = os.path.join(g_pool.rec_dir,'eye.avi')
            eye0_timestamps_path = os.path.join(g_pool.rec_dir,'eye_timestamps.npy')
        else:
            eye0_video_path = os.path.join(g_pool.rec_dir,'eye0.mkv')
            eye0_timestamps_path = os.path.join(g_pool.rec_dir,'eye0_timestamps.npy')
            eye1_video_path = os.path.join(g_pool.rec_dir,'eye1.mkv')
            eye1_timestamps_path = os.path.join(g_pool.rec_dir,'eye1_timestamps.npy')


        # Initialize capture -- for now we just try with monocular
        try:
            self.cap = autoCreateCapture(eye0_video_path,timestamps=eye0_timestamps_path)
        except FileCaptureError:
            logger.error("Could not load eye video.")
            self.alive = False
            return

        #initialize capture for second eye
        try:
            self.cap = autoCreateCapture(eye1_video_path,timestamps=eye1_timestamps_path)
        except:
            logger.error("There is only 1 eye")
            self.showeyes = "only eye1"

        #finding first frame for eye0
        self._frame = self.cap.get_frame()
        self.width, self.height = self.cap.frame_size

        eye0_timestamps = list(np.load(eye0_timestamps_path))
        self.eye0_world_frame_map = correlate_eye_world(eye0_timestamps,g_pool.timestamps)

        #finding first frame for eye1
        if 'eye2' in self.showeyes:
            self._frame1 =self.cap.get_frame()

            eye1_timestamps = list(np.load(eye1_timestamps_path))
            self.eye1_world_frame_map = correlate_eye_world(eye1_timestamps,g_pool.timestamps)

    def unset_alive(self):
        self.alive = False

    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Eye Video Overlay')
        self.menu.append(ui.Info_Text('Show the eye video overlaid on top of the world video.'))
        self.menu.append(ui.Slider('opaqueness',self,min=0.0,step=0.05,max=1.0,label='Opacity'))
        self.menu.append(ui.Slider('eyesize',self,min=0.2,step=0.1,max=1.0,label='Scale of Video'))
        self.menu.append(ui.Switch('move_around',self,label="Move Overlay Around"))
        self.menu.append(ui.Switch('mirror0',self,label="Eye 1: Horiz. Flip"))
        self.menu.append(ui.Switch('flip0',self,label="Eye 1: Vert. Flip"))
        if 'both' in self.showeyes:
            self.menu.append(ui.Selector('showeyes',self,label='Show',selection=['both eye1','both eye2','both eye1 and eye2'],labels= ['eye 1','eye 2','both']))
        if 'eye2' in self.showeyes:
            self.menu.append(ui.Switch('mirror1',self,label="Eye 2: Horiz Flip"))
            self.menu.append(ui.Switch('flip1',self,label="Eye 2: Vert Flip"))
        self.menu.append(ui.Button('close',self.unset_alive))
        # add menu to the window
        self.g_pool.gui.append(self.menu)

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def update(self,frame,events):

        """ For the first eye! """
        if 'eye1' in self.showeyes:
            requested_eye_frame_idx = self.eye0_world_frame_map[frame.index]

            #1. do we need a new frame?
            if requested_eye_frame_idx != self._frame.index:
                # do we need to seek?
                if requested_eye_frame_idx == self.cap.get_frame_index()+1:
                    # if we just need to seek by one frame, its faster to just read one and and throw it away.
                    _ = self.cap.get_frame()
                if requested_eye_frame_idx != self.cap.get_frame_index():
                    # only now do I need to seek
                    self.cap.seek_to_frame(requested_eye_frame_idx)
                # reading the new eye frame frame
                try:
                    self._frame = self.cap.get_frame()
                except EndofVideoFileError:
                    logger.warning("Reached the end of the eye video.")
            else:
                #our old frame is still valid because we are doing upsampling
                pass

            #2. drawing the eye0 overlay
            eyeimage0 = cv2.resize(self._frame.img,(0,0),fx=self.eyesize-0.1, fy=self.eyesize-0.1) 

            #3. resizing
            if self.drag_offset0 is not None:
                pos = glfwGetCursorPos(glfwGetCurrentContext())
                pos = normalize(pos,glfwGetWindowSize(glfwGetCurrentContext()))
                pos = denormalize(pos,(frame.img.shape[1],frame.img.shape[0]) ) # Position in img pixels
                self.pos[0][0] = pos[0]+self.drag_offset0[0]
                self.pos[0][1] = pos[1]+self.drag_offset0[1]
            else:
                #self.pos[0] = [round((self.eyesize-0.1)*self.width+2*pad), pad] #makes right eye move towards left corner as scale decreases
                self.size = [round(self.width*(self.eyesize- 0.1)), round(self.height*(self.eyesize-0.1))]

            #5. keep in image bounds, do this even when not dragging because the image sizes could change.
            self.pos[0][1] = min(frame.img.shape[0]-self.size[1],max(self.pos[0][1],0)) #frame.img.shape[0] is height, frame.img.shape[1] is width of screen
            self.pos[0][0] = min(frame.img.shape[1]-self.size[0],max(self.pos[0][0],0))

            #4. flipping images and stuff
            if self.mirror0:
                eyeimage0 = np.fliplr(eyeimage0)
            if self.flip0:
                eyeimage0 = np.flipud(eyeimage0)
            temp = cv2.cvtColor(eyeimage0,cv2.COLOR_BGR2GRAY) #auto grey scaling
            eyeimage0 = cv2.cvtColor(temp,cv2.COLOR_GRAY2BGR)

            #6. finally overlay the image
            transparent_image_overlay(self.pos[0],eyeimage0,frame.img,self.opaqueness)


        """ For the Second Eye! """
        if 'eye2' in self.showeyes:
            requested_eye_frame_idx = self.eye1_world_frame_map[frame.index]

            #1. do we need a new frame?
            if requested_eye_frame_idx != self._frame1.index:
                # do we need to seek?
                if requested_eye_frame_idx == self.cap.get_frame_index()+1:
                    # if we just need to seek by one frame, its faster to just read one and and throw it away.
                    _ = self.cap.get_frame()
                if requested_eye_frame_idx != self.cap.get_frame_index():
                   # only now do I need to seek
                   self.cap.seek_to_frame(requested_eye_frame_idx)
                # reading the new eye frame frame
                try:
                   self._frame1 = self.cap.get_frame()
                except EndofVideoFileError:
                    logger.warning("Reached the end of the eye video.")
            else:
                #our old frame is still valid because we are doing upsampling
                pass

            #2. resizing or dragging around the window
            if self.drag_offset1 is not None:
                pos = glfwGetCursorPos(glfwGetCurrentContext())
                pos = normalize(pos,glfwGetWindowSize(glfwGetCurrentContext()))
                pos = denormalize(pos,(frame.img.shape[1],frame.img.shape[0]) ) # Position in img pixels
                self.pos[1][0] = pos[0]+self.drag_offset1[0]
                self.pos[1][1] = pos[1]+self.drag_offset1[1]

            #3. drawing eye1 overlay (the second eye)
            eyeimage1 = cv2.resize(self._frame1.img,(0,0),fx=self.eyesize-0.1, fy=self.eyesize-0.1) 

            #4. flipping image
            if self.mirror1:
                eyeimage1 = np.fliplr(eyeimage1)
            if self.flip1:
                eyeimage1 = np.flipud(eyeimage1)
            temp = cv2.cvtColor(eyeimage1,cv2.COLOR_BGR2GRAY) #auto grey scaling
            eyeimage1 = cv2.cvtColor(temp,cv2.COLOR_GRAY2BGR)

            #5. keep in image bounds, do this even when not dragging because the image sizes could change.
            self.pos[1][1] = min(frame.img.shape[0]-self.size[1],max(self.pos[1][1],0)) #frame.img.shape[0] is height, frame.img.shape[1] is width of screen
            self.pos[1][0] = min(frame.img.shape[1]-self.size[0],max(self.pos[1][0],0))

            #6. finally draw the image
            transparent_image_overlay(self.pos[1],eyeimage1,frame.img,self.opaqueness)


    def on_click(self,pos,button,action):
        if self.move_around == 1 and action == 1:
            #eye0
            if 'eye1' in self.showeyes:
                if self.pos[0][0] < pos[0] < self.pos[0][0]+self.size[0] and self.pos[0][1] < pos[1] < self.pos[0][1] + self.size[1]:
                    self.drag_offset0 = self.pos[0][0]-pos[0],self.pos[0][1]-pos[1]
            #eye1
            if 'eye2' in self.showeyes:
                if self.pos[1][0] < pos[0] < self.pos[1][0]+self.size[0] and self.pos[1][1] < pos[1] < self.pos[1][1] + self.size[1]:
                    self.drag_offset1 = self.pos[1][0]-pos[0],self.pos[1][1]-pos[1]
        else:
            self.drag_offset0 = None
            self.drag_offset1 = None



    def get_init_dict(self):
        return {'opaqueness':self.opaqueness,'eyesize':self.eyesize,'mirror0':self.mirror0,'mirror1':self.mirror1,'flip0':self.flip0,'flip1':self.flip1,'pos':self.pos,'move_around':self.move_around}

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()
