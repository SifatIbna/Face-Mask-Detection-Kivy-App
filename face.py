from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

import cv2
import tensorflow as tf
import os
import numpy as np

from deeplearning import face_mask_prediction

class CamApp(App):
    
    def build(self):
        self.web_cam = Image(size_hint=(1,.8))
        # self.button = Button(text="verify",on_press=self.verify, size_hint = (1,.1))
        self.verification_label = Label(text='Verification Uninitiated',size_hint=(1,.1))

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        # layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update,1.0/33.0)


        return layout
    
    def update(self,*args):
        self.ret , self.frame = self.capture.read() # BGR Format
        face_mask_prediction(self.frame,self.verification_label)
        
        buf = cv2.flip(self.frame,0).tostring()
        img_texture = Texture.create(size=(self.frame.shape[1],self.frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf,colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # def verify(self,*args):
    #     prediction_img,label = face_mask_prediction(self.frame,self.verification_label)
        
        

if __name__ == '__main__':
    CamApp().run()