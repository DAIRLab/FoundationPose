import lcm
from lcm_systems.lcm_types.lcm_pose import lcmt_rgb_image
import numpy as np
import time

class RGBImagePublisher:
    def __init__(self):
        self.lc = lcm.LCM()

    def publish_rgb_image(self, rgb_image = None):
        # Instantiate the message object.
        self.rgb_msg = lcmt_rgb_image()

        # Set the message fields.
        self.rgb_msg.utime = int(time.time() * 1000000)
        self.rgb_msg.height = rgb_image.shape[0]
        self.rgb_msg.width = rgb.shape[1]
        self.rgb_msg.data = rgb_image

        # Publish the RGB image.
        self.lc.publish("RGB_IMAGE", self.rgb_msg.encode())
        print("Published RGB image message")
