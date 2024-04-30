import rospy # type: ignore
import convert2D_to_3D
from sensor_msgs.msg import Image # type: ignore
import math
import numpy as np # type: ignore
from cv_bridge import CvBridge # type: ignore
import cv2

CAMERA_HEIGHT = 0.5 #metre

def depth_referance(x,y):
    depth_referance_matrix = []
    X_SIZE = x
    Y_SIZE = y
    for i in range(Y_SIZE):
        row = []
        for j in range(X_SIZE):
            converted_pixel = convert2D_to_3D.convert_to_3d((i,j),(X_SIZE,Y_SIZE))
            depth = math.hypot(converted_pixel[0],converted_pixel[1],CAMERA_HEIGHT)
            row.append(depth)
        depth_referance_matrix.append(row)
        
    return np.array(depth_referance_matrix)


def depth_diff_callback(depth_image_msg):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(depth_image_msg, desired_encoding="16UC1")
    depth_image = np.asarray(cv_image)

    diff = depth_image - depth_referance_matrix
    
    cv_image = cv2.cvtColor(diff, "bgr8") 
    cv2.imshow("Image", cv_image)
    cv2.waitKey(1)

    
    
        
def depth_diff_subscriber():
    rospy.init_node('depth_diff_subscirber')
    rospy.Subscriber("/camera/depth/color/points", Image, depth_diff_callback)
    rospy.spin()
    
    
if __name__ == '__main__':
    depth_referance_matrix = depth_referance(1920,1080)
    depth_diff_subscriber()
    