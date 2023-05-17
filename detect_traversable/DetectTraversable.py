#! /usr/bin/env python
# Basic ROS 2 program to subscribe to real-time streaming
# video from your built-in webcam
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com

# Import the necessary libraries
import rclpy # Python library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

class ImageSubscriber(Node):
    """
    Create an ImageSubscriber class, which is a subclass of the Node class.
    """
    def __init__(self):
        """
        Class constructor to set up the node
        """
        # Initiate the Node class's constructor and give it a name
        super().__init__('detect_traversable_sub')

        # Create the subscriber. This subscriber will receive an Image
        # from the video_frames topic. The queue size is 10 messages.
        self.subscription = self.create_subscription(
            Image,
            'video_frames',
            self.listener_callback,
            10)
        self.subscription # prevent unused variable warning

        # Used to convert between ROS and OpenCV images
        self.cv_br = CvBridge()
        self.cfg = get_cfg()

        self.cfg.merge_from_file(model_zoo.get_config_file("Base-RCNN-FPN.yaml"))
        self.cfg.MODEL.META_ARCHITECTURE = "SemanticSegmentor"
        self.cfg.MODEL.RESNETS.DEPTH = 50
        self.cfg.INPUT.MIN_SIZE_TRAIN = 480
        self.cfg.INPUT.MAX_SIZE_TRAIN = 640
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        self.cfg.MODEL.WEIGHTS = "/home/ettore/ros2_humble_ws/src/detect_traversable/model_final.pth"
        self.cfg.MODEL.DEVICE = "cpu"
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.predictor = DefaultPredictor(self.cfg)

        stuff_classes = ["traversable", "obstacle"]
        stuff_colors = [(0,200,0),(200,0,0)]
        MetadataCatalog.get("dataset").set(stuff_classes=stuff_classes)
        MetadataCatalog.get("dataset").set(stuff_colors=stuff_colors)

        self.slower = 0

    def listener_callback(self, data):
        """
        Callback function.
        """
        # Display the message on the console

        self.slower = (self.slower + 1 ) % 5
        if slower != 0:
            return

        self.get_logger().info('Receiving video frame')

        # Convert ROS Image message to OpenCV image
        frame = self.cv_br.imgmsg_to_cv2(data)
        # frame = cv2.imread("/home/ettore/Downloads/vineyard.jpg")
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        outputs = self.predictor(frame)

        viz = Visualizer(frame[:, :, ::-1],
                           scale=0.8,
                           metadata=MetadataCatalog.get("dataset"),
                           instance_mode=ColorMode.SEGMENTATION
            )

        viz = viz.draw_sem_seg((outputs["sem_seg"].argmax(dim=0)).to("cpu"))

        # Display image
        cv2.imshow("camera", viz.get_image()[:, :, ::-1])

        cv2.waitKey(1000)

def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)

    # Create the node
    image_subscriber = ImageSubscriber()

    # Spin the node so the callback function is called.
    rclpy.spin(image_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    image_subscriber.destroy_node()

    # Shutdown the ROS client library for Python
    rclpy.shutdown()

if __name__ == '__main__':
    main()
