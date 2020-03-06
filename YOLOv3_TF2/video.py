"""
Written by: Rahmad Sadli
Website : https://machinelearningspace.com

I finally made this program simple and readable
Hopefully, this program will help some beginners like me to understand better object detection.
If you want to redistribute it, just keep the author's name.

In oder to execute this program, you need to install TensorFlow 2.0 and opencv 4.x

For more details about how this program works. I explained well about it, just click the link below:
https://machinelearningspace.com/the-beginners-guide-to-implementing-yolo-v3-in-tensorflow-2-0-part-1/

Credit to:
Ayoosh Kathuria who shared his great work using pytorch, really appreaciated it.
https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
"""


import tensorflow as tf

from utils import load_class_names, output_boxes, draw_outputs, resize_image
import cv2
import time

from yolov3 import YOLOv3Net

#If you don't have enough GPU hardware device available in your machine, uncomment the following three lines:
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


model_size = (416, 416,3)
num_classes = 80
class_name = './data/coco.names'
max_output_size = 100
max_output_size_per_class= 20
iou_threshold = 0.5
confidence_threshold = 0.5


cfgfile = 'cfg/yolov3.cfg'
weightfile = 'weights/yolov3_weights.tf'

def main():

    model = YOLOv3Net(cfgfile,model_size,num_classes)

    model.load_weights(weightfile)

    class_names = load_class_names(class_name)



    win_name = 'Yolov3 detection'
    cv2.namedWindow(win_name)

    #specify the vidoe input.
    # 0 means input from cam 0.
    # For vidio, just change the 0 to video path
    cap = cv2.VideoCapture(0)
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                  cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        while True:
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = tf.expand_dims(frame, 0)
            resized_frame = resize_image(resized_frame, (model_size[0],model_size[1]))

            pred = model.predict(resized_frame)

            boxes, scores, classes, nums = output_boxes( \
                pred, model_size,
                max_output_size=max_output_size,
                max_output_size_per_class=max_output_size_per_class,
                iou_threshold=iou_threshold,
                confidence_threshold=confidence_threshold)

            img = draw_outputs(frame, boxes, scores, classes, nums, class_names)
            cv2.imshow(win_name, img)

            stop = time.time()

            seconds = stop - start
            # print("Time taken : {0} seconds".format(seconds))

            # Calculate frames per second
            fps = 1 / seconds
            print("Estimated frames per second : {0}".format(fps))

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        cap.release()
        print('Detections have been performed successfully.')


if __name__ == '__main__':
    main()
