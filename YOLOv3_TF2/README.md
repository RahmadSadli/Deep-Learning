# YOLO_v3_tutorial
Accompanying code for Medium tutorial series ["The beginner’s guide to implementing YOLO (v3) in TensorFlow 2.0"](https://medium.com/@rahmadsadli/the-beginners-guide-to-implementing-yolo-v3-in-tensorflow-2-0-part-1-fcdb64b04a91)

Here's the output example:

![Detection Example](https://machinelearningspace.com/wp-content/uploads/2020/01/val2.jpg)


Requirements:
- TensorFLow 2.0
- cudatoolkit 10.0
- cudnn 7.6
- opencv 3.4

Execution:
Download orignial weights file, yolov3.eights from official yolo website:
https://pjreddie.com/media/files/yolov3.weights
to folder weights.

-For image:
python image.py

-For video/cam:
python video.py
