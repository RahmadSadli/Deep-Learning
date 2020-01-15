# YOLO_v3_tutorial
Accompanying code for Medium tutorial series ["The beginner’s guide to implementing YOLO (v3) in TensorFlow 2.0"](https://medium.com/@rahmadsadli/the-beginners-guide-to-implementing-yolo-v3-in-tensorflow-2-0-part-1-fcdb64b04a91)

This program has been tested on Windows 10 using Anaconda without any problem.

Here's the output example:

![Detection Example](https://machinelearningspace.com/wp-content/uploads/2020/01/val2.jpg)

Requirements:
- TensorFLow 2.0
- cudatoolkit 10.0
- cudnn 7.6
- opencv 3.4

Execution Steps:
1. Download the original weights, "yolov3.weights", from:
https://pjreddie.com/media/files/yolov3.weights
to the weights folder.

2. Execute convert_weights.py:

python convert_weights.py

Make sure that the weights already convert to TF2 format.
The converted weights also stored in the weights folder.

3. Test an image:

python image.py

4. Test video/cam:

python video.py
