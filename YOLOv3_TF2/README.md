# YOLO_v3_tutorial
Accompanying code for tutorial series:

https://machinelearningspace.com/yolov3-tensorflow-2-part-1/

https://machinelearningspace.com/yolov3-tensorflow-2-part-2/

https://machinelearningspace.com/yolov3-tensorflow-2-part-3/

https://machinelearningspace.com/yolov3-tensorflow-2-part-4/

This program has been tested on Windows 10 using Anaconda without any problem.

Here's the output example:

![Detection Example](https://machinelearningspace.com/wp-content/uploads/2020/01/val2.jpg)


Requirements:
- TensorFLow 2.0
- cudatoolkit 10.0
- cudnn 7.6 
- opencv 4.2

If you use Anaconda these two lines will solve the Requirements.

  For GPU users:
  
     conda install -c conda-forge tensorflow-gpu=2.0
     
     conda install -c conda-forge opencv
     
  For CPU users:
  
     conda install -c conda-forge tensorflow=2.0
     
     conda install -c conda-forge opencv


Execution Steps:

1. Download the "yolov3.weights", from:
https://pjreddie.com/media/files/yolov3.weights
to the weights folder.

2. Execute the convert_weights.py:

   python convert_weights.py

   Before testing, make sure that the weights have been converted to TF2 format.
   The converted weights file is saved in the weights folder.

#Testing

3. For the image:

   python image.py

4. For the video/cam:

   python video.py

Good luck
