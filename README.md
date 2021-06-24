# Team Members
* William Hemminger (wzh44@psu.edu)
  * Software Engineering
* Jacob Karabin (jxk5861@psu.edu)
  * Software Engineering, Mathematics
* Kevin Wang (kvw5720@psu.edu)
  * Software Engineering, Mathematics

# Project Description
LionTrack is a system designed to detect faces and determine whether the person being observed is recognized as friendly or is unfamiliar, while adhering to certain timing constraints. The goal for our system was to recognize each face in a frame in under 50 ms. This system is written in C++ and makes use of the OpenCV computer vision library.  Furthermore, LionTrack can run on a Raspberry Pi 4 with a simple USB webcam.  The system’s camera detects which user is being recognized, or if the user is unknown, and saves the response time for the recognition to a log file, recording whether the recognition deadline was met.  This deadline is dynamic and depends on the number of faces in the observed frame.  For every face detected, the system adds 50 ms to the recognition deadline.  LionTrack is designed to allow for a max number of 3 detected faces to be processed to prevent freezing.  This system can be applied to security situations where only recognized users should have access to certain rooms or administrative systems.

# Details of functionality
For our system we utilized a producer-consumer pattern with interrupts. The way this is done is because the USB camera sends an interrupt to the system whenever a frame is ready to be processed. The system then places that frame in OpenCV’s VideoCapture buffer so that whenever we are ready to process a frame for faces we just consume it from the buffer. 

