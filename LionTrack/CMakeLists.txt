cmake_minimum_required(VERSION 3.11)
project(LionTrack)

set(CMAKE_CXX_STANDARD 17)
set(OpenCV_DIR "/home/pi/opencv/build/opencv2")

set(CMAKE_CXX_FLAGS "-std=c++17 -lstdc++fs")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_objdetect opencv_face)

# Change the .cpp file depending on which c++ class is used.
add_executable(LionTrack face_recognizer.cpp)

target_link_libraries(LionTrack ${OpenCV_LIBS} stdc++fs)