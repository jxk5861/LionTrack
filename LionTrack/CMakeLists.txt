cmake_minimum_required(VERSION 3.19)
project(LionTrack)

set(CMAKE_CXX_STANDARD 17)
set(OpenCV_DIR "C:\\opencv\\vs-x64-build")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_objdetect opencv_face)

# Change the .cpp file depending on which c++ class is used.
add_executable(LionTrack training.cpp)

target_link_libraries(LionTrack ${OpenCV_LIBS})