#include <opencv2/face.hpp>

using namespace cv::face;

Ptr<LBPHFaceRecognizer> recognizer = face::LBPHFaceRecognizer::create();
recognizer.read(FileNode("../trainer/trainer.yml"));//***Going to need a trainer folder and trainer.yml file with it

CascadeClassifier faceCascade("../opencv/haarcascades/haarcascade_frontalface_default.xml");
