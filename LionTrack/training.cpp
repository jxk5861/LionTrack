#include <iostream>
#include <vector>
#include <map>
#include <sys/time.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/face.hpp>
#include <opencv2/core/mat.hpp>
#include <fstream>

using namespace std;
using namespace cv;

int main() {
    // Now we train a model to recognize the friendly faces stored in the dataset.

    // The location of the stored images.
    string path = "../dataset";

    // Recognizer
    Ptr<LBPHFaceRecognizer> recognizer = face::LBPHFaceRecognizer::create();

    // Load a the downloaded opencv cascade https://github.com/opencv/opencv/tree/master/data/haarcascades
    CascadeClassifier faceCascade("../opencv/haarcascades/haarcascade_frontalface_default.xml");


    return 0;
}


