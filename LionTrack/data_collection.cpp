#include <iostream>
#include <vector>
#include <sys/time.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>

using namespace std;
using namespace cv;

int main() {
    // First we gather the data of a friendly face. This will allow the us to train the recognizer to recognize a face.

    // Load a the downloaded opencv cascade https://github.com/opencv/opencv/tree/master/data/haarcascades
    CascadeClassifier faceCascade("../opencv/haarcascades/haarcascade_frontalface_default.xml");
    VideoCapture capture = VideoCapture(0);
    capture.set(CAP_PROP_FRAME_WIDTH, 640);
    capture.set(CAP_PROP_FRAME_HEIGHT, 480);

    capture.open(0);
    if(!capture.isOpened()){
        return 1;
    }

    Mat frame, gray;
    int face_count = 0;
    auto face_id = "Kevin";// Change this variable depending on person being data-mined.
    int startTime = 0, finishTime = 0;
    ofstream logfile;

    while(true){
        // Read the cameras current frame.
        capture >> frame;

        startTime = time(0); //getting the start time of the detection process

        if(frame.empty()) {
            break;
        }

        flip(frame, frame, 1);
//        imshow("Video", frame);
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector<Rect> faces;
        double scaleFactor = 1.2;
        int minNeighbors = 5;
        int flags = 0;
        Size minSize(30, 30);

        // Search for faces in the image.
        faceCascade.detectMultiScale(gray, faces, scaleFactor, minNeighbors, flags, minSize);

        // A face has been detected at (x,y) with size (w,h)
        for(Rect r : faces) {
            int x = r.x;
            int y = r.y;
            int w = r.width;
            int h = r.height;

            face_count++;

            Scalar color(255, 0, 0);
            // Draw a rectangle on the image where the face is detected.
            rectangle(frame, r, color);
            Mat detected_face_grayscale = gray.colRange(x, x + w).rowRange(y, y + h);
            stringstream path;
            path << "../dataset/Users." << face_id << "." << face_count << ".jpg";
            imwrite(path.str(), detected_face_grayscale);
            finishTime = time(0);


            //logging response time to a text file
            logfile.open("logfile.txt", ios_base::app);
            logfile << "Response time: " << finishTime - startTime << "ms" << endl;

            logfile.close();
        }
        // Show the image window.
        imshow("detected face", frame);

        // Press q to exit from window. Also this provides a needed delay between displaying each frame.
        char c = (char) waitKey(30);
        if( c == 27 || c == 'q' || c == 'Q' ) {
            break;
        }
        if(face_count > 100){
            break;
        }
    }

    return 0;
}
