#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

int main() {
    VideoCapture capture = VideoCapture(0);
    capture.set(CAP_PROP_FRAME_WIDTH, 640);
    capture.set(CAP_PROP_FRAME_HEIGHT, 480);

    capture.open(0);
    if(!capture.isOpened()){
        return 1;
    }

    Mat frame, flipped;
    while(true){
        auto ret = (capture >> frame);
        if(frame.empty()) {
            break;
        }

        flip(frame, flipped, 1);
        imshow("Video", frame);

        char c = (char)waitKey(10);

        // Press q to exit from window
        if( c == 27 || c == 'q' || c == 'Q' )
            break;
    }

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
