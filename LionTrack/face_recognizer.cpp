#include <opencv2/face.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <map>
#include <sstream>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>

using namespace std;
using namespace cv;
using namespace face;
using namespace chrono;

int main(){
    // Create a LBPH face recognizer. This is a good recognizer for our type of application.
    Ptr<LBPHFaceRecognizer> recognizer = face::LBPHFaceRecognizer::create();
    ofstream logfile;

    // Open the logfile for logging when deadlines are met or missed.
    logfile.open("logfile.txt", ios_base::app);

    // Read the training model in from the /trainer/trainer.yml directory. This is the model used to recognize friendly faces.
    recognizer->read("../trainer/trainer.yml");

    // Create the face cascade for detecting where a face is on the camera. (any face, not just friendly ones).
    CascadeClassifier faceCascade("../opencv/haarcascades/haarcascade_frontalface_default.xml");

    map<int, string> id_to_name = { {0, "Kevin"}, {1, "Jacob"} };

    // Initialize the video capture from usb.
    VideoCapture capture = VideoCapture(0);
    capture.set(CAP_PROP_FRAME_WIDTH, 640);
    capture.set(CAP_PROP_FRAME_HEIGHT, 480);

    capture.open(0);

    // Exit if the camera is not open.
    if(!capture.isOpened()){
        return 1;
    }

    // The image frame and image grayscale frame to store data to.
    Mat frame, gray;

    while(true){
        // Store the current frame into the buffer.
        capture >> frame;

        if(frame.empty()) {
            break;
        }

        // Flip the frame so it appears like a mirror.
        flip(frame, frame, 1);

        // Convert the image to grayscale for image processing. The classifiers work on grayscale images.
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // The list to store the faces to.
        vector<Rect> faces;
        double scaleFactor = 1.2;
        int minNeighbors = 5;
        int flags = 0;
        Size minSize(30, 30);

        // Search for faces in the image.
        faceCascade.detectMultiScale(gray, faces, scaleFactor, minNeighbors, flags, minSize);

        // Save the start time for determining if the deadline is met in the future.
        // The deadline is met if the image is read, classified, and recognized in the 50ms*num_faces window.
        milliseconds startTime = duration_cast< milliseconds >(
                system_clock::now().time_since_epoch()
        );

        // Set the minimum face count.
        int face_count_min = min((int)faces.size(), 3);

        for(int i = 0; i < face_count_min; i++) {
            Rect r = faces[i];
            int x = r.x;
            int y = r.y;
            int w = r.width;
            int h = r.height;

            Scalar color(255, 0, 0);
            // Draw a rectangle on the image where the face is detected.
            rectangle(frame, r, color);

            // Crop the image to just the detected face.
            Mat detected_face_grayscale = gray.colRange(x, x + w).rowRange(y, y + h);

            int id;
            double confidence;
            // Find the confidence that the image is a friendly face.
            recognizer->predict(detected_face_grayscale, id, confidence);

            string label;
            stringstream stream;

            //confidence = 100 - confidence;

            if(confidence < 50){
                stream << id_to_name[id];
//                cout << id_to_name[id] << " confidence: " << 100 - confidence << endl;
            } else {
                stream << "Unknown";
//                cout << "Unknown confidence: " << 100 - confidence << endl;
            }


            stream << " " << id << " "  << round(100 - confidence);
            label = stream.str();

            // Put the recognized face label on the image.
            cv::putText(frame, label, Point(x+5, y-5), FONT_HERSHEY_PLAIN, 1, color);
        }

        // The task has finished.
        milliseconds finishTime = duration_cast< milliseconds >(
                system_clock::now().time_since_epoch()
        );

        // Only log data if at least one face is present.
        if(faces.size() > 0)
        {
            int totalTime = finishTime.count() - startTime.count();
            logfile << "Response time: " << totalTime << "ms";

            // Determine if the deadline was met and log it.
            if(totalTime < 50*face_count_min)
            {
                logfile << " Deadline Met" << endl;
            }
            else
            {
                logfile << " Deadline Missed" << endl;
            }
        }


        // Show the frame with the draw on labels.
        imshow("detected face", frame);

        // Press q to exit from window. Also this provides a needed delay between displaying each frame.
        char c = (char) waitKey(1000 / 60);
        if( c == 27 || c == 'q' || c == 'Q' ) {

            if(logfile)
            {
                logfile << "--------" << endl;
                logfile.close();
            }
            break;
        }
    }

    if(logfile)
    {
        logfile << "--------" << endl;
        logfile.close();
    }
}
