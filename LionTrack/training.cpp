#include <iostream>
#include <filesystem>
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
#include <cstring>

using namespace std;
using namespace cv;
using namespace cv::face;
using filesystem::directory_iterator;

int main() {
    // Now we train a model to recognize the friendly faces stored in the dataset.

    // The location of the stored images.
    string path = "../dataset";

    // Recognizer
    Ptr<LBPHFaceRecognizer> recognizer = face::LBPHFaceRecognizer::create();

    // Load a the downloaded opencv cascade https://github.com/opencv/opencv/tree/master/data/haarcascades
    CascadeClassifier faceCascade("../opencv/haarcascades/haarcascade_frontalface_default.xml");

    // ../dataset/Users.Jacob.1.jpg
    // Get images and label data
    vector<Mat> dataset_faces;
    vector<int> dataset_face_labels;

    map<string, int> name_to_id;
    map<int, string> id_to_name;
    int current_id = 0;

    // read all images in the dataset for training.
    string image_path;
    int name_start;
    for (const auto & file : directory_iterator(path))
    {
        //User.William.1.jpg
        image_path = file.path();
        name_start = image_path.find_last_of('/');
        string image_path_end = image_path.substr(name_start);

        name_start = image_path_end.find_first_of('.');
        string face_id = image_path_end.substr(name_start + 1, image_path_end.find_first_of('.', name_start + 1) - name_start - 1);

        if(name_to_id.find(face_id) == name_to_id.end()){
            name_to_id[face_id] = current_id;
            id_to_name[current_id] = face_id;
            current_id++;
        }

        Mat face = imread(image_path, IMREAD_GRAYSCALE);
//        cout << face.size().width << " " << face.size().height << endl;
//        dataset_faces.push_back(face);
//        dataset_face_labels.push_back(name_to_id[face_id]);
        vector<Rect> faces;

        vector<Rect> detected_faces;
        faceCascade.detectMultiScale(face, detected_faces);

        for(Rect r : detected_faces){
            int x = r.x;
            int y = r.y;
            int w = r.width;
            int h = r.height;

            Mat detected_face_grayscale = face.colRange(x, x + w).rowRange(y, y + h);

            dataset_faces.push_back(face);
            dataset_face_labels.push_back(name_to_id[face_id]);

            cout << "Adding face: " << face_id << " (" << name_to_id[face_id] << ") File: " << image_path << endl;
        }
    }
//    cout << dataset_faces.size() << " " << dataset_face_labels.size() << endl;

    recognizer->train(dataset_faces, dataset_face_labels);
    recognizer->write("../trainer/trainer.yml");

    printf("%lu faces trained (%lu labels)\n", dataset_faces.size(), dataset_face_labels.size());

    return 0;
}