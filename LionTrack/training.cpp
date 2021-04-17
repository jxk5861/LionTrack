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

    // ../dataset/Users.Jacob.1.jpg
    // Get images and label data
    vector<Rect> dataset_faces;
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
        cout << image_path << endl;
        name_start = image_path.find_last_of('/');
        cout << "File name: " << image_path << endl;
        image_path = image_path.substr(name_start);

        name_start = image_path.find_first_of('.');
        string face_id = image_path.substr(name_start + 1, image_path.find_first_of('.', name_start + 1) - name_start - 1);

        if(name_to_id.find(face_id) == name_to_id.end()){
            name_to_id[face_id] = current_id;
            id_to_name[current_id] = face_id;
            current_id++;
        }

        Mat img = imread(image_path, IMREAD_GRAYSCALE);
        vector<Rect> faces;
        faceCascade.detectMultiScale(img, faces);

        for(Rect face : faces){
            dataset_faces.push_back(face);
            dataset_face_labels.push_back(name_to_id[face_id]);
        }
    }

    recognizer->train(dataset_faces, dataset_face_labels);
    recognizer->write("trainer/trainer.yml");

    printf("%lu faces trained\n", dataset_faces.size());

    return 0;
}


