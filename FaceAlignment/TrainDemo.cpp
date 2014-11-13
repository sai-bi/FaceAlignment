/*
Author: Bi Sai 
Date: 2014/06/18
This program is a reimplementation of algorithms in "Face Alignment by Explicit 
Shape Regression" by Cao et al.
If you find any bugs, please email me: soundsilencebisai-at-gmail-dot-com

Copyright (c) 2014 Bi Sai 
The MIT License (MIT)
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "FaceAlignment.h"
using namespace std;
using namespace cv;

int main(){
    int img_num = 1345;
    int candidate_pixel_num = 400;
    int fern_pixel_num = 5;
    int first_level_num = 10;
    int second_level_num = 500; 
    int landmark_num = 29;
    int initial_number = 20;
    vector<Mat_<uchar> > images;
    vector<BoundingBox> bbox; 
    
    cout<<"Read images..."<<endl;
    for(int i = 0;i < img_num;i++){
        string image_name = "./../../../Data/COFW_Dataset/trainingImages/";
        image_name = image_name + to_string(i+1) + ".jpg";
        Mat_<uchar> temp = imread(image_name,0);
        images.push_back(temp);
    }
    
    vector<Mat_<double> > ground_truth_shapes;
    vector<BoundingBox> bounding_box;
    ifstream fin;
    fin.open("./../../../Data/COFW_Dataset/boundingbox.txt");
    for(int i = 0;i < img_num;i++){
        BoundingBox temp;
        fin>>temp.start_x>>temp.start_y>>temp.width>>temp.height;
        temp.centroid_x = temp.start_x + temp.width/2.0;
        temp.centroid_y = temp.start_y + temp.height/2.0; 
        bounding_box.push_back(temp);
    }
    fin.close(); 

    fin.open("./../../../Data/COFW_Dataset/keypoints.txt");
    for(int i = 0;i < img_num;i++){
        Mat_<double> temp(landmark_num,2);
        for(int j = 0;j < landmark_num;j++){
            fin>>temp(j,0); 
        }
        for(int j = 0;j < landmark_num;j++){
            fin>>temp(j,1); 
        }
        ground_truth_shapes.push_back(temp);
    }

    fin.close(); 
    
    ShapeRegressor regressor;
    regressor.Train(images,ground_truth_shapes,bounding_box,first_level_num,second_level_num,
                    candidate_pixel_num,fern_pixel_num,initial_number);
    regressor.Save("./data/model.txt");

    return 0;
}

