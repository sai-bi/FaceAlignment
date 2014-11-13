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
    vector<Mat_<uchar> > test_images;
    vector<BoundingBox> test_bounding_box;
    int test_img_num = 507;
    int initial_number = 20;
    int landmark_num = 29;
    ifstream fin;

    for(int i = 0;i < test_img_num;i++){
        string image_name = "./../../../Data/COFW_Dataset/testImages/";
        image_name = image_name + to_string(i+1) + ".jpg";
        Mat_<uchar> temp = imread(image_name,0);
        test_images.push_back(temp);
    }
    fin.open("./../../../Data/COFW_Dataset/boundingbox_test.txt");
    for(int i = 0;i < test_img_num;i++){
        BoundingBox temp;
        fin>>temp.start_x>>temp.start_y>>temp.width>>temp.height;
        temp.centroid_x = temp.start_x + temp.width/2.0;
        temp.centroid_y = temp.start_y + temp.height/2.0; 
        test_bounding_box.push_back(temp);
    }
    fin.close(); 
    
    ShapeRegressor regressor;
    regressor.Load("./data/model.txt");
    while(true){
        int index = 1;
        cout<<"Input index:"<<endl;
        cin>>index;

        Mat_<double> current_shape = regressor.Predict(test_images[index],test_bounding_box[index],initial_number);
        Mat test_image_1 = test_images[index].clone();
        for(int i = 0;i < landmark_num;i++){
            circle(test_image_1,Point2d(current_shape(i,0),current_shape(i,1)),3,Scalar(255,0,0),-1,8,0);
        }
        imshow("result",test_image_1);
        waitKey(0);
    }
    return 0;
}


