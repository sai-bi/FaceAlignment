/**
 * @author 
 * @version 2014/02/18
 */
#include "cv.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <stdlib.h>
#include <fstream>
#include <algorithm>
#include "face.h" 
using namespace std;
using namespace cv;



int main(){
    
    vector<Mat_<double> > shapes;
    vector<Bbox> bounding_box;


    int landmark_num = 29;
    int img_num = 1345;
    Mat_<uchar> img = imread("./../../CRP/rcpr_v2/data/trainingImages/16.jpg",0);
     
    ifstream fin;
    fin.open("./../../CRP/rcpr_v2/data/boundingbox.txt");
    
    for(int i = 0;i < img_num;i++){
        Bbox temp;
        fin>>temp.start_x>>temp.start_y>>temp.width>>temp.height;
        bounding_box.push_back(temp);
    }

    fin.close();

    fin.open("./../../CRP/rcpr_v2/data/keypoints.txt");
    for(int i = 0;i < img_num;i++){
        Mat_<double> temp(landmark_num,2);
        for(int j = 0;j < landmark_num;j++){
            fin>>temp(j,0); 
        }
        for(int j = 0;j < landmark_num;j++){
            fin>>temp(j,1); 
        }
        shapes.push_back(temp);
    }
    for(int j = 0;j < img_num;j++){
        Mat_<uchar> img = imread("./../../CRP/rcpr_v2/data/trainingImages/" + to_string(j+1) + ".jpg",0);
        Bbox temp = bounding_box[j];  
        rectangle(img,Point2d(temp.start_x,temp.start_y), Point2d(temp.start_x + temp.width, temp.start_y + temp.height), Scalar(255,0,0),2); 

        Mat_<double> temp1 = shapes[j];
        for(int i = 0;i < landmark_num;i++){
            circle(img, Point2d(temp1(i,0),temp1(i,1)), 3, Scalar(255,0,0));
        }
        imshow("result",img);
        waitKey(0);
    } 

    return 0; 
}


