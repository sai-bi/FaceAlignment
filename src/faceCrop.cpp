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
using namespace std;
using namespace cv;

string face_cascade_name = "./../data/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
string window_name = "Capture - Face detection";


int validImageCount = 0;
ofstream fout;
RNG rng(12345);

double min(double a, double b){
    return (a < b ? a : b);
}

double max(double a, double b){
    return (a > b ? a : b);
}
void detectAndDisplay(Mat frame,string fileName,const vector<double>& xCor, const vector<double>& yCor)
{
    Mat croppedImage;
    Mat img = frame.clone();
    Rect cropRectangle;
    double minX = *min_element(xCor.begin(),xCor.end());
    double minY = *min_element(yCor.begin(),yCor.end());
    double maxX = *max_element(xCor.begin(),xCor.end());
    double maxY = *max_element(yCor.begin(),yCor.end());

    cropRectangle.x = max(minX - (maxX - minX) / 10,1);
    cropRectangle.y = max(minY - (maxY - minY) / 2,1);
    cropRectangle.width = min((maxX - minX)*1.2,frame.size().width - cropRectangle.x-1);
    cropRectangle.height = min((maxY - minY) * 1.6,frame.size().height-cropRectangle.y-1);

    cout<<cropRectangle.x<<" "<<cropRectangle.y<<" "<<cropRectangle.width<<" "<<cropRectangle.height<<endl;

    if(cropRectangle.width < 0 || cropRectangle.height < 0){
        validImageCount--;
        return;
    }

    try{
        img(cropRectangle).copyTo(croppedImage);
        imwrite(fileName,croppedImage);
        
        fout<<cropRectangle.x<<" "<<cropRectangle.y<<endl;
        for(int i = 0;i < xCor.size();i++){
            fout<<xCor[i]<<" "<<yCor[i];
        }  
        fout<<endl;
    }catch(int e){
        validImageCount--;
    }

    

}


int main(){
    ifstream fin;
    fin.open("./../data/LFPW/kbvt_lfpw_v1_train.csv.csv");
    fout.open("./../data/LFPW/keypointsInfor.txt");

    int imgCount = 0;
    string url;
    string worker;

    Mat img;

    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    while(fin>>url>>worker){
        vector<double> xCor;
        vector<double> yCor;
        double temp1;
        double temp2;
        double temp3;

        for(int i = 0;i < 35;i++){
            fin>>temp1>>temp2>>temp3;
            xCor.push_back(temp1);
            yCor.push_back(temp2);
        }

        if(worker != "average")
            continue;
        imgCount++;
        try{
            string temp = to_string(imgCount);
            string imgName = "./../data/LFPW/lfpwImages/" + temp + ".jpg";

            

            img = imread(imgName.c_str()); 
            if(img.empty()){
                continue;
            }
        }catch(int e){
            cout<<"Fail to read images"<<endl;
            continue;
        }
        validImageCount++; 
        // cout<<validImageCount<<" "<<imgCount<<endl;
        string temp = to_string(validImageCount);
        string imgName = "./../data/LFPW/lfpwFaces/" + temp + ".jpg";

        detectAndDisplay(img,imgName,xCor,yCor);
    } 

    fin.close();
    fout.close();

    return 0;

}


