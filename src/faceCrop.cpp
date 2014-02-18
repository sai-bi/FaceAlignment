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

    // cout<<minX<<"   "<<maxX<<"  "<<minY<<"  "<<maxY<<endl;
    cropRectangle.x = max(minX - (maxX - minX) / 10,1);
    cropRectangle.y = max(minY - (maxY - minY) / 2,1);
    cropRectangle.width = min((maxX - minX)*1.2,frame.size().width - cropRectangle.x-1);
    cropRectangle.height = min((maxY - minY) * 1.6,frame.size().height-cropRectangle.y-1);

    cout<<cropRectangle.x<<" "<<cropRectangle.y<<" "<<cropRectangle.width<<" "<<cropRectangle.height<<endl;

    img(cropRectangle).copyTo(croppedImage);
    imwrite(fileName,croppedImage);  
    // imshow(window_name,frame);
    std::vector<Rect> faces;
    Mat frame_gray;
    /*
       cvtColor( frame, frame_gray, CV_BGR2GRAY );
       equalizeHist( frame_gray, frame_gray );

       try{
       face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
       }
       catch(int e){
       ofstream fout;
       fout.open("./../data/LFPW/multiFaces.txt",std::fstream::out | std::fstream::app);
       fout<<fileName<<" "<<faces.size()<<endl;
       fout.close();
       return; 
       }

    // cout<<faces.size()<<endl;
    Mat img = frame.clone();

    if(faces.size() == 0 || faces.size() > 1){
    ofstream fout;
    fout.open("./../data/LFPW/multiFaces.txt",std::fstream::out | std::fstream::app);
    fout<<fileName<<" "<<faces.size()<<endl;
    fout.close();
    return;
    }

    for( size_t i = 0; i < faces.size(); i++ )
    {
    Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    Point2d p1(center.x-0.6 * faces[i].width, center.y-0.65*faces[i].height);
    Point2d p2(center.x+0.6 * faces[i].width, center.y+0.65*faces[i].height);
    rectangle(frame,p1,p2,Scalar(255,0,255),4,8,0);
    Mat facesROI = frame_gray( faces[i] );

    Mat croppedImage;
    Rect cropRectangle;
    // cropRectangle.x = max(center.x - 0.55 * faces[i].width,0);
    // cropRectangle.y = max(center.y - 0.65 * faces[i].width,0); 
    // cropRectangle.width = min(faces[i].width * 1.1, frame.size().width - cropRectangle.x-1);
    // cropRectangle.height = min(faces[i].height * 1.3,frame.size().height - cropRectangle.y-1);

    double minX = *min_element(xCor.begin(),xCor.end());
    double minY = *min_element(yCor.begin(),yCor.end());
    double maxX = *max_element(xCor.begin(),xCor.end());
    double maxY = *max_element(yCor.begin(),yCor.end());

    cout<<minX<<"   "<<maxX<<"  "<<minY<<"  "<<maxY<<endl;
    cropRectangle.x = max(minX - (maxX - minX) / 10,0);
    cropRectangle.y = max(minY - (maxY - minY) / 2,0);
    cropRectangle.width = min((maxX - minX)*1.2,frame.size().width - cropRectangle.x-1);
    cropRectangle.height = min((maxY - minY) * 1.6,frame.size().height-cropRectangle.y-1);



    img(cropRectangle).copyTo(croppedImage);
    imwrite(fileName,croppedImage); 
    // imshow(window_name,frame);
    // waitKey(0);
    }
    */


}


int main(){
    ifstream fin;
    fin.open("./../data/LFPW/kbvt_lfpw_v1_train.csv.csv");

    int imgCount = 0;
    string url;
    string worker;

    Mat img;

    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    int validImageCount = 0;
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
            // cout<<imgName<<endl;
            img = imread(imgName.c_str()); 
            if(img.empty()){
                // cout<<"empty"<<endl;
                continue;
            }
        }catch(int e){
            cout<<"Fail to read images"<<endl;
            continue;
        }
        validImageCount++; 
        cout<<validImageCount<<" "<<imgCount<<endl;
        string temp = to_string(validImageCount);
        string imgName = "./../data/LFPW/lfpwFaces/" + temp + ".jpg";

        detectAndDisplay(img,imgName,xCor,yCor);
    } 

}


