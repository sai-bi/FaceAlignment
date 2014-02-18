/**
 * @author 
 * @version 2014/02/17
 */

#include "cv.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <fstream>
#include <algorithm>
using namespace std;
using namespace cv;

void detectAndDisplay( Mat frame,char* fileName);

double min(double a, double b){
    return (a < b ? a : b);
}

double max(double a, double b){
    return (a > b ? a : b);
}

string face_cascade_name = "./../data/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

int main( int argc, const char** argv )
{
    Mat frame;

    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };


    DIR*     dir;
    dirent*  pdir;

    dir = opendir("../data/LFPW/lfpwImages");    


    int count = 0;
    while ((pdir = readdir(dir))) {
        count++;
        if(count <= 3)
            continue;
        cout<<count<<endl;
        char temp[] = "./../data/LFPW/lfpwImages/";  
        char* temp1 = strcat(temp,pdir->d_name);
        frame = imread(temp1);
        
        char dirName[] = "./../data/LFPW/lfpwFaces/";
        char* temp2 = strcat(dirName,pdir->d_name);
        detectAndDisplay(frame,temp2); 
    }
    closedir(dir);

    return 0;
}

void detectAndDisplay(Mat frame,char* fileName)
{

    // imshow(window_name,frame);
    std::vector<Rect> faces;
    Mat frame_gray;

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
        cropRectangle.x = max(center.x - 0.55 * faces[i].width,0);
        cropRectangle.y = max(center.y - 0.65 * faces[i].width,0); 
        cropRectangle.width = min(faces[i].width * 1.1, frame.size().width - cropRectangle.x-1);
        cropRectangle.height = min(faces[i].height * 1.3,frame.size().height - cropRectangle.y-1);

        img(cropRectangle).copyTo(croppedImage);
        imwrite(fileName,croppedImage); 
    }


}


