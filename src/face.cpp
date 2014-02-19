/**
 * @author 
 * @version 2014/02/16
 */

#include "face.h"

 
Face::Face(){
    
}

void Face::getMeanShape(){
    // initialize
}


void Face::readData(){
    ifstream fin;
    fin.open("./../data/LFPW/keypointsInfor.txt");
    
    double imageX;
    double imageY;
    double x;
    double y;
    
    while(fin>>imageX>>imageY>>x>>y){
        imageStartCor.push_back(Point2d(imageX,imageY));
        imgSize.push_back(Point2d(x,y)); 
        vector<Point2d> keypoint;
        for(int i = 0;i < 35;i++){
            fin>>x>>y;
            keypoint.push_back(Point2d(x-imageX,y-imageY));
        }
        
        facekeypoints.push_back(keypoint);
    }
    
    fin.close();

    // read all image
    Mat img; 
    string imgName = "./../data/LFPW/lfpwFaces/";
    for(int i = 1;i < imgNum+1;i++){  
        imgName = imgName + to_string(i) + ".jpg";
        img = imread(imgName.c_str());
        faceImages.push_back(img);
    }
}


void Face::getMeanShape(){
    // first scale all images to the same size
    // change the keypoints coordinates 
    // average all keypoints in the same position
    
    //get average size
    int averageWidth;
    int averageHeight;
    for(int i = 0;i < imgSize.size();i++){
        averageWidth += imgSize[i].x;
        averageHeight += imgSize[i].y;  
    }    
    
    averageWidth = averageWidth / imgSize.size();
    averageHeight = averageHeight / imgSize.size();
    
    // scale all images
    for(int i = 0;i < faceImages.size();i++){
        resize(faceImages[i],faceImages[i],Size(averageWidth,averageHeight)); 
    }  

    // change the keypoint coordinates
    for(int i = 0;i < facekeypoints[0].size();i++){
        meanShape.push_back(Point2d(0,0));
    }
    for(int i = 0;i < facekeypoints.size();i++){
        for(int j = 0;j < facekeypoints[i].size();j++){
            double x = facekeypoints[i][j].x * averageWidth / imgSize[i].x;
            double y = facekeypoints[i][j].y * averageHeight / imgSize[i].y;  
            facekeypoints[i][j].x = x;
            facekeypoints[i][j].y = y;  
            
            meanShape[j].x += x;
            meanShape[j].y += y;
        }
    } 
   
    // get the mean shape  
    for(int i = 0;i < meanShape.size();i++){
        meanShape[i].x = meanShape[i].x / facekeypoints.size();
        meanShape[i].y = meanShape[i].y / facekeypoints.size();
    }
}





