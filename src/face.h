#ifndef FACE_H
#define FACE_H
#include <iostream>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <limits>
#include <cstdlib>
// #include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
using namespace boost::numeric::ublas;
using namespace std;


#define MAX numeric_limits<double>::max(); 
#define MIN numeric_limits<double>::min();

class Face{
    public:
        //preloaded parameters
        int imgNum;
        int featurePixelNum;


        //grayscale training images
        vector<matrix<double>> trainingImages;
        //grayscale testing images 
        vector<matrix<double>> testingImages;
        //mean shape of training images
        vector<Point2d> meanShape;
        //face keypoints locations of each face 
        vector<vector<Point2d>> facekeypoints;
        //start coordinates of each image
        vector<Point2d> imageStartCor;
        //the size of each image
        vector<Point2d> imgSize; 
        //face images
        vector<Mat> faceImages;
        //average width and height of input images
        int averageHeight;
        int averageWidth;

        //the coordinates of each feature pixel, relative to the nearest
        //keypoint 
        vector<Point2d> featurePixelCoordinates;
        //index of nearest coordinates
        vector<int> nearestKeypointIndex;
        
        vector<vector<double>> trainingFeatures;
        vector<vector<double>> testingFeatures;
        
        Face();

        //get the meanShape
        void getMeanShape();
        
        void readData();
         
        //extract feature
        void extractFeature(matrix<double> iamge);
        
        
        //read parameters from parameters.ini
        void readParameters();

        void faceDetector(); 
        
          
};



#endif


