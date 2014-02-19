#ifndef FACE_H
#define FACE_H
#include <iostream>
#include <cmath>
#include <fstream>


#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
using namespace boost::numeric::ublas;
using namespace std;


class Face{
    public:
        //grayscale training images
        vector<matrix<double>> trainingImages;
        //grayscale testing images 
        vector<matrix<double>> testingImages;
        //mean shape of training images
        vector<Point2d> meanShape;
        vector<vector<Point2d>> facekeypoints;
        vector<Point2d> imageStartCor;
        vector<Point2d> imgSize; 
        vector<Mat> faceImages;
        int imgNum;
        
        vector<vector<double>> trainingFeatures;
        vector<vector<double>> testingFeatures;
        
        Face();

        //get the meanShape
        void getMeanShape();
        
        //extract feature
        void extractFeature(matrix<double> iamge);
        
        //read from file
        void readDataFromFile();
        
        //read parameters from parameters.ini
        void readParameters();

        void faceDetector(); 
        
          
};



#endif


