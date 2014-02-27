#ifndef FACE_H
#define FACE_H
#include <iostream>
#include <cmath>
#include <fstream>
#include <algorithm>
// #include <vector>
#include <ctime>
#include <limits>
#include <cstdlib>

#include <boost/numeric/ublas/vector.hpp>
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
        int keypointNum;
        int firstLevelNum;
        int secondLevelNum;
        int featureNumInFern;
        double shrinkage;

        //grayscale training images
        vector<matrix<double>> trainingImages;
        //grayscale testing images 
        vector<matrix<double>> testingImages;
        //mean shape of training images
        vector<Point2d> meanShape;
        //face keypoints locations of each face 
        vector<vector<Point2d>> targetShape;
        //current shape 
        vector<vector<Point2d>> currentShape;
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

        void  getFeaturePixelLocation(); 
        void  extractFeature(const matrix<double>& covariance,const vector<vector<double>>& pixelDensity,
                const vector<Point2i> selectedFeatureIndex);

        double  product(const vector<double>& v1, const vector<double>& v2);
        void  getDeltaShape(vector<vector<double>& deltaShape);
        void  getRandomDirection(vector<double>& randomDirection);
        void  firstLevelRegression();
        void  secondLevelRegression(const matrix<double>& covariance,const vector<vector<double>>& pixelDensity);
        void  constructFern(const vector<Point2i>& selectedFeatureIndex,
                const vector<vector<double>>& pixelDensity);
        double  getCovariance(const vector<double>& v1, const vector<double>& v2);
        vector<Point2d>  vectorMinus(const vector<Point2d>& shape1, const vector<Point2d>& shape2);
        vector<Point2d>  vectorPlus(const vector<Point2d>& shape1, const vector<Point2d>& shape2);



        //read parameters from parameters.ini
        // void readParameters();

        // void faceDetector(); 


};
/* 
class Fern{
    public:
        // the index of feature pixel pairs 
        vector<Point2i> featurePixelIndex;

        // output of delta shape
        vector<Point2d> deltaShape;

        vector<Point2d> threholds; 
};
 */


#endif


