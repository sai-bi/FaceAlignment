#ifndef FACE_H
#define FACE_H
const double PI =  3.141592653589793238463;
#include <iostream>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <limits>
#include <cstdlib>
#include "cv.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <boost/numeric/ublas/io.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
using namespace cv;
using namespace std;


#define MAXNUM numeric_limits<double>::max(); 
#define MINNUM numeric_limits<double>::min();

class SimilarTransform{
    public:
        double a;
        double b;
        SimilarTransform inverse();
        SimilarTransform();
};

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
        int debug;
        int mode;

        double pixel_range;


        vector<Point2d> meanShape;
        //face keypoints locations of each face 
        vector<vector<Point2d> > targetShape;
        //current shape 
        vector<vector<Point2d> > currentShape;
        //start coordinates of each image
        vector<Point2d> imageStartCor;
        //the size of each image
        vector<Point2d> imgSize; 
        //face images
        vector<Mat> faceImages;
        //average width and height of input images
        int averageHeight;
        int averageWidth;
        string currentFileName;

        //the coordinates of each feature pixel, relative to the nearest keypoint 
        vector<Point2d> featurePixelCoordinates;
        //index of nearest coordinates
        vector<int> nearestKeypointIndex;
    
        // normalize targets
        vector<vector<Point2d> > normalize_targets;
        // similar transform for normalize targets
        vector<SimilarTransform> targets_similar_transform;

        vector<SimilarTransform> current_shape_similar_transform;         

        //
        vector<Point2d> target_gravity_center; 
        vector<Point2d> current_shape_gravity_center; 
        
        vector<vector<Point2d> > augment_target_shapes;
        //  

        Face();

        void run();

        void preprocessing();

        void readData();
        
        void  getFeaturePixelLocation(); 
        void  extractFeature(const Mat& covariance,const vector<vector<double> >& pixelDensity,
                vector<Point2i>& selectedFeatureIndex,
                vector<double>& threhold);

        double  product(const vector<double>& v1, const vector<double>& v2);
        void  getDeltaShape(vector<vector<double> >& deltaShape);
        void  getRandomDirection(vector<double>& randomDirection);
        void  firstLevelRegression();
        void  secondLevelRegression(const Mat& covariance,const vector<vector<double> >& pixelDensity);
        void  constructFern(const vector<Point2i>& selectedFeatureIndex,
                const vector<vector<double> > & pixelDensity,
                const vector<double>& thresh);
        double  getCovariance(const vector<double>& v1, const vector<double>& v2);
        vector<Point2d>  vectorMinus(const vector<Point2d>& shape1, const vector<Point2d>& shape2);
        vector<Point2d>  vectorPlus(const vector<Point2d>& shape1, const vector<Point2d>& shape2);

        // test 
        void faceTest();
        void getRandomThrehold(vector<int>& threhold);

        // read parameters from parameters.ini
        void readParameters();
        void secondLevelTest(int currLevelNum, vector<Point2d>& testCurrentShape, 
                const vector<Point2d>& inputPixelCoordinates,const vector<int>& inputNearestIndex,
                const Mat& testImg, const Point2d& mean_point,SimilarTransform& transform);

        void apply_similar_transform(vector<Point2d>& src, const SimilarTransform& transform);
        void calculate_mean_shape();
        void scale_shape(vector<Point2d>& input_shape);
        double cal_vector_norm(const vector<Point2d>& input_vector);
        void align(const vector<Point2d>& src, const vector<Point2d>& dst, SimilarTransform& transform);
        Point2d get_mean(const vector<Point2d>& point);
        Point2d apply_similar_transform_point(const Point2d& point, const SimilarTransform& transform);
};



#endif


