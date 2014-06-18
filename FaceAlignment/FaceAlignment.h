#ifndef FACE_ALIGNMENT_H
#define FACE_ALIGNMENT_H

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "cv.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <ctime>
#include <string>
#include <limits>
#include <algorithm>
#include <cmath>
#include <vector>
#include <fstream>
#include <numeric>   
using namespace std;
using namespace cv;

class BoundingBox{
    public:
        double start_x;
        double start_y;
        double width;
        double height;
        double centroid_x;
        double centroid_y;
        BoundingBox(){
            start_x = 0;
            start_y = 0;
            width = 0;
            height = 0;
            centroid_x = 0;
            centroid_y = 0;
        }; 
};


class Fern{
    private:
        int fern_pixel_num_;
        int landmark_num_;
        Mat_<int> selected_nearest_landmark_index_;
        Mat_<double> threshold_;
        Mat_<double> selected_pixel_index_;
        Mat_<double> selected_pixel_locations_;
        vector<Mat_<double> > bin_output_;
    public:
        vector<Mat_<double> > Train(const Mat_<double>& candidate_pixel_intensity, 
                                    const Mat_<double>& covariance,
                                    const Mat_<double>& candidate_pixel_locations,
                                    const Mat_<int>& nearest_landmark_index,
                                    const vector<Mat_<double> >& regression_targets,
                                    int fern_pixel_num);
        Mat_<double> Predict(const Mat_<uchar>& image,
                             const Mat_<double>& shape,
                             const Mat_<double>& rotation,
                             const BoundingBox& bounding_box,
                             double scale);
        void Read(ifstream& fin);
        void Write(ofstream& fout);
};

class FernCascade{
    public:
        vector<Mat_<double> > Train(const vector<Mat_<uchar> >& images,
                                    const vector<Mat_<double> >& current_shapes,
                                    const vector<Mat_<double> >& ground_truth_shapes,
                                    const vector<BoundingBox> & bounding_box,
                                    const Mat_<double>& mean_shape,
                                    int second_level_num,
                                    int candidate_pixel_num,
                                    int fern_pixel_num);  
        Mat_<double> Predict(const Mat_<uchar>& image, 
                          const BoundingBox& bounding_box, 
                          const Mat_<double>& mean_shape,
                          const Mat_<double>& shape);
        void Read(ifstream& fin);
        void Write(ofstream& fout);
    private:
        vector<Fern> ferns_;
        int second_level_num_;
};

class ShapeRegressor{
    public:
        ShapeRegressor(); 
        void Train(const vector<Mat_<uchar> >& images, 
                   const vector<Mat_<double> >& ground_truth_shapes,
                   const vector<BoundingBox>& bounding_box,
                   int first_level_num, int second_level_num,
                   int candidate_pixel_num, int fern_pixel_num,
                   int initial_num);
        Mat_<double> Predict(const Mat_<uchar>& image, const BoundingBox& bounding_box, int initial_num);
        void Read(ifstream& fin);
        void Write(ofstream& fout);
        void Load(string path);
        void Save(string path);
    private:
        int first_level_num_;
        int landmark_num_;
        vector<FernCascade> fern_cascades_;
        Mat_<double> mean_shape_;
        vector<Mat_<double> > training_shapes_;
        vector<BoundingBox> bounding_box_;
};

Mat_<double> GetMeanShape(const vector<Mat_<double> >& shapes,
                          const vector<BoundingBox>& bounding_box);
Mat_<double> ProjectShape(const Mat_<double>& shape, const BoundingBox& bounding_box);
Mat_<double> ReProjectShape(const Mat_<double>& shape, const BoundingBox& bounding_box);
void SimilarityTransform(const Mat_<double>& shape1, const Mat_<double>& shape2, 
                         Mat_<double>& rotation,double scale);
double calculate_covariance(const Mat_<double>& v_1, const Mat_<double>& v_2);
#endif
