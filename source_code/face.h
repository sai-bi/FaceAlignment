#ifndef FACE_H_
#define FACE_H_

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
using namespace std;
using namespace cv;


class Fern{
    private:
        int pixel_pair_num_in_fern_;
        int landmark_num_;
        Mat_<int> nearest_keypoint_index_;
        Mat_<double> threshold_;
        Mat_<double> selected_x_;
        Mat_<double> selected_y_;
        Mat_<double> pixel_coordinates_;
        vector<Mat_<double> > bin_output_;
        Mat_<int> pixel_pair_selected_index_;
    public:
        Fern();
        void train(const vector<vector<double> >& pixel_density,
                const Mat_<double>& covariance,
                const Mat_<double>& pixel_coordinates,
                const Mat_<int>& nearest_keypoint_index,
                vector<Mat_<double> >& current_shapes,
                int pixel_pair_num_in_fern,
                vector<Mat_<double> >& normalized_targets,
                const vector<Mat_<double> >& invert_normalized_matrix);
        void predict(const Mat_<uchar>& image, Mat_<double>& shape,
                const Mat_<double>& invert_normalized_matrix);
        void write(ofstream& fout);
        void read(ifstream& fin);

};
class FernCascade{
    private:
        vector<Fern> primary_fern_;
        int second_level_num_;
    public:
        FernCascade();
        void train(const vector<Mat_<uchar> >& images,
                const vector<Mat_<double> >& normalize_matrix,
                const vector<Mat_<double> >& target_shapes,
                const Mat_<double>& mean_shape,
                int second_level_num,
                vector<Mat_<double> >& current_shapes,
                int pixel_pair_num,
                vector<Mat_<double> >& normalized_targets,
                int pixel_pair_in_fern);
        void predict(const Mat_<uchar>& image, Mat_<double>& shape,
                const Mat_<double>& mean_shape);
        void write(ofstream& fout);
        void read(ifstream& fin);        
};
class ShapeRegressor{
    private:
        Mat_<double> mean_shape_;
        vector<Mat_<uchar> > images_;
        vector<Mat_<double> > current_shapes_;
        vector<Mat_<double> > target_shapes_;
        vector<FernCascade> fern_cascades_;
        int first_level_num_;
        int second_level_num_;
        int pixel_pair_num_;
        int training_num_; 
        int landmark_num_;
        int img_width_;
        int img_height_;
        int pixel_pair_in_fern_; 
        void read(ifstream& fin);
        void write(ofstream& fout);
        void calcuate_normalized_matrix(vector<Mat_<double> >&);
    public:
        ShapeRegressor();
        ShapeRegressor(const Mat_<double>& mean_shape,
                const vector<Mat_<uchar> >& images,
                const vector<Mat_<double> >& target_shapes,
                vector<Mat_<double> >& current_shapes,
                int first_level_num,
                int second_level_num,
                int pixel_pair_num,
                int pixel_pair_in_fern);
        void load(const char* file_name);
        void save(const char* file_name);
        void train();
        void predict(const Mat_<uchar>& image, Mat_<double>& shape,
                const Mat_<double>& mean_shape);
        void calcSimil(const Mat_<float> &src,const Mat_<float> &dst,
                float &a,float &b,float &tx,float &ty);
        void invSimil(float a1,float b1,float tx1,float ty1,
                float& a2,float& b2,float& tx2,float& ty2);
};





double calculate_covariance(const vector<double>& v_1, const
        vector<double>& v_2);

void train(const vector<Mat_<uchar> >& input_images,                  
        const vector<Mat_<double> >& target_shapes,
        const Mat_<double>& mean_shape,
        int initial_number,
        int pixel_pair_num,
        int pixel_pair_in_fern,
        int first_level_num,
        int second_level_num);

Mat_<double> test(ShapeRegressor& regressor,const Mat_<uchar>& image, const vector<Mat_<double> > target_shapes,
        const Mat_<double>& mean_shape,
        int initial_number);

void show_image(const Mat_<uchar>& input_image, const Mat_<double>&  points);


#endif
