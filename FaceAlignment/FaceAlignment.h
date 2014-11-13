/*
Author: Bi Sai 
Date: 2014/06/18
This program is a reimplementation of algorithms in "Face Alignment by Explicit 
Shape Regression" by Cao et al.
If you find any bugs, please email me: soundsilencebisai-at-gmail-dot-com

Copyright (c) 2014 Bi Sai 
The MIT License (MIT)
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/


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
#include <utility> 

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
        cv::Mat_<int> selected_nearest_landmark_index_;
        cv::Mat_<double> threshold_;
        cv::Mat_<int> selected_pixel_index_;
        cv::Mat_<double> selected_pixel_locations_;
        std::vector<cv::Mat_<double> > bin_output_;
    public:
        std::vector<cv::Mat_<double> > Train(const std::vector<std::vector<double> >& candidate_pixel_intensity, 
                                             const cv::Mat_<double>& covariance,
                                             const cv::Mat_<double>& candidate_pixel_locations,
                                             const cv::Mat_<int>& nearest_landmark_index,
                                             const std::vector<cv::Mat_<double> >& regression_targets,
                                             int fern_pixel_num);
        cv::Mat_<double> Predict(const cv::Mat_<uchar>& image,
                                 const cv::Mat_<double>& shape,
                                 const cv::Mat_<double>& rotation,
                                 const BoundingBox& bounding_box,
                                 double scale);
        void Read(std::ifstream& fin);
        void Write(std::ofstream& fout);
};

class FernCascade{
    public:
        std::vector<cv::Mat_<double> > Train(const std::vector<cv::Mat_<uchar> >& images,
                                             const std::vector<cv::Mat_<double> >& current_shapes,
                                             const std::vector<cv::Mat_<double> >& ground_truth_shapes,
                                             const std::vector<BoundingBox> & bounding_box,
                                             const cv::Mat_<double>& mean_shape,
                                             int second_level_num,
                                             int candidate_pixel_num,
                                             int fern_pixel_num,
                                             int curr_level_num,
                                             int first_level_num);  
        cv::Mat_<double> Predict(const cv::Mat_<uchar>& image, 
                                 const BoundingBox& bounding_box, 
                                 const cv::Mat_<double>& mean_shape,
                                 const cv::Mat_<double>& shape);
        void Read(std::ifstream& fin);
        void Write(std::ofstream& fout);
    private:
        std::vector<Fern> ferns_;
        int second_level_num_;
};

class ShapeRegressor{
    public:
        ShapeRegressor(); 
        void Train(const std::vector<cv::Mat_<uchar> >& images, 
                   const std::vector<cv::Mat_<double> >& ground_truth_shapes,
                   const std::vector<BoundingBox>& bounding_box,
                   int first_level_num, int second_level_num,
                   int candidate_pixel_num, int fern_pixel_num,
                   int initial_num);
        cv::Mat_<double> Predict(const cv::Mat_<uchar>& image, const BoundingBox& bounding_box, int initial_num);
        void Read(std::ifstream& fin);
        void Write(std::ofstream& fout);
        void Load(std::string path);
        void Save(std::string path);
    private:
        int first_level_num_;
        int landmark_num_;
        std::vector<FernCascade> fern_cascades_;
        cv::Mat_<double> mean_shape_;
        std::vector<cv::Mat_<double> > training_shapes_;
        std::vector<BoundingBox> bounding_box_;
};

cv::Mat_<double> GetMeanShape(const std::vector<cv::Mat_<double> >& shapes,
                              const std::vector<BoundingBox>& bounding_box);
cv::Mat_<double> ProjectShape(const cv::Mat_<double>& shape, const BoundingBox& bounding_box);
cv::Mat_<double> ReProjectShape(const cv::Mat_<double>& shape, const BoundingBox& bounding_box);
void SimilarityTransform(const cv::Mat_<double>& shape1, const cv::Mat_<double>& shape2, 
                         cv::Mat_<double>& rotation,double& scale);
double calculate_covariance(const std::vector<double>& v_1, 
                            const std::vector<double>& v_2);
#endif
