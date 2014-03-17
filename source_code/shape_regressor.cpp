/**
 * @author 
 * @version 2014/03/17
 */

#include "face.h"

void ShapeRegressor::ShapeRegressor(const Mat_<double>& mean_shape,
                       const vector<Mat_<vec3b> >& images,
                       const vector<Mat_<double> >& target_shapes,
                       vector<Mat_<double> > current_shapes,
                       int first_level_num,
                       int second_level_num,
                       int img_height,
                       int img_width,
                       int pixel_pair_num){

    mean_shape_ = mean_shape;
    images_ = images;
    target_shapes_ = target_shapes;
    current_shapes_ = current_shapes;
    first_level_num_ = first_level_num;
    second_level_num_ = second_level_num;
    img_width_ = img_width;
    img_height_ = img_heigh;
    pixel_pair_num_ = pixel_pair_num; 
    training_num_ = images.size();
    landmark_num_ = target_shapes_[0].rows;
    fern_cascades_.resize(first_level_num_);
}

void ShapeRegressor::train(){
    for(int i = 0;i < first_level_num;i++){
        Mat_<double> pixel_coordinates(2*pixel_pair_num_,1);
        Mat_<int> nearest_keypoint_index(pixel_pair_num,1);
        vector<Mat_<double> > = calcuate_normalized_matrix(); 
        vector<Mat_<double> > normalized_targets(training_num_);
        for(int j = 0;j < training_num_;j++){
            normalized_targets[j] = (target_shapes_[j] - current_shapes_[j]) * 
                normalize_matrix[j];
        }
        fern_cascades_[i].train(images_,normalize_matrix,target_shapes_,mean_shape,
                second_level_num_,current_shapes_,pixel_pair_num_); 
    }   
}


void ShapeRegressor::calcuate_normalized_matrix(vector<Mat_<double> >& normalize_matrix){
    normalize_matrix.clear();
    for(int i = 0;i < training_num_;i++){
        Mat_<double> output_matrix;
        solve(current_shapes_[i],mean_shape_,output_matrix,DECOMP_SVD);
        normalize_matrix.push_back(output_matrix);
    } 
}

