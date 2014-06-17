/**
 * @author 
 * @version 2014/06/17
 */
#include "FaceAlignment.h"

ShapeRegressor::ShapeRegressor(){
    first_level_num_ = 0;
    second_level_num_ = 0;
    candidate_pixel_num_ = 0;
    fern_pixel_num_ = 0;
    initial_num_ = 0;
}


void ShapeRegressor::Train(const vector<Mat_<uchar> >& images, 
                   const vector<Mat_<double> >& ground_truth_shapes,
                   const vector<BoundingBox>& bounding_box,
                   int first_level_num, int second_level_num,
                   int candidate_pixel_num, int fern_pixel_num,
                   int initial_num){
    // data augmentation and multiple initialization 
    vector<Mat_<uchar> > augmented_images;
    vector<Mat_<BoundingBox> > augmented_bounding_box;
    vector<Mat_<double> > augmented_ground_truth_shapes;
    vecot<Mat_<double> > current_shapes;
     
    RNG random_generator(getTickCount());
    for(int i = 0;i < ground_truth_images.size();i++){
        for(int j = 0;j < initial_num;j++){
            int index = 0;
            do{
                index = random_generator.uniform(0,ground_truth_images.size());
            }while(index == i);
            augmented_images.push_back(images[index]);
            augmented_ground_truth_shapes.push_back(ground_truth_shapes[i]);
            augmented_bounding_box.push_back(bounding_box[i]); 
            // 1. Select ground truth shapes of other images as initial shapes
            // 2. Project current shape to bounding box of ground truth shapes 
            Mat_<double> temp = ground_truth_shape[index];
            temp = ProjectShape(temp, bounding_box[index]);
            temp = ReProjectShape(temp, bounding_box[i]);
            current_shapes.push_back(temp); 
        } 
    }
    
    // get mean shape
    mean_shape_ = GetMeanShape(ground_truth_shape,bounding_box); 
    
    // get normalized targets
    vector<Mat_<double> > normalized_targets;
    normalized_targets.resize(current_shapes.size()); 
    for(int i = 0;i < current_shapes.size();i++){
        normalized_targets[i] = ProjectShape(current_shapes[i],augmented_bounding_box[i]);
        normalized_targets[i] = ProjectShape(augmented_ground_truth_shapes[i],augmented_bounding_box[i]) - normalized_targets[i];
    } 
    
    // train fern cascades
    fern_cascades_.resize(first_level_num);
    vector<Mat_<double> > prediction;
    for(int i = 0;i < first_level_num;i++){
        prediction = fern_cascades_[i].train(augmented_images,current_shapes,
                augmented_ground_truth_shapes,augmented_bounding_box,mean_shape_,second_level_num,candidate_pixel_num,fern_pixel_num);
        
        // update current shape 
        for(int j = 0;j < prediction.size();j++){
            current_shapes[j] = prediction[j] + ProjectShape(current_shapes[j], augmented_bounding_box[j]);
            current_shapes[j] = ReProjectShape(current_shapes[j],augmented_bounding_box[j]);
        }
    } 
    
}


Mat_<double> ShapeRegressor::GetMeanShape(const vector<Mat_<double> >& shapes,
                          const vector<BoundingBox>& bounding_box){
    vector<Mat_<double> > temp;
    Mat_<double> result(shapes[0].rows,2,CV_64FC1);

    temp = ProjectShape(shapes,bounding_box);
    result = std::accumulate(temp.begin(),temp.end(),result);    
    
    return (1.0 / shapes.size() * result); 
}

Mat_<double> ShapeRegressor::ProjectShape(const Mat_<double>& shape, const BoundingBox& bounding_box){
    Mat_<double> temp(shape.rows,2);
    for(int j = 0;j < shape.rows;j++){
        temp(j,0) = (shape(j,0)-bounding_box.centroid_x) / (bounding_box.width / 2.0);
        temp(j,1) = (shape(j,1)-bounding_box.centroid_y) / (bounding_box.height / 2.0);  
    } 
    return temp;  
}

Mat_<double> ShapeRegressor::ReProjectShape(const Mat_<double>& shape, const BoundingBox& bounding_box){
    Mat_<double> temp(shape.rows,2);
    for(int j = 0;j < shape.rows;j++){
        temp(j,0) = (shape(j,0) * bounding_box.width / 2.0 + bounding_box.centroid_x);
        temp(j,1) = (shape(j,1) * bounding_box.height / 2.0 + bounding_box.centroid_y);
    } 
    return temp; 
}


