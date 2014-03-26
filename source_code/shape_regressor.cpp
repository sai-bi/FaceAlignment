/**
 * @author Bi Sai 
 * @version 2014/03/17
 */

#include "face.h"

/**
 * Constructors
 * @param mean_shape mean shapes 
 * @param images input training images
 * @param target_shapes target shapes of each face image
 * @param current_shapes shapes of each face
 * @param first_level_num number of levels for first level regression
 * @param second_level_num number of level for second level regression
 * @param pixel_pair_num pixel pair number to be selected in first level 
 * @param pixel_pair_in_fern pixel pair number in each primary fern regressor
 */
void ShapeRegressor::ShapeRegressor(const Mat_<double>& mean_shape,
                       const vector<Mat_<uchar> >& images,
                       const vector<Mat_<double> >& target_shapes,
                       vector<Mat_<double> > current_shapes,
                       int first_level_num,
                       int second_level_num,
                       int pixel_pair_num,
                       int pixel_pair_in_fern){

    mean_shape_ = mean_shape;
    images_ = images;
    target_shapes_ = target_shapes;
    current_shapes_ = current_shapes;
    first_level_num_ = first_level_num;
    second_level_num_ = second_level_num;
    pixel_pair_num_ = pixel_pair_num; 
    training_num_ = images.size();
    landmark_num_ = target_shapes_[0].rows;
    fern_cascades_.resize(first_level_num_);
    img_height_ = images_[0].cols;
    img_width_ = images_[0].rows;
    pixel_pair_in_fern_ = pixel_pair_in_fern;
}

/**
 * Training function
 */
void ShapeRegressor::train(){
    for(int i = 0;i < first_level_num;i++){
        vector<Mat_<double> > normalize_matrix;
        calcuate_normalized_matrix(normalize_matrix);
        vector<Mat_<double> > normalized_targets(training_num_);
        for(int j = 0;j < training_num_;j++){
            normalized_targets[j] = (target_shapes_[j] - current_shapes_[j]) * 
                normalize_matrix[j];
        }
        fern_cascades_[i].train(images_,normalize_matrix,target_shapes_,mean_shape,
                second_level_num_,current_shapes_,pixel_pair_num_,normalized_targets); 
    }   
}

/**
 * Calculate a similarity matrix from each initial shape to mean shape.
 * @param normalize_matrix  result of similarity matrix
 */
void ShapeRegressor::calcuate_normalized_matrix(vector<Mat_<double> >& normalize_matrix){
    normalize_matrix.clear();
    for(int i = 0;i < training_num_;i++){
        Mat_<double> output_matrix;
        solve(current_shapes_[i],mean_shape_,output_matrix,DECOMP_SVD);
        normalize_matrix.push_back(output_matrix);
    } 
}

/**
 * Read training model from file
 * @param fin file operator 
 */
void ShapeRegressor::read(ifstream& fin){
    fin>>first_level_num_;
    for(int i = 0;i < first_level_num_;i++){
        fern_cascades_[i].read(fin);
    }  
}  

/**
 * Write training model to file
 * @param fout file operator
 */
void ShapeRegressor::write(ofstream& fout)
    fout<<first_level_num_<<endl;
    for(int i = 0;i < first_level_num_;i++){
        fern_cascades_[i].write(fout);  
    } 
}

/**
 * Predict the shape of a given face image.
 * @param image input face image in grayscale
 * @param shape initial shape
 */
void ShapeRegressor::predict(const Mat_<uchar>& image, Mat_<double>& shape, const Mat_<double>& mean_shape){
    for(int i = 0;i < first_level_num_;i++){
        fern_cascades_[i].predict(image,shape,mean_shape);
    }
}

/**
 * Load training model from file
 * @param file_name file to be read
 */
void ShapeRegressor::load(const char* file_name){
    ifstream fin;
    fin.open(file_name);
    this.read(fin);
    fin.close();
}

/**
 * Save the training model to file
 * @param file_name file to be written
 */
void ShapeRegressor::save(const char* file_name){
    ofstream fout;
    fout.open(file_name);
    this.write(fout);
    fout.close();
}

