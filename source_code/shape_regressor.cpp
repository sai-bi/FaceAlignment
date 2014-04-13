/**
 * @author Bi Sai 
 * @version 2014/03/17
 */

#include "face.h"

ShapeRegressor::ShapeRegressor(){
    // to be added
}
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
ShapeRegressor::ShapeRegressor(const Mat_<double>& mean_shape,
        const vector<Mat_<uchar> >& images,
        const vector<Mat_<double> >& target_shapes,
        vector<Mat_<double> >& current_shapes,
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
    cout<<"ShapeRegressor train..."<<endl;
    for(int i = 0;i < first_level_num_;i++){
        cout<<i<<" ";
        vector<Mat_<double> > normalize_matrix;
        // calculate normalized matrix
        calcuate_normalized_matrix(normalize_matrix);
        // normalize targets = (target - current) * normalize_matrix;
        vector<Mat_<double> > normalized_targets(training_num_);
        for(int j = 0;j < training_num_;j++){
            normalized_targets[j] = (target_shapes_[j] - current_shapes_[j]) * 
                normalize_matrix[j];
            // show_image(images_[j], current_shapes_[j] * normalize_matrix[j]);
        }
        fern_cascades_[i].train(images_,normalize_matrix,target_shapes_,mean_shape_,
                second_level_num_,current_shapes_,pixel_pair_num_,normalized_targets,
                pixel_pair_in_fern_); 
    }   
}

/**
 * Calculate a similarity matrix from each initial shape to mean shape.
 * @param normalize_matrix  result of similarity matrix
 */
void ShapeRegressor::calcuate_normalized_matrix(vector<Mat_<double> >& normalize_matrix){
    normalize_matrix.clear();
    for(int i = 0;i < training_num_;i++){
        Mat_<double> output_matrix = Mat_<double>::eye(2,2);
        // solve(current_shapes_[i],mean_shape_,output_matrix,DECOMP_SVD);
        // calcSimil() 
        normalize_matrix.push_back(output_matrix);
    } 
}

/**
 * Read training model from file
 * @param fin file operator 
 */
void ShapeRegressor::read(ifstream& fin){
    fin>>first_level_num_;
    fern_cascades_.resize(first_level_num_);
    for(int i = 0;i < first_level_num_;i++){
        fern_cascades_[i].read(fin);
    }  
}  

/**
 * Write training model to file
 * @param fout file operator
 */
void ShapeRegressor::write(ofstream& fout){
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
    read(fin);
    fin.close();
}

/**
 * Save the training model to file
 * @param file_name file to be written
 */
void ShapeRegressor::save(const char* file_name){
    ofstream fout;
    fout.open(file_name);
    write(fout);
    fout.close();
}


void ShapeRegressor::calcSimil(const Mat_<double> &src,const Mat_<double> &dst,
        double &a,double &b,double &tx,double &ty){
    Mat_<double> H = Mat_<double>::zeros(4,4),g = Mat_<double>::zeros(4,1),p(4,1);
    for(int i = 0; i < src.rows/2; i++){
        double x1 = src(2*i),y1 = src(2*i+1);
        double x2 = dst(2*i),y2 = dst(2*i+1);
        H(0,0) += x1*x1 + y1*y1; H(0,2) += x1; H(0,3) += y1;
        g(0,0) += x1*x2 + y1*y2; g(1,0) += x1*y2 - y1*x2;
        g(2,0) += x2; g(3,0) += y2;
    }
    H(1,1) = H(0,0); H(1,2) = H(2,1) = -1.0*(H(3,0) = H(0,3));
    H(1,3) = H(3,1) = H(2,0) = H(0,2); H(2,2) = H(3,3) = src.rows/2;
    solve(H,g,p,DECOMP_CHOLESKY);
    a = p(0,0); b = p(1,0); tx = p(2,0); ty = p(3,0); return;
}

void ShapeRegressor::invSimil(double a1,double b1,double tx1,double ty1,
        double& a2,double& b2,double& tx2,double& ty2){
    Mat_<double> M = (cv::Mat_<double>(2,2) << a1, -b1, b1, a1);
    Mat_<double> N = M.inv(cv::DECOMP_SVD); a2 = N(0,0); b2 = N(1,0);
    tx2 = -1.0*(N(0,0)*tx1 + N(0,1)*ty1);
    ty2 = -1.0*(N(1,0)*tx1 + N(1,1)*ty1); 
    return;
}



