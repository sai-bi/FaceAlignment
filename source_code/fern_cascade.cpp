/**
 * @author Bi Sai 
 * @version 2014/03/17
 */

#include "face.h"

FernCascade::FernCascade(){

}

/**
 * Train a fern cascade.
 * @param images training images in gray scale
 * @param normalize_matrix similarity matrix
 * @param target_shapes target shapes of each face image
 * @param mean_shape mean shape
 * @param second_level_num level number for second level regression 
 * @param current_shapes current shapes of training images
 * @param pixel_pair_num number of pair of pixels to be selected
 * @param normalized_targets (target - current) * normalize_matrix
 */
void FernCascade::train(const vector<Mat_<uchar> >& images,
        const vector<Mat_<double> >& normalize_matrix,
        const vector<Mat_<double> >& target_shapes,
        const Mat_<double>& mean_shape,
        int second_level_num,
        vector<Mat_<double> >& current_shapes,
        int pixel_pair_num,
        vector<Mat_<double> >& normalized_targets,
        int pixel_pair_in_fern){
    cout<<"FernCascade train..."<<endl;
    second_level_num_ = second_level_num;
	// coordinates of selected pixels
    Mat_<double> pixel_coordinates(pixel_pair_num,2);
    // the corresponding nearest keypoint index of each selected pixel
	Mat_<int> nearest_keypoint_index(pixel_pair_num,1);
    RNG random_generator(getTickCount());
    primary_fern_.resize(second_level_num);
    int landmark_num = mean_shape.rows;   
    int training_num = images.size();
    int image_width = images[0].cols;
    int image_height = images[0].rows;
    // generate local coordinates
    for(int i = 0;i < pixel_pair_num;i++){
        int x_coordinates = random_generator.uniform(-20,20);
        int y_coordinates = random_generator.uniform(-20,20);
        int index = random_generator.uniform(0,landmark_num);
        pixel_coordinates(i,0) = x_coordinates;
        pixel_coordinates(i,1) = y_coordinates; 
        nearest_keypoint_index(i) = index;
    }
	// calculate the inverse of normalize matrix
    vector<Mat_<double> > inverse_normalize_matrix;
    for(int i = 0;i < normalize_matrix.size();i++){
        Mat_<double> temp = Mat_<double>::eye(2,2);
        // invert(normalize_matrix[i],temp,DECOMP_SVD);
        inverse_normalize_matrix.push_back(temp);
    }

    vector<vector<double> > pixel_density;
    for(int i = 0;i < pixel_pair_num;i++){
        int index = nearest_keypoint_index(i);
        Mat_<double> landmark_coordinates(1,2);
        landmark_coordinates(0,0) = pixel_coordinates(i,0);
        landmark_coordinates(0,1) = pixel_coordinates(i,1);
        vector<double> curr_pair_pixel_density;
        for(int j = 0;j < training_num;j++){
			// project the coordinates back into original system
            Mat_<double> global_coordinates = landmark_coordinates * inverse_normalize_matrix[j];
            global_coordinates(0,0) += current_shapes[j](index,0);
            global_coordinates(0,1) += current_shapes[j](index,1);
            int temp_x = global_coordinates(0,0);
            int temp_y = global_coordinates(0,1);   
            if(temp_x < 0){
				temp_x = 0;
			}
			if(temp_y < 0){
				temp_y = 0;
			}
			if(temp_x >= image_width){
				temp_x = image_width - 1;
			}
			if(temp_y >= image_height){
				temp_y = image_height - 1;
			}
			curr_pair_pixel_density.push_back(int(images[j](temp_y,temp_x)));
        }
		pixel_density.push_back(curr_pair_pixel_density);
    }
	// calculate the correlation between pixels 
    Mat_<double> correlation(pixel_pair_num,pixel_pair_num);
    for(int i = 0;i < pixel_pair_num;i++){
        for(int j = i;j< pixel_pair_num;j++){
            double correlation_result = calculate_covariance(pixel_density[i],pixel_density[j]);
            correlation(i,j) = correlation_result;
            correlation(j,i) = correlation_result;
        }
    }
	// train ferns
    primary_fern_.resize(second_level_num);
    for(int i = 0;i < second_level_num;i++){
		cout<<"Training fern "<<i<<endl;
        primary_fern_[i].train(pixel_density,correlation,pixel_coordinates,nearest_keypoint_index, current_shapes,pixel_pair_in_fern, normalized_targets,inverse_normalize_matrix); 
    }
}


void FernCascade::write(ofstream& fout){
    fout<<second_level_num_<<endl;
    for(int i = 0;i < second_level_num_;i++){
        primary_fern_[i].write(fout);
    }
}
void FernCascade::read(ifstream& fin){
    fin>>second_level_num_;
	second_level_num_ = 500;
	primary_fern_.resize(second_level_num_);
    for(int i = 0;i < second_level_num_;i++){
        primary_fern_[i].read(fin);
    }

	return;
}
void FernCascade::predict(const Mat_<uchar>& image, Mat_<double>& shape,const Mat_<double>& mean_shape){
    Mat_<double> normalize_matrix = Mat_<double>::eye(2,2);
    Mat_<double> invert_normalized_matrix = Mat_<double>::eye(2,2);
    // solve(shape,mean_shape,normalize_matrix,DECOMP_SVD);
    // invert(normalize_matrix,invert_normalized_matrix,DECOMP_SVD);

    for(int i = 0;i < second_level_num_;i++){
        primary_fern_[i].predict(image,shape,invert_normalized_matrix);
    }
}

