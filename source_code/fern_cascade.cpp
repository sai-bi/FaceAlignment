/**
 * @author 
 * @version 2014/03/17
 */

#include "face.h"

FernCascade::FernCascade(){
     
}


void FernCascade::train(const vector<Mat_<uchar> >& images,
                        const vector<Mat_<double> >& normalize_matrix,
                        const vector<Mat_<double> >& target_shapes,
                        const vector<Mat_<double> >& mean_shape,
                        int second_level_num,
                        vector<Mat_<double> >& current_shapes,
                        int pixel_pair_num,
                        vector<Mat_<double> >& normalized_targets){
    Mat_<double> pixel_coordinates(pixel_pair_num,2);
    Mat_<int> nearest_keypoint_index(pixel_pair_num,1);
    RNG random_generator(getTickCount());
    primary_fern.resize(second_level_num);
    int landmark_num = mean_shape.rows;   
    int training_num = images.size();
    for(int i = 0;i < pixel_pair_num;i++){
        int x_coordinates = random_generator.uniform(-20,20);
        int y_coordinates = random_generator.uniform(-20,20);
        int index = random_generator.uniform(0,landmark_num);
        pixel_coordinates(i,0) = x;
        pixel_coordinates(i,1) = y; 
        nearest_keypoint_index(i) = index;
    }
    vector<Mat_<double> > inverse_normalize_matrix;
    for(int i = 0;i < normalize_matrix.size();i++){
        Mat_<double> temp;
        invert(normalize_matrix[i],temp,DECOMP_SVD);
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
            Mat_<double> global_coordinates = landmark_coordinates *
                inverse_normalize_matrix[j];
            global_coordinates(0,0) += current_shapes[j](index,0);
            global_coordinates(0,1) += current_shapes[j](index,1);
            int temp_x = global_coordinates(0,0);
            int temp_y = global_coordinates(0,1);   
            curr_pair_pixel_density.push_back(int(images[j](temp_y,temp_x)));
        }
    }
    Mat_<double> correlation(pixel_pair_num,pixel_pair_num);
    for(int i = 0;i < pixel_pair_num;i++){
        for(int j = i;j< pixel_pair_num;j++){
            double correlation_result = calculate_correlation(pixel_density[i],pixel_density[j]);
            correlation(i,j) = correlation_result;
            correlation(j,i) = correlation_result;
        }
    }
    primary_fern_.resize(second_level_num);
    for(int i = 0;i < second_level_num;i++){
        primary_fern_[i].train(pixel_density,correlation,pixel_coordinates,nearest_keypoint_index,
                               current_shapes,target_shapes); 
    }
}


double FernCascade::calculate_covariance(const vector<double>& v_1, const
        vector<double>& v_2){
    double sum_1 = 0;
    double sum_2 = 0;
    double exp_1 = 0;
    double exp_2 = 0;
    double exp_3 = 0;
    for(int i = 0;i < v_1.size();i++){
        sum_1 += v_1[i];
        sum_2 += v_2[i];
    }
    exp_1 = sum_1 / v_1.size();
    exp_2 = sum_2 / v_2.size();
    for(int i = 0;i < v_1.size();i++){
        exp_3 = exp_3 + (v_1[i] - exp_1) * (v_2[i] - exp_2);
    }
    return exp_3 / v_1.size();
}


void FernCascade::write(ofstream& fout){
    fout<<second_level_num_;
    for(int i = 0;i < second_level_num_;i++){
        primary_fern_[i].write(fout);
    }
}

void FernCascade::read(ifstream& fin){
    fin>>second_level_num_;
    for(int i = 0;i < second_level_num_;i++){
        primary_fern_[i].read(fin);
    }
}
