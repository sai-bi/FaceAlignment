/**
 * @author 
 * @version 2014/03/17
 */

#include "face.h"

Fern::Fern(){

}

void Fern::train(const vector<vector<double> >& pixel_density,
                 const Mat_<double>& covariance,
                 const Mat_<double>& pixel_coordinates,
                 const Mat_<int>& nearest_keypoint_index,
                 vector<Mat_<double> >& current_shapes,
                 const vector<Mat_<double> >& target_shapes){
    int pixel_pair_num_in_fern = 5;
    int pixel_pair_num = pixel_density.size();
    int landmark_num = current_shapes[0].rows;
    vector<Mat_<double> > delta_shapes(current_shapes.size());
    for(int i = 0;i < current_shapes.size();i++){
        delta_shapes[i] = target_shapes[i] - current_shapes[i];
    }
    Mat_<int> pixel_pair_selected_index(pixel_pair_num_in_fern,2);
    for(int i = 0;i < pixel_pair_num_in_fern;i++){
        RNG random_generator;
        Mat_<double> random_direction(landmark_num,2);
        random_generator.fill(random_direction,RNG::UNIFORM,-1,1);
        normalize(random_direction,random_direction);
        vector<double> project_result;
        for(int j = 0;j < delta_shapes.size();j++){
            Mat temp = random_direction.mul(delta_shapes[j]);
            project_result.push_back(sum(temp));
        }
        Mat_<double> covariance_pixel_shape(pixel_pair_num,1);
        for(int j = 0;j < pixel_pair_num;j++){
            covariance_pixel_shape(j) = calculate_covariance(project_result,pixel_density[i]); 
        }
        double max_correlation = -1;
        double max_pixel_pair_index_1 = 0;
        double max_pixel_pair_index_2 = 0;
        for(int j = 0;j < pixel_pair_num;j++){
            for(int k = j+1;k < pixel_pair_num;k++){
                double temp = (covariance_pixel_shape(j) - covariance_pixel_shape(k))
                              / (covariance(j,j) + covariance(k,k) - 2*covariance(j,k));
                if(abs(temp) > max_correlation){
                    max_correlation = abs(temp);
                    max_pixel_pair_index_1 = j;
                    max_pixel_pair_index_2 = k; 
                } 
            }
        } 
        pixel_pair_selected_index(i,0) = max_pixel_pair_index_1;
        pixel_pair_selected_index(i,1) = max_pixel_pair_index_2; 
    }     
    vector<Mat_<double> > bin_output;
    vector<int> bin_of_shape;
    for(int i = 0;i < current_shapes.size();i++){
        for(int j = 0;j < pixel_pair_num_in_fern;j++){
            int index1 = pixel_pair_selected_index(j,0);
            int index2 = pixel_pair_selected_index(j,1);
             
        } 
    }
}

double Fern::calculate_covariance(const vector<double>& v_1, const
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
