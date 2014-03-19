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
    RNG random_generator(getTickCount());
    vector<Mat_<double> > delta_shapes(current_shapes.size());
    for(int i = 0;i < current_shapes.size();i++){
        delta_shapes[i] = target_shapes[i] - current_shapes[i];
    }
    Mat_<int> pixel_pair_selected_index(pixel_pair_num_in_fern,2);
    for(int i = 0;i < pixel_pair_num_in_fern;i++){
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
    int bin_number = pow(2.0,pixel_pair_num_in_fern);
    vector<Mat_<double> > bin_output(bin_number);
    vector<vector<int> > bin_of_shape(bin_number);

    Mat_<double> density_difference_range(pixel_pair_num_in_fern,2);
    
    for(int i = 0;i < pixel_pair_num_in_fern;i++){
        int index1 = pixel_pair_selected_index(i,0);
        int index2 = pixel_pair_selected_index(i,1);
        double min_value = numeric_limits<double>::max();
        double max_value = numeric_limits<double>::min(); 
        for(int j = 0;j < current_shapes.size();j++){
            double temp = pixel_density[index1][j] - pixel_density[index2][j];
            if(temp > max_value){
                max_value = temp;  
            }   
            if(temp < min_value){
                min_value = temp;
            }
        }  
        density_difference_range(i,0) = min_value;
        density_difference_range(i,1) = max_value;
    }
    Mat_<double> threshold(pixel_pair_num_in_fern,1);
    for(int i = 0;i < pixel_pair_num_in_fern;i++){
        double lower_value = 0.7 * density_difference_range(i,0) + 0.3 * density_difference_range(i,1);
        double upper_value = 0.3 * density_difference_range(i,0) + 0.7 * density_difference_range(i,1); 
        threshold(i) = random_generator.uniform(lower_value,upper_value);  
    }
    for(int i = 0;i < current_shapes.size();i++){
        int bin_index = 0;
        for(int j = 0;j < pixel_pair_num_in_fern;j++){
            int index1 = pixel_pair_selected_index(j,0);
            int index2 = pixel_pair_selected_index(j,1);
            if(pixel_density[index1][i] - pixel_density[index2][i] >= threshold[j]){
                bin_index = bin_index + (int)(pow(2.0,j)); 
            } 
        }
        bin_of_shape[bin_index].push_back(i);
    }    
    for(int i = 0;i < bin_of_shape.size();i++){
        Mat_<double> temp = Mat::zeros(landmark_num,2);
        int bin_size = bin_of_shape[i].size();
        if(bin_size() == 0){
            bin_output[i] = temp;
            continue;
        }
        for(int j = 0;j < bin_size;j++){  
            temp = temp + delta_shapes[bin_of_shape[i][j]];
        }
        bin_output[i] = (1.0/((1+1000/bin_size) * bin_size)) * temp;
        for(int j = 0;j < bin_size;j++){
            current_shapes[bin_of_shape[i][j]] = current_shapes[bin_of_shape[i][j]]
                + bin_output[i];
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
