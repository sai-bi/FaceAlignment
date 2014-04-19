/**
 * @author Bi Sai 
 * @version 2014/03/17
 */

#include "face.h"

Fern::Fern(){

}
/**
 * Train a fern.
 * @param pixel_density Each vector stores a vector of pixel density in
 * corresponding position of each image.
 * @param covariance covariance between pixels
 * @param nearest_keypoint_index for each pixel, the index of its nearest
 * keypoint
 * @param pixel_pair_in_fern number of pixel pairs in a fern
 * @param normalized_targets (target - current) * normalize_matrix
 * @param invert_normalized_matrix inverse of normalize_matrix
 */
void Fern::train(const vector<vector<double> >& pixel_density,
        const Mat_<double>& covariance,
        const Mat_<double>& pixel_coordinates,
        const Mat_<int>& nearest_keypoint_index,
        vector<Mat_<double> >& current_shapes,
        int pixel_pair_num_in_fern,
        vector<Mat_<double> >& normalized_targets,
        const vector<Mat_<double> >& invert_normalized_matrix){
    pixel_pair_num_in_fern_ = pixel_pair_num_in_fern;
    nearest_keypoint_index_ = nearest_keypoint_index.clone();
    pixel_coordinates_ = pixel_coordinates.clone();
    int pixel_pair_num = pixel_density.size();
    int landmark_num = current_shapes[0].rows;
    landmark_num_ = landmark_num;
    RNG random_generator(getTickCount());
    pixel_pair_selected_index_.create(pixel_pair_num_in_fern,2);
    for(int i = 0;i < pixel_pair_num_in_fern_;i++){
		// get a random direction
        Mat_<double> random_direction(landmark_num * 2,1);
        random_generator.fill(random_direction,RNG::UNIFORM,-1.1,1.1);
        normalize(random_direction,random_direction);
        // random_generator.fill(random_direction,RNG::,-1,1);
        // normalize(random_direction,random_direction);
        vector<double> project_result;
        
		// project the normalize targets to random direction
        for(int j = 0;j < normalized_targets.size();j++){
            double temp = 0;
            for(int k = 0;k < landmark_num_;k++){
                temp += random_direction(2*k) * normalized_targets[j](k,0);
                temp += random_direction(2*k+1) * normalized_targets[j](k,1);
            } 
            project_result.push_back(temp); 
        }

        Mat_<double> covariance_pixel_shape(pixel_pair_num,1);
        for(int j = 0;j < pixel_pair_num;j++){
            covariance_pixel_shape(j) = calculate_covariance(project_result,pixel_density[j]); 
        }
        double max_correlation = -1;
        double max_pixel_pair_index_1 = 0;
        double max_pixel_pair_index_2 = 0;
		// find max correlation
        for(int j = 0;j < pixel_pair_num;j++){
            for(int k = 0;k < pixel_pair_num;k++){
                bool flag = false;
                if(j == k){
                    continue;
                }
                for(int p = 0;p < i;p++){
                    if(j == pixel_pair_selected_index_(p,0) && k == pixel_pair_selected_index_(p,1)){
                        flag = true;
                        break; 
                    }else if(j == pixel_pair_selected_index_(p,1) && k == pixel_pair_selected_index_(p,0)){
                        flag = true;
                        break;
                    }
                }

                if(flag){
                    continue;
                }

                double temp1 = covariance(j,j) + covariance(k,k) - 2 * covariance(j,k);
                if(temp1 == 0){
                    // cout<<"covariance is 0"<<endl;
                    continue;
                }


                double temp = (covariance_pixel_shape(j) - covariance_pixel_shape(k))
                    / sqrt((covariance(j,j) + covariance(k,k) - 2*covariance(j,k)));
                if(abs(temp) > max_correlation){
                    max_correlation = temp;
                    max_pixel_pair_index_1 = j;
                    max_pixel_pair_index_2 = k; 
                }

            }
        } 
        // assert(max_pixel_pair_index_1 != max_pixel_pair_index_2);
        if(max_pixel_pair_index_1 == max_pixel_pair_index_2){
            cout<<max_pixel_pair_index_1 << "   "<<max_pixel_pair_index_2<<endl;
        }

        pixel_pair_selected_index_(i,0) = max_pixel_pair_index_1;
        pixel_pair_selected_index_(i,1) = max_pixel_pair_index_2; 
    }  

    int bin_number = pow(2.0,pixel_pair_num_in_fern);
    bin_output_.resize(bin_number);
	// each vector stores the index of shapes that belong to that bin
	vector<vector<int> > bin_of_shape(bin_number);
	// get the range of pixel difference in each bin
    Mat_<double> density_difference_range(pixel_pair_num_in_fern,2);
    for(int i = 0;i < pixel_pair_num_in_fern;i++){
        int index1 = pixel_pair_selected_index_(i,0);
        int index2 = pixel_pair_selected_index_(i,1);
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
	// get threshold
    //Mat_<double> threshold(pixel_pair_num_in_fern,1);
	threshold_.create(pixel_pair_num_in_fern_,1);
	for(int i = 0;i < pixel_pair_num_in_fern;i++){
        // double lower_value = 0.7 * density_difference_range(i,0) + 0.3 * density_difference_range(i,1);
        // double upper_value = 0.3 * density_difference_range(i,0) + 0.7 * density_difference_range(i,1); 
        double temp1 = abs(density_difference_range(i,0));
        double temp2 = abs(density_difference_range(i,1));
        double temp3 = temp1 > temp2 ? temp1 : temp2; 
        threshold_(i) = random_generator.uniform(-0.2 * temp3, 0.2 * temp3);  
    }
	// determine the bin for each shape
    for(int i = 0;i < current_shapes.size();i++){
        int bin_index = 0;
        for(int j = 0;j < pixel_pair_num_in_fern;j++){
            int index1 = pixel_pair_selected_index_(j,0);
            int index2 = pixel_pair_selected_index_(j,1);
            if(pixel_density[index1][i] - pixel_density[index2][i] >= threshold_(j)){
                bin_index = bin_index + (int)(pow(2.0,j)); 
            } 
        }
        bin_of_shape[bin_index].push_back(i);
    }    
	// get bin output
    for(int i = 0;i < bin_of_shape.size();i++){
        Mat_<double> temp = Mat::zeros(landmark_num,2,CV_64F);
        int bin_size = bin_of_shape[i].size();
        if(bin_size == 0){
            bin_output_[i] = temp;
            continue;
        }
        for(int j = 0;j < bin_size;j++){  
            temp = temp + normalized_targets[bin_of_shape[i][j]];
        }
        bin_output_[i] = (1.0/((1.0+1000.0/bin_size) * bin_size)) * temp;
        for(int j = 0;j < bin_size;j++){
            int index = bin_of_shape[i][j];
            current_shapes[index] = current_shapes[index] + bin_output_[i]
                * invert_normalized_matrix[index];
            normalized_targets[index] = normalized_targets[index] - bin_output_[i];
        } 
    }
}



void Fern::read(ifstream& fin){
    fin>>pixel_pair_num_in_fern_;
    fin>>landmark_num_;
	selected_x_.create(pixel_pair_num_in_fern_,2);
    selected_y_.create(pixel_pair_num_in_fern_,2);
    nearest_keypoint_index_.create(pixel_pair_num_in_fern_,2);
    threshold_.create(pixel_pair_num_in_fern_,1);
    for(int i = 0;i < pixel_pair_num_in_fern_;i++){
        fin>>selected_x_(i,0)>>selected_y_(i,0)
            >>selected_x_(i,1)>>selected_y_(i,1);
        fin>>nearest_keypoint_index_(i,0);
        fin>>nearest_keypoint_index_(i,1);
        fin>>threshold_(i);
    }
    int bin_num = pow(2.0,pixel_pair_num_in_fern_);
    for(int i = 0;i < bin_num;i++){
        double x = 0;
        double y = 0;
        Mat_<double> temp(landmark_num_,2);
        for(int j = 0;j < landmark_num_;j++){
            fin>>x>>y;
            temp(j,0) = x;
            temp(j,1) = y;    
        } 
        bin_output_.push_back(temp); 
    }  
}


void Fern::write(ofstream& fout){
    fout<<pixel_pair_num_in_fern_<<endl; 
    fout<<landmark_num_<<endl;
    for(int i = 0;i < pixel_pair_num_in_fern_;i++){
        int index1 = pixel_pair_selected_index_(i,0);
        int index2 = pixel_pair_selected_index_(i,1);
        fout<<pixel_coordinates_(index1,0)<<" "<<pixel_coordinates_(index1,1)<<" "
            <<pixel_coordinates_(index2,0)<<" "<<pixel_coordinates_(index2,1)
            <<endl;
        fout<<(nearest_keypoint_index_(index1))<<endl;
        fout<<(nearest_keypoint_index_(index2))<<endl;
        fout<<(threshold_(i))<<endl;
    }
    for(int i = 0;i < bin_output_.size();i++){
        for(int j = 0;j < bin_output_[i].rows;j++){
            double temp = bin_output_[i](j,0);
            fout<<temp<<" ";
            // fout<<(bin_output_[i](j,0))<<" "<<(bin_output_[i](j,1))<<endl; 
            temp = bin_output_[i](j,1);
            fout<<temp<<endl;
        }
    }
}


void Fern::predict(const Mat_<uchar>& image, Mat_<double>& shape,
        const Mat_<double>& invert_normalized_matrix){  
    int bin_index = 0;
    for(int i = 0;i < pixel_pair_num_in_fern_;i++){
        int keypoint_index1 = nearest_keypoint_index_(i,0);
        int keypoint_index2 = nearest_keypoint_index_(i,1);
        Mat_<double> coordinates(1,2);
        Mat_<double> pixel_1;
        Mat_<double> pixel_2;
        coordinates(0,0) = selected_x_(i,0);
        coordinates(0,1) = selected_y_(i,0);
        pixel_1 = coordinates * invert_normalized_matrix;
        pixel_1(0,0) += shape(keypoint_index1,0);
        pixel_1(0,1) += shape(keypoint_index1,1);

        coordinates(0,0) = selected_x_(i,1);
        coordinates(0,1) = selected_y_(i,1); 
        pixel_2 = coordinates * invert_normalized_matrix;
        pixel_2(0,0) += shape(keypoint_index1,0);
        pixel_2(0,1) += shape(keypoint_index2,1);

		int x = pixel_1(0,0);
		int y = pixel_1(0,1);
		if(x < 0){
			x = 0;
		}
		if(y < 0){
			y = 0;
		}
		if(x >= image.cols){
			x = image.cols-1;
		}
		if(y >= image.rows){
			y = image.rows-1;
		}
        double intensity_1 = image(y,x);
		x = pixel_2(0,0);
		y = pixel_2(0,1);
		if(x < 0){
			x = 0;
		}
		if(y < 0){
			y = 0;
		}
		if(x >= image.cols){
			x = image.cols-1;
		}
		if(y >= image.rows){
			y = image.rows-1;
		}
		
		double intensity_2 = image(y,x); 
        if(intensity_1 - intensity_2 >= threshold_(i)){
            bin_index = bin_index + (int)(pow(2.0,i));
        }
    } 
    shape = shape + bin_output_[bin_index] * invert_normalized_matrix;
    // show_image(image,shape);
}




