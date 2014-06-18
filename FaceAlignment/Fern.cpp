/**
 * @author 
 * @version 2014/06/18
 */

#include "FaceAlignment.h"

vector<Mat_<double> > Fern::Train(const Mat_<double>& candidate_pixel_intensity, 
                                  const Mat_<double>& covariance,
                                  const Mat_<double>& candidate_pixel_locations,
                                  const Mat_<int>& nearest_landmark_index,
                                  const vector<Mat_<double> >& regression_targets,
                                  int fern_pixel_num){
    fern_pixel_num_ = fern_pixel_num;
    landmark_num_ = target_shapes[0].rows;
    selected_pixel_index_.create(fern_pixel_num,2);
    selected_pixel_locations_.create(fern_pixel_num,4);
    selected_nearest_landmark_index_.create(fern_pixel_num,2);
    int candidate_pixel_num = candidate_pixel_locations.rows;

    // select pixel pairs from candidate pixels 
    RNG random_generator(getTickCount());
    threshold_.create(fern_pixel_num,1);
    for(int i = 0;i < fern_pixel_num;j++){
        // get a random direction
        Mat_<double> random_direction(landmark_num_ * 2,1);
        random_generator.fill(random_direction,RNG::UNIFORM,-1.1,1.1);
        normalize(random_direction,random_direction);
        Mat_<double> projection_result(regression_targets.size(),1);
        
        for(int j = 0;j < regression_targets.size();j++){
            projection_result(j) = random_direction.dot(regression_targets[j]); 
        } 
         
        Mat_<double> covariance_projection_density(candidate_pixel_num,1);
        for(int j = 0;j < candidate_pixel_num;j++){
            Mat_<double> temp = candidate_pixel_intensity(Range::all(),Range(j,j+1));
            covariance_projection_density(j) = calculate_covariance(projection_result,temp);
        }
        
        // find max correlation
        double max_correlation = -1;
        int max_pixel_index_1 = 0;
        int max_pixel_index_2 = 0;
        for(int j = 0;j < candidate_pixel_num;j++){
            for(int k = 0;k < candidate_pixel_num;k++){
                if(j == k){
                    continue;
                }  
                double temp1 = covariance(j,j) + covariance(k,k) - 2*covariance(j,k);
                if(temp1 < 1e-10){
                    continue;
                }

                double temp = (covariance_projection_density(j) - covariance_projection_density(k))
                    / temp1;
                if(abs(temp) > max_correlation){
                    max_correlation = temp;
                    max_pixel_index_1 = j;
                    max_pixel_index_2 = k; 
                } 
            }
        }
        selected_pixel_index_(i,0) = max_pixel_index_1;
        selected_pixel_index_(i,1) = max_pixel_index_2; 
        selected_pixel_locations_(i,0) = candidate_pixel_locations(max_pixel_index_1,0);
        selected_pixel_locations_(i,1) = candidate_pixel_locations(max_pixel_index_1,1);
        selected_pixel_locations_(i,2) = candidate_pixel_locations(max_pixel_index_2,0);
        selected_pixel_locations_(i,3) = candidate_pixel_locations(max_pixel_index_2,1);
        selected_nearest_landmark_index_(i,0) = nearest_landmark_index(max_pixel_index_1); 
        selected_nearest_landmark_index_(i,1) = nearest_landmark_index(max_pixel_index_2); 

        // get threshold for this pair
        Mat_<double> density_1 = candidate_pixel_intensity(Range::all(), Range(max_pixel_index_1,max_pixel_index_1+1));
        Mat_<double> density_2 = candidate_pixel_intensity(Range::all(), Range(max_pixel_index_2,max_pixel_index_2+1));
        density_1 = cv::abs(density_1 - density_2);
        double max_diff;
        cv::max(density_1,max_diff);
        threshold_(i) = random_generator.uniform(-0.2 * max_diff, 0.2 * max_diff);     
    }
    
    // determine the bins of each shape
    vector<vector<int> > shapes_in_bin;
    int bin_num = pow(2.0,fern_pixel_num)
    shapes_in_bin.resize(bin_num);
    for(int i = 0;i < regression_targets.size();i++){
        int index = 0;
        for(int j = 0;j < fern_pixel_num;j++){
            double density_1 = candidate_pixel_intensity(i,selected_pixel_index_(j,0));
            double density_2 = candidate_pixel_intensity(i,selected_pixel_index_(j,1));
            if(density_1 - density_2 >= threshold_(j)){
                index = index + pow(2.0,j);
            } 
        }
        shapes_in_bin[index].push_back(i);
    }
     
    // get bin output
    vector<Mat_<double> > prediction;
    prediction.resize(regression_targets.size());
    bin_output_.resize(bin_num);
    for(int i = 0;i < bin_num;i++){
        Mat_<double> temp(landmark_num_,2);
        int bin_size = shapes_in_bin[i].size();
        for(int j = 0;j < bin_size;j++){
            int index = shapes_in_bin[i][j];
            temp = temp + regression_targets[index]; 
        }
        if(bin_size == 0){
            bin_output_[i] = temp;
            continue; 
        }
        temp = (1.0/((1.0+1000.0/bin_size) * bin_size)) * temp;
        bin_output_[i] = temp;
        for(int j = 0;j < bin_size;j++){
            int index = shapes_in_bin[i][j];
            prediction[index] = temp;
        }
    }
    return prediction;
}


void Fern::Write(ofstream& fout){
    fout<<fern_pixel_num_<<endl;
    fout<<landmark_num_<<endl;
    for(int i = 0;i < fern_pixel_num_;i++){
        fout<<selected_nearest_landmark_index_(i,0)<<" "<<selected_nearest_landmark_index_(i,1)<<endl;
        fout<<selected_pixel_locations_(i,0)<<" "<<selected_pixel_locations_(i,1)<<" "
            <<selected_pixel_locations_(i,2)<<" "<<selected_pixel_locations_(i,3)<<" "<<endl;
        fout<<threshold_(i)<<endl;
    }        
    for(int i = 0;i < bin_output_.size();i++){
        for(int j = 0;j < bin_output_[i].rows;j++){
            fout<<bin_output_[i](j,0)<<" "<<bin_output_[i](j,1)<<" ";
        }
        fout<<endl;
    } 
}

void Fern::Read(ifstream& fin){
    fin>>fern_pixel_num_>>endl;
    fin>>landmark_num_>>endl;
    selected_nearest_landmark_index_.create(fern_pixel_num_,2);
    selected_pixel_locations_.create(fern_pixel_num_,4);
    threshold_.create(fern_pixel_num,1);
    for(int i = 0;i < fern_pixel_num_;i++){
        fout>>selected_nearest_landmark_index_(i,0)>>" ">>selected_nearest_landmark_index_(i,1)>>endl;
        fout>>selected_pixel_locations_(i,0)>>" ">>selected_pixel_locations_(i,1)>>" "
            >>selected_pixel_locations_(i,2)>>" ">>selected_pixel_locations_(i,3)>>" ">>endl;
        fout>>threshold_(i)>>endl;
    }       
     
    int bin_num = pow(2.0,fern_pixel_num_);
    for(int i = 0;i < bin_num;i++){
        Mat_<double> temp(landmark_num_,2);
        for(int j = 0;j < landmark_num_;j++){
            fin>>temp(j,0)>>temp(j,1);
        }
        bin_output_.push_back(temp);
    }
}

Mat_<double> Fern::Predict(const Mat_<uchar>& image,
                     const Mat_<double>& shape,
                     const Mat_<double>& rotation,
                     const BoundingBox& bounding_box,
                     int double scale){
    int index = 0;
    for(int i = 0;i < fern_pixel_num_;i++){
        int nearest_landmark_index_1 = selected_nearest_landmark_index_(i,0);
        int nearest_landmark_index_2 = selected_nearest_landmark_index_(i,1);
        double x = selected_pixel_locations_(i,0);
        double y = selected_pixel_locations_(i,1);
        x = scale * (rotation(0,0)*x + rotation(0,1)*y) * bounding_box.width + shape(nearest_landmark_index_1,0);
        y = scale * (rotation(1,0)*x + rotation(1,1)*y) * bounding_box.height + shape(nearest_landmark_index_1,1);
        double intensity_1 = (int)(image((int)y,(int)x));

        double x = selected_pixel_locations_(i,2);
        double y = selected_pixel_locations_(i,3);
        x = scale * (rotation(0,0)*x + rotation(0,1)*y) * bounding_box.width + shape(nearest_landmark_index_2,0);
        y = scale * (rotation(1,0)*x + rotation(1,1)*y) * bounding_box.height + shape(nearest_landmark_index_2,1);
        double intensity_2 = (int)(image((int)y,(int)x));

        if(intensity_1 - intensity_2 >= threshold_(i)){
            index = index + (int)(pow(2,i));
        } 
    }

    return bin_output_[index];
}

