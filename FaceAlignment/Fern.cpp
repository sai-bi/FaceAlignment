/*
Author: Bi Sai 
Date: 2014/06/18
This program is a reimplementation of algorithms in "Face Alignment by Explicit 
Shape Regression" by Cao et al.
If you find any bugs, please email me: soundsilencebisai-at-gmail-dot-com

Copyright (c) 2014 Bi Sai 
The MIT License (MIT)
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "FaceAlignment.h"

vector<Mat_<double> > Fern::Train(const vector<vector<double> >& candidate_pixel_intensity, 
                                  const Mat_<double>& covariance,
                                  const Mat_<double>& candidate_pixel_locations,
                                  const Mat_<int>& nearest_landmark_index,
                                  const vector<Mat_<double> >& regression_targets,
                                  int fern_pixel_num,
                                  bool model_compress_flag,
                                  const Mat_<double>& sparse_basis){
    // selected_pixel_index_: fern_pixel_num*2 matrix, the index of selected pixels pairs in fern
    // selected_pixel_locations_: fern_pixel_num*4 matrix, the locations of selected pixel pairs
    //                            stored in the format (x_1,y_1,x_2,y_2) for each row 
    model_compress_flag_ = model_compress_flag;
    fern_pixel_num_ = fern_pixel_num;
    landmark_num_ = regression_targets[0].rows;
    selected_pixel_index_.create(fern_pixel_num,2);
    selected_pixel_locations_.create(fern_pixel_num,4);
    selected_nearest_landmark_index_.create(fern_pixel_num,2);
    int candidate_pixel_num = candidate_pixel_locations.rows;

    // select pixel pairs from candidate pixels, this selection is based on the correlation between pixel 
    // densities and regression targets
    // for details, please refer to "Face Alignment by Explicit Shape Regression" 
    // threshold_: thresholds for each pair of pixels in fern 
    
    threshold_.create(fern_pixel_num,1);
    // get a random direction
    RNG random_generator(getTickCount());
    for(int i = 0;i < fern_pixel_num;i++){
        // RNG random_generator(i);
        Mat_<double> random_direction(landmark_num_ * 2,1);
        random_generator.fill(random_direction,RNG::UNIFORM,-1.1,1.1);

        normalize(random_direction,random_direction);
        // Mat_<double> projection_result(regression_targets.size(),1);
        vector<double> projection_result; 
        // project regression targets along the random direction 
        for(int j = 0;j < regression_targets.size();j++){
            double temp = 0;
            for(int k = 0;k < regression_targets[j].rows;k++){
                temp = temp + regression_targets[j](k,0) * random_direction(2*k) 
                    + regression_targets[j](k,1) * random_direction(2*k+1); 
            }
            projection_result.push_back(temp);
        } 

        Mat_<double> covariance_projection_density(candidate_pixel_num,1);
        for(int j = 0;j < candidate_pixel_num;j++){
            covariance_projection_density(j) = calculate_covariance(projection_result,candidate_pixel_intensity[j]);
        }

        // find max correlation
        double max_correlation = -1;
        int max_pixel_index_1 = 0;
        int max_pixel_index_2 = 0;
        for(int j = 0;j < candidate_pixel_num;j++){
            for(int k = 0;k < candidate_pixel_num;k++){
                double temp1 = covariance(j,j) + covariance(k,k) - 2*covariance(j,k);
                if(abs(temp1) < 1e-10){
                    continue;
                }
                bool flag = false;
                for(int p = 0;p < i;p++){
                    if(j == selected_pixel_index_(p,0) && k == selected_pixel_index_(p,1)){
                        flag = true;
                        break; 
                    }else if(j == selected_pixel_index_(p,1) && k == selected_pixel_index_(p,0)){
                        flag = true;
                        break;
                    } 
                }
                if(flag){
                    continue;
                } 
                double temp = (covariance_projection_density(j) - covariance_projection_density(k))
                    / sqrt(temp1);
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
        double max_diff = -1;
        for(int j = 0;j < candidate_pixel_intensity[max_pixel_index_1].size();j++){
            double temp = candidate_pixel_intensity[max_pixel_index_1][j] - candidate_pixel_intensity[max_pixel_index_2][j];
            if(abs(temp) > max_diff){
                max_diff = abs(temp);
            }
        }

        threshold_(i) = random_generator.uniform(-0.2*max_diff,0.2*max_diff); 
    } 


    // determine the bins of each shape
    vector<vector<int> > shapes_in_bin;
    int bin_num = pow(2.0,fern_pixel_num);
    shapes_in_bin.resize(bin_num);
    for(int i = 0;i < regression_targets.size();i++){
        int index = 0;
        for(int j = 0;j < fern_pixel_num;j++){
            double density_1 = candidate_pixel_intensity[selected_pixel_index_(j,0)][i];
            double density_2 = candidate_pixel_intensity[selected_pixel_index_(j,1)][i];
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
    sparse_output_.resize(bin_num);
    for(int i = 0;i < bin_num;i++){
        Mat_<double> temp = Mat::zeros(landmark_num_,2, CV_64FC1);
        int bin_size = shapes_in_bin[i].size();
        for(int j = 0;j < bin_size;j++){
            int index = shapes_in_bin[i][j];
            temp = temp + regression_targets[index]; 
        }
        if(bin_size == 0){
            bin_output_[i] = temp; 
            // sparse_output_[i] = Mat::zeros(1,2,CV_64FC1);
            continue; 
        }
         
        temp = (1.0/((1.0+1000.0/bin_size) * bin_size)) * temp;
        bin_output_[i] = temp;
        
        if(model_compress_flag){
            OthogonalMatchingPursuit(sparse_basis,temp,5,sparse_output_[i]);
            // non_zero_index: stores the index and value of non-zero elements in sparse output.
            vector<double> non_zero_index;
            for(int j = 0;j < sparse_output_[i].rows;j++){
                if(abs(sparse_output_[i](j)) > 1e-12){
                    non_zero_index.push_back(j);
                    non_zero_index.push_back(sparse_output_[i](j)); 
                }  
            }
            // real output 
            temp = sparse_basis * sparse_output_[i];
        }
        
        for(int j = 0;j < bin_size;j++){
            int index = shapes_in_bin[i][j];
            prediction[index] = temp;
        }
    }

    return prediction;
}


void Fern::Write(ofstream& fout){
    fout<<model_compress_flag_<<endl;
    fout<<fern_pixel_num_<<endl;
    fout<<landmark_num_<<endl;
    for(int i = 0;i < fern_pixel_num_;i++){
        fout<<selected_pixel_locations_(i,0)<<" "<<selected_pixel_locations_(i,1)<<" "
            <<selected_pixel_locations_(i,2)<<" "<<selected_pixel_locations_(i,3)<<" "<<endl;
        fout<<selected_nearest_landmark_index_(i,0)<<endl;
        fout<<selected_nearest_landmark_index_(i,1)<<endl;
        fout<<threshold_(i)<<endl;
    }
    if(model_compress_flag_ == false){
        for(int i = 0;i < bin_output_.size();i++){
            for(int j = 0;j < bin_output_[i].rows;j++){
                fout<<bin_output_[i](j,0)<<" "<<bin_output_[i](j,1)<<" ";
            }
            fout<<endl;
        }
    }
    else{
        // if compressed, just output the index and value of non-zero elements
        for(int i = 0;i < sparse_output_.size();i++){
            // non-zero elements size
            fout<<sparse_output_[i].size()<<endl;
            for(int j = 0;j < sparse_output_[i].size();j=j+2){
                // output: index, value
                fout<<sparse_output_[i][j]<<" "<<sparse_output_[i][j+1]<<endl;
            }        
        }
    } 
}

void Fern::Read(ifstream& fin){
    fin>>model_compress_flag_;
    fin>>fern_pixel_num_;
    fin>>landmark_num_;
    selected_nearest_landmark_index_.create(fern_pixel_num_,2);
    selected_pixel_locations_.create(fern_pixel_num_,4);
    threshold_.create(fern_pixel_num_,1);
    for(int i = 0;i < fern_pixel_num_;i++){
        fin>>selected_pixel_locations_(i,0)>>selected_pixel_locations_(i,1)
            >>selected_pixel_locations_(i,2)>>selected_pixel_locations_(i,3);
        fin>>selected_nearest_landmark_index_(i,0)>>selected_nearest_landmark_index_(i,1);
        fin>>threshold_(i);
    }       
    int bin_num = pow(2.0,fern_pixel_num_);
    if(model_compress_flag_ == 0){ 
        for(int i = 0;i < bin_num;i++){
            Mat_<double> temp(landmark_num_,2);
            for(int j = 0;j < landmark_num_;j++){
                fin>>temp(j,0)>>temp(j,1);
            }
            bin_output_.push_back(temp);
        }
    }
    else{
        sparse_output_.resize(bin_num);
        for(int i = 0;i < bin_num;i++){
            int non_zero_size;
            fin>>non_zero_size;
            for(int j = 0;j < non_zero_size;j++){
                int index;
                int value;
                fin>>index>>value;
                sparse_output_[i].push_back(index);
                sparse_output_[i].push_back(value); 
            }        
        }        
    }
}

Mat_<double> Fern::Predict(const Mat_<uchar>& image,
                     const Mat_<double>& shape,
                     const Mat_<double>& rotation,
                     const BoundingBox& bounding_box,
                     double scale,
                     const Mat_<double>& sparse_basis){
    int index = 0;
    for(int i = 0;i < fern_pixel_num_;i++){
        int nearest_landmark_index_1 = selected_nearest_landmark_index_(i,0);
        int nearest_landmark_index_2 = selected_nearest_landmark_index_(i,1);
        double x = selected_pixel_locations_(i,0);
        double y = selected_pixel_locations_(i,1);
        double project_x = scale * (rotation(0,0)*x + rotation(0,1)*y) * bounding_box.width/2.0 + shape(nearest_landmark_index_1,0);
        double project_y = scale * (rotation(1,0)*x + rotation(1,1)*y) * bounding_box.height/2.0 + shape(nearest_landmark_index_1,1);

        project_x = std::max(0.0,std::min((double)project_x,image.cols-1.0));
        project_y = std::max(0.0,std::min((double)project_y,image.rows-1.0)); 
        double intensity_1 = (int)(image((int)project_y,(int)project_x));

        x = selected_pixel_locations_(i,2);
        y = selected_pixel_locations_(i,3);
        project_x = scale * (rotation(0,0)*x + rotation(0,1)*y) * bounding_box.width/2.0 + shape(nearest_landmark_index_2,0);
        project_y = scale * (rotation(1,0)*x + rotation(1,1)*y) * bounding_box.height/2.0 + shape(nearest_landmark_index_2,1);
        project_x = std::max(0.0,std::min((double)project_x,image.cols-1.0));
        project_y = std::max(0.0,std::min((double)project_y,image.rows-1.0));
        double intensity_2 = (int)(image((int)project_y,(int)project_x));

        if(intensity_1 - intensity_2 >= threshold_(i)){
            index = index + (int)(pow(2,i));
        }
    }

    if(model_compress_flag_ == false){
        return bin_output_[index];
    }

    Mat_<double> sparse_result = Mat::zeros(sparse_basis.cols,1,CV_64FC1);
    for(int i = 0;i < sparse_output_[index].size();i=i+2){
        sparse_result((int)sparse_output_[index][i]) = sparse_output_[index][i+1];
    }
    return sparse_basis * sparse_result; 
}

Mat_<double> Fern::GetFernOutput(int index){
    return bin_output_[index];
}

