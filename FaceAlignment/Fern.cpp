/**
 * @author 
 * @version 2014/06/18
 */

#include "FaceAlignment.h"

vector<Mat_<double> > Fern::Train(const Mat_<double>& candidate_pixel_intensity, 
                                  const Mat_<double>& covariance,
                                  const Mat_<double>& candidate_pixel_locations,
                                  const vector<Mat_<double> >& regression_targets,
                                  int fern_pixel_num){
    fern_pixel_num_ = fern_pixel_num;
    landmark_num_ = target_shapes[0].rows;
    int candidate_pixel_num = candidate_pixel_locations.rows;

    // select pixel pairs from candidate pixels 
    RNG random_generator(getTickCount());
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
                
        }
    } 
}



