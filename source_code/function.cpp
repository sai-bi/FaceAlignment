/** 
 * @author 
 * @version 2014/03/29
 */

#include "face.h"

void train(const vector<Mat_<uchar> >& input_images,                  
                const vector<Mat_<double> >& target_shapes,
                 const Mat_<double>& mean_shape,
                 int initial_number,
                 int pixel_pair_num,
                 int pixel_pair_in_fern,
                 int first_level_num,
                 int second_level_num){
    vector<Mat_<double> > augment_target_shapes;
    vector<Mat_<uchar> > images; 
    vector<Mat_<double> > augment_current_shapes; 
    RNG random_generator(getTickCount());
    for(int i = 0;i < input_images.size();i++){
        Mat_<uchar> temp = input_images[i].clone(); 
        for(int j = 0;j < initial_number;j++){
            images.push_back(temp);
            augment_target_shapes.push_back(target_shapes[i]);
            int index = 0;
            do{
                index = random_generator.uniform(0,image_path.size()); 
            }while(index == i);
            augment_current_shapes.push_back(target_shapes[index]);
        }
    }
    
    ShapeRegressor regressor(mean_shape,images,augment_target_shapes,
                augment_current_shapes,first_level_num,
                second_level_num, pixel_pair_num,
                pixel_pair_in_fern);
    regressor.train();
    regressor.save("model.data");
}

Mat_<double> test(string image_path, const vector<Mat_<double> > target_shapes,
        const Mat_<double>& mean_shape,
        int initial_number){
    ShapeRegressor regressor;
    regressor.load("model.data");
    RNG random_generator(getTickCount()); 
    Mat_<uchar> image = imread(image_path,0);
    Mat_<double> combine_shape;
    for(int i = 0;i < initial_number;i++){
        int index = 0;
        do{
            index = random_generator.uniform(0,target_shapes.size());  
        }while(index == i);
        Mat_<double> shape = target_shapes[index].clone();
        regressor.predict(image,shape,mean_shape);
        if(i == 0){
            combine_shape = shape.clone();
        }else{
            combine_shape = combine_shape + shape;
        }
    }
    return combine_shape; 
}


double calculate_covariance(const vector<double>& v_1, const
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
};



