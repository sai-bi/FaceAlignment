/** 
 * @author Bi Sai 
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
    cout<<"Start training..."<<endl;
    vector<Mat_<double> > augment_target_shapes;
    vector<Mat_<uchar> > images; 
    vector<Mat_<double> > augment_current_shapes; 
    RNG random_generator(getTickCount());
    for(int i = 0;i < input_images.size();i++){
        Mat_<uchar> temp = input_images[i].clone(); 
        for(int j = 0;j < initial_number;j++){
            images.push_back(temp);
            // augment_target_shapes.push_back(target_shapes[i]);
            int index = 0;
            do{
                index = random_generator.uniform(0,input_images.size()); 
            }while(index == i);
            // augment_current_shapes.push_back(mean_shape);
            augment_current_shapes.push_back(target_shapes[index]);
            augment_target_shapes.push_back(target_shapes[i]);
        }
    }

    ShapeRegressor regressor(mean_shape,images,augment_target_shapes,
            augment_current_shapes,first_level_num,
            second_level_num, pixel_pair_num,
            pixel_pair_in_fern);
    regressor.train();
    regressor.save("./data/model.txt");
}

Mat_<double> test(ShapeRegressor& regressor, const Mat_<uchar>& image, const vector<Mat_<double> > target_shapes,
        Bbox& bounding_box,
        int initial_number){
    RNG random_generator(getTickCount()); 
    Mat_<double> combine_shape;
    for(int i = 0;i < initial_number;i++){
        int index = 0;
        do{
            index = random_generator.uniform(0,target_shapes.size());  
        }while(index == i);
        Mat_<double> shape = target_shapes[index].clone();
        // Mat_<double> shape = mean_shape.clone();
        Bbox bounding_box_1 = get_bounding_box(shape);
        shape = shape_normalize(shape,bounding_box_1);
        shape = reproject_shape_single(shape,bounding_box); 

        regressor.predict(image,shape,bounding_box);
        if(i == 0){
            combine_shape = shape.clone();
        }else{
            combine_shape = combine_shape + shape;
        }
    }
    return (1.0/initial_number * combine_shape); 
}

// calculate the covariance of two vectors
// cov(x,y) = E((x - E(x)*(y-E(y))
double calculate_covariance(const vector<double>& v_1, const
        vector<double>& v_2){
    assert(v_1.size() == v_2.size());
    assert(v_1.size() != 0);
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


void show_image(const Mat_<uchar>& input_image, const Mat_<double>&  points){
    Mat_<uchar> image = input_image.clone();
    for(int i = 0;i < points.rows;i++){
        circle(image,Point2d(points(i,0),points(i,1)),3,Scalar(255,0,0),-1,8,0); 
    }
    imshow("image",image);
    // waitKey(2);
}


vector<Mat_<double> > inverse_shape(const vector<Mat_<double> >& shapes, const vector<Bbox>& bounding_box){
    vector<Mat_<double> > result;   

    result = project_shape(shapes, bounding_box); 
    
    for(int i = 0;i < result.size();i++){
        result[i] = -1 * result[i];
    }

    return result;
}

vector<Mat_<double> > compose_shape(const vector<Mat_<double> >& shape1, const vector<Mat_<double> >& shape2, 
        const vector<Bbox>& bounding_box){
   
    assert(shape1.size() == shape2.size());
    vector<Mat_<double> > result;
    result = project_shape(shape2,bounding_box);
    
    for(int i = 0;i < shape1.size();i++){
        result[i] = result[i] + shape1[i]; 
    } 
    return result;
}


vector<Mat_<double> > project_shape(const vector<Mat_<double> >& shapes, const vector<Bbox>& bounding_box){
    vector<Mat_<double> > result;   

    for(int i = 0;i < shapes.size();i++){
        Mat_<double> temp(shapes[i].rows,2);
        for(int j = 0;j < shapes[i].rows;j++){
            temp(j,0) = (shapes[i](j,0)-bounding_box[i].centroid_x) / (bounding_box[i].width / 2.0);
            temp(j,1) = (shapes[i](j,1)-bounding_box[i].centroid_y) / (bounding_box[i].height / 2.0);  
        } 
        result.push_back(temp);
    }
    return result; 
}

vector<Mat_<double> > reproject_shape(const vector<Mat_<double> >& shapes, const vector<Bbox>& bounding_box){
    vector<Mat_<double> > result;   

    for(int i = 0;i < shapes.size();i++){
        Mat_<double> temp(shapes[i].rows,2);
        for(int j = 0;j < shapes[i].rows;j++){
            temp(j,0) = (shapes[i](j,0) * bounding_box[i].width / 2.0 + bounding_box[i].centroid_x);
            temp(j,1) = (shapes[i](j,1) * bounding_box[i].height / 2.0 + bounding_box[i].centroid_y);
        } 
        result.push_back(temp);
    }
    return result; 
}

Mat_<double> reproject_shape_single(const Mat_<double>& shape, const Bbox& bounding_box){
    Mat_<double> result(shape.rows,2);   
    for(int j = 0;j < shapes.rows;j++){
        result(j,0) = (shape(j,0) * bounding_box.width / 2.0 + bounding_box.centroid_x);
        result(j,1) = (shape(j,1) * bounding_box.height / 2.0 + bounding_box.centroid_y);
    } 
    return result;  
}

Mat_<double> shape_normalize(const Mat_<double>& shape, const Bbox& bounding_box){    
    // Bbox bounding_box = get_bounding_box(shape);
    
    Mat_<double> result(shape.rows,2);
    for(int i = 0;i < shape.rows;i++){
        result(i,0) = (shape(i,0) - bounding_box.centroid_x) / (bounding_box.width / 2.0);
        result(i,1) = (shape(i,1) - bounding_box.centroid_y) / (bounding_box.height / 2.0);  
    }
    return result; 
} 

Bbox get_bounding_box(const Mat_<double>& shape){
    double min_x = shape(i,0);
    double min_y = shape(i,1);
    double max_x = shape(i,0);
    double max_y = shape(i,1);
    for(int i = 0;i < shape.rows;i++){
        if(shape(i,0) < min_x){
            min_x = shape(i,0);
        } 
        if(shape(i,0) > max_x){
            max_x = shape(i,0);
        }
        if(shape(i,1) < min_y){
            min_y = shape(i,1);
        }
        if(shape(i,1) > max_y){
            max_y = shape(i,1);
        }
    }

    Bbox result;
    result.start_x = max(0,min_x-10);
    result.start_y = max(0,min_y-10);
    double end_x = max_x;
    double end_y = max_y;
    result.width = end_x - start_x;
    result.height = end_y - start_y;
    result.centroid_x = result.start_x + result.width / 2.0; 
    result.centroid_y = result.start_y + result.height / 2.0;
    return result;
}




void translate_scale_rotate(const Mat_<double>& shape1, const Mat_<double>& shape2, 
        Mat_<double>& translation, double &scale, Mat_<double>& rotation){
    translation = Mat::zeros(shape1.rows,2,CV_64FC1);
    rotation = Mat::zeros(2,2,CV_64FC1);
    scale = 0;
    
    // center the data
    double center_x_1 = 0;
    double center_y_1 = 0;
    double center_x_2 = 0;
    double center_y_2 = 0;
    for(int i = 0;i < shape1.rows;i++){
        center_x_1 += shape1(i,0);
        center_y_1 += shape1(i,1);
        center_x_2 += shape2(i,0);
        center_y_2 += shape2(i,1); 
    }
    center_x_1 /= shape1.rows;
    center_y_1 /= shape1.rows;
    center_x_2 /= shape2.rows;
    center_y_2 /= shape2.rows;
    
    Mat_<double> temp1 = shape1.clone();
    Mat_<double> temp2 = shape2.clone();
    
    for(int i = 0;i < shape1.rows;i++){
        temp1(i,0) -= center_x_1;
        temp1(i,1) -= center_y_1;
        temp2(i,0) -= center_x_2;
        temp2(i,1) -= center_y_2;
    } 
    Mat_<double> covariance1, covariance2;
    Mat_<double> mean1,mean2;

    calcCovarMatrix(temp1,covariance1,mean1,CV_COVAR_COLS);
    calcCovarMatrix(temp2,covariance2,mean2,CV_COVAR_COLS);

    double s1 = sqrt(norm(covariance1));
    double s2 = sqrt(norm(covariance2));
    
    scale = s1 / s2; 

    temp1 = 1.0 / s1 * temp1;
    temp2 = 1.0 / s2 * temp2;

    double num = 0;
    double den = 0;
    
    for(int i = 0;i < shape1.rows;i++){
        num = num + temp1(i,1) * temp2(i,0) - temp1(i,0) * temp2(i,1);
        den = den + temp1(i,0) * temp2(i,0) + temp1(i,1) * temp2(i,1);      
    }
    
    double norm = sqrt(num*num + den*den);    
    
    double sin_theta = num / norm;
    double cos_theta = den / norm;

    rotation(0,0) = cos_theta;
    rotation(0,1) = -sin_theta;
    rotation(1,0) = sin_theta;
    rotation(1,1) = cos_theta;
}


