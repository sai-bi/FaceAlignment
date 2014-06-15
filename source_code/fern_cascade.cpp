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
        const vector<Mat_<double> >& target_shapes,
        int second_level_num,
        vector<Mat_<double> >& current_shapes,
        int pixel_pair_num,
        vector<Mat_<double> >& normalized_targets,
        int pixel_pair_in_fern){
    cout<<"FernCascade train..."<<endl;

    second_level_num_ = second_level_num;
	// coordinates of selected pixels
    Mat_<double> pixel_coordinates(pixel_pair_num,2);
	Mat_<int> nearest_keypoint_index(pixel_pair_num,1);
    RNG random_generator(getTickCount());
    int landmark_num = current_shapes[0].rows;
    int training_num = images.size();
    int image_width = images[0].cols;
    int image_height = images[0].rows;


    vector<Bbox> bounding_box;

    for(int i = 0;i < current_shapes.size();i++){
        Bbox temp;
        temp = get_bounding_box(current_shapes[i]);
        bounding_box.push_back(temp);
    }

    // calculate normalized targets
    normalized_targets = inverse_shape(current_shapes,bounding_box);
    normalized_targets = compose_shape(normalized_targets,target_shapes,bounding_box); 
    


    // calculate current mean shape 
    vector<Mat_<double> > normalized_shapes;
    // mean_shape_ = Mat::zeros(landmark_num,2,CV_64FC1);
    mean_shape_.create(landmark_num,2);
    for(int i = 0;i < current_shapes.size();i++){
        Mat_<double> temp_shape(landmark_num,2);
        for(int j = 0;j < landmark_num;j++){
            double temp1 = (current_shapes[i](j,0) - bounding_box[i].centroid_x) / (bounding_box[i].width / 2.0);
            double temp2 = (current_shapes[i](j,1) - bounding_box[i].centroid_y) / (bounding_box[i].height / 2.0);
            temp_shape(j,0) = temp1;
            temp_shape(j,1) = temp2;
            mean_shape_(j,0) += temp1;
            mean_shape_(j,1) += temp2;
        }
        normalized_shapes.push_back(temp_shape);
    }
    mean_shape_ = 1.0 / current_shapes.size() * mean_shape_; 

    // generate feature pixel location 
    for(int i = 0;i < pixel_pair_num;i++){
        double x = random_generator.uniform(-1.0,1.0);
        double y = random_generator.uniform(-1.0,1.0);
        
        // get its nearest landmark
        double min_dist = 1e10;
        int min_index = 0;
        for(int j = 0;j < landmark_num;j++){
            double temp = pow(mean_shape_(j,0) - x,2.0) + pow(mean_shape_(j,1) - y,2.0);
            if(temp < min_dist){
                min_dist = temp;
                min_index = j;
            } 
        } 
        nearest_keypoint_index(i) = min_index;
        pixel_coordinates(i,0) = x;
        pixel_coordinates(i,1) = y;
    }

    // get feature pixel location for each image
    // for pixel_density, each vector in it stores the pixel value for each image on the corresponding pixel locations
    vector<vector<double> > pixel_density;
    pixel_density.resize(pixel_pair_num);
    for(int i = 0;i < normalized_shapes.size();i++){
        // similarity transform from normalized_shapes to mean shape     
        Mat_<double> rotation(2,2);
        Mat_<double> translation(landmark_num,2);
        double scale = 0;

        translate_scale_rotate(normalized_shapes[i],mean_shape_,translation,scale,rotation); 
       
        for(int j = 0;j < pixel_pair_num;j++){
            double x = pixel_coordinates(j,0);
            double y = pixel_coordinates(j,1);
            double project_x = rotation(0,0) * x + rotation(0,1) * y;
            double project_y = rotation(1,0) * x + rotation(1,1) * y;
            project_x = project_x * scale;
            project_y = project_y * scale;
            
            // resize according to bounding_box
            project_x = project_x * bounding_box[i].width/2.0;
            project_y = project_y * bounding_box[i].height/2.0; 
            
            int index = nearest_keypoint_index(j); 
            int real_x = project_x + current_shapes[i](index,0);
            int real_y = project_y + current_shapes[i](index,1);
            
             
            if(real_x < 0){
                real_x = 0;
            } 
            if(real_y < 0){
                real_y = 0;
            }
            if(real_x > images[i].cols-1){
                real_x = images[i].cols-1;
            }
            if(real_y > images[i].rows - 1){
                real_y = images[i].rows - 1;
            }
            pixel_density[j].push_back(int(images[i](real_y,real_x)));    
        }
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

    // predications for each shape 
    vector<Mat_<double> > prediction;
    prediction.resize(current_shapes.size());
    for(int i = 0;i < current_shapes.size();i++){
        prediction[i] = Mat::zeros(landmark_num,2,CV_64FC1);
    }
    
    for(int i = 0;i < second_level_num;i++){
		cout<<"Training fern "<<i<<endl;
        primary_fern_[i].train(pixel_density,correlation,pixel_coordinates,nearest_keypoint_index, current_shapes,pixel_pair_in_fern,normalized_targets,
                prediction); 
    }
    
    current_shapes = compose_shape(prediction, current_shapes, bounding_box); 
    current_shapes = reproject_shape(current_shapes,bounding_box); 
}


void FernCascade::write(ofstream& fout){
    fout<<second_level_num_<<endl;
    fout<<mean_shape_.rows<<endl; 
    // write mean shape 
    for(int i = 0;i < mean_shape_.rows;i++){
        fout<<mean_shape_(i,0)<<" "<<mean_shape_(i,1)<<" "; 
    } 
    fout<<endl;

    for(int i = 0;i < second_level_num_;i++){
        primary_fern_[i].write(fout);
    }
}
void FernCascade::read(ifstream& fin){
    fin>>second_level_num_;
	second_level_num_ = 500;
	primary_fern_.resize(second_level_num_);
    
    int landmark_num = 0;
    fin>>landmark_num;

    // read mean shape
    for(int i = 0;i < landmark_num;i++){
        fin>>mean_shape_(i,0)>>mean_shape_(i,1);
    }


    for(int i = 0;i < second_level_num_;i++){
        primary_fern_[i].read(fin);
    }

	return;
}
void FernCascade::predict(const Mat_<uchar>& image, Mat_<double>& shape, Bbox& bounding_box){
    // Mat_<double> normalize_matrix = Mat_<double>::eye(2,2);
    // Mat_<double> invert_normalized_matrix = Mat_<double>::eye(2,2);
    // solve(shape,mean_shape,normalize_matrix,DECOMP_SVD);
    // invert(normalize_matrix,invert_normalized_matrix,DECOMP_SVD);
    Mat_<double> normalize_shape = shape_normalize(shape, bounding_box); 
    Mat_<double> rotation;
    double scale;
    Mat_<double> translation;
    translate_scale_rotate(shape,mean_shape_,translation,scale,rotation); 

    for(int i = 0;i < second_level_num_;i++){
        primary_fern_[i].predict(image,shape, bounding_box,mean_shape_,scale, rotation);
    }
}

