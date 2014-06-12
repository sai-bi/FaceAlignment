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
    // the corresponding nearest keypoint index of each selected pixel
	Mat_<int> nearest_keypoint_index(pixel_pair_num,1);
    RNG random_generator(getTickCount());
    // primary_fern_.resize(second_level_num);
    int landmark_num = current_shapes.rows;
    int training_num = images.size();
    int image_width = images[0].cols;
    int image_height = images[0].rows;


    vector<Bbox> bounding_box;

    for(int i = 0;i < current_shapes.size();i++){
        // find max x coordinates and max y coordinates 
        double max_x = -1e10;
        double max_y = -1e10;
        double min_x = 1e10;
        double min_y = 1e10;
        for(int j = 0;j < landmark_num;j++){
            if(current_shapes[i](j,0) > max_x){
                max_x = current_shapes[i](j,0);
            } 
            if(current_shapes[i](j,0) < min_x){
                min_x = current_shapes[i](j,0);
            } 
            if(current_shapes[i](j,1) > max_y){
                max_y = current_shapes[i](j,1);
            } 
            if(current_shapes[i](j,1) < min_y){
                min_y = current_shapes[i](j,1);
            } 
        }
        Bbox temp;
        temp.start_x = min_x - 10;
        temp.start_y = min_y - 10;
        temp.width = max_x - min_x + 20;
        temp.height = max_y - min_y + 20;
        temp.centroid_x = min_x + temp.width/2.0; 
        temp.centroid_y = min_y + temp.height/2.0
        bounding_box.push_back(temp);
    }

    vector<Mat_<double> > normalized_shapes;
    mean_shape_ = Mat::zeros(landmark_num_,2,CV_64FC1);
    for(int i = 0;i < current_shapes_.size();i++){
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
    mean_shape_ = 1.0 / current_shapes_.size() * mean_shape; 

    // get feature pixel location for each image
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
                real_x = 0
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

     


     



	// calculate the inverse of normalize matrix
    vector<Mat_<double> > inverse_normalize_matrix;
    for(int i = 0;i < normalize_matrix.size();i++){
        Mat_<double> temp = Mat_<double>::eye(2,2);
        // invert(normalize_matrix[i],temp,DECOMP_SVD);
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
			// project the coordinates back into original system
            Mat_<double> global_coordinates = landmark_coordinates * inverse_normalize_matrix[j];
            global_coordinates(0,0) += current_shapes[j](index,0);
            global_coordinates(0,1) += current_shapes[j](index,1);
            int temp_x = global_coordinates(0,0);
            int temp_y = global_coordinates(0,1);   
            if(temp_x < 0){
				temp_x = 0;
			}
			if(temp_y < 0){
				temp_y = 0;
			}
			if(temp_x >= image_width){
				temp_x = image_width - 1;
			}
			if(temp_y >= image_height){
				temp_y = image_height - 1;
			}
			curr_pair_pixel_density.push_back(int(images[j](temp_y,temp_x)));
        }
		pixel_density.push_back(curr_pair_pixel_density);
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
        fout<<mean_shape_(i,0)<<" "<<mean_shape_(j,1)<<" "; 
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
    Mat_<double> normalized_shapes;


    for(int i = 0;i < second_level_num_;i++){
        primary_fern_[i].predict(image,shape, bounding_box);
    }
}

