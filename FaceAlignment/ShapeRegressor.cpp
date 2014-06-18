/**
 * @author 
 * @version 2014/06/17
 */
#include "FaceAlignment.h"

ShapeRegressor::ShapeRegressor(){
    first_level_num_ = 0;
    second_level_num_ = 0;
    candidate_pixel_num_ = 0;
    fern_pixel_num_ = 0;
    initial_num_ = 0;
}


void ShapeRegressor::Train(const vector<Mat_<uchar> >& images, 
                   const vector<Mat_<double> >& ground_truth_shapes,
                   const vector<BoundingBox>& bounding_box,
                   int first_level_num, int second_level_num,
                   int candidate_pixel_num, int fern_pixel_num,
                   int initial_num){
    bounding_box_ = bounding_box;
    training_shapes_ = ground_truth_shapes;
    // data augmentation and multiple initialization 
    vector<Mat_<uchar> > augmented_images;
    vector<Mat_<BoundingBox> > augmented_bounding_box;
    vector<Mat_<double> > augmented_ground_truth_shapes;
    vecot<Mat_<double> > current_shapes;
     
    RNG random_generator(getTickCount());
    for(int i = 0;i < ground_truth_images.size();i++){
        for(int j = 0;j < initial_num;j++){
            int index = 0;
            do{
                index = random_generator.uniform(0,ground_truth_images.size());
            }while(index == i);
            augmented_images.push_back(images[index]);
            augmented_ground_truth_shapes.push_back(ground_truth_shapes[i]);
            augmented_bounding_box.push_back(bounding_box[i]); 
            // 1. Select ground truth shapes of other images as initial shapes
            // 2. Project current shape to bounding box of ground truth shapes 
            Mat_<double> temp = ground_truth_shape[index];
            temp = ProjectShape(temp, bounding_box[index]);
            temp = ReProjectShape(temp, bounding_box[i]);
            current_shapes.push_back(temp); 
        } 
    }
    
    // get mean shape
    mean_shape_ = GetMeanShape(ground_truth_shape,bounding_box); 
    
    // get normalized targets
    vector<Mat_<double> > normalized_targets;
    normalized_targets.resize(current_shapes.size()); 
    for(int i = 0;i < current_shapes.size();i++){
        normalized_targets[i] = ProjectShape(current_shapes[i],augmented_bounding_box[i]);
        normalized_targets[i] = ProjectShape(augmented_ground_truth_shapes[i],augmented_bounding_box[i]) - normalized_targets[i];
    } 
    
    // train fern cascades
    fern_cascades_.resize(first_level_num);
    vector<Mat_<double> > prediction;
    for(int i = 0;i < first_level_num;i++){
        prediction = fern_cascades_[i].train(augmented_images,current_shapes,
                augmented_ground_truth_shapes,augmented_bounding_box,mean_shape_,second_level_num,candidate_pixel_num,fern_pixel_num);
        
        // update current shapes 
        for(int j = 0;j < prediction.size();j++){
            current_shapes[j] = prediction[j] + ProjectShape(current_shapes[j], augmented_bounding_box[j]);
            current_shapes[j] = ReProjectShape(current_shapes[j],augmented_bounding_box[j]);
        }
    } 
    
}


void ShapeRegressor::Write(ofstream& fout){
    fout<<first_level_num_<<endl;
    fout<<landmark_num_<<endl;
    for(int i = 0;i < landmark_num_;i++){
        fout<<mean_shape_(i,0)<<" "<<mean_shape_(i,1)<<" "; 
    }
    fout<<endl;

    fout<<training_shapes_.size()<<endl;
    for(int i = 0;i < training_shapes_.size();i++){
        fout<<bounding_box_[i].start_x<<" "<<bounding_box_[i].start_y<<" "
            <<bounding_box_[i].width<<" "<<bounding_box_[i].height<<" "
            <<bounding_box_[i].centroid_x<<" "<<bounding_box_[i].centroid_y<<endl;
        for(int j = 0;j < training_shapes_[i].rows;j++){
            fout<<training_shapes_[i](j,0)<<" "<<training_shapes_[i](j,1)<<" "; 
        }
        fout<<endl;
    }
    
    for(int i = 0;i < first_level_num_;i++){
        fern_cascades_[i].Write(fout);
    } 
}

void ShapeRegressor::Read(ifstream& fin){
    fin>>first_level_num_;
    fin>>landmark_num_;
    for(int i = 0;i < landmark_num_;i++){
        fin>>mean_shape_(i,0)>>mean_shape_(i,1);
    }
    
    int training_num;
    cin>>training_num;
    training_shapes_.resize(training_num);
    bounding_box_.resize(training_num);

    for(int i = 0;i < training_num;i++){
        BoundingBox temp;
        fin>>temp.start_x>>temp.start_y>>temp.width>>temp.height>>temp.centroid_x>>temp.centroid_y>>endl;
        bounding_box_[i] = temp;
        
        Mat_<double> temp1(landmark_num_,2);
        for(int j = 0;j < landmark_num_;j++){
            fin>>temp1(j,0)>>temp1(j,1);
        }
        training_shapes_.push_back(temp1); 
    }

    fern_cascades_.resize(first_level_num_);
    for(int i = 0;i < first_level_num_;i++){
        fern_cascades_.Read(fin);
    }
} 


Mat_<double> ShapeRegressor::Predict(const Mat_<uchar>& image, const BoundingBox& bounding_box, int initial_num){
    // generate multiple initializations
    Mat_<double> result = Mat::zeros(landmark_num_,2, CV_64FC1);
    RNG random_generator(getTickCount());
    for(int i = 0;i < initial_num;i++){
        int index = random_generator.uniform(0,training_shapes_.size());
        Mat_<double> current_shape = training_shapes_[index];
        BoundingBox current_bounding_box = bounding_box_[index];
        
        current_shape = ProjectShape(current_shape,current_bounding_box);
        current_shape = ReProjectShape(current_shape,bounding_box);
        
        for(int j = 0;j < first_level_num_;j++){
            Mat_<double> prediction = fern_cascades_[j].Predict(image,bounding_box,mean_shape_,current_shape);
            // update current shape
            current_shape = prediction + ProjectShape(current_shape,bounding_box);
            current_shape = ReProjectShape(current_shape,bounding_box); 
        }
        result = result + current_shape; 
    }    

    return 1.0 / initial_num * result;
}


