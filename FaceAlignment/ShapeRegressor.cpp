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
using namespace std;
using namespace cv;

ShapeRegressor::ShapeRegressor(){
    first_level_num_ = 0;
}

/**
 * @param images gray scale images
 * @param ground_truth_shapes a vector of N*2 matrix, where N is the number of landmarks
 * @param bounding_box BoundingBox of faces
 * @param first_level_num number of first level regressors
 * @param second_level_num number of second level regressors
 * @param candidate_pixel_num number of pixels to be selected as features
 * @param fern_pixel_num number of pixel pairs in a fern
 * @param initial_num number of initial shapes for each input image
 */
void ShapeRegressor::Train(const vector<Mat_<uchar> >& images, 
                   const vector<Mat_<double> >& ground_truth_shapes,
                   const vector<BoundingBox>& bounding_box,
                   int first_level_num, int second_level_num,
                   int candidate_pixel_num, int fern_pixel_num,
                   int initial_num){
    cout<<"Start training..."<<endl;
    bounding_box_ = bounding_box;
    training_shapes_ = ground_truth_shapes;
    first_level_num_ = first_level_num;
    landmark_num_ = ground_truth_shapes[0].rows; 
    // data augmentation and multiple initialization 
    vector<Mat_<uchar> > augmented_images;
    vector<BoundingBox> augmented_bounding_box;
    vector<Mat_<double> > augmented_ground_truth_shapes;
    vector<Mat_<double> > current_shapes;
     
    RNG random_generator(getTickCount());
    for(int i = 0;i < images.size();i++){
        for(int j = 0;j < initial_num;j++){
            int index = 0;
            do{
                // index = (i+j+1) % (images.size()); 
                index = random_generator.uniform(0, images.size());
            }while(index == i);
            augmented_images.push_back(images[i]);
            augmented_ground_truth_shapes.push_back(ground_truth_shapes[i]);
            augmented_bounding_box.push_back(bounding_box[i]); 
            // 1. Select ground truth shapes of other images as initial shapes
            // 2. Project current shape to bounding box of ground truth shapes 
            Mat_<double> temp = ground_truth_shapes[index];
            temp = ProjectShape(temp, bounding_box[index]);
            temp = ReProjectShape(temp, bounding_box[i]);
            current_shapes.push_back(temp); 
        } 
    }
    
    // get mean shape from training shapes
    mean_shape_ = GetMeanShape(ground_truth_shapes,bounding_box); 
    
    // train fern cascades
    fern_cascades_.resize(first_level_num);
    vector<Mat_<double> > prediction;
    for(int i = 0;i < first_level_num;i++){
        cout<<"Training fern cascades: "<<i+1<<" out of "<<first_level_num<<endl;
        prediction = fern_cascades_[i].Train(augmented_images,current_shapes,
                augmented_ground_truth_shapes,augmented_bounding_box,mean_shape_,second_level_num,candidate_pixel_num,fern_pixel_num, i+1, first_level_num);
        
        // update current shapes 
        for(int j = 0;j < prediction.size();j++){
            current_shapes[j] = prediction[j] + ProjectShape(current_shapes[j], augmented_bounding_box[j]);
            current_shapes[j] = ReProjectShape(current_shapes[j],augmented_bounding_box[j]);
        }
    } 
    
}


void ShapeRegressor::Write(ofstream& fout){
    fout<<first_level_num_<<endl;
    fout<<mean_shape_.rows<<endl;
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
    mean_shape_ = Mat::zeros(landmark_num_,2,CV_64FC1);
    for(int i = 0;i < landmark_num_;i++){
        fin>>mean_shape_(i,0)>>mean_shape_(i,1);
    }
    
    int training_num;
    fin>>training_num;
    training_shapes_.resize(training_num);
    bounding_box_.resize(training_num);

    for(int i = 0;i < training_num;i++){
        BoundingBox temp;
        fin>>temp.start_x>>temp.start_y>>temp.width>>temp.height>>temp.centroid_x>>temp.centroid_y;
        bounding_box_[i] = temp;
        
        Mat_<double> temp1(landmark_num_,2);
        for(int j = 0;j < landmark_num_;j++){
            fin>>temp1(j,0)>>temp1(j,1);
        }
        training_shapes_[i] = temp1; 
    }

    fern_cascades_.resize(first_level_num_);
    for(int i = 0;i < first_level_num_;i++){
        fern_cascades_[i].Read(fin);
    }
} 


Mat_<double> ShapeRegressor::Predict(const Mat_<uchar>& image, const BoundingBox& bounding_box, int initial_num){
    // generate multiple initializations
    Mat_<double> result = Mat::zeros(landmark_num_,2, CV_64FC1);
    RNG random_generator(getTickCount());
    for(int i = 0;i < initial_num;i++){
        random_generator = RNG(i);
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

void ShapeRegressor::Load(string path){
    cout<<"Loading model..."<<endl;
    ifstream fin;
    fin.open(path);
    this->Read(fin); 
    fin.close();
    cout<<"Model loaded successfully..."<<endl;
}

void ShapeRegressor::Save(string path){
    cout<<"Saving model..."<<endl;
    ofstream fout;
    fout.open(path);
    this->Write(fout);
    fout.close();
}


