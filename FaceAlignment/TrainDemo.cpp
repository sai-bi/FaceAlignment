/**
 * @author 
 * @version 2014/06/18
 */

#include "FaceAlignment.h"

int main(){
    int img_num = 1345;
    int candidate_pixel_num = 400;
    int fern_pixel_num = 5;
    int first_level_num = 10;
    int second_level_num = 500; 
    int landmark_num = 29;
    int initial_number = 20;
    vector<Mat_<uchar> > images;
    vector<BoundingBox> bbox; 
    
    cout<<"Read images..."<<endl;
    for(int i = 0;i < img_num;i++){
        string image_name = "./../../CRP/rcpr_v2/data/trainingImages/";
        image_name = image_name + to_string(i+1) + ".jpg";
        Mat_<uchar> temp = imread(image_name,0);
        images.push_back(temp);
    }
    
    vector<Mat_<double> > ground_truth_shapes;
    vector<BoundingBox> bounding_box;
    ifstream fin;
    fin.open("./../../CRP/rcpr_v2/data/boundingbox.txt");
    for(int i = 0;i < img_num;i++){
        BoundingBox temp;
        fin>>temp.start_x>>temp.start_y>>temp.width>>temp.height;
        temp.centroid_x = temp.start_x + temp.width/2.0;
        temp.centroid_y = temp.start_y + temp.height/2.0; 
        bounding_box.push_back(temp);
    }
    fin.close(); 

    fin.open("./../../CRP/rcpr_v2/data/keypoints.txt");
    for(int i = 0;i < img_num;i++){
        Mat_<double> temp(landmark_num,2);
        for(int j = 0;j < landmark_num;j++){
            fin>>temp(j,0); 
        }
        for(int j = 0;j < landmark_num;j++){
            fin>>temp(j,1); 
        }
        ground_truth_shapes.push_back(temp);
    }        
    fin.close(); 
    
    ShapeRegressor regressor;
    regressor.Train(images,ground_truth_shapes,bounding_box,first_level_num,second_level_num,
                    candidate_pixel_num,fern_pixel_num,initial_number);
    regressor.Save("./data/model.txt");

    return 0;
}

