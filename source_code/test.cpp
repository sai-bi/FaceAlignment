/**
 * @author 
 * @version 2014/03/31
 */

#include "face.h"

int main(){
    int img_num = 716;
    int pixel_pair_num = 400;
    int pixel_pair_in_fern = 5;
    int first_level_num = 10;
    int second_level_num = 500; 
    int landmark_num = 35;
    int initial_number = 1;


    vector<Mat_<uchar> > images;
   
    // read images 
    cout<<"Read images..."<<endl;
    for(int i = 0;i < img_num;i++){
        string image_name = "./../data/LFPW/lfpwFaces/";
        image_name = image_name + to_string(i+1) + ".jpg";
        Mat_<uchar> temp = imread(image_name,0);
    }

    // read keypoints information
    ifstream fin;
    fin.open("./../data/LFPW/keypointsInfor.txt"); 
    double start_x;
    double start_y;
    double curr_width;
    double curr_height;
    Mat_<double> mean_shape(landmark_num,2);
    vector<Bbox> bbox;
    vector<Mat_<double> > target_shapes;
    for(int i = 0;i < landmark_num;i++){
        mean_shape(i,0) = 0;
        mean_shape(i,1) = 0;
    }
    while(fin>>start_x>>start_y>>curr_width>>curr_height){
        Bbox temp_bbox;
        temp_bbox.start_x = start_x;
        temp_bbox.start_y = start_y;
        temp_bbox.width = curr_width;
        temp_bbox.height = curr_height;
        temp_bbox.centroid_x = start_x + curr_width/2.0;
        temp_bbox.centroid_y = start_y + curr_height/2.0; 
        bbox.push_back(temp_bbox);

        Mat_<double> temp(landmark_num,2);
        double keypoint_x;
        double keypoint_y;
        for(int i = 0;i < 35;i++){
            fin>>keypoint_x>>keypoint_y;
            temp(i,0) = keypoint_x + start_x;
            temp(i,1) = keypoint_y + start_y;
        }
        target_shapes.push_back(temp);
    }

    for(int i = 0;i < img_num;i++){
        Bbox temp = bbox[i];
        for(int j = 0;j < landmark_num;j++){
            double temp1 = (target_shapes[i](j,0) - temp.centroid_x) / (temp.width/2.0);
            double temp2 = (target_shapes[i](j,1) - temp.centroid_y) / (temp.height/2.0);
            mean_shape(j,0) += temp1;
            mean_shape(j,1) += temp2;
        } 
    }        
    mean_shape = 1.0/img_num * mean_shape;

    cout<<"Load model..."<<endl;
    ShapeRegressor regressor;
    regressor.load("./data/model.txt");
    cout<<"Model loaded..."<<endl;

    while(true){
        int index = 1;
        cout<<"Input index:"<<endl;
        cin>>index;
        Mat_<uchar> test_image = imread("./../data/LFPW/lfpwFaces/" + to_string(index) + ".jpg",0);
        Mat test_image_1 = imread("./../data/LFPW/lfpwFaces/" + to_string(index) + ".jpg");
        
        Bbox box = get_bounding_box(target_shapes[index-1]); 

        Mat_<double> current_shape = test(regressor,test_image,target_shapes,box,1);

        cout<<current_shape<<endl;

        for(int i = 0;i < landmark_num;i++){
            circle(test_image_1,Point2d(current_shape(i,0),current_shape(i,1)),3,Scalar(255,0,0),-1,8,0);
        }
        for(int i = 0;i < landmark_num;i++){
            circle(test_image_1,Point2d(target_shapes[index-1](i,0),target_shapes[index-1](i,1)),3,Scalar(0,0,255),-1,8,0);
        }
        imshow("result",test_image_1);
        // waitKey(0);
    }
    return 0;

}
