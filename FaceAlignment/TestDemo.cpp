/**
 * @author 
 * @version 2014/06/18
 */

#include "FaceAlignment.h"

int main(){
    vector<Mat_<uchar> > test_images;
    vector<BoundingBox> test_bounding_box;
    int test_img_num = 507;
    int initial_number = 20;

    for(int i = 0;i < test_img_num;i++){
        string image_name = "./../../CRP/rcpr_v2/data/testImages/";
        image_name = image_name + to_string(i+1) + ".jpg";
        Mat_<uchar> temp = imread(image_name,0);
        test_images.push_back(temp);
    }
    fin.open("./../../CRP/rcpr_v2/data/boundingbox_test.txt");
    for(int i = 0;i < test_img_num;i++){
        Bbox temp;
        fin>>temp.start_x>>temp.start_y>>temp.width>>temp.height;
        temp.centroid_x = temp.start_x + temp.width/2.0;
        temp.centroid_y = temp.start_y + temp.height/2.0; 
        test_bounding_box.push_back(temp);
    }
    fin.close(); 
    
    cout<<"Load model..."<<endl;
    ShapeRegressor regressor;
    regressor.load("./data/model_cofw_2.txt");
    cout<<"Model loaded successfully..."<<endl;
    while(true){
        int index = 1;
        cout<<"Input index:"<<endl;
        cin>>index;

        Mat_<double> current_shape = regressor.Predict(test_images[index],bounding_box[index],initial_number);
        cout<<current_shape<<endl;
        Mat test_image_1 = test_images[index].clone();
        for(int i = 0;i < landmark_num;i++){
            circle(test_image_1,Point2d(current_shape(i,0),current_shape(i,1)),3,Scalar(255,0,0),-1,8,0);
        }
        imshow("result",test_image_1);
    }
    return 0;
}


