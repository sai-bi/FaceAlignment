/**
 * @author Bi Sai 
 * @version 2014/03/26
 */
#include "face.h"
using namespace std;
using namespace cv;
int main(){

    // parameters
    int img_num = 1345;
    int pixel_pair_num = 400;
    int pixel_pair_in_fern = 5;
    int first_level_num = 10;
    int second_level_num = 500; 
    int landmark_num = 29;
    int initial_number = 20;
    vector<Mat_<uchar> > images;
    
    // bounding box for each face 
    vector<Bbox> bbox; 
    
    // read images
    cout<<"Read images..."<<endl;
    for(int i = 0;i < img_num;i++){
        string image_name = "./../../CRP/rcpr_v2/data/trainingImages/";
        image_name = image_name + to_string(i+1) + ".jpg";
        Mat_<uchar> temp = imread(image_name,0);
        images.push_back(temp);
    }

    // read keypoints information
    vector<Mat_<double> > target_shapes;
    vector<Bbox> bounding_box;


    ifstream fin;
    fin.open("./../../CRP/rcpr_v2/data/boundingbox.txt");

    for(int i = 0;i < img_num;i++){
        Bbox temp;
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
        target_shapes.push_back(temp);
    }        
    fin.close();
    
    
    vector<Mat_<uchar> > test_images;
    vector<Bbox> test_bounding_box;
    int test_img_num = 507;

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
    regressor.load("./data/model_cofw_1.txt");
    cout<<"Model loaded..."<<endl;
    
    while(true){
        int index = 1;
        cout<<"Input index:"<<endl;
        cin>>index;

        Mat_<double> current_shape = test(regressor,test_images[index],target_shapes,test_bounding_box[index],20,bounding_box);

        cout<<current_shape<<endl;

        Mat test_image_1 = test_images[index].clone();

        for(int i = 0;i < landmark_num;i++){
            circle(test_image_1,Point2d(current_shape(i,0),current_shape(i,1)),3,Scalar(255,0,0),-1,8,0);
        }
        imshow("result",test_image_1);
        waitKey(0);
    }
	
    
	

    return 0;
}


















