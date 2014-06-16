/**
 * @author Bi Sai 
 * @version 2014/03/26
 */
#include "face.h"

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
    

	// start training 
	train(images,target_shapes,initial_number,pixel_pair_num,
			pixel_pair_in_fern,first_level_num,second_level_num, bounding_box);
	

    return 0;
}

















