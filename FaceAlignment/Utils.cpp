/**
 * @author 
 * @version 2014/06/17
 */

#include "FaceAlignment.h"

Mat_<double> GetMeanShape(const vector<Mat_<double> >& shapes,
                          const vector<BoundingBox>& bounding_box){
    vector<Mat_<double> > temp;
    Mat_<double> result(shapes[0].rows,2,CV_64FC1);

    temp = ProjectShape(shapes,bounding_box);
    result = std::accumulate(temp.begin(),temp.end(),result);    
    
    return (1.0 / shapes.size() * result); 
}

Mat_<double> ProjectShape(const Mat_<double>& shape, const BoundingBox& bounding_box){
    Mat_<double> temp(shape.rows,2);
    for(int j = 0;j < shape.rows;j++){
        temp(j,0) = (shape(j,0)-bounding_box.centroid_x) / (bounding_box.width / 2.0);
        temp(j,1) = (shape(j,1)-bounding_box.centroid_y) / (bounding_box.height / 2.0);  
    } 
    return temp;  
}

Mat_<double> ReProjectShape(const Mat_<double>& shape, const BoundingBox& bounding_box){
    Mat_<double> temp(shape.rows,2);
    for(int j = 0;j < shape.rows;j++){
        temp(j,0) = (shape(j,0) * bounding_box.width / 2.0 + bounding_box.centroid_x);
        temp(j,1) = (shape(j,1) * bounding_box.height / 2.0 + bounding_box.centroid_y);
    } 
    return temp; 
}





