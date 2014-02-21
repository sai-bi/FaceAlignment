/**
 * @author 
 * @version 2014/02/16
 */

#include "face.h"

 
Face::Face(){
    
}


void Face::readData(){
    ifstream fin;
    fin.open("./../data/LFPW/keypointsInfor.txt");
    
    double imageX;
    double imageY;
    double x;
    double y;
    
    while(fin>>imageX>>imageY>>x>>y){
        imageStartCor.push_back(Point2d(imageX,imageY));
        imgSize.push_back(Point2d(x,y)); 
        vector<Point2d> keypoint;
        for(int i = 0;i < 35;i++){
            fin>>x>>y;
            keypoint.push_back(Point2d(x-imageX,y-imageY));
        }
        
        targetShape.push_back(keypoint);
    }
    
    fin.close();

    // read all image
    Mat img; 
    string imgName = "./../data/LFPW/lfpwFaces/";
    for(int i = 1;i < imgNum+1;i++){  
        imgName = imgName + to_string(i) + ".jpg";
        img = imread(imgName.c_str());
        faceImages.push_back(img);
    }
}


void Face::getMeanShape(){
    // first scale all images to the same size
    // change the keypoints coordinates 
    // average all keypoints in the same position
    
    //get average size
    for(int i = 0;i < imgSize.size();i++){
        averageWidth += imgSize[i].x;
        averageHeight += imgSize[i].y;  
    }    
    
    averageWidth = averageWidth / imgSize.size();
    averageHeight = averageHeight / imgSize.size();
    
    // scale all images
    for(int i = 0;i < faceImages.size();i++){
        resize(faceImages[i],faceImages[i],Size(averageWidth,averageHeight)); 
    }  

    // change the keypoint coordinates
    for(int i = 0;i < targetShape[0].size();i++){
        meanShape.push_back(Point2d(0,0));
    }
    for(int i = 0;i < targetShape.size();i++){
        for(int j = 0;j < targetShape[i].size();j++){
            double x = targetShape[i][j].x * averageWidth / imgSize[i].x;
            double y = targetShape[i][j].y * averageHeight / imgSize[i].y;  
            targetShape[i][j].x = x;
            targetShape[i][j].y = y;  
            
            meanShape[j].x += x;
            meanShape[j].y += y;
        }
    } 
   
    // get the mean shape  
    for(int i = 0;i < meanShape.size();i++){
        meanShape[i].x = meanShape[i].x / targetShape.size();
        meanShape[i].y = meanShape[i].y / targetShape.size();
    }
}

void Face::getFeaturePixelLocation(){
    // sample a number of pixels from the face images
    // get their coordinates related to the nearest face keypoints
    
    // random face pixels selected
    vector<int> allIndex;
    for(int i = 0;i < averageHeight * averageWidth;i++){
        allIndex.push_back(i); 
    }    
    
    srand(time(NULL));

    random_shuffle(allIndex.begin(),allIndex.end());

    for(int i = 0;i < featurePixelNum;i++){
        int x = allIndex[i] % averageWidth;
        int y = allIndex[i] / averageWidth;
        
        vector<Point2d> temp1;
        vector<int> temp2;
        // find the nearest keypoint
        for(int j = 0;j < targetShape.size();j++){
            double dist = MAX;
            int minIndex = 0; 
            for(int k = 0;k < targetShape[j].size();k++){
                double dist1 = norm(Point2d(x,y) - targetShape[j][k]);
                if(dist1 < dist){
                    dist = dist1;
                    minIndex = j;
                }
            } 
            temp1.push_back(Point2d(x,y) - targetShape[j][minIndex]);
            temp2.push_back(minIndex);
        } 
        featurePixelCoordinates.push_back(temp1);
        nearestKeypointIndex.push_back(temp2);
    }  
}


void Face::extractFeature(){
    vector<vector<double>> deltaShape; 
    getDeltaShape(deltaShape);
    //get a random direction
    vector<double> randomDirection;
    getRandomDirection(randomDirection); 

    //project
    vector<double> scalar;
    for(int i = 0;i < deltaShape.size();i++){
        double product = 0;
        for(int j = 0;j < randomDirection.size();j++){
            product = product + randomDirection[j] * deltaShape[i][j];
        }   
        scalar.push_back(product);
    }

    vector<vector<double>> pixelValue;
    
    for(int i = 0;i < featurePixelNum;i++){
        vector<double> temp;
    } 

}

void Face::getDeltaShape(vector<vector<double>& deltaShape){
    //calculate the difference between current shape and target shape
    for(int i = 0;i < currentShape.size();i++){
        vector<Point2d> temp;
        for(int j = 0;j < currentShape[i].size();j++){
            // temp.push_back(currentShape[i][j] - targetShape[i][j]); 
            Point2d delta = currentShape[i][j] - targetShape[i][j];
            deltaShape.push_back(delta.x);
            deltaShape.push_back(delta.y);
        }
        deltaShape.push_back(temp);
    }
}

void Face::getRandomDirection(vector<double>& randomDirection){
    srand(time(NULL));
    double sum = 0;
    for(int i = 0;i < 2 * keypointNum;i++){
        int temp = rand()%100 + 1; 
        randomDirection.push_back(temp);
        sum = sum + temp * temp;
    }

    sum = sqrt(sum);
    
    // normalize it;
    for(int i = 0;i < randomDirection.size();i++){
        randomDirection[i] = randomDirection[i] / sum;  
    }
}


void Face::firstLevelRegression(){
    for(int i = 0;i < firstLevelNum;i++){
        // get the feature pixel location based on currentShape             
        vector<vector<Point2d>> currentFeatureLocation;
        vector<vector<double>> pixelDensity;
        
        for(int j = 0;j < featurePixelCoordinates.size();j++){
            vector<Point2d> temp;
            vector<double> temp2;
            for(int k = 0;k < featurePixelCoordinates[j].size();k++){
                int nearestIndex = nearestKeypointIndex[j][k];
                Point2d temp1 = currentShape[k][nearestIndex] + featurePixelCoordinates[j][k];
                temp.push_back(temp1);

                double tempDensity =    
            }
            currentFeatureLocation.push_back(temp);
        }   
            
        // select the best feature
        vector<vector<double>> pixelDensity;
        
        
    }

}

void Face::secondLevelRegression(){

}
