/**
 * @author 
 * @version 2014/02/16
 */

#include "face.h"

void Face::run(){
    // read parameters
    cout<<"Read parameters..."<<endl;
    readParameters();

    // read training data
    cout<<"Read data..."<<endl;
    readData();

    // calculate mean shape;
    cout<<"Get mean shape..."<<endl;
    getMeanShape();
    
        
    // Mat myImage = Mat::zeros(averageHeight,averageWidth,CV_8UC3);
    // for(int i = 0;i < meanShape.size();i++){
        // circle(myImage,meanShape[i],2,Scalar(255,0,0),-1,8,0);  
    // }
    // imshow("image",myImage);        
    // waitKey(0); 

    // get feature location
    cout<<"Get feature pixel locations..."<<endl;
    getFeaturePixelLocation();

    // regression
    cout<<"First level regression..."<<endl;
    firstLevelRegression();

}

void Face::readParameters(){
    boost::property_tree::ptree pt;
    boost::property_tree::ini_parser::read_ini("./parameters.ini",pt);
    
    imgNum = pt.get<int>("Training.imgNum");
    featurePixelNum = pt.get<int>("Training.featurePixelNum");
    firstLevelNum = pt.get<int>("Training.firstLevelNum");
    secondLevelNum = pt.get<int>("Training.secondLevelNum");
    keypointNum = pt.get<int>("Training.keypointNum");
    featureNumInFern = pt.get<int>("Training.featureNumInFern");
    shrinkage = pt.get<int>("Training.shrinkage");
    debug = pt.get<int>("Training.debug");

    // cout<<"Shrinkage:"<<shrinkage<<endl;

}


Face::Face(){
    srand(time(NULL));
    averageHeight = 0;
    averageWidth = 0;    
}

void Face::readData(){
    ifstream fin;
    fin.open("./../data/LFPW/keypointsInfor.txt");
    
    double imageX;
    double imageY;
    double x;
    double y;
    
    // read image info
    while(fin>>imageX>>imageY>>x>>y){
        imageStartCor.push_back(Point2d(imageX,imageY));
        imgSize.push_back(Point2d(x,y)); 
        vector<Point2d> keypoint;
        for(int i = 0;i < 35;i++){
            fin>>x>>y;
            keypoint.push_back(Point2d(x,y));
        }
        
        targetShape.push_back(keypoint);
    }
    
    fin.close();

    // read all image
    Mat img; 
    for(int i = 1;i < imgNum+1;i++){  
        string imgName = "./../data/LFPW/lfpwFaces/";
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

    if(debug){
        cout<<"Average width and height "<<averageWidth<<" "<<averageHeight<<endl;
    }
    
    // scale all images
    for(int i = 0;i < faceImages.size();i++){
        resize(faceImages[i],faceImages[i],Size(averageWidth,averageHeight)); 
    }  

    // change the keypoint coordinates
    for(int i = 0;i < keypointNum;i++){
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
    

    random_shuffle(allIndex.begin(),allIndex.end());

    for(int i = 0;i < featurePixelNum;i++){
        int x = allIndex[i] % averageWidth;
        int y = allIndex[i] / averageWidth;
        
        double dist = MAXNUM;
        int minIndex = 0;
        for(int j=0;j < meanShape.size();j++){
            double temp = norm(Point2d(x,y) - meanShape[j]);
            if(temp < dist){
                dist = temp;
                minIndex = j;
            } 
        } 
        featurePixelCoordinates.push_back(Point2d(x,y) - meanShape[minIndex]);
        nearestKeypointIndex.push_back(minIndex);
    }  
}


void Face::extractFeature(const Mat& covariance,const vector<vector<double> >& pixelDensity,
        vector<Point2i>& selectedFeatureIndex){

    vector<vector<double> > deltaShape; 
    getDeltaShape(deltaShape);

    //get a random direction
    for(int i = 0;i < featureNumInFern;i++){    
        
        // get a random direction
        vector<double> randomDirection;
        getRandomDirection(randomDirection);
        
        // cout<<endl;
        // for(int j = 0;j < randomDirection.size();j++){
            // cout<<randomDirection[j]<<" ";
        // } 
        // cout<<endl;

        // calculate the product
        vector<double> projectResult;
        for(int j = 0;j < deltaShape.size();j++){
            double temp = product(deltaShape[j],randomDirection);
            // cout<<"Product result:"<<temp<<endl;
            projectResult.push_back(temp);
        }
        
        // calculate cov(y,f_i)
        vector<double> covarianceYF;
        for(int j = 0;j < pixelDensity.size();j++){
            double temp = getCovariance(projectResult,pixelDensity[j]);
            // covarianceYF.push_back(getCovariance(projectResult,pixelDensity[j]));      
            // cout<<temp<<" ";
            covarianceYF.push_back(temp);
        }
        
        // get the pair with highest correlation corr(y,fi - fj);  
        // zero_matrix<double> correlation;
        // double stdY = sqrt(getCovariance(projectResult,projectResult)); 
        // cout<<stdY<<endl; 
        int selectedIndex1 = 0;
        int selectedIndex2 = 0;
        double highest = MINNUM;
        for(int j = 0;j < featurePixelNum;j++){
            for(int k = j+1;k < featurePixelNum;k++){
                bool flag = false;
                for(int p = 0;p < selectedFeatureIndex.size();p++){
                    if(j == selectedFeatureIndex[p].x && k == selectedFeatureIndex[p].y){
                        flag = true;
                        break;
                    }
                    if(k == selectedFeatureIndex[p].x && j == selectedFeatureIndex[p].y){
                        flag = true;
                        break;
                    }
                    
                }
                if(flag)
                    continue;


                double temp1 = covarianceYF[j];
                double temp2 = covarianceYF[k];
                double temp3 = covariance.at<double>(k,j);
                double temp4 = temp1 * temp2 / (sqrt(temp3));

                if(abs(temp4) > highest){
                    highest = abs(temp4);
                    // cout<<highest<<endl;
                    if(temp4 > 0){
                        selectedIndex1 = j;
                        selectedIndex2 = k;
                    }
                    else{
                        selectedIndex1 = k;
                        selectedIndex2 = j;
                    }
                }
            }
        }

        // cout<<selectedIndex1<<" "<<selectedIndex2<<endl;

        selectedFeatureIndex.push_back(Point2i(selectedIndex1,selectedIndex2));
    } 

    

}

double Face::product(const vector<double>& v1, const vector<double>& v2){
    double result = 0;
    for(int i = 0;i < v1.size();i++){
        // cout<<"wow "<<v1[i]<<" "<<v2[i]<<endl;
        result = result + v1[i] * v2[i]; 
    }

    return result;
}


void Face::getDeltaShape(vector<vector<double> >& deltaShape){
    //calculate the difference between current shape and target shape
    for(int i = 0;i < currentShape.size();i++){
        vector<double> temp;
        for(int j = 0;j < currentShape[i].size();j++){
            // Point2d delta = currentShape[i][j] - targetShape[i][j];
            Point2d delta = targetShape[i][j] - currentShape[i][j]; 
            temp.push_back(delta.x);
            temp.push_back(delta.y);
        }
        deltaShape.push_back(temp);
    }
}

void Face::getRandomDirection(vector<double>& randomDirection){
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

    // first initial currentShape
    for(int i = 0;i < targetShape.size();i++){
        currentShape.push_back(meanShape);
    }

    for(int i = 0;i < firstLevelNum;i++){
        cout<<endl;
        cout<<"First level regression: "<<i<<endl;
        cout<<endl;
        // get the feature pixel location based on currentShape             
        vector<vector<Point2d> > currentFeatureLocation;
        vector<vector<double> > pixelDensity;
        
        for(int j = 0;j < featurePixelCoordinates.size();j++){
            vector<Point2d> temp;
            vector<double> temp1;
            for(int k = 0;k < currentShape.size();k++){
                Point2d currLocation;
                currLocation = featurePixelCoordinates[j] + currentShape[k][nearestKeypointIndex[j]];
                temp.push_back(currLocation);
                // temp1.push_back(trainingImages((int)(currLocation.y),(int)(currLocation.x))); 
                Vec3b color = faceImages[k].at<Vec3b>((int)(currLocation.y),(int)(currLocation.x));
                int b = color.val[0];
                int g = color.val[1];
                int r = color.val[2];
                double density = 0.2126 * r +  0.7152 * g + 0.0722 * b;
                temp1.push_back(density);
            }
            currentFeatureLocation.push_back(temp);
            pixelDensity.push_back(temp1);
        }     
    
        // select feature
        
        // calculate the covariance of f_i and f_j
        // zero_matrix<double> covariance(featurePixelNum,featurePixelNum);
        Mat covariance = Mat::zeros(featurePixelNum,featurePixelNum,CV_64F);
    
        for(int j = 0;j < featurePixelNum;j++){
            for(int k = j+1;k < featurePixelNum;k++){
                double temp = getCovariance(pixelDensity[j],pixelDensity[k]);
                // covariance(j,k) = temp;
                // covariance(k,j) = temp;
                covariance.at<double>(j,k) = temp;
                covariance.at<double>(k,j) = temp;
            }
        }
        // cout<<covariance<<endl;
        secondLevelRegression(covariance,pixelDensity);    
    }

    
}

void Face::secondLevelRegression(const Mat& covariance,const vector<vector<double> >& pixelDensity){
    for(int i = 0;i < secondLevelNum;i++){
        // select best feature
        vector<Point2i> selectedFeatureIndex;  
        extractFeature(covariance,pixelDensity,selectedFeatureIndex); 
        
        // record selected feature index 
        ofstream fout;
        fout.open("trainingOutput.txt",std::ofstream::out | std::ofstream::app);

        for(int i = 0;i < selectedFeatureIndex.size();i++){
            // cout<<"hello world"<<endl;
            fout<<selectedFeatureIndex[i].x<<" "<< selectedFeatureIndex[i].y<<" ";
        }
        fout<<endl;
        fout.close();

        //construct a fern using selected best features 
        constructFern(selectedFeatureIndex,pixelDensity); 
    }   
}

void Face::constructFern(const vector<Point2i>& selectedFeatureIndex,
        const vector<vector<double> >& pixelDensity){
    // turn each shape into a scalar according to relative intensity of pixels
    // generate random threholds
    // divide shapes into bins based on threholds and scalars
    // for each bin, calculate its output
    
    vector<int> fernResult;
    vector<vector<Point2d> > fernOutput;
    int binNum = pow(2.0,featureNumInFern);
    vector<vector<int> > bins;

    for(int i = 0;i < binNum;i++){
        vector<int> temp;
        bins.push_back(temp);
    }

    for(int i = 0;i < currentShape.size();i++){
        int tempResult = 0;
        for(int j = 0;j < selectedFeatureIndex.size();j++){
            double density1 = pixelDensity[selectedFeatureIndex[j].x][i];
            double density2 = pixelDensity[selectedFeatureIndex[j].y][i];

            // binary number
            if(density1 > density2){
                tempResult = tempResult + 2 ^ j; 
            }
        }
        fernResult.push_back(tempResult);
        bins[tempResult].push_back(i); 
    }

    // get random threhold, the number of bins is 2^featureNumInFern; 
    
    // get output
    for(int i = 0;i < bins.size();i++){
        vector<Point2d> currFernOutput;

        if(bins[i].size() == 0){
            for(int j = 0;j < keypointNum;j++){
                currFernOutput.push_back(Point2d(0,0));
            } 
            fernOutput.push_back(currFernOutput);
            continue;
        }
        
        for(int j = 0;j < bins[i].size();j++){
            int shapeIndex = bins[i][j];
            if(j == 0){
                currFernOutput = vectorMinus(targetShape[shapeIndex], currentShape[shapeIndex]);
            }
            else{
                currFernOutput = vectorPlus(currFernOutput, vectorMinus(targetShape[shapeIndex],
                            currentShape[shapeIndex]));
            }
        }

        for(int j = 0;j < currFernOutput.size();j++){
            double temp = 1.0/((1 + shrinkage/bins[i].size()) * bins[i].size());
            currFernOutput[j] = temp * currFernOutput[j]; 
        }

         


        fernOutput.push_back(currFernOutput);
    }
    
    ofstream fout;
    fout.open("trainingOutput.txt",std::ofstream::out | std::ofstream::app);
    for(int i = 0;i < fernOutput.size();i++){
        for(int j = 0;j < fernOutput[i].size();i++){
            fout<<fernOutput[i][j]<<" "; 
        }
        fout<<endl;
    }
    
    // update current shape
    for(int i = 0;i < currentShape.size();i++){
        int binIndex = fernResult[i];
        currentShape[i] = vectorPlus(currentShape[i],fernOutput[binIndex]); 
    }
}


double Face::getCovariance(const vector<double>& v1, const vector<double>& v2){
    // double expV1 = accumulate(v1.begin(),v1.end(),0);
    // double expV2 = accumulate(v2.begin(),v2.end(),0);
    double expV1 = 0;
    double expV2 = 0;

    for(int i = 0;i < v1.size();i++){
        expV1 = expV1 + v1[i];
    }

    for(int i = 0;i < v2.size();i++){
        expV2 = expV2 + v2[i];
    }

    expV1 = expV1 / v1.size();
    expV2 = expV2 / v2.size();

    double total = 0;
    for(int i = 0;i < v1.size();i++){
        total = total + (v1[i] - expV1) * (v2[i] - expV2); 
    }
    return total / v1.size();
}

vector<Point2d> Face::vectorMinus(const vector<Point2d>& shape1, const vector<Point2d>& shape2){
    vector<Point2d> result;

    for(int i = 0;i < shape1.size();i++){
        result.push_back(shape1[i] - shape2[i]);
    }

    return result;
}

vector<Point2d> Face::vectorPlus(const vector<Point2d>& shape1, const vector<Point2d>& shape2){
    vector<Point2d> result;

    for(int i = 0;i < shape1.size();i++){
        result.push_back(shape1[i] + shape2[i]);
    }

    return result;
}

