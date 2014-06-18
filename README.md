###Face Alignment
This program is a C++ reimplementation of algorithms in paper "Face Alignment by Explicit Regression"
by Cao et al. This program can be used to train models for detecting facial keypoints, and it is 
extremely fast both in training and testing. 

Please go to folder `FaceAlignment` to see the source files.

###Usage
To train a new model:
``` C++
ShapeRegressor regressor;
regressor.Train(images,ground_truth_shapes,bounding_box,first_level_num,second_level_num,
                    candidate_pixel_num,fern_pixel_num,initial_number);
regressor.Save("./data/model.txt");
```
To predict a new input:
``` C++
ShapeRegressor regressor;
regressor.load("./data/model_cofw_2.txt");
regressor.Predict(test_images[index],bounding_box[index],initial_number);
```
For details, please see `TrainDemo.cpp` and `TestDemo.cpp`.

###Sample Results
![Sample Results](https://dl.dropboxusercontent.com/u/47747425/Photo/point1.png)


###Reference papers:
[Face Alignment by Explicit Shape Regression](http://research.microsoft.com/pubs/192097/cvpr12_facealignment.pdf)




