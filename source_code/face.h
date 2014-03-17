#ifndef FACE_H_
#define FACE_H_

class ShapeRegressor{
    private:
        Mat_<double> mean_shape_;
        vector<Mat_<vec3b> > images_;
        vector<Mat_<double> > current_shapes_;
        vector<Mat_<double> > target_shapes_;
        vector<FernCascade> fern_cascades_;

        ShapeRegressor();
        void load_model(const char* file_name);
        void save_model(const char* file_name);
        void train();
        void predict();
                 
};

class FernCascade{
    private:
        FernCascade();
        void train();
        void predict();
        
};

class Fern{
    private:
        Fern();
        void train();
        void predict();

};


#endif
