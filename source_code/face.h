#ifndef FACE_H_
#define FACE_H_


class ShapeRegressor{
    private:
        Mat_<double> mean_shape_;
        vector<Mat_<uchar> > images_;
        vector<Mat_<double> > current_shapes_;
        vector<Mat_<double> > target_shapes_;
        vector<FernCascade> fern_cascades_;
        int first_level_num_;
        int second_level_num_;
        int pixel_pair_num_;
        int training_num_; 
        
    public:
        ShapeRegressor(const Mat_<double>& mean_shape,
                       const vector<Mat_<uchar> >& images,
                       const vector<Mat_<double> >& target_shapes,
                       const vector<Mat_<double> >& current_shapes,
                       int first_level_num,
                       int second_level_num,
                       int pixel_pair_num);
        void load_model(const char* file_name);
        void save_model(const char* file_name);
        void train();
        void predict();
};

class FernCascade{
    private:
        vector<Fern> primary_fern;
    public:
        FernCascade();
        void train();
        void predict();
        
};

class Fern{
    private:

    public:
        Fern();
        void train();
        void predict();

};


#endif
