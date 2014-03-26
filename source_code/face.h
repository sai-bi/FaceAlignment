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
        int img_width_;
        int img_height_;
        int pixel_pair_in_fern_; 
        void read(ifstream& fin);
        void write(ofstream& fout);
        void calcuate_normalized_matrix(vector<Mat_<double> >&);
    public:
        ShapeRegressor(const Mat_<double>& mean_shape,
                       const vector<Mat_<uchar> >& images,
                       const vector<Mat_<double> >& target_shapes,
                       const vector<Mat_<double> >& current_shapes,
                       int first_level_num,
                       int second_level_num,
                       int pixel_pair_num,
                       int pixel_pair_in_fern);
        void load(const char* file_name);
        void save(const char* file_name);
        void train();
        void predict(const Mat_uchar>& image, Mat_<double>& shape);
};

class FernCascade{
    private:
        vector<Fern> primary_fern_;
        int second_level_num_;
    public:
        FernCascade();
        void train(const vector<Mat<uchar> >& images,
                const vector<Mat_<double> >& normalize_matrix,
                const vetor<Mat_<double> >& target_shapes,
                const Mat_<double>& mean_shape,
                int second_level_num,
                vector<Mat_<double> >& current_shapes,
                int second_level_num,
                vector<Mat_<double> >& normalized_targets);
        void predict(const Mat_<uchar>& image, Mat_<double>& shape);
        void write(ofstream& fout);
        void read(ifstream& fin);        
};

class Fern{
    private:

    public:
        Fern();
        void train();
        void predict();

};


double calculate_covariance(const vector<double>& v_1, const
        vector<double>& v_2){
    double sum_1 = 0;
    double sum_2 = 0;
    double exp_1 = 0;
    double exp_2 = 0;
    double exp_3 = 0;
    for(int i = 0;i < v_1.size();i++){
        sum_1 += v_1[i];
        sum_2 += v_2[i];
    }
    exp_1 = sum_1 / v_1.size();
    exp_2 = sum_2 / v_2.size();
    for(int i = 0;i < v_1.size();i++){
        exp_3 = exp_3 + (v_1[i] - exp_1) * (v_2[i] - exp_2);
    }
    return exp_3 / v_1.size();
};




#endif
