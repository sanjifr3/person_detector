#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <image_utils/Utilities.h>
#include <image_utils/ImageProcessor.h>

#include <opencv2/cudaobjdetect.hpp>

#include <dlib/gui_widgets.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include <person_detector/face_recognition.h>
#include <person_detector/faceDetails.h>
#include <person_detector/dlibRect.h>

#include <dynamic_reconfigure/server.h>
#include <person_detector/FaceDetectorRQTConfig.h>

namespace FD
{
  struct detectInfo{
    cv::Rect loc;
    std::vector<int> ctr = {0,0,0};
    cv::Point2f center;
    cv::Scalar color;
    std::string orient;
    double angle;
  };

  struct recInfo{
    std::string name = "Unknown";
    double conf;
    cv::Rect loc;
  };

  struct person{
    person(std::string nm, std::string phone) : name(nm), pronounciation(phone) {}
    
    std::string name, pronounciation;
  };

  struct classifier{
    cv::CascadeClassifier detector;
    double scale_factor = 1.1;
    int min_neighbors = 3;
    cv::Size min_size = cv::Size(50,50);
    cv::Size max_size = cv::Size(140,140);
  };

  enum FACIAL_LANDMARKS_IDX{
    JAW=0,
    RIGHT_EYEBROW=17,
    LEFT_EYEBROW=22,
    NOSE=27,
    RIGHT_EYE=36,
    LEFT_EYE=42,
    MOUTH=48
  };

  struct landmarkInfo{
    landmarkInfo() : jawPts(16), rEyeBrowPts(5), lEyeBrowPts(5), nosePts(9), 
                     rEyePts(6), lEyePts(6), mouthPts(20) {}

    cv::Rect loc;
    std::vector<cv::Point> jawPts, rEyeBrowPts, lEyeBrowPts, nosePts, 
                           rEyePts, lEyePts, mouthPts;
  };
}

class FaceDetector{
  public:
    FaceDetector(ros::NodeHandle nh, std::string f1="", std::string f2="");
    FaceDetector(ros::NodeHandle nh, ros::NodeHandle rqt_nh, std::string f1="", std::string f2="");
    ~FaceDetector();
    
    // Single Detector
    std::vector<cv::Rect> detect(cv::Mat& im, bool modded=false);
    std::vector<cv::Rect> detect(cv::cuda::GpuMat& im, bool modded=false);
       
    // Full Detector 
    void detect(cv::Mat &im, bool modded, std::vector<cv::Rect> &faces, 
                          std::vector<cv::Rect> &Lprofiles, std::vector<cv::Rect> &Rprofiles);
    void detect(cv::cuda::GpuMat& im, bool modded, std::vector<cv::Rect> &faces, 
                          std::vector<cv::Rect> &Lprofiles, std::vector<cv::Rect> &Rprofiles); 
                                                    
    // Merge Detector
    std::vector<FD::detectInfo> mergeDetect(cv::Mat &im, bool modded);
    std::vector<FD::detectInfo> mergeDetect(cv::cuda::GpuMat &im, bool modded);
    void mergeDetect(std::vector<RPS::personInfo> &people, cv::Mat &im, bool modded, bool first_time=true, double scale_size=2.0);
    void mergeDetect(std::vector<RPS::personInfo> &people, cv::cuda::GpuMat &im, bool modded, bool first_time=true, double scale_size=2.0);

    // Mixed
    void mixedDetect(cv::cuda::GpuMat& im, bool modded, std::vector<cv::Rect> &faces, 
                          std::vector<cv::Rect> &Lprofiles, std::vector<cv::Rect> &Rprofiles);
    void mixedDetect2(cv::cuda::GpuMat& im, bool modded, std::vector<cv::Rect> &faces, 
                          std::vector<cv::Rect> &Lprofiles, std::vector<cv::Rect> &Rprofiles);

    // HOG
    void hogDetect(std::vector<RPS::personInfo>& targets, cv::cuda::GpuMat& im);
    std::vector<cv::Rect> hogDetect(cv::cuda::GpuMat& im);                                               
    std::vector<cv::Rect> hogDetect(cv::Mat& im);

    // Dlib
    std::vector<dlib::rectangle> dlibFaceDetect(cv::Mat& im, bool modded=false);
    std::vector<cv::Rect> dlibDetect(cv::Mat& im);
    void dlibDetect(std::vector<RPS::personInfo> &targets, cv::Mat& im, bool first_time=true, double scale_size=2.0);

    void dlibLandmarkDetect(std::vector<RPS::personInfo> &targets, cv::cuda::GpuMat &im, bool first_time=true);
    void dlibLandmarkDetect(std::vector<RPS::personInfo> &targets, cv::Mat &im, bool first_time=true);
    std::vector<FD::landmarkInfo> dlibLandmarkDetect(cv::Mat& im);

    // OpenFace
    void recognize(std::vector<RPS::personInfo> &targets, cv::Mat& im, cv::cuda::GpuMat& gim, bool modded=false);
    void recognize(std::vector<RPS::personInfo> &targets, cv::Mat &im, bool first_time=true);
    std::vector<FD::recInfo> recognize(cv::Mat& im);

    // Update
    void updateDetectors(bool front, bool profile);
    void updateHog();
                         
  private:
    void updatePeopleFound(std::vector<RPS::personInfo>& people, std::vector<FD::detectInfo>& matches);       
    
    // Preprocess images
    void preprocess(cv::Mat& im);
    void preprocess(cv::cuda::GpuMat &im);
    
    // Detect
    std::vector<cv::Rect> detect(FD::classifier& detector, cv::Mat& im);
    cv::cuda::GpuMat detect(cv::Ptr<cv::cuda::CascadeClassifier>& detector, cv::cuda::GpuMat& im);
    
    // Merge Matches
    std::vector<FD::detectInfo> mergeMatches(std::vector<cv::Rect> &faces, 
                          std::vector<cv::Rect> &Lprofiles, std::vector<cv::Rect> &Rprofiles);
    void mergeMatches(std::vector<RPS::personInfo> &targets, std::vector<cv::Rect> &faces, 
                          std::vector<cv::Rect> &Lprofiles, std::vector<cv::Rect> &Rprofiles);    
    void getDirection(FD::detectInfo &target);
    void getDirection(RPS::personInfo &target);

    // Utility
    std::vector<cv::Rect> toRect(cv::cuda::GpuMat &objs);
    void fixFlippedRects(std::vector<cv::Rect>& rects, int im_width);

    // Load cascades
    void init(std::string f);
    void init(std::string f1, std::string f2);

    // Load recognizer data
    void populateFaceRecCtrVectors(std::string filename);

    // Load hog
    void loadHog();
    void loadDlib();
    
    void rqtCb(person_detector::FaceDetectorRQTConfig &config, uint32_t level);
    void loadROSParams(std::string ns=ros::this_node::getName());

  private:
    ros::NodeHandle nh_;

    // Service Clients
    ros::ServiceClient face_recognition_client_;

    // Classes
    ImageProcessor ip_;

    // RQT Reconfigure       
    dynamic_reconfigure::Server<person_detector::FaceDetectorRQTConfig> server_;
    dynamic_reconfigure::Server<person_detector::FaceDetectorRQTConfig>::CallbackType f_;

    // OpenCV
    FD::classifier fcDetector_, pcDetector_;
    cv::Ptr<cv::cuda::CascadeClassifier> fDetector_, pDetector_;
    cv::Ptr<cv::HOGDescriptor> chog_;
    cv::Ptr<cv::cuda::HOG> hog_;
    
    // DLIB
    dlib::frontal_face_detector dlDetector_;
    dlib::shape_predictor sp_;

    // Paths
    std::string package_path_ = "person_detector";
    std::string cascade_path_ = "/models/haarcascades/";
    std::string dlib_path_ = "/models/dlib/";
    
    std::string front_classifier_file_ = "haarcascade_frontalface_alt";
    std::string profile_classifier_file_ = "haarcascade_profileface2";
    std::string sp_file_ = "shape_predictor_68_face_landmarks";
    std::string net_file_ = "dlib_face_recognition_resnet_model_v1";
            
    // Globals
    cv::cuda::GpuMat g_results_;
    cv::Mat temp_im_;

    // Control Parameters
    bool use_gpu_ = true;
    double min_dist_ = 0.0625;

    // HOG Parameters
    cv::Size hog_win_size_ = cv::Size(64,128);
    cv::Size hog_win_stride_ = cv::Size(8,8);
    cv::Size hog_block_size_ = cv::Size(16,16);
    cv::Size hog_block_stride_ = cv::Size(8,8);
    cv::Size hog_cell_size_ = cv::Size(8,8);
    int hog_bins_ = 9;
    int hog_nlvls_ = 13;
    double hog_scale_factor_ = 1.05;
    double hog_hit_thresh_ = 0;

    // Loading Parameters
    bool load_cpu_ = true;
    bool load_gpu_ = true;
    bool hog_loaded_ = false;
    bool dlib_loaded_ = false;

    std::vector<FD::person> all_people_;
};

#endif //FACE_DETECTOR_H
