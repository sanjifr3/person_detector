#include <person_detector/FaceDetector.h>

// Constructor //

FaceDetector::FaceDetector(ros::NodeHandle nh, std::string f1, std::string f2){
  this->nh_ = nh;
  package_path_ = ros::package::getPath(package_path_);
  cascade_path_ = package_path_ + cascade_path_;

  loadROSParams();
  
  init(f1,f2);
  populateFaceRecCtrVectors(package_path_ + "/models/openface/user_mapping.csv");

  face_recognition_client_ = nh_.serviceClient<person_detector::face_recognition>("FaceRecognition");

  if(f1==""&&f2=="") ROS_INFO("[FaceDetector] Initialized with default cascades!");  
  else if(f2=="") ROS_INFO("[FaceDetector] Initialized with %s cascade!",f1);  
  else ROS_INFO("[FaceDetector] Initialized with %s and %s!",f1.c_str(),f2.c_str());
}

FaceDetector::FaceDetector(ros::NodeHandle nh, ros::NodeHandle rqt_nh, std::string f1, std::string f2) : server_(rqt_nh){
  this->nh_ = nh;
  package_path_ = ros::package::getPath(package_path_);
  cascade_path_ = package_path_ + cascade_path_;

  loadROSParams();
  
  init(f1,f2);
  populateFaceRecCtrVectors(package_path_ + "/models/openface/user_mapping.csv");

  face_recognition_client_ = nh_.serviceClient<person_detector::face_recognition>("FaceRecognition");
  
  f_ = boost::bind(&FaceDetector::rqtCb, this, _1, _2);
  server_.setCallback(f_);

  if(f1==""&&f2=="") ROS_INFO("[FaceDetector] Initialized with default cascades!");  
  else if(f2=="") ROS_INFO("[FaceDetector] Initialized with %s cascade!",f1);  
  else ROS_INFO("[FaceDetector] Initialized with %s and %s!",f1.c_str(),f2.c_str());
}

// Destructor //

FaceDetector::~FaceDetector(){
}

// Simple Detect -- only first classifier //

std::vector<cv::Rect> FaceDetector::detect(cv::Mat& im, bool modded){
  std::vector<cv::Rect> results;

  if(!im.empty()){
    if(use_gpu_){
      cv::cuda::GpuMat gpu_im = ip_.toGpuMat(im);
      if (!modded) preprocess(gpu_im);
      g_results_ = detect(fDetector_, gpu_im);
      results = toRect(g_results_);
    }

    else{
      if (!modded) preprocess(im);
      results = detect(fcDetector_, im);
    }
  }

  return results;
}

std::vector<cv::Rect> FaceDetector::detect(cv::cuda::GpuMat& im, bool modded){
  std::vector<cv::Rect> results;

  if(!im.empty()){
    if(use_gpu_){
      if (!modded) preprocess(im);
      g_results_ = detect(fDetector_, im);
      results = toRect(g_results_);
    }

    else{
      cv::Mat cpu_im = ip_.toMat(im);
      if (!modded) preprocess(cpu_im);
      results = detect(fcDetector_, cpu_im);
    }
  }

  return results;
}

// Full Detector w/ Full Results //

void FaceDetector::detect(cv::Mat& im, bool modded, std::vector<cv::Rect> &faces, 
                          std::vector<cv::Rect> &Lprofiles, std::vector<cv::Rect> &Rprofiles){
  if(!im.empty()){
    if(use_gpu_){
      cv::cuda::GpuMat gpu_im = ip_.toGpuMat(im);
      if (!modded) preprocess(gpu_im);
      g_results_ = detect(fDetector_, gpu_im);     faces = toRect(g_results_);
      g_results_ = detect(pDetector_, gpu_im);     Lprofiles = toRect(g_results_);
      ip_.flip(gpu_im);
      g_results_ = detect(pDetector_, gpu_im);     Rprofiles = toRect(g_results_);
      
    }
    else{
      if (!modded) preprocess(im);
      faces = detect(fcDetector_, im);
      Lprofiles = detect(pcDetector_, im);
      cv::flip(im, im, 1);
      Rprofiles = detect(pcDetector_, im);
      //if (modded) cv::flip(im, im, 1);
    }
  }

  fixFlippedRects(Rprofiles, im.cols);
}

void FaceDetector::detect(cv::cuda::GpuMat& im, bool modded, std::vector<cv::Rect> &faces, 
                          std::vector<cv::Rect> &Lprofiles, std::vector<cv::Rect> &Rprofiles){
  if(!im.empty()){
    if(use_gpu_){
      if (!modded) preprocess(im);
      g_results_ = detect(fDetector_, im);     faces = toRect(g_results_);
      g_results_ = detect(pDetector_, im);     Lprofiles = toRect(g_results_);

      ip_.flip(im);
      g_results_ = detect(pDetector_, im);     Rprofiles = toRect(g_results_);
      //if (modded) ip_.flip(im);
    }
    else{
      cv::Mat cpu_im = ip_.toMat(im);
      if (!modded) preprocess(cpu_im);
      faces = detect(fcDetector_, cpu_im);
      Lprofiles = detect(pcDetector_, cpu_im);
      cv::flip(cpu_im, cpu_im, 1);
      Rprofiles = detect(pcDetector_, cpu_im);
    }
  }

  fixFlippedRects(Rprofiles, im.cols);
}

// Full Detector w/ Merged Results //

std::vector<FD::detectInfo> FaceDetector::mergeDetect (cv::Mat &im, bool modded){
  // Correct min_dist to percentage of image
  if(min_dist_ < 1) min_dist_ = min_dist_ * im.cols;
  std::vector<cv::Rect> faces, Lprofiles, Rprofiles;
  detect(im, modded, faces, Lprofiles, Rprofiles);
  return mergeMatches(faces,Lprofiles, Rprofiles);
}

std::vector<FD::detectInfo> FaceDetector::mergeDetect (cv::cuda::GpuMat &im, bool modded){
  // Correct min_dist to percentage of image
  if(min_dist_ < 1) min_dist_ = min_dist_ * im.cols;
  std::vector<cv::Rect> faces, Lprofiles, Rprofiles;
  detect(im, modded, faces, Lprofiles, Rprofiles);
  return mergeMatches(faces,Lprofiles, Rprofiles);
}

void FaceDetector::mergeDetect(std::vector<RPS::personInfo> &people, cv::Mat &im, bool modded, bool first_time, double scale_size){
  if(first_time){
    std::vector<FD::detectInfo> matches = mergeDetect(im, modded);
    updatePeopleFound(people, matches);
  }
  else if(!im.empty()){
    if(use_gpu_){
      preprocess(im);
      for(std::vector<RPS::personInfo>::iterator it = people.begin(); it != people.end(); ++it){
        cv::Rect search_frame = utils::rescale(it->loc, scale_size, scale_size, im.cols, im.rows);

        cv::Mat temp_search_window = im(search_frame);
        cv::cuda::GpuMat search_window = ip_.toGpuMat(temp_search_window);

        std::vector<FD::detectInfo> detections = mergeDetect(search_window, true);

        for(std::vector<FD::detectInfo>::iterator d_it = detections.begin(); d_it != detections.end(); ++d_it){
          it->loc = utils::searchToGlobalFrame(search_frame,d_it->loc);
          it->color = d_it->color;
          it->name = "Unknown";
          it->rec_conf = 0.0;

          it->orient = d_it->orient;
          it->angle = d_it->angle;

          it->d_ctr += std::accumulate(d_it->ctr.begin(), d_it->ctr.end(), 0);
          it->ctr = utils::operator+(it->ctr, d_it->ctr);
        }
      }
    }
    else{
      preprocess(im);
      for(std::vector<RPS::personInfo>::iterator it = people.begin(); it != people.end(); ++it){
        cv::Rect search_frame = utils::rescale(it->loc, scale_size, scale_size, im.cols, im.rows);
        cv::Mat search_window = im(search_window);
        std::vector<FD::detectInfo> detections = mergeDetect(search_window, true);
        for(std::vector<FD::detectInfo>::iterator d_it = detections.begin(); d_it != detections.end(); ++d_it){
          it->loc = utils::searchToGlobalFrame(search_frame,d_it->loc);
          it->color = d_it->color;
          it->name = "Unknown";
          it->rec_conf = 0.0;

          it->orient = d_it->orient;
          it->angle = d_it->angle;

          it->d_ctr += std::accumulate(d_it->ctr.begin(), d_it->ctr.end(), 0);
          it->ctr = utils::operator+(it->ctr, d_it->ctr);
        }        
      }
    }
  }
  return;
}

void FaceDetector::mergeDetect(std::vector<RPS::personInfo> &people, cv::cuda::GpuMat &im, bool modded, bool first_time, double scale_size){
  if(first_time){
    std::vector<FD::detectInfo> matches = mergeDetect(im, modded);
    updatePeopleFound(people, matches);
  }
  else if(!im.empty()){
    if(use_gpu_){
      preprocess(im);
      cv::Mat temp_im = ip_.toMat(im);

      for(std::vector<RPS::personInfo>::iterator it = people.begin(); it != people.end(); ++it){
        cv::Rect search_frame = utils::rescale(it->loc, scale_size, scale_size, im.cols, im.rows);

        cv::Mat temp_search_window = temp_im(search_frame);
        cv::cuda::GpuMat search_window = ip_.toGpuMat(temp_search_window);

        std::vector<FD::detectInfo> detections = mergeDetect(search_window, true);

        for(std::vector<FD::detectInfo>::iterator d_it = detections.begin(); d_it != detections.end(); ++d_it){
          it->loc = utils::searchToGlobalFrame(search_frame,d_it->loc);
          it->color = d_it->color;
          it->name = "Unknown";
          it->rec_conf = 0.0;

          it->orient = d_it->orient;
          it->angle = d_it->angle;

          it->d_ctr += std::accumulate(d_it->ctr.begin(), d_it->ctr.end(), 0);
          it->ctr = utils::operator+(it->ctr, d_it->ctr);
        }
      }
    }
    else{
      cv::Mat cpu_im = ip_.toMat(im);
      preprocess(cpu_im);     
      for(std::vector<RPS::personInfo>::iterator it = people.begin(); it != people.end(); ++it){
        cv::Rect search_frame = utils::rescale(it->loc, scale_size, scale_size, im.cols, im.rows);
        cv::Mat search_window = cpu_im(search_window);
        std::vector<FD::detectInfo> detections = mergeDetect(search_window, true);
        for(std::vector<FD::detectInfo>::iterator d_it = detections.begin(); d_it != detections.end(); ++d_it){
          it->loc = utils::searchToGlobalFrame(search_frame,d_it->loc);
          it->color = d_it->color;
          it->name = "Unknown";
          it->rec_conf = 0.0;

          it->orient = d_it->orient;
          it->angle = d_it->angle;

          it->d_ctr += std::accumulate(d_it->ctr.begin(), d_it->ctr.end(), 0);
          it->ctr = utils::operator+(it->ctr, d_it->ctr);
        }        
      }
    }
  }
  return;
}

void FaceDetector::updatePeopleFound(std::vector<RPS::personInfo>& people, std::vector<FD::detectInfo>& matches){
  for(unsigned int i = 0; i < matches.size(); i++){
    cv::Point2f a_ctr = utils::getCenter(matches[i].loc);
    bool associated = false;
    for(unsigned int j = 0; j < people.size(); j++){
      cv::Point2f m_ctr = utils::getCenter(people[j].loc);
      if(utils::euclideanDistance(m_ctr,a_ctr) < utils::im_dist_tol_){
        associated = true;

        people[j].loc = matches[i].loc;
        people[j].color = matches[i].color;
        people[j].name = "Unknown";
        people[j].rec_conf = 0.0;
        
        people[j].orient = matches[i].orient;
        people[j].angle = matches[i].angle;
        
        people[j].d_ctr+= std::accumulate(matches[i].ctr.begin(), matches[i].ctr.end(), 0);
        people[j].ctr = utils::operator+(people[j].ctr, matches[i].ctr);
      
        break;
      }
    }
    if(!associated){
      RPS::personInfo temp = RPS::personInfo(matches[i].loc, matches[i].color, "Unknown", 0.0);
      temp.orient = matches[i].orient;
      temp.angle = matches[i].angle;
      temp.d_ctr = std::accumulate(matches[i].ctr.begin(), matches[i].ctr.end(), 0);
      temp.ctr = matches[i].ctr;
      people.push_back(temp);
    }
  }
}

// Mixed CPU/GPU Detector //

void FaceDetector::mixedDetect(cv::cuda::GpuMat& im, bool modded, std::vector<cv::Rect> &faces, 
                                std::vector<cv::Rect> &Lprofiles, std::vector<cv::Rect> &Rprofiles){
  if(!im.empty()){
    if (!modded) preprocess(im);
    g_results_ = detect(fDetector_, im);     faces = toRect(g_results_);

    temp_im_ = ip_.toMat(im);
    Lprofiles = detect(pcDetector_, temp_im_);

    cv::flip(temp_im_, temp_im_, 1);
    Rprofiles = detect(pcDetector_, temp_im_);
    fixFlippedRects(Rprofiles, im.cols);
  }
}

void FaceDetector::mixedDetect2(cv::cuda::GpuMat& im, bool modded, std::vector<cv::Rect> &faces, 
                                std::vector<cv::Rect> &Lprofiles, std::vector<cv::Rect> &Rprofiles){
  if(!im.empty()){
    if (!modded) preprocess(im);
    g_results_ = detect(fDetector_, im);     faces = toRect(g_results_);

    temp_im_ = ip_.toMat(im);
    Lprofiles = detect(pcDetector_, temp_im_);

    ip_.flip(im);
    temp_im_ = ip_.toMat(im);
    Rprofiles = detect(pcDetector_, temp_im_);
    fixFlippedRects(Rprofiles, im.cols);
  }
}

void FaceDetector::hogDetect(std::vector<RPS::personInfo>& targets, cv::cuda::GpuMat& im){
  if(!hog_loaded_) loadHog();
  std::vector<cv::Rect> people;

  if(!im.empty()){
    if(use_gpu_){
      ip_.convertTo(im);
      hog_ -> detectMultiScale(im, people);
    }
    else{
      cv::Mat cpu_im = ip_.toMat(im);      
      cv::cvtColor(cpu_im, cpu_im, CV_BGR2GRAY);
      chog_ -> detectMultiScale(cpu_im, people, hog_hit_thresh_, hog_win_stride_, cv::Size(0,0),
        hog_scale_factor_);
    }

    utils::updateStoredPpl(targets, people);
  }
}

std::vector<cv::Rect> FaceDetector::hogDetect(cv::cuda::GpuMat& im){
  if (!hog_loaded_) loadHog();
  std::vector<cv::Rect> targets;

  if (!im.empty()){
    if(use_gpu_){
      ip_.convertTo(im);
      hog_ -> detectMultiScale(im, targets);
    }
    else{
      cv::Mat cpu_im = ip_.toMat(im);      
      cv::cvtColor(cpu_im, cpu_im, CV_BGR2GRAY);
      chog_ -> detectMultiScale(cpu_im, targets, hog_hit_thresh_, hog_win_stride_, cv::Size(0,0),
        hog_scale_factor_);
    }
  }

  return targets;
}

std::vector<cv::Rect> FaceDetector::hogDetect(cv::Mat& im){
  if (!hog_loaded_) loadHog();
  std::vector<cv::Rect> targets;

  if (!im.empty()){
    if(use_gpu_){
      cv::cuda::GpuMat gpu_im = ip_.toGpuMat(im);
      ip_.convertTo(gpu_im);
      hog_ -> detectMultiScale(gpu_im, targets);
    }
    else{
      cv::cvtColor(im, im, CV_BGR2GRAY);
      chog_ -> detectMultiScale(im, targets, hog_hit_thresh_, hog_win_stride_, cv::Size(0,0),
        hog_scale_factor_);
    }
  }

  return targets;
}

std::vector<dlib::rectangle> FaceDetector::dlibFaceDetect(cv::Mat& im, bool modded){
  if (!dlib_loaded_) loadDlib();

  if(!im.empty()){
    if(!modded)
      cv::cvtColor(im, im, CV_BGR2GRAY);
    dlib::array2d<uchar> dlib_im = ip_.toDlib(im);
    return dlDetector_(dlib_im);
  }
  return std::vector<dlib::rectangle>(0);
}

std::vector<cv::Rect> FaceDetector::dlibDetect(cv::Mat& im){
  std::vector<dlib::rectangle> dlib_results = dlibFaceDetect(im);
  std::vector<cv::Rect> results (dlib_results.size());

  for(unsigned int i = 0; i < dlib_results.size(); i++)
    results[i] = ip_.toCvRect(dlib_results[i]);

  return results;
}

void FaceDetector::dlibDetect(std::vector<RPS::personInfo> &targets, cv::Mat& im, bool first_time, double scale_size){
  if(first_time){
    std::vector<cv::Rect> faces = dlibDetect(im);
    utils::updateStoredPpl(targets, faces, CV_RGB(125,125,125), "Unknown", 0.0);
  }
  else{
    if(im.empty() || targets.size() == 0) return;

    cv::Rect search_frame = cv::Rect(5000,5000,-5000,-5000);
    std::vector<RPS::personInfo>::iterator it;
    for(it = targets.begin(); it != targets.end(); ++it){
      cv::Rect loc = utils::rescale(it->loc, scale_size, scale_size, im.cols, im.rows);

      search_frame.x = std::min(loc.x, search_frame.x);
      search_frame.y = std::min(loc.y, search_frame.y);

      // Using width and height parameter to store max x, and y for now
        // correct after for loop

      search_frame.width = std::max(loc.x + loc.width, search_frame.width);
      search_frame.height = std::max(loc.y + loc.height, search_frame.height);
    }

    search_frame.width = search_frame.width - search_frame.x;
    search_frame.height = search_frame.height - search_frame.y;

    im = im(search_frame);
    std::vector<dlib::rectangle> dlib_results = dlibFaceDetect(im);

    for(unsigned int i = 0; i < dlib_results.size(); i++){
      cv::Rect face = ip_.toCvRect(dlib_results[i]);
      cv::Point2f face_center = utils::getCenter(face);
      for(unsigned int j = 0; j < targets.size(); j++){
        cv::Point2f head_center = utils::getCenter(targets[j].loc);
        if(utils::euclideanDistance(face_center, head_center) < utils::im_dist_tol_){
          targets[j].loc = face;
          targets[j].name = "Unknown";
          targets[j].rec_conf = 0.0;
          targets[j].color = CV_RGB(125,125,125);
          targets[j].d_ctr++;
        }
      }
    }
  }

  return;
}

void FaceDetector::dlibLandmarkDetect(std::vector<RPS::personInfo> &targets, cv::cuda::GpuMat& im, bool first_time){
  if (!dlib_loaded_) loadDlib();

  preprocess(im);
  cv::Mat gray_im = ip_.toMat(im);
  
  std::vector<FD::detectInfo> matches;
  if(first_time){
    matches = mergeDetect(im, true);
    if(matches.size() == 0)
      return;
  }

  dlib::array2d<uchar> dlib_im = ip_.toDlib(gray_im);

  if(!first_time){
    for(std::vector<RPS::personInfo>::iterator it = targets.begin(); it != targets.end(); ++it){
      if(it->name == "Unknown"){
        dlib::full_object_detection shapes = sp_(dlib_im, ip_.toDlibRect(it->loc));

        FD::landmarkInfo ld_info;
        int idx = 0;
        for(unsigned int k = 0; k < shapes.num_parts(); k++){
          if (k < FD::RIGHT_EYEBROW)
            ld_info.jawPts[k-FD::JAW] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
          else if (k < FD::LEFT_EYEBROW)
            ld_info.rEyeBrowPts[k-FD::RIGHT_EYEBROW] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
          else if (k < FD::NOSE)
            ld_info.lEyeBrowPts[k-FD::LEFT_EYEBROW] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
          else if (k < FD::RIGHT_EYE)
            ld_info.nosePts[k-FD::NOSE] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
          else if (k < FD::LEFT_EYE)
            ld_info.rEyePts[k-FD::RIGHT_EYE] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
          else if (k < FD::MOUTH)
            ld_info.lEyePts[k-FD::LEFT_EYE] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
          else
            ld_info.mouthPts[k-FD::MOUTH] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
          idx++;
        }

        it->jawPts = ld_info.jawPts;
        it->rEyeBrowPts = ld_info.rEyeBrowPts;
        it->lEyeBrowPts = ld_info.lEyeBrowPts;
        it->nosePts = ld_info.nosePts;
        it->rEyePts = ld_info.rEyePts;
        it->lEyePts = ld_info.lEyePts;
        it->mouthPts = ld_info.mouthPts;
      }
    }
  }

  else{
    for(unsigned int i = 0; i < matches.size(); i++){
      dlib::full_object_detection shapes = sp_(dlib_im, ip_.toDlibRect(matches[i].loc));

      cv::Point2f center = utils::getCenter(matches[i].loc);
      bool associated = false;
      for(unsigned int j = 0; j < targets.size(); j++){
        cv::Point2f old_center = utils::getCenter(targets[j].loc);
        if(utils::euclideanDistance(center, old_center) < utils::im_dist_tol_){
          associated = true;
          targets[j].loc = matches[i].loc;

          targets[j].color = matches[i].color;
          targets[j].orient = matches[i].orient;
          targets[j].angle = matches[i].angle;

          targets[j].d_ctr += std::accumulate(matches[i].ctr.begin(), matches[i].ctr.end(), 0);
          targets[j].ctr = utils::operator+(targets[j].ctr, matches[i].ctr);

          FD::landmarkInfo ld_info;
          int idx = 0;
          for(unsigned int k = 0; k < shapes.num_parts(); k++){
            if (k < FD::RIGHT_EYEBROW)
              ld_info.jawPts[k-FD::JAW] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
            else if (k < FD::LEFT_EYEBROW)
              ld_info.rEyeBrowPts[k-FD::RIGHT_EYEBROW] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
            else if (k < FD::NOSE)
              ld_info.lEyeBrowPts[k-FD::LEFT_EYEBROW] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
            else if (k < FD::RIGHT_EYE)
              ld_info.nosePts[k-FD::NOSE] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
            else if (k < FD::LEFT_EYE)
              ld_info.rEyePts[k-FD::RIGHT_EYE] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
            else if (k < FD::MOUTH)
              ld_info.lEyePts[k-FD::LEFT_EYE] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
            else
              ld_info.mouthPts[k-FD::MOUTH] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
            idx++;
          }

          targets[j].jawPts = ld_info.jawPts;
          targets[j].rEyeBrowPts = ld_info.rEyeBrowPts;
          targets[j].lEyeBrowPts = ld_info.lEyeBrowPts;
          targets[j].nosePts = ld_info.nosePts;
          targets[j].rEyePts = ld_info.rEyePts;
          targets[j].lEyePts = ld_info.lEyePts;
          targets[j].mouthPts = ld_info.mouthPts;

          break;
        }
      }

      if(!associated){
        RPS::personInfo temp;

        temp.name = "Unknown";
        temp.rec_conf = 0.0;
        temp.loc = matches[i].loc;

        temp.color = matches[i].color;
        temp.orient = matches[i].orient;
        temp.angle = matches[i].angle;

        temp.d_ctr = std::accumulate(matches[i].ctr.begin(), matches[i].ctr.end(), 0);
        temp.ctr = matches[i].ctr;

        FD::landmarkInfo ld_info;
        int idx = 0;
        for(unsigned int k = 0; k < shapes.num_parts(); k++){
          if (k < FD::RIGHT_EYEBROW)
            ld_info.jawPts[k-FD::JAW] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
          else if (k < FD::LEFT_EYEBROW)
            ld_info.rEyeBrowPts[k-FD::RIGHT_EYEBROW] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
          else if (k < FD::NOSE)
            ld_info.lEyeBrowPts[k-FD::LEFT_EYEBROW] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
          else if (k < FD::RIGHT_EYE)
            ld_info.nosePts[k-FD::NOSE] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
          else if (k < FD::LEFT_EYE)
            ld_info.rEyePts[k-FD::RIGHT_EYE] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
          else if (k < FD::MOUTH)
            ld_info.lEyePts[k-FD::LEFT_EYE] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
          else
            ld_info.mouthPts[k-FD::MOUTH] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
          
          idx++;
        }

        temp.jawPts = ld_info.jawPts;
        temp.rEyeBrowPts = ld_info.rEyeBrowPts;
        temp.lEyeBrowPts = ld_info.lEyeBrowPts;
        temp.nosePts = ld_info.nosePts;
        temp.rEyePts = ld_info.rEyePts;
        temp.lEyePts = ld_info.lEyePts;
        temp.mouthPts = ld_info.mouthPts;
        targets.push_back(temp);
      }
    }
  }
}

void FaceDetector::dlibLandmarkDetect(std::vector<RPS::personInfo> &targets, cv::Mat& im, bool first_time){
  if (!dlib_loaded_) loadDlib();

  if(!im.empty()){
    cv::cvtColor(im, im, CV_BGR2GRAY);
    dlib::array2d<uchar> dlib_im = ip_.toDlib(im);

    std::vector<dlib::rectangle> dlib_results;
    if(first_time)
      dlib_results = dlDetector_(dlib_im);

    if(first_time){
      for(std::vector<RPS::personInfo>::iterator it = targets.begin(); it != targets.end(); ++it){
        if(it->name == "Unknown"){
          dlib::full_object_detection shapes = sp_(dlib_im, ip_.toDlibRect(it->loc));

          FD::landmarkInfo ld_info;
          int idx = 0;
          for(unsigned int k = 0; k < shapes.num_parts(); k++){
            if (k < FD::RIGHT_EYEBROW)
              ld_info.jawPts[k-FD::JAW] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
            else if (k < FD::LEFT_EYEBROW)
              ld_info.rEyeBrowPts[k-FD::RIGHT_EYEBROW] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
            else if (k < FD::NOSE)
              ld_info.lEyeBrowPts[k-FD::LEFT_EYEBROW] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
            else if (k < FD::RIGHT_EYE)
              ld_info.nosePts[k-FD::NOSE] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
            else if (k < FD::LEFT_EYE)
              ld_info.rEyePts[k-FD::RIGHT_EYE] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
            else if (k < FD::MOUTH)
              ld_info.lEyePts[k-FD::LEFT_EYE] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
            else
              ld_info.mouthPts[k-FD::MOUTH] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
            idx++;
          }

          it->jawPts = ld_info.jawPts;
          it->rEyeBrowPts = ld_info.rEyeBrowPts;
          it->lEyeBrowPts = ld_info.lEyeBrowPts;
          it->nosePts = ld_info.nosePts;
          it->rEyePts = ld_info.rEyePts;
          it->lEyePts = ld_info.lEyePts;
          it->mouthPts = ld_info.mouthPts;
        }
      }
    }

    else{
      for(unsigned int i = 0; i < dlib_results.size(); i++){
        dlib::full_object_detection shapes = sp_(dlib_im, dlib_results[i]);

        cv::Rect loc = ip_.toCvRect(dlib_results[i]);
        cv::Point2f center = utils::getCenter(loc);
        
        bool associated = false;
        for(unsigned int j = 0; j < targets.size(); j++){
          cv::Point2f old_center = utils::getCenter(targets[j].loc);
          if(utils::euclideanDistance(center, old_center) < utils::im_dist_tol_){
            associated = true;
            targets[j].loc = loc;
            FD::landmarkInfo ld_info;
            int idx = 0;
            for(unsigned int k = 0; k < shapes.num_parts(); k++){
              if (k < FD::RIGHT_EYEBROW)
                ld_info.jawPts[k-FD::JAW] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
              else if (k < FD::LEFT_EYEBROW)
                ld_info.rEyeBrowPts[k-FD::RIGHT_EYEBROW] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
              else if (k < FD::NOSE)
                ld_info.lEyeBrowPts[k-FD::LEFT_EYEBROW] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
              else if (k < FD::RIGHT_EYE)
                ld_info.nosePts[k-FD::NOSE] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
              else if (k < FD::LEFT_EYE)
                ld_info.rEyePts[k-FD::RIGHT_EYE] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
              else if (k < FD::MOUTH)
                ld_info.lEyePts[k-FD::LEFT_EYE] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
              else
                ld_info.mouthPts[k-FD::MOUTH] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
              idx++;
            }

            targets[j].jawPts = ld_info.jawPts;
            targets[j].rEyeBrowPts = ld_info.rEyeBrowPts;
            targets[j].lEyeBrowPts = ld_info.lEyeBrowPts;
            targets[j].nosePts = ld_info.nosePts;
            targets[j].rEyePts = ld_info.rEyePts;
            targets[j].lEyePts = ld_info.lEyePts;
            targets[j].mouthPts = ld_info.mouthPts;

            break;
          }
        }

        if(!associated){
          RPS::personInfo temp;
          temp.color = CV_RGB(125,125,125);
          temp.name = "Unknown";
          temp.rec_conf = 0.0;
          temp.loc = loc;

          FD::landmarkInfo ld_info;
          int idx = 0;
          for(unsigned int k = 0; k < shapes.num_parts(); k++){
            if (k < FD::RIGHT_EYEBROW)
              ld_info.jawPts[k-FD::JAW] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
            else if (k < FD::LEFT_EYEBROW)
              ld_info.rEyeBrowPts[k-FD::RIGHT_EYEBROW] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
            else if (k < FD::NOSE)
              ld_info.lEyeBrowPts[k-FD::LEFT_EYEBROW] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
            else if (k < FD::RIGHT_EYE)
              ld_info.nosePts[k-FD::NOSE] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
            else if (k < FD::LEFT_EYE)
              ld_info.rEyePts[k-FD::RIGHT_EYE] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
            else if (k < FD::MOUTH)
              ld_info.lEyePts[k-FD::LEFT_EYE] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
            else
              ld_info.mouthPts[k-FD::MOUTH] = cv::Point(shapes.part(k).x(), shapes.part(k).y());
            
            idx++;
          }

          temp.jawPts = ld_info.jawPts;
          temp.rEyeBrowPts = ld_info.rEyeBrowPts;
          temp.lEyeBrowPts = ld_info.lEyeBrowPts;
          temp.nosePts = ld_info.nosePts;
          temp.rEyePts = ld_info.rEyePts;
          temp.lEyePts = ld_info.lEyePts;
          temp.mouthPts = ld_info.mouthPts;
          targets.push_back(temp);
        }
      }
    }    
  }
}

std::vector<FD::landmarkInfo> FaceDetector::dlibLandmarkDetect(cv::Mat& im){
  if (!dlib_loaded_) loadDlib();
  std::vector<FD::landmarkInfo> results;

  if (!im.empty()){
    cv::cvtColor(im, im, CV_BGR2GRAY);
    dlib::array2d<uchar> dlib_im = ip_.toDlib(im);
    std::vector<dlib::rectangle> dlib_results = dlDetector_(dlib_im);

    results.resize(dlib_results.size());
    std::vector<dlib::full_object_detection> shapes (dlib_results.size());

    // Get pose of each face we detect
    for(unsigned int i = 0; i < dlib_results.size(); i++){
      shapes[i] = sp_(dlib_im, dlib_results[i]);

      // Store bounding box of face
      results[i].loc = ip_.toCvRect(dlib_results[i]);

      int idx = 0;
      for(unsigned int j = 0; j < shapes[i].num_parts(); j++){
        if (j < FD::RIGHT_EYEBROW)
          results[i].jawPts[j-FD::JAW] = cv::Point(shapes[i].part(j).x(), shapes[i].part(j).y());
        else if (j < FD::LEFT_EYEBROW)
          results[i].rEyeBrowPts[j-FD::RIGHT_EYEBROW] = cv::Point(shapes[i].part(j).x(), shapes[i].part(j).y());
        else if (j < FD::NOSE)
          results[i].lEyeBrowPts[j-FD::LEFT_EYEBROW] = cv::Point(shapes[i].part(j).x(), shapes[i].part(j).y());
        else if (j < FD::RIGHT_EYE)
          results[i].nosePts[j-FD::NOSE] = cv::Point(shapes[i].part(j).x(), shapes[i].part(j).y());
        else if (j < FD::LEFT_EYE)
          results[i].rEyePts[j-FD::RIGHT_EYE] = cv::Point(shapes[i].part(j).x(), shapes[i].part(j).y());
        else if (j < FD::MOUTH)
          results[i].lEyePts[j-FD::LEFT_EYE] = cv::Point(shapes[i].part(j).x(), shapes[i].part(j).y());
        else
          results[i].mouthPts[j-FD::MOUTH] = cv::Point(shapes[i].part(j).x(), shapes[i].part(j).y());
        
        idx++;
      }
    }
  }

  return results;
}

void FaceDetector::recognize(std::vector<RPS::personInfo> &targets, cv::Mat& im, cv::cuda::GpuMat& gim, bool modded){
  std::vector<FD::detectInfo> matches = mergeDetect(im, modded);
  if (matches.size() == 0) 
    return;

  person_detector::face_recognition face_rec_msg;
  face_rec_msg.request.rects.resize(matches.size());

  cv::Rect search_frame = cv::Rect(1920,1080,-1920,-1080);
  for(unsigned int i = 0; i < matches.size(); i++){

    search_frame.x = std::min(matches[i].loc.x, search_frame.x);
    search_frame.y = std::min(matches[i].loc.y, search_frame.y);

    // Using width and height parameter to store max x, and y for now
      // correct after for loop
    search_frame.width = std::max(matches[i].loc.x + matches[i].loc.width, search_frame.width);
    search_frame.height = std::max(matches[i].loc.y + matches[i].loc.height, search_frame.height);
  }

  search_frame.width = search_frame.width - search_frame.x;
  search_frame.height = search_frame.height - search_frame.y;

  for(unsigned int i = 0; i < matches.size(); i++){
    cv::Rect loc = matches[i].loc;
    loc.x = loc.x - search_frame.x;
    loc.y = loc.y - search_frame.y;

    dlib::rectangle rect = ip_.toDlibRect(loc);

    face_rec_msg.request.rects[i].l = rect.left();
    face_rec_msg.request.rects[i].t = rect.top();
    face_rec_msg.request.rects[i].r = rect.right();
    face_rec_msg.request.rects[i].b = rect.bottom();
  }

  if(!im.empty()){
    cv::Mat search_window = im(search_frame);
    sensor_msgs::Image image_msg;
    cv_bridge::CvImage(std_msgs::Header(), "bgr8", search_window).toImageMsg(image_msg);

    face_rec_msg.request.im = image_msg;

    ros::service::waitForService("FaceRecognition");
    face_recognition_client_.call(face_rec_msg);

    for(unsigned int i = 0; i < face_rec_msg.response.person.size(); i++){
      std::string *name = &face_rec_msg.response.person[i].name;
      float *conf = &face_rec_msg.response.person[i].confidence;

      cv::Point2f center = utils::getCenter(matches[i].loc);
      bool associated = false;
      for(unsigned int j = 0; j < targets.size(); j++){
        cv::Point2f old_center = utils::getCenter(targets[j].loc);
        if(utils::euclideanDistance(center, old_center) < utils::im_dist_tol_){
          associated = true;

          targets[j].loc = matches[i].loc;
          targets[j].color = matches[i].color;
          
          targets[j].orient = matches[i].orient;
          targets[j].angle = matches[i].angle;
          
          targets[j].d_ctr+= std::accumulate(matches[i].ctr.begin(), matches[i].ctr.end(), 0);
          targets[j].ctr = utils::operator+(targets[j].ctr, matches[i].ctr);

          if (*conf > targets[j].rec_conf){
            targets[j].rec_conf = *conf;
            targets[j].name = *name;
          }
        }
      }

      if(!associated){
        RPS::personInfo temp;
        temp.loc = matches[i].loc;
        temp.color = matches[i].color;
        temp.orient = matches[i].orient;
        temp.angle = matches[i].angle;
        temp.d_ctr = std::accumulate(matches[i].ctr.begin(), matches[i].ctr.end(), 0);
        temp.ctr = matches[i].ctr;
        temp.name = *name;
        temp.rec_conf = *conf;
        targets.push_back(temp);
      }
    }
  }

  return;
}

void FaceDetector::recognize(std::vector<RPS::personInfo> &targets, cv::Mat& im, bool first_time){
  if (!im.empty() && first_time){
    sensor_msgs::Image image_msg;
    cv_bridge::CvImage(std_msgs::Header(), "bgr8", im).toImageMsg(image_msg);

    person_detector::face_recognition face_rec_msg;
    face_rec_msg.request.im = image_msg;

    ros::service::waitForService("FaceRecognition");
    face_recognition_client_.call(face_rec_msg);

    for(unsigned int i = 0; i < face_rec_msg.response.person.size(); i++){
      int left = face_rec_msg.response.person[i].left;
      int right = face_rec_msg.response.person[i].right;
      int top = face_rec_msg.response.person[i].top;
      int bottom = face_rec_msg.response.person[i].bottom;

      cv::Rect loc = cv::Rect(cv::Point2d(left,top), cv::Point2d(right,bottom));
      cv::Point2f center = utils::getCenter(loc);

      std::string *name = &face_rec_msg.response.person[i].name;
      float *conf = &face_rec_msg.response.person[i].confidence;

      bool associated = false;
      for(unsigned int j = 0; j < targets.size(); j++){
        cv::Point2f old_center = utils::getCenter(targets[j].loc);
        if(utils::euclideanDistance(center, old_center) < utils::im_dist_tol_){
          associated = true;
          targets[j].d_ctr ++;
          targets[j].loc = loc;
          if (*conf > targets[j].rec_conf){
            targets[j].rec_conf = *conf;
            targets[j].name = *name;
          }
        }
      }
      if(!associated)
        targets.push_back(RPS::personInfo(loc,CV_RGB(125,125,125),*name,*conf));
    }
  }
  else if (!im.empty() && !first_time){
    cv::Rect search_frame = cv::Rect(5000,5000,-5000,-5000);
    person_detector::face_recognition face_rec_msg;
    std::vector<int> id_map;

    for(unsigned int i = 0; i < targets.size(); i++){
      if(targets[i].name == "Unknown"){
        search_frame.x = std::min(targets[i].loc.x, search_frame.x);
        search_frame.y = std::min(targets[i].loc.y, search_frame.y);

        // Using width and height parameter to store max x, and y for now
          // correct after for loop

        search_frame.width = std::max(targets[i].loc.x + targets[i].loc.width, search_frame.width);
        search_frame.height = std::max(targets[i].loc.y + targets[i].loc.height, search_frame.height);

        dlib::rectangle dlib_rect = ip_.toDlibRect(targets[i].loc);
        person_detector::dlibRect rect;

        rect.l = dlib_rect.left();
        rect.t = dlib_rect.top();
        rect.r = dlib_rect.right();
        rect.b = dlib_rect.bottom();
        face_rec_msg.request.rects.push_back(rect);
        id_map.push_back(i);
      }
    }

    search_frame.width = search_frame.width - search_frame.x;
    search_frame.height = search_frame.height - search_frame.y;
    
    for(unsigned int i = 0; i < face_rec_msg.request.rects.size(); i++){
      face_rec_msg.request.rects[i].l -= search_frame.x;
      face_rec_msg.request.rects[i].t -= search_frame.y;
      face_rec_msg.request.rects[i].r -= search_frame.x;
      face_rec_msg.request.rects[i].b -= search_frame.y;
    }

    if(face_rec_msg.request.rects.size() > 0){
      cv::Mat search_window = im(search_frame);
      sensor_msgs::Image image_msg;
      cv_bridge::CvImage(std_msgs::Header(), "bgr8", search_window).toImageMsg(image_msg);

      face_rec_msg.request.im = image_msg;

      ros::service::waitForService("FaceRecognition");
      face_recognition_client_.call(face_rec_msg);

      for(unsigned int i = 0; i < face_rec_msg.response.person.size(); i++){
        std::string *name = &face_rec_msg.response.person[i].name;
        float *conf = &face_rec_msg.response.person[i].confidence;

        if (*conf > targets[id_map[i]].rec_conf){
          targets[id_map[i]].rec_conf = *conf;
          targets[id_map[i]].name = *name;
        }
      }
    }
  }

  return;
}
/*
person_detector::face_recognition face_rec_msg;
  face_rec_msg.request.rects.resize(matches.size());

  cv::Rect search_frame = cv::Rect(640,480,-640,-480);
  for(unsigned int i = 0; i < matches.size(); i++){
    if ( matches[i].loc.x < search_frame.x)
      search_frame.x = matches[i].loc.x;
    if ( matches[i].loc.y < search_frame.y)
      search_frame.y = matches[i].loc.y;
    if ( matches[i].loc.x + matches[i].loc.width > search_frame.x + search_frame.width)
      search_frame.width = (matches[i].loc.x + matches[i].loc.width) - search_frame.x;
    if ( matches[i].loc.y + matches[i].loc.height > search_frame.y + search_frame.height)
      search_frame.height = (matches[i].loc.y + matches[i].loc.height) - search_frame.y;
  }

  
  for(unsigned int i = 0; i < matches.size(); i++){
    cv::Rect loc = matches[i].loc;
    loc.x = loc.x - search_frame.x;
    loc.y = loc.y - search_frame.y;

    dlib::rectangle rect = ip_.toDlibRect(loc);

    face_rec_msg.request.rects[i].l = rect.left();
    face_rec_msg.request.rects[i].t = rect.top();
    face_rec_msg.request.rects[i].r = rect.right();
    face_rec_msg.request.rects[i].b = rect.bottom();
  }

  if(!im.empty()){
    cv::Mat search_window = im(search_frame);
    sensor_msgs::Image image_msg;
    cv_bridge::CvImage(std_msgs::Header(), "bgr8", search_window).toImageMsg(image_msg);

    face_rec_msg.request.im = image_msg;

    ros::service::waitForService("FaceRecognition");
    face_recognition_client_.call(face_rec_msg);

    for(unsigned int i = 0; i < face_rec_msg.response.person.size(); i++){
      std::string *name = &face_rec_msg.response.person[i].name;
      float *conf = &face_rec_msg.response.person[i].confidence;

      cv::Point2f center = utils::getCenter(matches[i].loc);
      bool associated = false;
      for(unsigned int j = 0; j < targets.size(); j++){
        cv::Point2f old_center = utils::getCenter(targets[j].loc);
        if(utils::euclideanDistance(center, old_center) < utils::im_dist_tol_){
          associated = true;

          targets[j].loc = matches[i].loc;
          targets[j].color = matches[i].color;
          
          targets[j].orient = matches[i].orient;
          targets[j].angle = matches[i].angle;
          
          targets[j].d_ctr+= std::accumulate(matches[i].ctr.begin(), matches[i].ctr.end(), 0);
          targets[j].ctr = utils::operator+(targets[j].ctr, matches[i].ctr);

          if (*conf > targets[j].rec_conf){
            targets[j].rec_conf = *conf;
            targets[j].name = *name;
          }
        }
      }

      if(!associated){
        RPS::personInfo temp;
        temp.loc = matches[i].loc;
        temp.color = matches[i].color;
        temp.orient = matches[i].orient;
        temp.angle = matches[i].angle;
        temp.d_ctr = std::accumulate(matches[i].ctr.begin(), matches[i].ctr.end(), 0);
        temp.ctr = matches[i].ctr;
        temp.name = *name;
        temp.rec_conf = *conf;
        targets.push_back(temp);
      }
    }
  }
  */

std::vector<FD::recInfo> FaceDetector::recognize(cv::Mat& im){
  std::vector<FD::recInfo> targets;
  if (!im.empty()){
    sensor_msgs::Image image_msg;
    cv_bridge::CvImage(std_msgs::Header(), "bgr8", im).toImageMsg(image_msg);

    person_detector::face_recognition face_rec_msg;
    face_rec_msg.request.im = image_msg;

    ros::service::waitForService("FaceRecognition");
    face_recognition_client_.call(face_rec_msg);

    targets.resize(face_rec_msg.response.person.size());

    for(unsigned int i = 0; i < face_rec_msg.response.person.size(); i++){
      int left = face_rec_msg.response.person[i].left;
      int right = face_rec_msg.response.person[i].right;
      int top = face_rec_msg.response.person[i].top;
      int bottom = face_rec_msg.response.person[i].bottom;

      targets[i].name = face_rec_msg.response.person[i].name;
      targets[i].conf = face_rec_msg.response.person[i].confidence;
      targets[i].loc = cv::Rect(cv::Point2d(left,top), cv::Point2d(right,bottom));
    }
  }

  return targets;
}

// Preprocessing //

// CPU Preprocess
void FaceDetector::preprocess(cv::Mat& im){
  if (im.channels() == 3){
    cv::cvtColor(im, im, CV_BGR2GRAY);
    cv::equalizeHist(im, im);
  }
}

// GPU Preprocess
void FaceDetector::preprocess(cv::cuda::GpuMat& im){
  if (im.channels() == 3){
    ip_.convertTo(im);
    ip_.equalizeHist(im);
  }
}

// Detect //

// CPU Detect
std::vector<cv::Rect> FaceDetector::detect(FD::classifier& detector, cv::Mat& im){
  std::vector<cv::Rect> results;
  detector.detector.detectMultiScale(im, results, detector.scale_factor, detector.min_neighbors, 
                                      0, detector.min_size, detector.max_size);
  return results;
}

// GPU Detect
cv::cuda::GpuMat FaceDetector::detect(cv::Ptr<cv::cuda::CascadeClassifier>& detector, cv::cuda::GpuMat& im){
  cv::cuda::GpuMat results;
  detector->detectMultiScale(im, results);
  return results;
}

// Merge Utility Functions //
std::vector<FD::detectInfo> FaceDetector::mergeMatches(std::vector<cv::Rect> &faces, 
                          std::vector<cv::Rect> &Lprofiles, std::vector<cv::Rect> &Rprofiles){

  std::vector<FD::detectInfo> targets;
  std::vector<cv::Rect> *type;

  for (int type_itter = 0; type_itter < 3; type_itter++){
    if (type_itter == 0) type = &Lprofiles;
    else if (type_itter == 1) type = &faces;
    else if (type_itter == 2) type = &Rprofiles;

    for(unsigned int j = 0; j < type->size(); j++){
      bool associated = false;
      cv::Point2f center = utils::getCenter((*type)[j]);

      for(unsigned int k = 0; k < targets.size(); k++){
        if (utils::euclideanDistance(center, targets[k].center) < min_dist_){
          associated = true;
          if (type_itter == 1) targets[k].loc = (*type)[j]; // Update location if center
          targets[k].ctr[type_itter]++;
          break;
        }
      }

      if(!associated){
        FD::detectInfo new_target;
        new_target.loc = (*type)[j];
        new_target.ctr[type_itter] = 1;
        new_target.center = center;
        targets.push_back(new_target);
      }
    }
  }

  for(unsigned int i = 0; i < targets.size(); i++)
    getDirection(targets[i]);

  return targets;
}

void FaceDetector::getDirection(FD::detectInfo &target){
  cv::Scalar color;
  int *f_ctr = &target.ctr[1];
  int *l_ctr = &target.ctr[0];
  int *r_ctr = &target.ctr[2];

  if ( *f_ctr > 0 && *l_ctr > 0 && *r_ctr > 0 ){
    target.orient = "F";
    target.angle = 0;
    target.color = CV_RGB(0,255,0);
  }
  else if ( *f_ctr > 0 && *l_ctr > 0 ){
    target.orient = "FL";
    target.angle = -30;
    target.color = CV_RGB(125,125,0);
  } 
  else if ( *f_ctr > 0 && *r_ctr > 0 ){
    target.orient = "FR";
    target.angle = 30;
    target.color = CV_RGB(0,125,125);
  } 
  else if ( *l_ctr > 0 && *r_ctr > 0 ){
    target.orient = "F";
    target.angle = 0;
    target.color = CV_RGB(0,255,0);
  }
  else if ( *f_ctr > 0 ){
    target.orient = "F";
    target.angle = 0;
    target.color = CV_RGB(0,255,0);
  }
  else if ( *l_ctr > 0 ){
    target.orient = "L";
    target.angle = -60;
    target.color = CV_RGB(255,0,0);
  } 
  else if ( *r_ctr > 0 ) {
    target.orient = "R";
    target.angle = 60;
    target.color = CV_RGB(0,0,255);
  }
  else ROS_ERROR("Invalid ctr combination: (f,l,r): (%d,%d,%d)", *f_ctr, *l_ctr, *r_ctr);
}

// Utility Functions // 
std::vector<cv::Rect> FaceDetector::toRect(cv::cuda::GpuMat& objs){
  std::vector<cv::Rect> rects;
  fDetector_->convert(objs, rects);
  return rects;
}

void FaceDetector::fixFlippedRects(std::vector<cv::Rect>& rects, int im_width){
  for ( unsigned int i = 0; i < rects.size(); i++ )
      rects[i].x = im_width - rects[i].x - rects[i].width;
}

void FaceDetector::updateDetectors(bool front, bool profile){
  if (front){
    fDetector_->setMaxObjectSize(fcDetector_.max_size);
    fDetector_->setMinObjectSize(fcDetector_.min_size);
    fDetector_->setMinNeighbors(fcDetector_.min_neighbors);
    fDetector_->setScaleFactor(fcDetector_.scale_factor);  
  }

  if (profile){
    pDetector_->setMaxObjectSize(pcDetector_.max_size);
    pDetector_->setMinObjectSize(pcDetector_.min_size);
    pDetector_->setMinNeighbors(pcDetector_.min_neighbors);
    pDetector_->setScaleFactor(pcDetector_.scale_factor);  
  }
}

// Load Cascades //
void FaceDetector::init(std::string f){
  if (load_gpu_){
    try{
      fDetector_ = cv::cuda::CascadeClassifier::create(cascade_path_ + "gpu/" + f + ".xml");
      updateDetectors(true, false); 
    }
    catch (cv::Exception& e){
      ROS_ERROR("OpenCV exception: %s", e.what());
      use_gpu_ = false;
      load_cpu_ = true; load_gpu_ = false;
    }
  }
  if(load_cpu_)
    if (!fcDetector_.detector.load(cascade_path_ + "cpu/" + f + ".xml") )
      ROS_ERROR("[FaceDetector] Could not load cascade classifier %s",(cascade_path_ + "cpu/" + f + ".xml").c_str());      
}

void FaceDetector::init(std::string f1, std::string f2){
  if(f1==""&&f2==""){
    f1 = front_classifier_file_;
    f2 = profile_classifier_file_;
  }
  else if(f2==""){
    init(f1);
    return;
  }
  
  if (load_gpu_){
    try{
      fDetector_ = cv::cuda::CascadeClassifier::create(cascade_path_ + "gpu/" + f1 + ".xml");
      pDetector_ = cv::cuda::CascadeClassifier::create(cascade_path_ + "gpu/" + f2 + ".xml");
    
      updateDetectors(true,true);
    }
    catch (cv::Exception& e){
      ROS_ERROR("OpenCV exception: %s", e.what());
      use_gpu_ = false;
      load_cpu_ = true; load_gpu_ = false;
    }
  }

  if (load_cpu_){
    if ( !fcDetector_.detector.load(cascade_path_ + "cpu/" + f1 + ".xml") )
      ROS_ERROR("[FaceDetector] Could not load cascade classifier %s",(cascade_path_ + "cpu/" + f1 + ".xml").c_str());
    
    if ( !pcDetector_.detector.load(cascade_path_ + "cpu/" + f2 + ".xml") )
      ROS_ERROR("[FaceDetector] Could not load cascade classifier %s",(cascade_path_ + "cpu/" + f2 + ".xml").c_str());    
  }
  
  if(!dlib_loaded_) loadDlib();
}

void FaceDetector::populateFaceRecCtrVectors(std::string filename){
  std::ifstream file ( filename.c_str(), std::ifstream::in );

  if ( !file )
  {
    std::cout << "Could not access " << filename << std::endl;
    return;
  }

  std::string line, index, real_name, phonetic_name;
  char separator = ',';
  bool first_line = true;

  while ( getline ( file, line ) )
  {
    std::stringstream liness ( line );
    getline ( liness, index, separator );
    getline ( liness, real_name, separator );
    getline ( liness, phonetic_name, separator );

    // Skip first line
    if ( first_line ) 
    {
      first_line = false;
      continue;
    }

    if ( !index.empty() && !real_name.empty() && !phonetic_name.empty() )
      all_people_.push_back(FD::person(real_name,phonetic_name));
  }

  // Display people in database
  std::cout << "People in model: " << std::endl;
  for ( std::size_t i = 0; i < all_people_.size(); i++ )
    std::cout << " (" << i+1 << ") " << all_people_[i].name << std::endl;

  // Add unknown
  all_people_.push_back(FD::person("Unknown","unknown"));
}

void FaceDetector::loadHog(){
  //hog_ = cv::cuda::HOG::create();
  hog_ = cv::cuda::HOG::create(hog_win_size_, hog_block_size_, hog_block_stride_, hog_cell_size_, hog_bins_);

  cv::Mat model = hog_ -> getDefaultPeopleDetector();
  hog_ -> setSVMDetector(model);

  hog_ -> setNumLevels(hog_nlvls_);
  hog_ -> setHitThreshold(hog_hit_thresh_);
  hog_ -> setScaleFactor(hog_scale_factor_);
  hog_ -> setWinStride(hog_win_stride_);

  chog_ = new cv::HOGDescriptor(hog_win_size_, hog_block_size_, hog_block_stride_, hog_cell_size_ , hog_bins_);
  chog_ -> setSVMDetector(model);
  chog_ -> nlevels = hog_nlvls_;

  ROS_INFO("[FaceDetector] Hog Models Initalized");

  hog_loaded_ = true;
}

void FaceDetector::updateHog(){
  hog_ = cv::cuda::HOG::create(hog_win_size_, hog_block_size_, hog_block_stride_, hog_cell_size_, hog_bins_);
  hog_ -> setHitThreshold(hog_hit_thresh_);
  hog_ -> setNumLevels(hog_nlvls_);
  hog_ -> setScaleFactor(hog_scale_factor_);
  hog_ -> setWinStride(hog_win_stride_);

  chog_ -> nlevels = hog_nlvls_;
}

void FaceDetector::loadDlib(){
  dlDetector_ = dlib::get_frontal_face_detector();
  dlib::deserialize(package_path_ + dlib_path_ + sp_file_ + ".dat") >> sp_;
  dlib_loaded_ = true;
}

void FaceDetector::loadROSParams(std::string ns){
  // Global Params
  nh_.param(ns + "/gpu", use_gpu_, use_gpu_);

  // Local Params
  ns += "/FaceDetector"; 
  nh_.param(ns + "/min_dist_tol", min_dist_, min_dist_);

  std::vector<int> temp;
  
  // Face Params
  nh_.param(ns + "/face" + "/type", front_classifier_file_, front_classifier_file_);
  nh_.param(ns + "/face" + "/scale_factor", fcDetector_.scale_factor, fcDetector_.scale_factor);
  nh_.param(ns + "/face" + "/min_neighbors", fcDetector_.min_neighbors, fcDetector_.min_neighbors);
  temp = utils::createVecFromCVSize(fcDetector_.min_size);
  nh_.param(ns + "/face" + "/min_size", temp, temp);

  std::cout << temp[0] << std::endl;

  fcDetector_.min_size = utils::createCVSizeFromVec(temp);
  temp = utils::createVecFromCVSize(fcDetector_.max_size);
  nh_.param(ns + "/face" + "/max_size", temp, temp);
  fcDetector_.max_size = utils::createCVSizeFromVec(temp);

  // // Profile Params
  nh_.param(ns + "/profile" + "/type", profile_classifier_file_, profile_classifier_file_);
  nh_.param(ns + "/profile" + "/scale_factor", pcDetector_.scale_factor, pcDetector_.scale_factor);
  nh_.param(ns + "/profile" + "/min_neighbors", pcDetector_.min_neighbors, pcDetector_.min_neighbors);
  temp = utils::createVecFromCVSize(pcDetector_.min_size);
  nh_.param(ns + "/profile" + "/min_size", temp, temp);
  pcDetector_.min_size = utils::createCVSizeFromVec(temp);
  temp = utils::createVecFromCVSize(pcDetector_.max_size);
  nh_.param(ns + "/profile" + "/max_size", temp, temp);
  pcDetector_.max_size = utils::createCVSizeFromVec(temp);

  // // DLIB Params
  nh_.param(ns + "/dlib" + "/sp_model", sp_file_, sp_file_);
  nh_.param(ns + "/dlib" + "/net_model", net_file_, net_file_);

  // // HOG Params
  temp = utils::createVecFromCVSize(hog_win_size_);
  nh_.param(ns + "/hog" + "/win_size", temp, temp);
  hog_win_size_ = utils::createCVSizeFromVec(temp);

  temp = utils::createVecFromCVSize(hog_block_size_);
  nh_.param(ns + "/hog" + "/block_size", temp, temp);
  hog_block_size_ = utils::createCVSizeFromVec(temp);
  
  temp = utils::createVecFromCVSize(hog_block_stride_);
  nh_.param(ns + "/hog" + "/block_stride", temp, temp);
  hog_block_stride_ = utils::createCVSizeFromVec(temp);

  int mult = 1;
  nh_.param(ns + "/hog" + "/win_stride", mult, mult);
  hog_win_stride_ = cv::Size(hog_block_stride_.width*mult, hog_block_stride_.height*mult);

  temp = utils::createVecFromCVSize(hog_cell_size_);
  nh_.param(ns + "/hog" + "/cell_size", temp, temp);
  hog_cell_size_ = utils::createCVSizeFromVec(temp);
  
  nh_.param(ns + "/hog" + "/bins", hog_bins_, hog_bins_);
  nh_.param(ns + "/hog" + "/nlvls", hog_nlvls_, hog_nlvls_);
  nh_.param(ns + "/hog" + "/scale_factor", hog_scale_factor_, hog_scale_factor_);
  nh_.param(ns + "/hog" + "/hit_thr", hog_hit_thresh_, hog_hit_thresh_);

  ROS_INFO("[FaceDetector] ROS Params Updated!");
}

void FaceDetector::rqtCb(person_detector::FaceDetectorRQTConfig &config, uint32_t level){
  if ( level == 0 ){
    use_gpu_ = config.gpu;
    ROS_INFO("[FaceDetector] %s is now active!",config.gpu?"GPU":"CPU");
  }

  else if ( level == 1 ){
    min_dist_ = config.min_dist_pct;
    ROS_INFO("[FaceDetector] Merge min dist set to %.2f of image width",min_dist_);
  }

  else if ( level == 2 ){
    fcDetector_.scale_factor = config.f_scale_factor;
    fcDetector_.min_neighbors = config.f_min_neighbors;
    fcDetector_.min_size = cv::Size(config.f_min_size, config.f_min_size);
    fcDetector_.max_size = cv::Size(config.f_max_size, config.f_max_size);
    updateDetectors(true, false);
    
    ROS_INFO("[FaceDetector] RQT Face Detector Params Updated: \n"
             "                  - Scale Factor: %.2f \n"
             "                  - Min Neighbors: %d \n"
             "                  - Min Size; (%d,%d) \n"
             "                  - Max Size: (%d,%d)",
             fcDetector_.scale_factor, fcDetector_.min_neighbors, 
             fcDetector_.min_size.width, fcDetector_.min_size.height,
             fcDetector_.max_size.width, fcDetector_.max_size.height);

  }

  else if ( level == 3 ){
    pcDetector_.scale_factor = config.p_scale_factor;
    pcDetector_.min_neighbors = config.p_min_neighbors;
    pcDetector_.min_size = cv::Size(config.p_min_size, config.p_min_size);
    pcDetector_.max_size = cv::Size(config.p_max_size, config.p_max_size);
    updateDetectors(false, true);
    
    ROS_INFO("[FaceDetector] RQT Profile Detector Params Updated: \n"
             "                  - Scale Factor: %.2f \n"
             "                  - Min Neighbors: %d \n"
             "                  - Min Size; (%d,%d) \n"
             "                  - Max Size: (%d,%d)",
             pcDetector_.scale_factor, pcDetector_.min_neighbors, 
             pcDetector_.min_size.width, pcDetector_.min_size.height,
             pcDetector_.max_size.width, pcDetector_.max_size.height);

  }

  else if ( level == 4 ){
    cv::Size prev_hog_win_size_ = hog_win_size_;
    cv::Size prev_hog_block_size_ = hog_block_size_;
    cv::Size prev_hog_block_stride_ = hog_block_stride_;
    cv::Size prev_hog_win_stride_ = hog_win_stride_;
    cv::Size prev_hog_cell_size_ = hog_cell_size_;

    hog_win_size_ = cv::Size(config.h_win_size, config.h_win_size*2);
    hog_block_size_ = cv::Size(config.h_block_size, config.h_block_size);
    hog_block_stride_ = cv::Size(config.h_block_stride, config.h_block_stride);
    hog_win_stride_ = cv::Size(config.h_win_stride*hog_block_stride_.width, 
                                config.h_win_stride*hog_block_stride_.width);
    hog_cell_size_ = cv::Size(config.h_cell_size,config.h_cell_size);      
    hog_bins_ = config.h_bins;
    hog_nlvls_ = config.h_nlvls;
    hog_scale_factor_ = config.h_scale_factor;
    hog_hit_thresh_ = config.h_hit_thresh;

    if ((hog_win_size_.width - hog_block_size_.width ) % hog_block_stride_.width != 0 || (hog_win_size_.height - hog_block_size_.height) % hog_block_stride_.height != 0){
      hog_win_size_ = prev_hog_win_size_ ;
      hog_win_size_ = prev_hog_win_size_ ;
      hog_block_size_ = prev_hog_block_size_ ;
      hog_block_stride_ = prev_hog_block_stride_ ;
      hog_win_stride_ = prev_hog_win_stride_ ;
      hog_cell_size_ = prev_hog_cell_size_ ;
    }
    else
      updateHog();

    ROS_INFO("[FaceDetector] RQT HOG Params Updated!");
  }

  return;
}
