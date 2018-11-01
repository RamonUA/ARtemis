#ifndef ARTEMIS_H
#define ARTEMIS_H

#include "ARtemisExports.h"

#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <ctime>

#include <opencv2/core/core.hpp>
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/video/tracking.hpp"


/*	==============================================================================================*/
namespace  ART{
	class ARTEMIS_LIB ARtemis{
		public:
			ARtemis(const std::string &trainImage_t, const std::string &trainImage_b, float markerWidth, float workingDistance, cv::Mat &intrinsics, cv::Mat &distCoeff);
			~ARtemis();

			bool findExtrinsics(const cv::Mat &inputFrame, const cv::Rect &ROI, std::vector<int> &inliers, cv::Mat &tvec, cv::Mat &rvec);

			void drawMarker(cv::Mat &image, const std::vector<int> &inliers);
			void drawRectangle(cv::Mat &image, const cv::Point2f &center, const cv::Size &size, const cv::Scalar &color);
			
		private:

			class HammingFlannMatcher : public cv::FlannBasedMatcher
			{
				public:
					HammingFlannMatcher( const cv::Ptr<cv::flann::IndexParams>& indexParams=cv::makePtr<cv::flann::KDTreeIndexParams>(),
						const cv::Ptr<cv::flann::SearchParams>& searchParams=cv::makePtr<cv::flann::SearchParams>() );
					virtual void train();
			};

			void trainMarker( int maxKeypoints );

			static void sortKeyPointsByDescriptors( const std::vector<cv::KeyPoint>& src, const std::vector<cv::Mat>& srcDesc, const size_t numMaxPoints, std::vector<cv::KeyPoint>& dst, std::vector<cv::Mat>& dstDesc );
			static void calcMeanDescriptors( std::vector<cv::Mat> &allDesc, cv::Mat &meanDesc );
			static float computeDistance(cv::Point2f& p0, cv::Point2f& p1);
			static cv::Point2f projectPointHomography(cv::Point2f& p, cv::Mat& h);
			static cv::Mat getHomography( cv::Size resolution, float degreesX, float degreesY, float degreesZ, float scale );			
			
			void filterKeyPoints(std::vector<cv::KeyPoint> &keyPoints, int maxKeyPoints);
			void filterMatches(std::vector<std::vector<cv::DMatch> > &matches, int maxMatches);
			void getKeyPointsUniform(std::vector<int> &inliers, int maxPoints, const cv::Size &gridSize, std::vector<cv::KeyPoint> &dst2d, std::vector<cv::Point3f> &dst3d);
			void projectPoints(const std::vector<cv::Point3f> &objPoints, std::vector<cv::Point2f> &projPoints);
			void trackKeyPoints(const cv::Mat &img, std::vector<cv::KeyPoint> &keyPoints, std::vector<cv::Point3f> &objectPoints, std::vector<cv::Point2f> &output2d);
			static cv::Point2f templateMatching(const cv::Mat1b &img, const cv::Mat &templ);
			cv::Mat getTransformationHomography(cv::Rect &BB);
			
			void resetKalman();
			void updateKalman();

		protected:
			cv::Ptr<cv::FeatureDetector> _detector;

			cv::Ptr<cv::FlannBasedMatcher> matcher_;
			//cv::Ptr<cv::BFMatcher> matcher_;

			cv::Mat _image;
			float _pxPmm;

			std::vector<cv::Point3f> _corners3d;
			std::vector<cv::Point3f> _objectPoints;
			std::vector<cv::KeyPoint> _keyPoints;
			cv::Mat _descriptors;

			cv::Mat _intrinsics;
			cv::Mat _distCoeff;
			
			cv::Mat _rvec;
			cv::Mat _tvec;

			cv::KalmanFilter _KF;
			int _tproc1, _tproc2, _tproc3;
			int _rproc1, _rproc2, _rproc3;
			int _tmeas;
			int _rmeas;
			std::clock_t _clock;			

			bool _tracking;
			int _areaSize;
			int _templSize;			

			//std::ofstream _file;
	};
};

#endif //ARTEMIS_H