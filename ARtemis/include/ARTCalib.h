#ifndef ARTEMIS_H
#define ARTEMIS_H

#include "ARtemisExports.h"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#include <opencv2/core/core.hpp>
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"


/*	==============================================================================================*/
namespace  ART{
	class ARTEMIS_LIB ARTCalib{
		public:
			ARTCalib();
			~ARTCalib();

			void initStereo(cv::Size imgSize, float scale, std::string intrinsicsFile, std::string extrinsicsFile);

			cv::Point3d getCameraCenter(int id);

			static bool showCorners(const cv::Mat &in1, const cv::Mat &in2, int scale, cv::Mat &out);
			static bool checkPattern(const cv::Mat &image, const cv::Size &boardSize, const float squareSize);			
			static void singleCalibration(const std::vector<cv::Mat> &imageList, const cv::Size &boardSize, const float squareSize, std::vector<bool> &inliers, std::vector<cv::Mat> &tvecs, std::vector<cv::Mat> &rvecs, cv::Mat &intrinsics, cv::Mat &distCoeffs);
			static float stereoCalibration(std::vector<std::vector<cv::Mat3b> > &imageList, cv::Size boardSize, float squareSize, bool singleCalibrate, std::vector<cv::Mat> &intrinsics, std::vector<cv::Mat> &distCoeffs, cv::Mat &tvec, cv::Mat &rvec);			
			//ARUCO
			void getCameraMatrix(cv::Mat &cm) {_M1.copyTo(cm);};
			void getCameraDistCoeffs(cv::Mat &dc) {_D1.copyTo(dc);};
		private:
			void loadCameraMatrices(std::string intrinsicsFile, std::string extrinsicsFile);

		private:

			int _scale;

			cv::Size _imgSize;

			cv::Mat _M1, _D1, _M2, _D2;
			cv::Mat _R, _T;
	};
};

#endif //ARTCALIB_H