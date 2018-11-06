#include <iostream>
#include <fstream>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include "ARtemisSampleConfig.h"
#include "ARTTrack.h"

#define LOG_TAG "Sample_ARtemis"

using namespace ART;

cv::VideoCapture _videoCapture;

cv::Mat3b _incomingFrame;
cv::Mat3b _outputImage;

int 	_captureWidth = 1920;//640;
int 	_captureHeight = 1080;//480;
int		_fps = 30;

double 	_cameraFps = 0.f;
int 	_cameraWidth = 0;
int 	_cameraHeight = 0;

int _scale = 1;

ART::ARTTrack* _tracker;

cv::Mat _rvec, _tvec;
cv::Mat _intrinsics, _distCoeff;
std::vector<int> _inliers; 

std::string _trainImage_t = ARTEMIS_SAMPLE_PATH_DATA"/trainImage_t_crop.jpg";
//std::string _trainImage_t = ARTEMIS_SAMPLE_PATH_DATA"/marker.jpg";
std::string _trainImage_b = ARTEMIS_SAMPLE_PATH_DATA"/trainImage_b_320.jpg";

float _markerWidth = 167.f;
float _workingDistance = 250.f;

std::clock_t _clock;

//int cont = 0;

void waitAndExit(int errorCode)
{
	std::cin.ignore();
	exit(errorCode);
}

void printFps(cv::Mat &src)
{
	double fps = (double) CLOCKS_PER_SEC / ( std::clock() - _clock );
	cv::putText(src, "Fps: "+std::to_string(fps), cv::Point(20, src.rows-20),
		 			1, 2.f/_scale, cv::Scalar(220,0,200), 2/_scale);
	
	_clock = std::clock();
}

void initMatrices()
{
	_intrinsics = cv::Mat1f::eye(3,3);
	_distCoeff = cv::Mat1f::zeros(4,1);
	_tvec = cv::Mat1f::zeros(3,1);
	_rvec = cv::Mat1f::zeros(3,1);	

	// 640x480
/*	_intrinsics.at<float>(0,0) = 628.396f;
	_intrinsics.at<float>(1,1) = 634.144f;
	_intrinsics.at<float>(0,2) = 297.513f;
	_intrinsics.at<float>(1,2) = 230.98f;

	_distCoeff.at<float>(0) = 0.1521f;
	_distCoeff.at<float>(1) = -0.1922f;
	_distCoeff.at<float>(2) = 0.0f;
	_distCoeff.at<float>(3) = -0.006f;
*/
	// 1920x1080
	_intrinsics.at<float>(0,0) = 1404.55f;
	_intrinsics.at<float>(1,1) = 1408.46f;
	_intrinsics.at<float>(0,2) = 991.67f;
	_intrinsics.at<float>(1,2) = 522.03f;

	_distCoeff.at<float>(0) = 0.092363f;
	_distCoeff.at<float>(1) = -0.171852f;
	_distCoeff.at<float>(2) = -0.001701f;
	_distCoeff.at<float>(3) = 0.001883f;
}

void init()
{
 	if(!_videoCapture.open(1 + CV_CAP_AVFOUNDATION))	// Try to start camera 1
	{									
		std::cout << "Cannot open Camera" << std::endl;
		//waitAndExit(-1);
	} else
		std::cout << "Camera opened!" << std::endl;

	_videoCapture.set(CV_CAP_PROP_FPS, _fps);

	_videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, _captureWidth);
	_videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, _captureHeight);

	_cameraFps = 	_videoCapture.get(CV_CAP_PROP_FPS);
	_cameraWidth = 	_videoCapture.get(CV_CAP_PROP_FRAME_WIDTH);
	_cameraHeight = _videoCapture.get(CV_CAP_PROP_FRAME_HEIGHT);

	std::cout << "FPS: " << std::to_string(_cameraFps) << std::endl;

	_incomingFrame = cv::Mat3b(_captureWidth/_scale, _captureHeight/_scale);
	_outputImage = cv::Mat3b(_captureWidth/_scale, _captureHeight/_scale);

	initMatrices();

	// INIT TRACKER
	_tracker = new ART::ARTTrack(_trainImage_t, _trainImage_b, _markerWidth, _workingDistance, _intrinsics, _distCoeff);
}

int main()
{
	init();

	_clock = std::clock();
int cont = 0;
	while( true )
	{
		if( _videoCapture.grab() )
		{
			try // grab the frame
			{
				_videoCapture.retrieve(_incomingFrame);
			}
			catch (cv::Exception& e)
			{
				const char* err_msg = e.what();
				std::cout << "Camera1: Error message: " << err_msg << std::endl;
			};
		}

		//_incomingFrame = cv::imread(std::to_string(cont)+".jpg");
/*cv::Mat res;
cv::resize(_incomingFrame, res, _incomingFrame.size()/2);
cv::imshow("marker", res);
uchar u = cv::waitKey(1);
if(u == 's')
{
	cv::imwrite(std::to_string(cont++)+".jpg", _incomingFrame);
}
}
{*/
		cv::Mat1b grayFrame;
		cv::cvtColor(_incomingFrame, grayFrame, CV_BGR2GRAY);

		bool find = _tracker->findExtrinsics(grayFrame, cv::Rect(_incomingFrame.cols/2 - 320, _incomingFrame.rows/2 - 240, 640, 480), _inliers, _tvec, _rvec);

		if( find )
		{
			_tracker->drawMarker(_incomingFrame, _inliers);
			_tracker->drawRectangle(_incomingFrame, cv::Point2f(86,35), cv::Size(10,10), cv::Scalar(255,0,255));
		}

		printFps(_incomingFrame);

		cv::resize(_incomingFrame, _outputImage, _incomingFrame.size()/2);
		cv::imshow("output", _outputImage);
		cv::waitKey(1);
	}

	return 0;
}