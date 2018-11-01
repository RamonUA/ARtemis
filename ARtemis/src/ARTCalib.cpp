#include "ARTCalib.h"

using namespace ART;
using namespace cv;
using namespace std;
		
ARTCalib::ARTCalib(){}

ARTCalib::~ARTCalib(){}

void ARTCalib::loadCameraMatrices(std::string intrinsicsFile, std::string extrinsicsFile)
{
    // reading intrinsic parameters
    cv::FileStorage fs(intrinsicsFile, cv::FileStorage::READ);
    if(!fs.isOpened())
    {
        std::printf("Failed to open file %s\n", intrinsicsFile.c_str());
        return;
    }

    fs["M1"] >> _M1;
    fs["D1"] >> _D1;
    fs["M2"] >> _M2;
    fs["D2"] >> _D2;

    _M1 *= _scale;
    _M2 *= _scale;

    fs.open(extrinsicsFile, FileStorage::READ);
    if(!fs.isOpened())
    {
        printf("Failed to open file %s\n", extrinsicsFile.c_str());
        return;
    }

    fs["R"] >> _R;
    fs["T"] >> _T;

    std::printf("Intrinsics and extrinsics files opened correctly!\n");
}

cv::Point3d ARTCalib::getCameraCenter(int id)
{
    cv::Point3d cameraCenter;
    switch (id)
    {
        case 0:
            cameraCenter.x = 0.f;//_P1.at<float>(0,3);
            cameraCenter.y = 0.f;//_P1.at<float>(1,3);
            cameraCenter.z = 0.f;//_P1.at<float>(2,3);
            break;
        case 1:
            cameraCenter.x = _T.at<double>(0,0);//_P2.at<float>(0,3);
            cameraCenter.y = 0.f;//_P2.at<float>(1,3);
            cameraCenter.z = 0.f;//_P2.at<float>(2,3);
            break;
    };

    return cameraCenter;
}

void ARTCalib::initStereo(cv::Size imgSize, float scale, std::string intrinsicsFile, std::string extrinsicsFile)
{
    printf("Configuring stereo...\n");

    _imgSize = imgSize;
    _scale = scale;

    if( (!intrinsicsFile.empty()) ^ (!extrinsicsFile.empty()) )
    {
        printf("Parameter error: either both intrinsic and extrinsic parameters must be specified, or none of them (when the stereo pair is already rectified)\n");
        return;
    }

    if( !intrinsicsFile.empty() && !extrinsicsFile.empty() )
    {
        loadCameraMatrices(intrinsicsFile, extrinsicsFile);
    }

    printf("Stereo configured properly!\n");
}

bool ARTCalib::showCorners(const cv::Mat &in1, const cv::Mat &in2, int scale, cv::Mat &out)
{
    cv::Mat1b resized1, resized2;
    cv::cvtColor(in1, resized1, CV_BGR2GRAY);
    cv::cvtColor(in2, resized2, CV_BGR2GRAY);

    cv::resize(resized1, resized1, resized1.size()/scale);
    cv::resize(resized2, resized2, resized2.size()/scale);

    std::vector<cv::Point2f> corners1, corners2;
    bool found1 = cv::findChessboardCorners(resized1, cv::Size(7,4), corners1,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
    bool found2 = cv::findChessboardCorners(resized2, cv::Size(7,4), corners2,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);

    cv::Mat3b draw1, draw2;
    cv::cvtColor(resized1, draw1, CV_GRAY2RGB);			
    cv::cvtColor(resized2, draw2, CV_GRAY2RGB);
    
    cv::drawChessboardCorners(draw1, cv::Size(7,4), corners1, found1);
    cv::drawChessboardCorners(draw2, cv::Size(7,4), corners2, found2);
    
    out = cv::Mat3b(draw1.rows, draw1.cols+draw2.cols);
    draw1.copyTo(out(cv::Range(0, draw1.rows), cv::Range(0, draw1.cols)));
    draw2.copyTo(out(cv::Range(0, draw2.rows), cv::Range(draw1.cols, draw1.cols+draw2.cols)));

    if(found1 && found2)
        return true;
    else
        return false;
}

bool ARTCalib::checkPattern(const cv::Mat &image, const cv::Size &boardSize, const float squareSize)
{
    cv::Mat img;
    cv::cvtColor(image, img, CV_BGR2GRAY);
    
    bool found = false;
    std::vector<cv::Point2f> corners;

    for( int scale = 1; scale <= 2; scale++ )
    {
        cv::Mat timg;
        if( scale == 1 )
            timg = img;
        else
            cv::resize(img, timg, cv::Size(), scale, scale);

        found = cv::findChessboardCorners(timg, boardSize, corners,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

        if( found )
            return found;
    }

    return found;
}

void ARTCalib::singleCalibration(const std::vector<cv::Mat> &imageList, const cv::Size &boardSize, const float squareSize, std::vector<bool> &inliers, std::vector<cv::Mat> &tvecs, std::vector<cv::Mat> &rvecs, cv::Mat &intrinsics, cv::Mat &distCoeffs)
{
    std::vector<std::vector<cv::Point2f> > imagePoints;
    std::vector<std::vector<cv::Point3f> > objectPoints;
    cv::Size imageSize;
    cv::Mat img;    

    inliers.resize(imageList.size(), false);

    for( int i = 0; i < imageList.size(); ++i )
    {
        cv::cvtColor(imageList[i], img, CV_BGR2GRAY);

        bool found = false;
        std::vector<cv::Point2f> corners;

        found = cv::findChessboardCorners(img, boardSize, corners,
                    CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
        
        if( found )
            inliers[i] = true;
        else
            continue;

        cv::cornerSubPix(img, corners, cv::Size(11,11), cv::Size(-1,-1),
                         cv::TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
                                      30, 0.01));
        
        objectPoints.resize(objectPoints.size()+1);
        for( int j = 0; j < boardSize.height; j++ )
            for( int k = 0; k < boardSize.width; k++ )
                objectPoints[objectPoints.size()-1].push_back(cv::Point3f(k*squareSize, j*squareSize, 0));
        
        imagePoints.push_back(corners);
	}

    intrinsics = cv::Mat::eye(3, 3, CV_64F);
    distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

    double rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, intrinsics,
                                    distCoeffs, rvecs, tvecs, CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);

    /*cv::FileStorage fs;
    fs.open("intrinsics_single.yml", FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "M" << intrinsics << "D" << distCoeffs;        
        fs.release();
    }
    else
        cout << "Error: can not save the intrinsic parameters\n";*/
}

float ARTCalib::stereoCalibration(std::vector<std::vector<cv::Mat3b> > &imageList, cv::Size boardSize, float squareSize, bool singleCalibrate, std::vector<cv::Mat> &intrinsics, std::vector<cv::Mat> &distCoeffs, cv::Mat &tvec, cv::Mat &rvec)
{
    if( imageList[0].size() != imageList[1].size() )
        return -1.f;

    const int maxScale = 2;

    std::vector<std::vector<bool> > detected(2);
    std::vector<std::vector<std::vector<cv::Point2f> > > imagePoints(2);
    std::vector<std::vector<cv::Point3f> > objectPoints;
    cv::Size imageSize = imageList[0][0].size();

    for( int i = 0; i < imageList.size(); ++i )
    {
        detected[i].resize(imageList[i].size(), false);
        imagePoints[i].resize(imageList[i].size());
        
        for( int j = 0; j < imageList[i].size(); ++j )
        {
            cv::Mat img;
            cv::cvtColor(imageList[i][j], img, CV_BGR2GRAY);
            
            bool found = false;
            std::vector<cv::Point2f>& corners = imagePoints[i][j];

            for( int scale = 1; scale <= maxScale; scale++ )
            {
                cv::Mat timg;
                if( scale == 1 )
                    timg = img;
                else
                    cv::resize(img, timg, cv::Size(), scale, scale);

                found = cv::findChessboardCorners(timg, boardSize, corners,
                    CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

                if( found )
                {
                    if( scale > 1 )
                    {
                        cv::Mat cornersMat(corners);
                        cornersMat *= 1./scale;
                    }
                    detected[i][j] = true;
                    break;
                }
            }
            if( found )
                cv::cornerSubPix(img, corners, cv::Size(11,11), cv::Size(-1,-1), cv::TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01));
        }
    }
          
    std::vector<cv::Point3f> markerPoints; 
    for( int i = 0; i < boardSize.height; i++ )
        for( int j = 0; j < boardSize.width; j++ )
            markerPoints.push_back(cv::Point3f(j*squareSize, i*squareSize, 0));

    std::vector<std::vector<std::vector<cv::Point2f> > > stereoImagePoints(2), singleImagePoints(2);        
    std::vector<std::vector<cv::Point3f> > stereoObjectPoints;        
    std::vector<std::vector<std::vector<cv::Point3f> > > singleObjectPoints(2);        
    for( int i = 0; i < imageList[0].size(); ++i )
    {
        if( detected[0][i] )
        {
            singleImagePoints[0].push_back(imagePoints[0][i]);
            singleObjectPoints[0].push_back(markerPoints);
        }
        if( detected[1][i] )
        {
            singleImagePoints[1].push_back(imagePoints[1][i]);
            singleObjectPoints[1].push_back(markerPoints);
        }
        if( detected[0][i] && detected[1][i] )
        {
            stereoImagePoints[0].push_back(imagePoints[0][i]);
            stereoImagePoints[1].push_back(imagePoints[1][i]);
            stereoObjectPoints.push_back(markerPoints);
        }
    }

    if( singleCalibrate )
    {
        if( singleImagePoints[0].size() < 5 || singleImagePoints[1].size() < 5 )
            return -1.f;

        intrinsics.resize(2);
        distCoeffs.resize(2);    
        std::vector<double> calibrationRMS(2);
        for( int i = 0; i < imageList.size(); ++i )
        {
            intrinsics[i] = cv::Mat::eye(3, 3, CV_64F);
            distCoeffs[i] = cv::Mat::zeros(8, 1, CV_64F);

            std::vector<cv::Mat> rvecs, tvecs;
            
            calibrationRMS[i] = cv::calibrateCamera(singleObjectPoints[i], singleImagePoints[i], imageList[i][0].size(), intrinsics[i],
                                distCoeffs[i], rvecs, tvecs/*, CV_CALIB_FIX_K3|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5*/);
        }
    }
 
    cv::Mat E, F;

    if( stereoObjectPoints.size() < 5 )
        return -1.f;

    double rms = cv::stereoCalibrate(stereoObjectPoints, stereoImagePoints[0], stereoImagePoints[1],
                    intrinsics[0], distCoeffs[0],
                    intrinsics[1], distCoeffs[1],
                    cv::Size(), rvec, tvec, E, F,
                    CV_CALIB_FIX_INTRINSIC,
                    cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 500, 1e-5) );

    return rms;
}