#include "ARTTrack.h"

using namespace ART;
using namespace cv;
using namespace std;

# define PI 3.14159265358979323846  /* pi */

ARTTrack::HammingFlannMatcher::HammingFlannMatcher( const Ptr<flann::IndexParams>& indexParams,
const Ptr<flann::SearchParams>& searchParams) : cv::FlannBasedMatcher(indexParams, searchParams) {}

void ARTTrack::HammingFlannMatcher::train()
{
    if( !flannIndex || mergedDescriptors.size() < addedDescCount )
    {
        // FIXIT: Workaround for 'utrainDescCollection' issue (PR #2142)
        if (!utrainDescCollection.empty())
        {
            CV_Assert(trainDescCollection.size() == 0);
            for (size_t i = 0; i < utrainDescCollection.size(); ++i)
                trainDescCollection.push_back(utrainDescCollection[i].getMat(ACCESS_READ));
        }
        mergedDescriptors.set( trainDescCollection );
        flannIndex = makePtr<flann::Index>( mergedDescriptors.getDescriptors(), *indexParams, cvflann::FLANN_DIST_HAMMING );
    }
}

ARTTrack::ARTTrack(const std::string &trainImage_t, const std::string &trainImage_b, float markerWidth, float workingDistance, cv::Mat &intrinsics, cv::Mat &distCoeff)
{
    //_file.open("out.csv");

    _intrinsics = intrinsics;
    _distCoeff = distCoeff;

    _areaSize = 64;
    _templSize = 32;

    _detector = cv::BRISK::create(30, 3, 1.f);
    //_detector = cv::ORB::create(2000, 1.5f, 6, 30);

    matcher_ = cv::makePtr<HammingFlannMatcher>(new cv::flann::HierarchicalClusteringIndexParams());
    //matcher_ = cv::makePtr<cv::BFMatcher>((int)cv::NORM_HAMMING, false);

    cv::Mat fileImage = cv::imread(trainImage_t);
    cv::Mat grayImage;
    cv::cvtColor(fileImage, grayImage, CV_BGR2GRAY);

    int trainingWidth = markerWidth * intrinsics.at<float>(0,0) / workingDistance;
    cv::Mat resizedImage;    
    cv::resize(grayImage, resizedImage, cv::Size(trainingWidth, grayImage.rows*trainingWidth/grayImage.cols));
//cv::imshow("resMarker", resizedImage);
//cv::waitKey(0);
    _pxPmm = markerWidth / trainingWidth;    

    _image = cv::Mat(resizedImage.rows+_templSize, resizedImage.cols+_templSize, CV_8UC1, cv::Scalar(127));
    resizedImage.copyTo(_image(cv::Range(_templSize/2, _templSize/2 + resizedImage.rows), 
                            cv::Range(_templSize/2, _templSize/2 + resizedImage.cols)));
    
    //trainMarker(1000);
// MASTER
    _detector->detect(_image, _keyPoints);
    filterKeyPoints(_keyPoints, 1000);
    _detector->compute(_image, _keyPoints, _descriptors);
  
    //store real keypoints with metric
    for(int i = 0; i < _keyPoints.size(); ++i)
        _objectPoints.push_back(cv::Point3f((_keyPoints[i].pt.x-(_templSize/2))*_pxPmm, 
                                            (_keyPoints[i].pt.y-(_templSize/2))*_pxPmm, 
                                            0.f));

    _corners3d.push_back(cv::Point3f(0.f, 0.f, 0.f));
    _corners3d.push_back(cv::Point3f((_image.cols-_templSize)*_pxPmm, 0.f, 0.f));
    _corners3d.push_back(cv::Point3f((_image.cols-_templSize)*_pxPmm, (_image.rows-_templSize)*_pxPmm, 0.f));
    _corners3d.push_back(cv::Point3f(0.f, (_image.rows-_templSize)*_pxPmm, 0.f));

    std::vector<cv::Mat> desc(1);
    desc[0] = _descriptors;
    matcher_->add(desc);
    
    matcher_->train();
    
/*    _tproc1 = 51;
    _tproc2 = 51;
    _tproc3 = 51;
    _rproc1 = 51;
    _rproc2 = 51;
    _rproc3 = 51;
    _tmeas = 81;
    _rmeas = 81;
    cv::namedWindow("Kalman Error", WINDOW_AUTOSIZE);
    cv::createTrackbar( "tprocNoise 1st", "Kalman Error", &_tproc1, 99, NULL );    
    cv::createTrackbar( "tprocNoise 2nd", "Kalman Error", &_tproc2, 99, NULL );    
    cv::createTrackbar( "tprocNoise 3rd", "Kalman Error", &_tproc3, 99, NULL );    
    cv::createTrackbar( "rprocNoise 1st", "Kalman Error", &_rproc1, 99, NULL );    
    cv::createTrackbar( "rprocNoise 2nd", "Kalman Error", &_rproc2, 99, NULL );    
    cv::createTrackbar( "rprocNoise 3rd", "Kalman Error", &_rproc3, 99, NULL );    
    cv::createTrackbar( "tmeasNoise", "Kalman Error", &_tmeas, 99, NULL );    
    cv::createTrackbar( "rmeasNoise", "Kalman Error", &_rmeas, 99, NULL ); 
*/    
    _tracking = false;    
}

ARTTrack::~ARTTrack(){}

void ARTTrack::trainMarker( int maxKeypoints )
{
    // Calculate rotations(30º, azimuth step = 45º)
    std::vector<std::pair<float,float> > rotations;

    // Store firstView angle
    std::pair<float,float> firstView;
    firstView.first = 0.f;
    firstView.second = 0.f;
    rotations.push_back( firstView );

    // Store rest of view angles
    for( int altitude = 30; altitude <= 30; altitude+=15 ){
        for( int azimuth = 0; azimuth < 360; azimuth+=15 ){
            std::pair<float,float> rot;
            rot.first = sin(altitude*PI/180)*cos(azimuth*PI/180);
            rot.second = sin(altitude*PI/180)*sin(azimuth*PI/180);
            rotations.push_back(rot);
        }
    }

    //create vectors of global keypoints and descriptors for all the transformations 
    std::vector<cv::KeyPoint> globalKps;
    std::vector<cv::Mat> globalDesc;

    cv::Size size = _image.size();
    cv::Size border = cv::Size(size.width/2,size.height/2);
    cv::Mat real = cv::Mat::zeros( size.height+border.height*2, size.width+border.width*2, CV_8U );
    _image.copyTo( real( cv::Rect(border.width,border.height,size.width,size.height) ) );

    for( int i = 0; i < rotations.size(); ++i ){

        float rotx = rotations[i].first*180/PI;
        float roty = rotations[i].second*180/PI;
        float scale = 1.f;

        //Generate transformations  and calculate  homographies
        cv::Mat jH = getHomography(real.size(), rotx, roty, 0.0, scale);
        cv::Mat invH = jH.inv();

        cv::Mat synthetic;
        cv::warpPerspective( real, synthetic, jH, real.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT );
//cv::imshow("warp", synthetic);
//cv::waitKey(200);
        std::vector<cv::KeyPoint> rotatedKp;
        cv::Mat rotatedDesc;

        _detector->detect(synthetic, rotatedKp);        

        filterKeyPoints(rotatedKp, 1000);

        _detector->compute(synthetic, rotatedKp, rotatedDesc);

        for( int j = 0; j < rotatedKp.size(); ++j )
            rotatedKp[j].pt = projectPointHomography(rotatedKp[j].pt, invH) - cv::Point2f(border.width,border.height);
                
        //compute coincidence between keypoints
        std::vector<bool> coincidenceKp( rotatedKp.size(), false );
        for( int j = 0; j < rotatedKp.size(); ++j ){
            float minDist = 1.f;
            int globalId = 0;
            for( int k = 0; k < globalKps.size(); ++k ){						
                cv::KeyPoint kp0 =  rotatedKp[j];
                kp0.pt = kp0.pt*pow(2.f, kp0.octave);
                cv::KeyPoint kp1 = globalKps[k];
                kp1.pt = kp1.pt*pow(2.f, kp0.octave);
                float distance = computeDistance(kp0.pt, kp1.pt);
                if( distance < minDist ) {
                    minDist = distance;
                    globalId = k;
                    coincidenceKp[j] = true;
                }
            }
            if( coincidenceKp[j] )
                globalDesc[globalId].push_back( rotatedDesc.row(j) );
        }

        //push back the keypoints and their descriptors with no coincidence with globalKps
        for( int j = 0; j < coincidenceKp.size(); ++j ){
            if( !coincidenceKp[j] ){
                globalKps.push_back( rotatedKp[j] );
                globalDesc.push_back( rotatedDesc.row(j) );
            }
        }
    }
//cv::destroyWindow("warp");
    //resize keypoints to maxKeypoints
    std::vector<cv::KeyPoint> finalKps;
    std::vector<cv::Mat> finalDesc;

    sortKeyPointsByDescriptors( globalKps, globalDesc, maxKeypoints, finalKps, finalDesc );

    calcMeanDescriptors(finalDesc, _descriptors);

    //store real keypoints with metric
    for(int i = 0; i < finalKps.size(); ++i)
        _keyPoints.push_back(finalKps[i]);
}

void ARTTrack::sortKeyPointsByDescriptors( const std::vector<cv::KeyPoint>& src, const std::vector<cv::Mat>& srcDesc, const size_t numMaxPoints, std::vector<cv::KeyPoint>& dst, std::vector<cv::Mat>& dstDesc ) 
{
	dst.reserve( numMaxPoints );
	dstDesc.reserve( numMaxPoints );

	std::vector<bool> sorted ( src.size(), false );
	for( size_t i = 0; i < numMaxPoints; ++i ) {
		int srcId;
		int rows = 0;
		for( size_t j = 0; j < src.size(); ++j ) {
			if( sorted[j] ) continue;

			if( srcDesc[j].rows > rows ) {
				srcId = j;
				rows = srcDesc[j].rows;
			}
        }
		sorted[srcId] = true;
		dst.push_back( src[srcId] );
		dstDesc.push_back( srcDesc[srcId] );
	}
}

void ARTTrack::calcMeanDescriptors( std::vector<cv::Mat> &allDesc, cv::Mat &meanDesc )
{
    meanDesc = cv::Mat(allDesc.size(), allDesc[0].cols, allDesc[0].type());
    cv::Mat mean;
    for(int i = 0; i < allDesc.size(); ++i)
    {
        switch (allDesc[i].type())
        {
        case CV_8U:
            {			
                std::vector<size_t> counter( allDesc[i].cols*8, 0 );				
                for( size_t j=0; j<(size_t)allDesc[i].rows; ++j ){
                    cv::Mat jrow = allDesc[i].row(j);				
                    size_t *counterPtr = &counter[0];
                    for( size_t k=0; k<(size_t)jrow.cols; ++k ){
                        uchar kval = jrow.at<uchar>(0,k);
                        counterPtr[0] += kval & 0x01;
                        counterPtr[1] += ((kval>>1) & 0x01);
                        counterPtr[2] += ((kval>>2) & 0x01);
                        counterPtr[3] += ((kval>>3) & 0x01);
                        counterPtr[4] += ((kval>>4) & 0x01);
                        counterPtr[5] += ((kval>>5) & 0x01);
                        counterPtr[6] += ((kval>>6) & 0x01);
                        counterPtr[7] += ((kval>>7) & 0x01);
                        counterPtr += 8;
                    }				
                }

                mean = cv::Mat(1,allDesc[i].cols,CV_8U);
                size_t *counterPtr = &counter[0];
                const size_t counterThr = allDesc[i].rows / 2;
                for( size_t k=0; k<(size_t)mean.cols; ++k ){				
                    uchar kval = 0;
                    kval = kval | (counterPtr[0]>counterThr ? 0x01 : 0);
                    kval = kval | (counterPtr[1]>counterThr ? 0x02 : 0);
                    kval = kval | (counterPtr[2]>counterThr ? 0x04 : 0);
                    kval = kval | (counterPtr[3]>counterThr ? 0x08 : 0);
                    kval = kval | (counterPtr[4]>counterThr ? 0x10 : 0);
                    kval = kval | (counterPtr[5]>counterThr ? 0x20 : 0);
                    kval = kval | (counterPtr[6]>counterThr ? 0x40 : 0);
                    kval = kval | (counterPtr[7]>counterThr ? 0x80 : 0);
                    counterPtr +=8;
                    mean.at<uchar>(k) = kval;
                }
            }
            break;
        case CV_32F:
            {
                mean = cv::Mat::zeros(1,allDesc[i].cols,CV_32F);
                for( size_t i=0; i<(size_t)allDesc[i].rows; ++i )
                    mean += allDesc[i].row(i);
                mean = mean / allDesc[i].rows;			
            }
            break;
        default:
            mean = cv::Mat::zeros(1,allDesc[i].cols,allDesc[i].type());
            break;
        }
        mean.copyTo(meanDesc.row(i));
    }
}

cv::Mat ARTTrack::getHomography( cv::Size resolution, float degreesX, float degreesY, float degreesZ, float scale )
{
	const float degreeToRadian = 3.1415f/180.f;
	cv::Mat Hx = cv::Mat::eye(3,3,CV_32F);
	((float*)Hx.data)[4] = cos( degreesX*degreeToRadian );
	((float*)Hx.data)[5] = -sin( degreesX*degreeToRadian );
	((float*)Hx.data)[7] = -((float*)Hx.data)[5]*0.001f;  // approximated perspective effect

	cv::Mat Hy = cv::Mat::eye(3,3,CV_32F);
	((float*)Hy.data)[0] = cos( degreesY*degreeToRadian );
	((float*)Hy.data)[2] = sin( degreesY*degreeToRadian );
	((float*)Hy.data)[6] = ((float*)Hy.data)[2]*0.001f; // approximated perspective effect

	cv::Mat Hz = cv::Mat::eye(3,3,CV_32F);
	((float*)Hz.data)[0] = cos( degreesZ*degreeToRadian );
	((float*)Hz.data)[1] = sin( degreesZ*degreeToRadian );
	((float*)Hz.data)[3] = -((float*)Hz.data)[1];
	((float*)Hz.data)[4] = ((float*)Hz.data)[0]; 

	cv::Mat HToCenterYNeg = cv::Mat::eye(3,3,CV_32F);
	((float*)HToCenterYNeg.data)[2] = -resolution.width*0.5f;
	((float*)HToCenterYNeg.data)[4] = -1;
	((float*)HToCenterYNeg.data)[5] = resolution.height*0.5f;

	cv::Mat HToTopLeftYPos = cv::Mat::eye(3,3,CV_32F);
	((float*)HToTopLeftYPos.data)[2] = resolution.width*0.5f;
	((float*)HToTopLeftYPos.data)[4] = -1;
	((float*)HToTopLeftYPos.data)[5] = resolution.height*0.5f;

	// scale
	cv::Mat S = cv::Mat::eye(3,3,CV_32F);
	((float*)S.data)[0] = scale;
	((float*)S.data)[4] = scale;

	cv::Mat H = S * HToTopLeftYPos * (Hz * Hx * Hy) * HToCenterYNeg;
	
	// center the warped image
	cv::Point2f center = cv::Point2f( resolution.width*0.5f, resolution.height*0.5f );
	float z = ((float*)H.data)[6] * center.x + ((float*)H.data)[7] * center.y + ((float*)H.data)[8];
	float x = (((float*)H.data)[0] * center.x + ((float*)H.data)[1] * center.y + ((float*)H.data)[2]) / z;
	float y = (((float*)H.data)[3] * center.x + ((float*)H.data)[4] * center.y + ((float*)H.data)[5]) / z;

	cv::Mat HToCenter = cv::Mat::eye(3,3,CV_32F);
	((float*)HToCenter.data)[2] = center.x - x;
	((float*)HToCenter.data)[5] = center.y - y;
	H = HToCenter * H;

	return H;
}

cv::Point2f ARTTrack::projectPointHomography(cv::Point2f& p, cv::Mat& h)
{    
    cv::Point2f result;
    float z = h.at<float>(2,0)*p.x + h.at<float>(2,1)*p.y + h.at<float>(2,2);
    result.x = (h.at<float>(0,0)*p.x + h.at<float>(0,1)*p.y + h.at<float>(0,2) )/z;
    result.y = (h.at<float>(1,0)*p.x + h.at<float>(1,1)*p.y + h.at<float>(1,2) )/z;

    return result;
}

float ARTTrack::computeDistance(cv::Point2f& p0, cv::Point2f& p1)
{    
    float distance = 0.f;

    cv::Point2f final = p0 - p1;

    distance = sqrt((final.x*final.x) + (final.y*final.y));
    return distance;
}

bool ARTTrack::findExtrinsics(const cv::Mat &inputFrame, const cv::Rect &ROI, std::vector<int> &inliers, cv::Mat &tvec, cv::Mat &rvec)
{
    cv::Mat frame = inputFrame(ROI);
    cv::Point2f offset = cv::Point2f(ROI.x, ROI.y);
    //frame(ROI) = inputFrame(ROI);
//cv::imshow("frame", frame);
    if( !_tracking )    // DETECTION
    {
        std::vector<cv::KeyPoint> queryKeyPoints;
        cv::Mat queryDescriptors;

        _detector->detect(frame, queryKeyPoints);

        if( queryKeyPoints.size() > 0 )
        {
            filterKeyPoints(queryKeyPoints, 1000);

            _detector->compute(frame, queryKeyPoints, queryDescriptors);

            std::vector<std::vector<cv::DMatch> > matches;
            
            matcher_->knnMatch(queryDescriptors, matches, 2);
            
            filterMatches(matches, 100);
            
            std::vector<cv::Point3f> objectPoints;            
            std::vector<cv::Point2f> matches2D;
            for(int i = 0; i < matches.size(); ++i)
            {
                matches2D.push_back(queryKeyPoints[matches[i][0].queryIdx].pt + offset);
                objectPoints.push_back(_objectPoints[matches[i][0].trainIdx]);                
            }                

            _rvec = cv::Mat1f::zeros(3, 1);
            _tvec = cv::Mat1f::zeros(3, 1); 

            inliers.clear();
            std::vector<int> auxInliers;
            _tracking = cv::solvePnPRansac(objectPoints, matches2D, _intrinsics, _distCoeff,
                                            _rvec, _tvec, false, 200, 2.0f, 0.97f, auxInliers);

            for(int i = 0; i < auxInliers.size(); ++i)
                inliers.push_back(matches[auxInliers[i]][0].trainIdx);

            //std::cout << "DETECTION   Inliers: " << inliers.size() << std::endl;    
            if( _tracking && inliers.size() > 10 )
            {
                //resetKalman();
                
                tvec = _tvec;
                rvec = _rvec;

                return true;
            } else {
                _tracking = false;
            }
        }
    } else {    // TRACKING

        std::vector<cv::KeyPoint> kps;
        std::vector<cv::Point3f> objectPoints;
        cv::Size gridSize = cv::Size(8,6);
        getKeyPointsUniform(inliers, 96, gridSize, kps, objectPoints);
        
        // TRACKPOINTS
        std::vector<cv::Point2f> uniform2d;
        trackKeyPoints(inputFrame, kps, objectPoints, uniform2d);

        if( objectPoints.size() != 0 )
        {
            std::vector<int> auxInliers;
            _tracking = cv::solvePnPRansac(objectPoints, uniform2d, _intrinsics, _distCoeff,
                                            _rvec, _tvec, _tracking, 200, 1.f, 0.99f, auxInliers);
    
            //std::cout << "TRACKING   Inliers: " << auxInliers.size() << std::endl;
            if( _tracking && auxInliers.size() > 10 )
            {
                //std::cout << "rvec: " << _rvec.at<double>(0) << ", " << _rvec.at<double>(1) << ", " << _rvec.at<double>(2) << std::endl;
                //std::cout << "tvec: " << _tvec.at<double>(0) << ", " << _tvec.at<double>(1) << ", " << _tvec.at<double>(2) << std::endl;
                
                //updateKalman();
        //_file << _tvec.at<double>(0) <<" "<< _tvec.at<double>(1) <<" "<< _tvec.at<double>(2) <<" "<< 
        //         _rvec.at<double>(0) <<" "<< _rvec.at<double>(1) <<" "<< _rvec.at<double>(2) <<"\n";
                tvec = _tvec;
                rvec = _rvec;

                return true;
            } else {
                _tracking = false;
            }
        } else {
            _tracking = false;
        }
    }

    return false;
}

void ARTTrack::trackKeyPoints(const cv::Mat &inputImage, std::vector<cv::KeyPoint> &keyPoints, std::vector<cv::Point3f> &objectPoints, std::vector<cv::Point2f> &output2d)
{
    // Area from inputImage, template from marker image
    cv::Mat1b searchArea = cv::Mat1b(_areaSize, _areaSize);
    cv::Mat1b templ = cv::Mat1b(_templSize, _templSize);

    cv::Rect BB;
    cv::Mat H = getTransformationHomography(BB);

    cv::Mat warpedImage, warpedBorder;
    cv::warpPerspective(_image, warpedImage, H, cv::Size(BB.width, BB.height), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(127));    
    warpedBorder = cv::Mat(warpedImage.rows+_templSize, warpedImage.cols+_templSize, warpedImage.type(), cv::Scalar(127));
    warpedImage.copyTo(warpedBorder(cv::Rect(_templSize/2, _templSize/2, warpedImage.cols, warpedImage.rows)));
//cv::imshow("warped", warpedBorder);
    std::vector<cv::Point2f> pt2d;
    projectPoints(objectPoints, pt2d);
    
    for( int i = 0; i < keyPoints.size(); ++i )
    {
        if(pt2d[i].x > _areaSize/2 && pt2d[i].x < inputImage.cols - _areaSize/2 &&
           pt2d[i].y > _areaSize/2 && pt2d[i].y < inputImage.rows - _areaSize/2 )
        {
            templ = warpedBorder(cv::Range((int)(pt2d[i].y-BB.y+_templSize/2)-(_templSize/2), (int)(pt2d[i].y-BB.y+_templSize/2)+(_templSize/2)), 
                                cv::Range((int)(pt2d[i].x-BB.x+_templSize/2)-(_templSize/2), (int)(pt2d[i].x-BB.x+_templSize/2)+(_templSize/2)));
            searchArea = inputImage(cv::Range((int)pt2d[i].y-(_areaSize/2), (int)pt2d[i].y+(_areaSize/2)), 
                                    cv::Range((int)pt2d[i].x-(_areaSize/2), (int)pt2d[i].x+(_areaSize/2)));
/*cv::Mat templRes, areaRes;
cv::resize(templ, templRes, templ.size()*4);
cv::resize(searchArea, areaRes, searchArea.size()*4);
cv::circle(templRes, cv::Point2f(templRes.cols/2, templRes.rows/2), 2, cv::Scalar(255), 2);
*/
            cv::Point2f newLoc = templateMatching(searchArea, templ);
/*cv::circle(areaRes, cv::Point2f(newLoc.x*4, newLoc.y*4), 2, cv::Scalar(255), 2);            
cv::imshow("template", templRes);
cv::imshow("area", areaRes);
cv::waitKey(0);
*/            pt2d[i].x = pt2d[i].x + newLoc.x - _areaSize/2;
            pt2d[i].y = pt2d[i].y + newLoc.y - _areaSize/2;

            output2d.push_back(pt2d[i]);
        } else {
            keyPoints.erase(keyPoints.begin()+i, keyPoints.begin()+i+1);
            pt2d.erase(pt2d.begin()+i, pt2d.begin()+i+1);
            objectPoints.erase(objectPoints.begin()+i, objectPoints.begin()+i+1);
            --i;
        }
    }
}

cv::Point2f ARTTrack::templateMatching(const cv::Mat1b &img, const cv::Mat &templ)
{
    cv::Mat1f ccorr;
    cv::matchTemplate(img, templ, ccorr, CV_TM_CCORR_NORMED);
    cv::normalize(ccorr, ccorr, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc( ccorr, &minVal, &maxVal, &minLoc, &maxLoc );

    cv::Point2f loc;
    loc.x = maxLoc.x + templ.rows/2;
    loc.y = maxLoc.y + templ.cols/2;

    return loc;
}

cv::Mat ARTTrack::getTransformationHomography(cv::Rect &BB)
{
    std::vector<cv::Point2f> inputPoints, projPoints;
    std::vector<cv::Point3f> points3d;

    inputPoints.push_back(cv::Point2f(_templSize/2,                 _templSize/2));
    inputPoints.push_back(cv::Point2f(_image.cols - _templSize/2,   _templSize/2));
    inputPoints.push_back(cv::Point2f(_image.cols - _templSize/2,   _image.rows - _templSize/2));
    inputPoints.push_back(cv::Point2f(_templSize/2,                 _image.rows - _templSize/2));

    points3d.push_back(_corners3d[0]);
    points3d.push_back(_corners3d[1]);
    points3d.push_back(_corners3d[2]);
    points3d.push_back(_corners3d[3]);

    projectPoints(points3d, projPoints);

    BB.x = projPoints[0].x;
    BB.y = projPoints[0].y;
    BB.width = projPoints[0].x;
    BB.height = projPoints[0].y;
    for(int i = 1; i < projPoints.size(); ++i)
    {
        if(projPoints[i].x < BB.x)
            BB.x = projPoints[i].x;
        if(projPoints[i].y < BB.y)
            BB.y = projPoints[i].y;
        if(projPoints[i].x > BB.width)
            BB.width = projPoints[i].x;
        if(projPoints[i].y > BB.height)
            BB.height = projPoints[i].y;
    }
    BB.width -= BB.x;
    BB.height -= BB.y;

    for(int i = 0; i < projPoints.size(); ++i)
    {
        projPoints[i].x -= BB.x;
        projPoints[i].y -= BB.y;
    }

    cv::Mat transf = cv::getPerspectiveTransform(inputPoints, projPoints);
    
    return transf;
}

void ARTTrack::resetKalman()
{
    _KF.init(18, 6, 0);

    setIdentity(_KF.processNoiseCov, Scalar::all(1e-5));       // set process noise
    setIdentity(_KF.measurementNoiseCov, Scalar::all(1e-2));   // set measurement noise
    setIdentity(_KF.errorCovPost, Scalar::all(1));             // error covariance

    double dt = (double) ( std::clock() - _clock ) / CLOCKS_PER_SEC;
    _clock = std::clock();    
    
    // traslation
    _KF.transitionMatrix.at<float>(0,3) = dt;
    _KF.transitionMatrix.at<float>(1,4) = dt;
    _KF.transitionMatrix.at<float>(2,5) = dt;
    _KF.transitionMatrix.at<float>(3,6) = dt;
    _KF.transitionMatrix.at<float>(4,7) = dt;
    _KF.transitionMatrix.at<float>(5,8) = dt;
    _KF.transitionMatrix.at<float>(0,6) = 0.5*pow(dt,2);
    _KF.transitionMatrix.at<float>(1,7) = 0.5*pow(dt,2);
    _KF.transitionMatrix.at<float>(2,8) = 0.5*pow(dt,2);

    // rotation
    _KF.transitionMatrix.at<float>(9,12) = dt;
    _KF.transitionMatrix.at<float>(10,13) = dt;
    _KF.transitionMatrix.at<float>(11,14) = dt;
    _KF.transitionMatrix.at<float>(12,15) = dt;
    _KF.transitionMatrix.at<float>(13,16) = dt;
    _KF.transitionMatrix.at<float>(14,17) = dt;
    _KF.transitionMatrix.at<float>(9,15) = 0.5*pow(dt,2);
    _KF.transitionMatrix.at<float>(10,16) = 0.5*pow(dt,2);
    _KF.transitionMatrix.at<float>(11,17) = 0.5*pow(dt,2);

    _KF.measurementMatrix.at<float>(0,0) = 1;  // x
    _KF.measurementMatrix.at<float>(1,1) = 1;  // y
    _KF.measurementMatrix.at<float>(2,2) = 1;  // z
    _KF.measurementMatrix.at<float>(3,9) = 1;  // roll
    _KF.measurementMatrix.at<float>(4,10) = 1; // pitch
    _KF.measurementMatrix.at<float>(5,11) = 1; // yaw

    // fill previous state with detected _tvec & _rvec
    _KF.statePost.at<float>(0) = _tvec.at<double>(0);
    _KF.statePost.at<float>(1) = _tvec.at<double>(1);
    _KF.statePost.at<float>(2) = _tvec.at<double>(2);
    _KF.statePost.at<float>(9) = _rvec.at<double>(0);
    _KF.statePost.at<float>(10) = _rvec.at<double>(1);
    _KF.statePost.at<float>(11) = _rvec.at<double>(2);

    _clock = std::clock();    
}

void ARTTrack::updateKalman()
{
    float tp1, tp2, tp3, rp1, rp2, rp3, tm, rm;
    tp1 = (_tproc1%10) / std::pow(10, 10 - std::floor(_tproc1/10));
    tp2 = (_tproc2%10) / std::pow(10, 10 - std::floor(_tproc2/10));
    tp3 = (_tproc3%10) / std::pow(10, 10 - std::floor(_tproc3/10));
    rp1 = (_rproc1%10) / std::pow(10, 10 - std::floor(_rproc1/10));
    rp2 = (_rproc2%10) / std::pow(10, 10 - std::floor(_rproc2/10));
    rp3 = (_rproc3%10) / std::pow(10, 10 - std::floor(_rproc3/10));
    tm = (_tmeas%10) / std::pow(10, 10 - std::floor(_tmeas/10));
    rm = (_rmeas%10) / std::pow(10, 10 - std::floor(_rmeas/10));    
    //std::cout << "tproc1: " << tp1 << "tproc2: " << tp2 << "tproc3: " << tp3 << ", rproc1: " << rp1 << ", rproc2: " << rp2 << ", rproc3: " << rp3 << ", tmeas: " << tm << ", rmeas: " << rm << std::endl;

    // Actualization of process and measurement errors  
    _KF.processNoiseCov.at<float>(0,0) = tp1;
    _KF.processNoiseCov.at<float>(1,1) = tp1;
    _KF.processNoiseCov.at<float>(2,2) = tp1;
    _KF.processNoiseCov.at<float>(3,3) = tp2;
    _KF.processNoiseCov.at<float>(4,4) = tp2;
    _KF.processNoiseCov.at<float>(5,5) = tp2;
    _KF.processNoiseCov.at<float>(6,6) = tp3;
    _KF.processNoiseCov.at<float>(7,7) = tp3;
    _KF.processNoiseCov.at<float>(8,8) = tp3;
    _KF.processNoiseCov.at<float>(9,9) = rp1;
    _KF.processNoiseCov.at<float>(10,10) = rp1;
    _KF.processNoiseCov.at<float>(11,11) = rp1;
    _KF.processNoiseCov.at<float>(12,12) = rp2;
    _KF.processNoiseCov.at<float>(13,13) = rp2;
    _KF.processNoiseCov.at<float>(14,14) = rp2;
    _KF.processNoiseCov.at<float>(15,15) = rp3;
    _KF.processNoiseCov.at<float>(16,16) = rp3;
    _KF.processNoiseCov.at<float>(17,17) = rp3;

    _KF.measurementNoiseCov.at<float>(0,0) = tm;
    _KF.measurementNoiseCov.at<float>(1,1) = tm;
    _KF.measurementNoiseCov.at<float>(2,2) = tm;
    _KF.measurementNoiseCov.at<float>(3,3) = rm;
    _KF.measurementNoiseCov.at<float>(4,4) = rm;
    _KF.measurementNoiseCov.at<float>(5,5) = rm;

    // Actualization of dt using frame rate
    double dt = (double) ( std::clock() - _clock ) / CLOCKS_PER_SEC;
    _clock = std::clock();    
    
    _KF.transitionMatrix.at<float>(0,3) = dt;
    _KF.transitionMatrix.at<float>(1,4) = dt;
    _KF.transitionMatrix.at<float>(2,5) = dt;
    _KF.transitionMatrix.at<float>(3,6) = dt;
    _KF.transitionMatrix.at<float>(4,7) = dt;
    _KF.transitionMatrix.at<float>(5,8) = dt;
    _KF.transitionMatrix.at<float>(0,6) = 0.5*pow(dt,2);
    _KF.transitionMatrix.at<float>(1,7) = 0.5*pow(dt,2);
    _KF.transitionMatrix.at<float>(2,8) = 0.5*pow(dt,2);

    // rotation
    _KF.transitionMatrix.at<float>(9,12) = dt;
    _KF.transitionMatrix.at<float>(10,13) = dt;
    _KF.transitionMatrix.at<float>(11,14) = dt;
    _KF.transitionMatrix.at<float>(12,15) = dt;
    _KF.transitionMatrix.at<float>(13,16) = dt;
    _KF.transitionMatrix.at<float>(14,17) = dt;
    _KF.transitionMatrix.at<float>(9,15) = 0.5*pow(dt,2);
    _KF.transitionMatrix.at<float>(10,16) = 0.5*pow(dt,2);
    _KF.transitionMatrix.at<float>(11,17) = 0.5*pow(dt,2);

    // Prediction
    cv::Mat prediction = _KF.predict();

    cv::Mat_<float> measurement(6,1);
    measurement(0) = _tvec.at<double>(0);
    measurement(1) = _tvec.at<double>(1);
    measurement(2) = _tvec.at<double>(2);
    measurement(3) = _rvec.at<double>(0);
    measurement(4) = _rvec.at<double>(1);
    measurement(5) = _rvec.at<double>(2);
     
    if( prediction.at<float>(11)*measurement(5) < -8.f )
    {
        _KF.statePre.at<float>(9)  = _rvec.at<double>(0);   
        _KF.statePre.at<float>(10) = _rvec.at<double>(1);           
        _KF.statePre.at<float>(11) = _rvec.at<double>(2);
    }
    
    cv::Mat estimated = _KF.correct(measurement);
/*  _file << _tvec.at<double>(0) <<" "<< _tvec.at<double>(1) <<" "<< _tvec.at<double>(2) <<" "<< 
             _rvec.at<double>(0) <<" "<< _rvec.at<double>(1) <<" "<< _rvec.at<double>(2) <<" "<<
             prediction.at<float>(10) <<" "<< prediction.at<float>(13) <<" "<< prediction.at<float>(16) <<" "<< 
             estimated.at<float>(10) <<" "<< estimated.at<float>(13) <<" "<< estimated.at<float>(16) <<" "<<
             "\n";
    std::cout << "dt: " << dt << std::endl;
    std::cout << "Measurement position: " << measurement(4) << std::endl;
    std::cout << "ROT Y pred, position: " << prediction.at<float>(10) << ", velocity: " << prediction.at<float>(13) << ", aceleration: " << prediction.at<float>(16) << std::endl;
    std::cout << "ROT Y post, position: " << estimated.at<float>(10) << ", velocity: " << estimated.at<float>(13) << ", aceleration: " << estimated.at<float>(16) << std::endl;
*/    _tvec.at<double>(0) = estimated.at<float>(0);
    _tvec.at<double>(1) = estimated.at<float>(1);
    _tvec.at<double>(2) = estimated.at<float>(2);
    _rvec.at<double>(0) = estimated.at<float>(9);
    _rvec.at<double>(1) = estimated.at<float>(10);
    _rvec.at<double>(2) = estimated.at<float>(11);
}

void ARTTrack::filterKeyPoints(std::vector<cv::KeyPoint> &keyPoints, int maxKeyPoints)
{
    if(keyPoints.size() > maxKeyPoints)
    {
        std::sort(keyPoints.begin(), keyPoints.end(), 
            [](const cv::KeyPoint & a, const cv::KeyPoint & b) -> bool {
                return a.response > b.response;
            });
        keyPoints.erase(keyPoints.begin()+maxKeyPoints, keyPoints.end());
    }
}

void ARTTrack::filterMatches(std::vector<std::vector<cv::DMatch> > &matches, int maxMatches)
{
    if(matches.size() > maxMatches)
    {
        std::sort(matches.begin(), matches.end(), 
            [](const std::vector<cv::DMatch> & a, const std::vector<cv::DMatch> & b) -> bool {
                return (a[1].distance-a[0].distance)/a[0].distance > (b[1].distance-b[0].distance)/b[0].distance;
            });
        matches.erase(matches.begin()+maxMatches, matches.end());
    } 
}

void ARTTrack::projectPoints(const std::vector<cv::Point3f> &objPoints, std::vector<cv::Point2f> &projPoints)
{
    cv::projectPoints(objPoints, _rvec, _tvec, _intrinsics, _distCoeff, projPoints);
}

void ARTTrack::drawMarker(cv::Mat &image, const std::vector<int> &inliers)
{
    std::vector<cv::Point2f> projCorners;
    projectPoints(_corners3d, projCorners);

    cv::line(image, projCorners[0], projCorners[1], cv::Scalar(0,255,0), 2);
    cv::line(image, projCorners[1], projCorners[2], cv::Scalar(0,255,0), 2);
    cv::line(image, projCorners[2], projCorners[3], cv::Scalar(0,255,0), 2);
    cv::line(image, projCorners[3], projCorners[0], cv::Scalar(0,255,0), 2);
    cv::line(image, projCorners[0], projCorners[2], cv::Scalar(0,255,0), 2);

    std::vector<cv::Point3f> inliers3d;
    for(int i = 0; i < inliers.size(); ++i)
        inliers3d.push_back(_objectPoints[i]);

    std::vector<cv::Point2f> projInliers;
    projectPoints(inliers3d, projInliers);

    for(int i = 0; i < projInliers.size(); ++i)
        cv::circle(image, projInliers[i], 1, cv::Scalar(0,0,255), 2);
}

void ARTTrack::drawRectangle(cv::Mat &image, const cv::Point2f &center, const cv::Size &size, const cv::Scalar &color)
{
    std::vector<cv::Point3f> corners3d(4);

    corners3d[0] = cv::Point3f(center.x-(size.width/2), center.y-(size.height/2), 0.f);
    corners3d[1] = cv::Point3f(center.x+(size.width/2), center.y-(size.height/2), 0.f);
    corners3d[2] = cv::Point3f(center.x+(size.width/2), center.y+(size.height/2), 0.f);
    corners3d[3] = cv::Point3f(center.x-(size.width/2), center.y+(size.height/2), 0.f);

    std::vector<cv::Point2f> projCorners;
    projectPoints(corners3d, projCorners);

    cv::line(image, projCorners[0], projCorners[1], color, 2);
    cv::line(image, projCorners[1], projCorners[2], color, 2);
    cv::line(image, projCorners[2], projCorners[3], color, 2);
    cv::line(image, projCorners[3], projCorners[0], color, 2);
}

void ARTTrack::getKeyPointsUniform(std::vector<int> &inliers, int maxPoints, const cv::Size &gridSize, std::vector<cv::KeyPoint> &dst2d, std::vector<cv::Point3f> &dst3d)
{
	/*if( inliers.size() < (size_t)maxPoints ){
        for(int i = 0; i < inliers.size(); ++i)
            dst.push_back(_keyPoints[inliers[i]]);
    	return;
    }*/

    std::vector<bool> kpsUsed( _keyPoints.size(), false );
    for(int i= 0; i < inliers.size(); ++i)
        kpsUsed.at(inliers[i]) = true;    
	
    int pointsLeft = maxPoints;
    std::vector<int> inliersLeft(inliers.size());
    inliersLeft = inliers;

	int pointsLeftInCell;
    int cellW = (_image.cols-_templSize) / gridSize.width;
	int cellH = (_image.rows-_templSize) / gridSize.height;	
	for( int i = 0; i < gridSize.width; ++i )
    {
        for(int j = 0; j < gridSize.height; ++j)
        {
            pointsLeftInCell = (int)(pointsLeft/(gridSize.width*gridSize.height - ( (j*gridSize.width) + i) ) );
            // First use the inliers
            for(int h = 0; h < inliersLeft.size(); ++h)
            {   
                if(_keyPoints[inliersLeft[h]].pt.x > (i*cellW)+_templSize/2 && _keyPoints[inliersLeft[h]].pt.x < ((i+1)*cellW)+_templSize/2 &&
                   _keyPoints[inliersLeft[h]].pt.y > (j*cellH)+_templSize/2 && _keyPoints[inliersLeft[h]].pt.y < ((j+1)*cellH)+_templSize/2 )
                {
                    if(_keyPoints[inliersLeft[h]].pt.x > _templSize/2 && _keyPoints[inliersLeft[h]].pt.x < _image.cols - _templSize/2 &&
                       _keyPoints[inliersLeft[h]].pt.y > _templSize/2 && _keyPoints[inliersLeft[h]].pt.y < _image.rows - _templSize/2)
                    {
                        dst2d.push_back(_keyPoints[inliersLeft[h]]);
                        dst3d.push_back(_objectPoints[inliersLeft[h]]);
                        inliersLeft.erase(inliersLeft.begin()+h, inliersLeft.begin()+h+1);
                        pointsLeftInCell--;
                        pointsLeft--;
                        h--;
                    }
                }
                if(pointsLeftInCell < 1)
                    break;
            }

            // Fill gaps with _keyPoints
            for(int h = 0; h < _keyPoints.size(); ++h)
            {   
                if( !kpsUsed[h] )
                {
                    if(_keyPoints[h].pt.x > (i*cellW)+_templSize/2 && _keyPoints[h].pt.x < ((i+1)*cellW)+_templSize/2 &&
                       _keyPoints[h].pt.y > (j*cellH)+_templSize/2 && _keyPoints[h].pt.y < ((j+1)*cellH)+_templSize/2 )
                    {
                        if(_keyPoints[h].pt.x > _templSize/2 && _keyPoints[h].pt.x < _image.cols - _templSize/2 &&
                            _keyPoints[h].pt.y > _templSize/2 && _keyPoints[h].pt.y < _image.rows - _templSize/2)
                        {
                            inliers.push_back(h);
                            dst2d.push_back(_keyPoints[h]);
                            dst3d.push_back(_objectPoints[h]);                            
                            pointsLeftInCell--;
                            pointsLeft--;
                            kpsUsed[h] = true;
                        }
                    }
                }
                if(pointsLeftInCell < 1)
                    break;
            }
        }
	}
}