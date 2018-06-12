/*!
 * \file detectBlockedRect.cpp
 *
 * \author Lewis
 * \date Jun 2018
 *
 *  This is a method for detect nearly horizontal rect, which blocked by other things. Because of the 
 *  contour of the rect is broken, it can not be recognized by findContours() method.
 */
#pragma once
#include "detectBlockedRect.hpp"

#ifdef _DEBUG
#include <opencv2/highgui/highgui.hpp>
#endif // _DEBUG

#include <fstream>

using namespace std;

namespace cv
{
	BlockedRect::BlockedRect()
	{
		lsd_detector = cv::line_descriptor::LSDDetector::createLSDDetector();
	}

	BlockedRect::~BlockedRect()
	{
		//delete lsd_detector;
	}
//-------------------------------------------------------------------
	bool BlockedRect::initParams(std::string filename)
	{
		if (filename.empty())
			return false;

		ifstream fin(filename.c_str());
		if (!fin.is_open())
			return true;

		fin >> params.th1 >> params.th2 >> params.th3;
		fin.close();

		if (params.th1 < 0.0 || params.th2 < 0.0 || params.th3 < 0.0)
			return false;

		return true;
	}
//-------------------------------------------------------------------
	void BlockedRect::detect(Mat image)
	{
		detectLinesHori(image);
		detectLinesVerti(image);

	}
//-------------------------------------------------------------------
	bool BlockedRect::getRect(vector<Point2f>& points)
	{
		return false;
	}
//-------------------------------------------------------------------
//-------------------------------------------------------------------
	bool BlockedRect::detectLinesHori(Mat src)
	{
		Mat gray;
		if (src.channels() == 3)
			cvtColor(src, gray, CV_BGR2GRAY);
		else
			src.copyTo(gray);

		Mat ydiff;
		Mat  abs_ydiff;
		//y方向梯度计算  
		Sobel(gray, ydiff, CV_16S, 0, 1, 3);
		convertScaleAbs(ydiff, abs_ydiff);

		Mat element = getStructuringElement(MORPH_CROSS, Size(25, 1));
		erode(abs_ydiff, abs_ydiff, element);
		dilate(abs_ydiff, abs_ydiff, element);

		Mat gradYBin;
		threshold(abs_ydiff, gradYBin, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

#ifdef _DEBUG
		namedWindow("gradY", 0);
		imshow("gradY", gradYBin);
		waitKey();
#endif // _DEBUG

		lsd_detector->detect(gradYBin, lines_Hori, 2, 1);

		return true;
	}
//-------------------------------------------------------------------	
	bool BlockedRect::detectLinesVerti(Mat src)
	{
		Mat gray;
		if (src.channels() == 3)
			cvtColor(src, gray, CV_BGR2GRAY);
		else
			src.copyTo(gray);

		Mat xdiff;
		Mat abs_xdiff;
		//x方向梯度计算
		Sobel(gray, xdiff, CV_16S, 1, 0, 3);
		convertScaleAbs(xdiff, abs_xdiff);

		Mat element = getStructuringElement(MORPH_CROSS, Size(1, 25));
		erode(abs_xdiff, abs_xdiff, element);
		dilate(abs_xdiff, abs_xdiff, element);

		Mat gradXBin;
		threshold(abs_xdiff, gradXBin, 0, 255, CV_THRESH_OTSU | CV_THRESH_BINARY);

#ifdef _DEBUG
		namedWindow("gradX", 0);
		imshow("gradX", gradXBin);
		waitKey();
#endif // _DEBUG

		lsd_detector->detect(gradXBin, lines_Verti, 2, 1);

		return true;
	}
//-------------------------------------------------------------------
	bool BlockedRect::mergeLinesHori()
	{
		return false;
	}
//-------------------------------------------------------------------
	bool BlockedRect::mergeLinesVerti()
	{
		return false;
	}
//-------------------------------------------------------------------
	void BlockedRect::drawLines(Mat& img)
	{
		drawLinesHori(img);
		drawLinesVerti(img);
	}
	//-------------------------------------------------------------------
#define  CV_DRAWLINES(w) \
	void BlockedRect::drawLines##w( Mat &img ) \
		{	   \
		for (int i = 0; i < lines_##w.size(); i++) \
				 {														\
		 KeyLine keyline = lines_##w[i]; \
		 if( 0 != keyline.octave ) continue; \
		 line(img, keyline.getStartPoint(), keyline.getEndPoint(),Scalar(0, 0, 255), 2); \
		 }				\
	}

	CV_DRAWLINES(Hori);
	CV_DRAWLINES(Verti);
//-------------------------------------------------------------------
}