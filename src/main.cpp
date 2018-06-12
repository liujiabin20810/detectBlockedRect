#pragma once
#include "detectBlockedRect.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int draw_x[2] = { 0, 1 };
int draw_y[2] = { 2, 3 };

#define  CV_DRAW(x,i)  draw_##x[i]

int main(int argc, char** argv)
{
	cout << CV_DRAW(x,0) << " " << CV_DRAW(y,1) << endl;

	if(argc < 2)
		return -1;
	
	Mat img = imread(argv[1],1);
	
	if(img.empty() )
		return -1;
	
	BlockedRect detect_bdRect;
	
	detect_bdRect.detect(img);
	
	Mat drawIm = img.clone();

	detect_bdRect.drawLines(drawIm);

	namedWindow("src", 0);
	namedWindow("draw", 0);

	imshow("src", img);
	imshow("draw", drawIm);
	waitKey();

	return 0;
}