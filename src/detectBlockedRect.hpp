/*!
 * \file detectBlockedRect.hpp
 *
 * \author Lewis
 * \date Jun 2018
 *
 * This is a method for detect nearly horizontal rect, which blocked by other things. Because of the
 *  contour of the rect is broken, it can not be recognized by findContours() method.
 */

#ifndef detectBlockedRect_h__
#define detectBlockedRect_h__

#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>

#include <iostream>
#include <vector>



namespace cv
{
	typedef  line_descriptor::KeyLine KeyLine;

	// method params
	struct BRectParams
	{
		float th1;
		float th2;
		float th3;
	};
	
	// line data: Cartesian coordinates points , polar points, cluster idx
	class LineAllData
	{
	public:

		LineAllData(){}
		LineAllData(int index, Vec2d polar_line, KeyLine line)
		{
			this->index = index;
			this->polar_line = polar_line;
			this->kline = line;
		}
		~LineAllData(){}

		bool operator<(const LineAllData & t) const
		{
			return index < t.index;
		}

	public:
		int index;
		Vec2d polar_line;
		KeyLine kline;
	};

	// the main class : detect the rect ,which blocked by other things
	class BlockedRect
	{
	public:
		BlockedRect();
		~BlockedRect();

		// init params
		bool initParams(std::string filename);

		void detect(Mat image);

		bool getRect(std::vector<Point2f>& points);

		void drawLines(Mat& img);

	private:

		Ptr<cv::line_descriptor::LSDDetector>  lsd_detector;
		BRectParams params;

		std::vector<KeyLine> lines_Hori, lines_Verti; // 保存线段端点，长度，宽度
		std::vector<int> index_Hori, index_Verti; //直线标签
		std::vector<Vec2d> polar_lines_Hori, polar_lines_Verti; // 保存直线的极坐标参数

		// each line have lots of little keyLine
		std::vector<std::vector<LineAllData> > linesData;

		bool detectLinesHori(Mat src); // detect horizontal lines
		bool detectLinesVerti(Mat src); // detect vertical lines

		bool mergeLinesHori();
		bool mergeLinesVerti();

		void drawLinesHori(Mat& img);
		void drawLinesVerti(Mat& img);
	};

}
#endif // detectBlockedRect_h__