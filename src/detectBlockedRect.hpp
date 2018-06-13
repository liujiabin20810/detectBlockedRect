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
		float lineLen_th; //  line minimum length ( lsd detect)
		float m_dTheta; // line polar angle variation range [-m_dTheta,+m_dTheta], it is Degree, not Radian.
		float theta_dt; // the  one line angle variation range
		float dR_th; // the one line radius variation range

		float merge_lineLen_th_Hori; // horizontal line merged minimum length
		float merge_lineLen_th_Verti; // vertical line merged minimum length
	};
	
	template <class PointT> class CLineParams
	{
	public:
		CLineParams(){};

		CLineParams(float r, float theta, PointT p1, PointT p2)
		{
			this->pr = r;
			this->ptheta = theta;
			this->start_vertex = p1;
			this->end_vertex = p2;
		}

	public:
		float pr;		// polar coordinates radius
		float ptheta; // polar coordinates angle
		PointT start_vertex, end_vertex; // line start point & end point in  Cartesian coordinates 
	};

	typedef CLineParams<Point2f> CLineParams2f;

	// line data: Cartesian coordinates points , polar coordinates points, cluster idx
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

		void mergeLines();

		bool getRect(std::vector<Point2f>& points);

		void drawLines(Mat& img);
		void drawMergeLines(Mat& img);

	private:

		int m_width, m_height;

		Ptr<cv::line_descriptor::LSDDetector>  lsd_detector;
		BRectParams params;

		std::vector<KeyLine> lines_Hori, lines_Verti; // 保存线段端点，长度，宽度
		std::vector<int> index_Hori, index_Verti; //直线标签
		std::vector<Vec2d> polar_lines_Hori, polar_lines_Verti; // 保存直线的极坐标参数

		std::vector<CLineParams2f> merge_lines_Hori, merge_lines_Verti;
		// each line have lots of little keyLine
		std::vector<std::vector<LineAllData> > linesData;

		bool detectLinesHori(Mat src); // detect horizontal lines
		bool detectLinesVerti(Mat src); // detect vertical lines

		bool mergeLinesHori();
		bool mergeLinesVerti();

		bool mergeLinesHori(std::vector<LineAllData>& vLda, std::vector<CLineParams2f>& plines);
		bool mergeLinesVerti(std::vector<LineAllData>& vLda, std::vector<CLineParams2f>& plines);

		void drawLinesHori(Mat& img);
		void drawLinesVerti(Mat& img);

		void drawMergeLinesHori(Mat& img);
		void drawMergeLinesVerti(Mat& img);

		Vec2d getPolarLine(Vec4d p);

		bool getLineIndx(std::vector<Vec2d> polarLine, std::vector<int>& index);
		
	};

}
#endif // detectBlockedRect_h__