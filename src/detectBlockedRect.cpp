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

#define DEG2RAD(d)  ((d)*CV_PI/180.0)

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

		fin >> params.lineLen_th >> params.m_dTheta >> params.theta_dt >> params.dR_th;
		fin >> params.merge_lineLen_th_Hori >> params.merge_lineLen_th_Verti;
		
		fin.close();

		if (params.lineLen_th <= 1e-6 || params.m_dTheta <= 1e-6 || params.theta_dt <= 1e-6)
			return false;

#ifdef _DEBUG
		cout << "0: " << DEG2RAD(params.m_dTheta) << endl;
		cout << "90: " << DEG2RAD(90 - params.m_dTheta) << "  " << DEG2RAD(90 + params.m_dTheta) << endl;
		cout << "180: " << DEG2RAD(180 - params.m_dTheta) << " " << DEG2RAD(180 + params.m_dTheta) << endl;
		cout << "270: " << DEG2RAD(270 - params.m_dTheta) << " " << DEG2RAD(270 + params.m_dTheta) << endl;
#endif // _DEBUG

		return true;
	}
//-------------------------------------------------------------------
	void BlockedRect::mergeLines()
	{
		mergeLinesHori();
		mergeLinesVerti();
	}
//-------------------------------------------------------------------
	void BlockedRect::detect(Mat image)
	{
		if (image.empty())
			return;

		m_width = image.cols;
		m_height = image.rows;

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
	Vec2d BlockedRect::getPolarLine(Vec4d p)
	{

		if (fabs(p[0] - p[2]) < 1e-5)//垂直直线
		{
			if (p[0] > 0)
				return Vec2d(fabs(p[0]), 0);
			else
				return Vec2d(fabs(p[0]), CV_PI);
		}

		if (fabs(p[1] - p[3]) < 1e-5) //水平直线
		{
			if (p[1] > 1e-5)
				return Vec2d(fabs(p[1]), CV_PI / 2);
			else
				return Vec2d(fabs(p[1]), 3 * CV_PI / 2);
		}

		float k = (p[1] - p[3]) / (p[0] - p[2]);
		float y_intercept = p[1] - k*p[0];

		float theta;
		// atan 值域范围[-pi/2,pi/2]; atan2 值域范围[-pi,pi]
		// 根据直线斜率与截距 判断角度所在象限
		if (k < 0 && y_intercept > 0)   // 第一象限
			theta = atan(-1 / k);
		else if (k > 0 && y_intercept > 0)  // 第二象限，
			theta = CV_PI + atan(-1 / k);
		else if (k < 0 && y_intercept < 0)  // 第三象限
			theta = CV_PI + atan(-1 / k);
		else if (k> 0 && y_intercept < 0) // 第四象限
			theta = 2 * CV_PI + atan(-1 / k);

		float _cos = cos(theta);
		float _sin = sin(theta);

		float r = fabs(p[0] * _cos + p[1] * _sin);

		return Vec2d(r, theta);
	}
//-------------------------------------------------------------------
	bool BlockedRect::getLineIndx(vector<Vec2d> polarLine, vector<int>& index)
	{
		int polar_num = polarLine.size();
		if (polar_num < 2)
		{
			return false;
		}

		index.clear();
		index.resize(polar_num);

		// init line index and the marked flag
		vector<int> line_indexed;
		line_indexed.resize(polar_num);
		for (int i = 0; i < polar_num; i++)
		{
			index[i] = i;
			line_indexed[i] = 0;
		}

		for (int i = 0; i < polar_num - 1; i++)
		{
			Vec2d pl1 = polarLine[i];
			for (int j = i + 1; j < polar_num; j++)
			{
				if ( 1 == line_indexed[j] )  // marked
					continue; 
				Vec2d pl2 = polarLine[j];

				// 计算两条直线的极坐标差
				float dTheta = fabs(pl2[1] - pl1[1]);
				float dR = fabs(pl2[0] - pl1[0]);

				float meanR = fabs( pl2[0] + pl1[0] ) / 2;
				// have nearly same angle
				if (dTheta < DEG2RAD(params.theta_dt) || fabs(2 * CV_PI - dTheta) < DEG2RAD(params.theta_dt) ) 
				{
					// have nearly same radius
					if (dR < params.dR_th)
					{
						// same index
						line_indexed[j] = 1;
						index[j] = index[i];
					}
				}

			} // for j
		} // for i


		return true;
	}
//-------------------------------------------------------------------
	bool BlockedRect::mergeLinesHori(std::vector<LineAllData>& vLda, vector<CLineParams2f>& plines)
	{
		if (vLda.size() < 2)
			return false;

		vector<int> index_used;
		int st = vLda[0].index;
		index_used.push_back(st);
		//////////////////////////////////////////////////////////////////////////
		// get all the differen index :  000111222333 -> 0123
		// count the different index : 4
		int n_line = 1;
		for (int i = 0; i < vLda.size(); i++)
		{
			if (vLda[i].index != st)
			{
				st = vLda[i].index;
				index_used.push_back(st);
				n_line++;
			}
		}

		if (n_line < 2) return false;
		////////////////////////////////////////////////////////////////////////// split to different pieces
		// split the vLda : 000111222333-> 000 \ 111 \ 222 \ 333
		int j = 0;
		linesData.clear();

		for (int i = 0; i < n_line; i++)
		{
			int idx = index_used[i];

			vector<LineAllData> split_line_data;
			float sum_length = 0;

			for (; j < vLda.size() && (vLda[j].index == idx); j++)
			{
				split_line_data.push_back(vLda[j]);
				sum_length += vLda[j].kline.lineLength;
			}
			linesData.push_back(split_line_data);

			cout << "length: " << sum_length << endl;
		}
		////////////////////////////////////////////////////////////////////////// merge each pieces to one line
		// merge  linesData
		CLineParams2f up_pline, down_pline;
		up_pline.start_vertex.y = m_height / 2;
		down_pline.start_vertex.y = m_height / 2;

		for (int i = 0; i < n_line; i++)
		{
			vector<LineAllData> cline_data = linesData[i];
			int m = cline_data.size();
			if (m <= 1)
				continue;

			// length, position,bound box[minx,miny,maxx,maxy]
			float sum_length = 0.0;
			float mx = 0.0;
			Point2f p1(m_width, m_height), p2(0, 0);
			float sum_r = 0.0;
			float sum_theta = 0.0;

			for (int j = 0; j < m; j++)
			{
				LineAllData cline = cline_data[j];

				if (cline.polar_line[0] < 5 ) // 过滤位于图片中间的直线
					continue;

				KeyLine line_segment = cline.kline;
				sum_length += line_segment.lineLength; // length

				sum_r += cline.polar_line[0];

				sum_theta += cline.polar_line[1];

				// 根据X坐标选择端点
				float min_x = MIN(line_segment.startPointX, line_segment.endPointX);
				float min_x_y = line_segment.startPointX < line_segment.endPointX ? line_segment.startPointY : line_segment.endPointY;
				float max_x = MAX(line_segment.startPointX, line_segment.endPointX);
				float max_x_y = line_segment.startPointX> line_segment.endPointX ? line_segment.startPointY : line_segment.endPointY;

				if (p1.x > min_x)
				{
					p1.x = min_x;
					p1.y = min_x_y;
				}

				if (p2.x < max_x)
				{
					p2.x = max_x;
					p2.y = max_x_y;
				}
			}

			// 同类线段上下两端的长度
			float line_sum_len = sqrtf((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));

			if (line_sum_len > params.merge_lineLen_th_Hori && sum_length > params.merge_lineLen_th_Hori / 2.0)
			{
				float r = sum_r / m;
				float theta = sum_theta / m;

				if (up_pline.start_vertex.y > p1.y)  // 上横线
				{
					up_pline.start_vertex = p1;
					up_pline.end_vertex = p2;
					up_pline.pr = r;
					up_pline.ptheta = theta;
				}

				if (down_pline.start_vertex.y < p1.y) // 下横线
				{
					down_pline.start_vertex = p1;
					down_pline.end_vertex = p2;
					down_pline.pr = r;
					down_pline.ptheta = theta;
				}
			}
		}

		if (up_pline.start_vertex.y < m_height / 2 - 1) 
			plines.push_back(up_pline);
		if (down_pline.start_vertex.y > m_width / 2 + 1)
			plines.push_back(down_pline);

		return plines.size() == 2;

	}
//-------------------------------------------------------------------
	bool BlockedRect::mergeLinesVerti(std::vector<LineAllData>& vLda, std::vector<CLineParams2f>& plines)
	{
		if (vLda.size() < 2)
			return false;

		vector<int> index_used;
		int st = vLda[0].index;
		index_used.push_back(st);
		//////////////////////////////////////////////////////////////////////////
		// get all the differen index :  000111222333 -> 0123
		// count the different index : 4
		int n_line = 1;
		for (int i = 0; i < vLda.size(); i++)
		{
			if (vLda[i].index != st)
			{
				st = vLda[i].index;
				index_used.push_back(st);
				n_line++;
			}
		}

		if (n_line < 2) return false;
		////////////////////////////////////////////////////////////////////////// split to different pieces
		// split the vLda : 000111222333-> 000 \ 111 \ 222 \ 333
		int j = 0;
		linesData.clear();

		for (int i = 0; i < n_line; i++)
		{
			int idx = index_used[i];

			vector<LineAllData> split_line_data;
			float sum_length = 0;

			for (; j < vLda.size() && (vLda[j].index == idx); j++)
			{
				split_line_data.push_back(vLda[j]);
				sum_length += vLda[j].kline.lineLength;
			}
			linesData.push_back(split_line_data);

			cout << "length: " << sum_length << endl;
		}
		////////////////////////////////////////////////////////////////////////// merge each pieces to one line
		// merge  linesData
		CLineParams2f left_pline, right_pline;
		left_pline.start_vertex.x = m_width / 2;
		right_pline.start_vertex.x = m_width / 2;

		for (int i = 0; i < n_line; i++)
		{
			vector<LineAllData> cline_data = linesData[i];
			int m = cline_data.size();
			if (m <= 1)
				continue;

			// length, position,bound box[minx,miny,maxx,maxy]
			float sum_length = 0.0;
			float mx = 0.0;
			Point2f p1(m_width, m_height), p2(0, 0);
			float sum_r = 0.0;
			float sum_theta = 0.0;

			for (int j = 0; j < m; j++)
			{
				LineAllData cline = cline_data[j];

				if (cline.polar_line[0] < 5) // 过滤位于图片中间的直线
					continue;

				KeyLine line_segment = cline.kline;
				sum_length += line_segment.lineLength; // length

				sum_r += cline.polar_line[0];

				sum_theta += cline.polar_line[1];

				// 根据Y坐标选择端点
				float min_y = MIN(line_segment.startPointY, line_segment.endPointY);
				float min_y_x = line_segment.startPointY < line_segment.endPointY ? line_segment.startPointX : line_segment.endPointX;
				float max_y = MAX(line_segment.startPointY, line_segment.endPointY);
				float max_y_x = line_segment.startPointY > line_segment.endPointY ? line_segment.startPointX : line_segment.endPointX;

				if (p1.y > min_y)
				{
					p1.y = min_y;
					p1.x = min_y_x;
				}

				if (p2.y < max_y)
				{
					p2.y = max_y;
					p2.x = max_y_x;
				}
			}

			// 同类线段上下两端的长度
			float line_sum_len = sqrtf((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));

			if (line_sum_len > params.merge_lineLen_th_Verti && sum_length > params.merge_lineLen_th_Verti / 2.0)
			{
				float r = sum_r / m;
				float theta = sum_theta / m;

				if (left_pline.start_vertex.x > p1.x)  // 左边直线
				{
					left_pline.start_vertex = p1;
					left_pline.end_vertex = p2;
					left_pline.pr = r;
					left_pline.ptheta = theta;
				}

				if (right_pline.start_vertex.x < p1.x) // 右边直线
				{
					right_pline.start_vertex = p1;
					right_pline.end_vertex = p2;
					right_pline.pr = r;
					right_pline.ptheta = theta;
				}
			}
		}

		if (left_pline.start_vertex.x < m_width / 2 - 1)
			plines.push_back(left_pline);
		if (right_pline.start_vertex.x > m_width / 2 + 1)
			plines.push_back(right_pline);

		return plines.size() == 2;
	}
//-------------------------------------------------------------------
//-------------------------------------------------------------------
//-------------------------------------------------------------------
//-------------------------------------------------------------------
//-------------------------------------------------------------------
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

		int width = src.cols;
		int height = src.rows;

		vector<KeyLine> lines;
		lsd_detector->detect(gradYBin, lines, 2, 1);

		size_t  n = lines.size();
		for (size_t  i = 0; i < n; i++)
		{
			KeyLine kl = lines[i];
			
			float line_len = kl.lineLength;
			if ( params.lineLen_th > line_len) continue; // ignore the too little item

			float x1, y1, x2, y2;
			x1 = kl.startPointX, y1 = kl.startPointY;
			x2 = kl.endPointX, y2 = kl.endPointY;

			Vec2d polarline = getPolarLine(Vec4d(x1 - width / 2.0, height / 2.0 - y1, x2 - width / 2.0, height / 2.0 - y2));

			//Line is nearly horizontal , so ignore the line which polar angle out of the defined range.
			if (polarline[1] > DEG2RAD(params.m_dTheta) && polarline[1] <= DEG2RAD(90 - params.m_dTheta ))
				continue;
			if (polarline[1] > DEG2RAD(90 + params.m_dTheta) && polarline[1] < DEG2RAD(180 - params.m_dTheta))
				continue;
			if (polarline[1] > DEG2RAD(180 + params.m_dTheta) && polarline[1] <= DEG2RAD(270 - params.m_dTheta) )
				continue;
			if (polarline[1] > DEG2RAD(270 + params.m_dTheta) && polarline[1] < DEG2RAD(360 - params.m_dTheta))
				continue;

			lines_Hori.push_back(kl);
			polar_lines_Hori.push_back(polarline);
		}

		return lines_Hori.size() >= 2;
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
		
		int width = src.cols;
		int height = src.rows;

		vector<KeyLine> lines;
		lsd_detector->detect(gradXBin, lines, 2, 1);
		
		size_t  n = lines.size();
		for (size_t i = 0; i < n; i++)
		{
			KeyLine kl = lines[i];

			float line_len = kl.lineLength;
			if (params.lineLen_th > line_len) continue; // ignore the too little item

			float x1, y1, x2, y2;
			x1 = kl.startPointX, y1 = kl.startPointY;
			x2 = kl.endPointX, y2 = kl.endPointY;

			Vec2d polarline = getPolarLine(Vec4d(x1 - width / 2.0, height / 2.0 - y1, x2 - width / 2.0, height / 2.0 - y2));

			//Line is nearly horizontal , so ignore the line which polar angle out of the defined range.
			if (polarline[1] > DEG2RAD(params.m_dTheta) && polarline[1] <= DEG2RAD(90 - params.m_dTheta))
				continue;
			if (polarline[1] > DEG2RAD(90 + params.m_dTheta) && polarline[1] < DEG2RAD(180 - params.m_dTheta))
				continue;
			if (polarline[1] > DEG2RAD(180 + params.m_dTheta) && polarline[1] <= DEG2RAD(270 - params.m_dTheta))
				continue;
			if (polarline[1] > DEG2RAD(270 + params.m_dTheta) && polarline[1] < DEG2RAD(360 - params.m_dTheta))
				continue;

			lines_Verti.push_back(kl);
			polar_lines_Verti.push_back(polarline);
		}

		return lines_Verti.size() >= 2;
	}
//-------------------------------------------------------------------
	bool BlockedRect::mergeLinesHori()
	{
		if (!getLineIndx(polar_lines_Hori, index_Hori))
			return false;

		size_t  line_num = lines_Hori.size();

		vector<LineAllData> line_data;
		for (size_t  i = 0; i < line_num; i++)
		{
			KeyLine kl = lines_Hori[i];
			Vec2d pl = polar_lines_Hori[i];
			int idx = index_Hori[i];

			LineAllData lda(idx,pl,kl);
			line_data.push_back(lda);
		}
		// sort by the line index
		sort(line_data.begin(), line_data.end());
		for (LineAllData lda : line_data)
			cout << lda.index << " ";
		cout << endl;

//		vector<CLineParams2f> merged_lines;
		mergeLinesHori(line_data, merge_lines_Hori);

		return false;
	}
//-------------------------------------------------------------------
	bool BlockedRect::mergeLinesVerti()
	{
		if (!getLineIndx(polar_lines_Verti, index_Verti))
			return false;

		size_t  line_num = lines_Verti.size();

		vector<LineAllData> line_data;
		for (size_t i = 0; i < line_num; i++)
		{
			KeyLine kl = lines_Verti[i];
			Vec2d pl = polar_lines_Verti[i];
			int idx = index_Verti[i];

			LineAllData lda(idx, pl, kl);
			line_data.push_back(lda);
		}
		// sort by the line index
		sort(line_data.begin(), line_data.end());
		for (LineAllData lda : line_data)
			cout << lda.index << " ";
		cout << endl;

		mergeLinesVerti(line_data, merge_lines_Verti);
		return false;
	}
//-------------------------------------------------------------------
	void BlockedRect::drawLines(Mat& img)
	{
		drawLinesHori(img);
		drawLinesVerti(img);
	}
//-------------------------------------------------------------------
	void BlockedRect::drawMergeLines(Mat& img)
	{
		drawMergeLinesHori(img);
		drawMergeLinesVerti(img);
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
#define  CV_DRAWMERGELINES(w) \
	void BlockedRect::drawMergeLines##w( Mat &img ) \
	{	   \
		for (int i = 0; i < merge_lines_##w.size(); i++) \
		 {														\
			 CLineParams2f ml = merge_lines_##w[i]; \
			line(img, ml.start_vertex, ml.end_vertex,Scalar(255, 0, 0), 2); \
		}				\
	}	
	CV_DRAWMERGELINES(Hori);
	CV_DRAWMERGELINES(Verti);
}