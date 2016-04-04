#ifndef TARGETDETECTOR_H
#define TARGETDETECTOR_H

#include<iostream>
#include"opencv2/opencv.hpp"
using namespace std;
using namespace cv;

typedef struct
{
	Point center;
	Rect bound;
	vector<Point>contour;
}ConnectObj;

class CObj
{
	public:
		CObj(ConnectObj Area);
		void imshowArea();
		int getAreaAsize();
	private:
		ConnectObj area;
		int areaSize;
		Mat img;
		
};

//class TrackObj
//{
//	public:
//		TrackObj();
//		int initObj(Mat src,Rect trackWindow);
//		int track(Mat src,Rect& outputWindow );
//		int updateObj(Mat src,Rect trackWindow);
//		int updateHist(Mat srcHist,Mat curHist,Mat& dstHist);
//		
//	private:
//		Rect trackWindow1,searchRange;
//		Mat hist1,hist2,hist3,hist4;
//		float hranges[2];
//		const float* phranges;
//		int hsize;
//		
//
//};
#endif
