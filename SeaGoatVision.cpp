#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#pragma once
#include <sstream>
#include <string>
#include <iostream>
#include <vector>
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>

#include "vision.h"

using namespace cv;
using namespace std;

//initial min and max HSV filter values.
//these will be changed using trackbars
// red bar HSV
int rbrH_MIN = 350, rbrH_MAX = 20;
int rbrS_MIN = 40,  rbrS_MAX = 100;
int rbrV_MIN = 80,  rbrV_MAX = 100; 
// red buoy HSV
int rbH_MIN = 350; int rbH_MAX = 35;
int rbS_MIN = 40;  int rbS_MAX = 100;
int rbV_MIN = 40;  int rbV_MAX = 100;
// green buoy HSV
int gbH_MIN = 110; 
int gbH_MAX = 130;
int gbS_MIN = 20;   // water will be noise if S is too small
int gbS_MAX = 99;
int gbV_MIN = 45;   // water will be noise if V is too small
int gbV_MAX = 99;
// yellow "buoy" HSV
int ybH_MIN = 50;
int ybH_MAX = 70;
int ybS_MIN = 25;
int ybS_MAX = 100;
int ybV_MIN = 45;
int ybV_MAX = 100;

int houghPara1 = 200;
int houghPara1_MAX = 300;
int houghPara2i = 25;
int houghPara2m = 20;
int houghPara2_MAX = 100;

int morphOpen_size = 2;  // if too big, nothing will appear after hsv->morph, but if too small, there will be noise. Prefer big.
int morphClose_size = 3; 

int const max_kernel_size = 21;
int blur_size = 1;  // 1 is the best, lol

const string trackbarWindowName = "Trackbars";

/* Function Headers */
void Morphology_Operations( Mat &thresh);

bool pause = 0;
int waitTime = 10;
char key = ' ';
int id = 1080; // image index in the image dataset

int connectedComponents(Mat src,vector<ConnectObj> &cO);

string intToString(int number){
	std::stringstream ss;
	ss << number;
	return ss.str();
}

void on_trackbar( int, void* )
{//This function gets called whenever a
	// trackbar position is changed
}

void Morphology_Operations( Mat &thresh)
{
  Mat element = getStructuringElement( 0, Size( 2*morphOpen_size + 1, 2*morphOpen_size+1 ), Point( morphOpen_size, morphOpen_size ) );
  morphologyEx( thresh, thresh, MORPH_OPEN, element );

  element = getStructuringElement( 0, Size( 2*morphClose_size + 1, 2*morphClose_size+1 ), Point( morphClose_size, morphClose_size ) );
  morphologyEx( thresh, thresh, MORPH_CLOSE, element ); 
}

void createTrackbars(){
	//create window for trackbars
	
    namedWindow(trackbarWindowName,0);
	//create memory to store trackbar name on window
	char TrackbarName[50];
	/*sprintf( TrackbarName, "rbH_MIN", rbH_MIN);
	sprintf( TrackbarName, "rbH_MAX", rbH_MAX);
	sprintf( TrackbarName, "rbS_MIN", rbS_MIN);
	sprintf( TrackbarName, "rbS_MAX", rbS_MAX);
	sprintf( TrackbarName, "rbV_MIN", rbV_MIN);
	sprintf( TrackbarName, "rbV_MAX", rbV_MAX);

	sprintf( TrackbarName, "gbH_MIN", gbH_MIN);
	sprintf( TrackbarName, "gbH_MAX", gbH_MAX);
	sprintf( TrackbarName, "gbS_MIN", gbS_MIN);
	sprintf( TrackbarName, "gbS_MAX", gbS_MAX);
	sprintf( TrackbarName, "gbV_MIN", gbV_MIN);
	sprintf( TrackbarName, "gbV_MAX", gbV_MAX);

	sprintf( TrackbarName, "ybH_MIN", ybH_MIN);
	sprintf( TrackbarName, "ybH_MAX", ybH_MAX);
	sprintf( TrackbarName, "ybS_MIN", ybS_MIN);
	sprintf( TrackbarName, "ybS_MAX", ybS_MAX);
	sprintf( TrackbarName, "ybV_MIN", ybV_MIN);
	sprintf( TrackbarName, "ybV_MAX", ybV_MAX);*/
	//sprintf( TrackbarName, "ObjArea_MIN", MIN_OBJECT_AREA);
	//sprintf( TrackbarName, "ObjArea_MAX", MAX_OBJECT_AREA);
	//create trackbars and insert them into window
	//3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
	//the max value the trackbar can move (eg. H_HIGH), 
	//and the function that is called whenever the trackbar is moved(eg. on_trackbar)
	//                                  ---->    ---->     ---->      

	createTrackbar( "rbrH_MIN", trackbarWindowName, &rbrH_MIN, 360, on_trackbar );
    createTrackbar( "rbrH_MAX", trackbarWindowName, &rbrH_MAX, 360, on_trackbar );
    createTrackbar( "rbrS_MIN", trackbarWindowName, &rbrS_MIN, 100, on_trackbar );
    createTrackbar( "rbrS_MAX", trackbarWindowName, &rbrS_MAX, 100, on_trackbar );
    createTrackbar( "rbrV_MIN", trackbarWindowName, &rbrV_MIN, 100, on_trackbar );
    createTrackbar( "rbrV_MAX", trackbarWindowName, &rbrV_MAX, 100, on_trackbar );

    createTrackbar( "rbH_MIN", trackbarWindowName, &rbH_MIN, 360, on_trackbar );
    createTrackbar( "rbH_MAX", trackbarWindowName, &rbH_MAX, 360, on_trackbar );
    createTrackbar( "rbS_MIN", trackbarWindowName, &rbS_MIN, 100, on_trackbar );
    createTrackbar( "rbS_MAX", trackbarWindowName, &rbS_MAX, 100, on_trackbar );
    createTrackbar( "rbV_MIN", trackbarWindowName, &rbV_MIN, 100, on_trackbar );
    createTrackbar( "rbV_MAX", trackbarWindowName, &rbV_MAX, 100, on_trackbar );

	createTrackbar( "gbH_MIN", trackbarWindowName, &gbH_MIN, 360, on_trackbar );
    createTrackbar( "gbH_MAX", trackbarWindowName, &gbH_MAX, 360, on_trackbar );
    createTrackbar( "gbS_MIN", trackbarWindowName, &gbS_MIN, 100, on_trackbar );
    createTrackbar( "gbS_MAX", trackbarWindowName, &gbS_MAX, 100, on_trackbar );
    createTrackbar( "gbV_MIN", trackbarWindowName, &gbV_MIN, 100, on_trackbar );
    createTrackbar( "gbV_MAX", trackbarWindowName, &gbV_MAX, 100, on_trackbar );

	createTrackbar( "ybH_MIN", trackbarWindowName, &ybH_MIN, 360, on_trackbar );
    createTrackbar( "ybH_MAX", trackbarWindowName, &ybH_MAX, 360, on_trackbar );
    createTrackbar( "ybS_MIN", trackbarWindowName, &ybS_MIN, 100, on_trackbar );
    createTrackbar( "ybS_MAX", trackbarWindowName, &ybS_MAX, 100, on_trackbar );
    createTrackbar( "ybV_MIN", trackbarWindowName, &ybV_MIN, 100, on_trackbar );
    createTrackbar( "ybV_MAX", trackbarWindowName, &ybV_MAX, 100, on_trackbar );
	//createTrackbar( "ObjArea_MIN", trackbarWindowName, &ObjArea_MIN, ObjArea_MAX, on_trackbar );
 //   createTrackbar( "ObjArea_MAX", trackbarWindowName, &ObjArea_MAX, ObjArea_MAX, on_trackbar );


createTrackbar( "blur 2n+1", trackbarWindowName,
                 &blur_size, max_kernel_size,
                 on_trackbar );

 createTrackbar( "KrnOp 2n+1", trackbarWindowName,
                 &morphOpen_size, max_kernel_size,
                 on_trackbar );

  /// Create Trackbar to choose kernel size
 createTrackbar( "KrnClo 2n+1", trackbarWindowName,
                 &morphClose_size, max_kernel_size,
                 on_trackbar );

   /// Create Trackbar to choose kernel size
  createTrackbar( "para1", trackbarWindowName,
                 &houghPara1, houghPara1_MAX,
                 on_trackbar );
 createTrackbar( "para2m", trackbarWindowName,
                 &houghPara2m, houghPara2_MAX,
                 on_trackbar );
  createTrackbar( "para2i", trackbarWindowName,
                 &houghPara2i, houghPara2_MAX,
                 on_trackbar );
}

void keyboardControl(char key){
	switch(key)
	{
		case 'p': case 'P': { // Pause video
			pause = !pause;
			break;
			}	
		case 'n': case 'N': { // Image Next 
			id++;
			break;
			}
		case 'b': case 'B': { // Image Before
			id--;
			break;
			}
	}
}

void hsvThreshold(Mat &HSV, Mat &thesh, int H_MAX, int H_MIN, int S_MAX, int S_MIN, int V_MAX, int V_MIN){
	Mat threshold, threshold1;
	if(H_MAX>=H_MIN){
		inRange(HSV,Scalar(H_MIN/2,S_MIN*2.55,V_MIN*2.55),Scalar(H_MAX/2,S_MAX*2.55,V_MAX*2.55),threshold);
		thesh = threshold;
	}else{
		inRange(HSV,Scalar(H_MIN/2,S_MIN*2.55,V_MIN*2.55),Scalar(360,S_MAX*2.55,V_MAX*2.55),threshold);
		inRange(HSV,Scalar(0,S_MIN*2.55,V_MIN*2.55),Scalar(H_MAX/2,S_MAX*2.55,V_MAX*2.55),threshold1);
		thesh = threshold|threshold1;
	}

}
void findCircles (Mat &grayImg, int &xc, int &yc, Mat &imgShow){

	vector<Vec3f> circles;
		int maxRadius = 0;
		HoughCircles(grayImg, circles, CV_HOUGH_GRADIENT,
			2, grayImg.rows/4, 200, houghPara2i, 4);
		for( size_t i = 0; i < circles.size(); i++ )
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			// draw the circle center
			circle( imgShow, center, 3, Scalar(0,255,0), -1, 8, 0 );
			// draw the circle outline
			circle( imgShow, center, radius, Scalar(0,0,255), 3, 8, 0 );
			if (maxRadius < radius){
				maxRadius = radius;
				xc = center.x;
				yc = center.y;
			}
		}
}

void detectBuoy(Mat &HSV, Mat &gray, int &xb, int &yb, int* hsv_paras, Mat &imgShow){
	Mat b, bROI, bRoiM;
	int x0,y0,w0,h0; 
	int x,y,w,h;
	int key = 0, r = 0;
	float areaRatio, areaRatioMax = 0.0f;
	int xt = -1, yt = -1;
	xb = yb = -1;
	string color;
	if (hsv_paras[1] == rbH_MIN)
		color = "R";
	else if (hsv_paras[1] == gbH_MIN)
		color = "G";
	else 
		color = "Y";
	

	//cout<<color+"para: "<< hsv_paras[0] << hsv_paras[1] << hsv_paras[2] << hsv_paras[3] << hsv_paras[4]<<hsv_paras[5]<<endl;
	HSV.copyTo(b);
	hsvThreshold(HSV, b, hsv_paras[0],hsv_paras[1],hsv_paras[2], hsv_paras[3], hsv_paras[4], hsv_paras[5]);
	imshow(color+"buoy hsv", b);

	Morphology_Operations(b); 
	cv::GaussianBlur( b, b, Size(2*blur_size+1, 2*blur_size+1), 0, 0, BORDER_DEFAULT );
	imshow(color+"buoy morph", b);
	
	// circles in the morph image
	vector<Vec3f> circlesM;
	Rect RectRoiM;
	HoughCircles(b, circlesM, CV_HOUGH_GRADIENT,
		2, b.rows/4, 200, 20, 3);
	for( size_t i = 0; i < circlesM.size(); i++ )
	{
		Point center(cvRound(circlesM[i][0]), cvRound(circlesM[i][1]));
		int radius = cvRound(circlesM[i][2]);
		// draw the circle center and outline
		//circle( imgShow, center, 6, Scalar(255,255,255), -1, 8, 0 );
		//circle( imgShow, center, radius, Scalar(255,255,255), 3, 8, 0 );

		// in Mat gb after morph, analyize each roi given by the circles on gb morph
		x = x0 = (int)center.x - 1.0f*radius; 
		y = y0 = (int)center.y - 1.0f*radius; 
		w= h = w0 = h0 = (int)(2.0f*radius);
		if (x0 < 0) x = 0;
		if (y0 < 0) y = 0;
		if ((x+w0) > b.size().width) w = b.size().width - x;
		if ((y+h0) > b.size().height) h = b.size().height - y;

		RectRoiM = cv::Rect(x, y, w, h);
		//rectangle(imgShow, RectRoiM, Scalar(255,0,0));
		cout<< color+"morphROI "<<i<<" "<<x<<" "<<y<<" "<<w<<" "<<h<<" "<<b.size().width<<" "<<b.size().height<<endl;
		
		bRoiM = b(RectRoiM);
		imshow(color+"bRoiM"+intToString(i), bRoiM);
				
		int white_num = cv::countNonZero(bRoiM);
		int black_num = bRoiM.rows * bRoiM.cols - white_num;
		areaRatio = (float)white_num / (float)( bRoiM.rows * bRoiM.cols);
		//cout<< " areaRatio "<< areaRatio <<endl;
		if (areaRatio > areaRatioMax && areaRatio != 1 && white_num > 200){
			areaRatioMax = areaRatio;
			xb = center.x;
			yb = center.y;
			key = i; r = radius;
		}
	}
	//cout<< "Morph areaRatioMax: "<<areaRatioMax<<endl;
	if (xb != -1){
		circle( imgShow, Point(xb, yb), 3, Scalar(0,0,0), -1, 8, 0 );
		circle( imgShow, Point(xb, yb), r, Scalar(0,0,0), -1, 8, 0 );
		cout<< color+"key: "<<key<<endl;
	}
	else{
		vector<Vec3f> circles;
		Rect gbRectROI;
		cv::GaussianBlur( gray, gray, Size(2*blur_size+1, 2*blur_size+1), 0, 0, BORDER_DEFAULT );
		// circles in image
		HoughCircles(gray, circles, CV_HOUGH_GRADIENT,
			2, gray.cols/8, 200, houghPara2i, 4, gray.cols/8);     // max radius

		for( size_t i = 0; i < circles.size(); i++ )
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			// draw the circle center and outline
			//circle( imgShow, center, 3, Scalar(0,255,0), -1, 8, 0 );
			//circle( imgShow, center, radius, Scalar(0,0,255), 3, 8, 0 );

			// in Mat gb after morph, analyize each roi given by the circles on the original image
			x = x0 = center.x - 0.8f*radius; 
			y = y0 = center.y - 0.8f*radius; 
			w= h = w0 = h0 = 1.6f*radius;
			if (x0 < 0) x = 0;
			if (y0 < 0) y = 0;
			if ((x+w0) > b.size().width) w = b.size().width - x;
			if ((y+h0) > b.size().height) h = b.size().height - y;

			gbRectROI = cv::Rect(x, y, w, h);
			//rectangle(imgShow, gbRectROI, Scalar(255,0,0));
			//cout<<i<<" "<<x<<" "<<y<<" "<<w<<" "<<h<<" "<<b.size().width<<" "<<gb.size().height<<endl;
		
			bROI = b(gbRectROI);
			//imshow("gbROI"+intToString(i), gbROI);
				
			int white_num = cv::countNonZero(bROI);

			int black_num = bROI.rows * bROI.cols - white_num;
			//cout<< " white pix: " << white_num;
			//cout<< " black pix: " << black_num << endl;

			areaRatio = (float)white_num / (float)( bROI.rows * bROI.cols);
			//cout<< i << " areaRatio "<< areaRatio <<endl;
			if (areaRatio > areaRatioMax && areaRatio != 1){
				areaRatioMax = areaRatio;
				xb = center.x;
				yb = center.y;
				key = i; r = radius;
			}

		}
		//cout<< color+"areaRatioMax: "<<areaRatioMax<<endl;
		if (xb != -1){
			circle( imgShow, Point(xb, yb), 5, Scalar(255,255,0), -1, 8, 0 );
			circle( imgShow, Point(xb, yb), r, Scalar(255,255,0), -1, 8, 0 );
			cout<< color+"key: "<<key<<endl;
		}
	}
	
		

}
void detectBuoys (Mat &HSV, Mat &gray, int &rb_x, int &rb_y, int &gb_x, int &gb_y, int &yb_x, int &yb_y, Mat &imgShow){
	
	rb_x = rb_y = gb_x = gb_y = yb_x = yb_y = -1;
	
	int rb_para[6] = {rbH_MAX, rbH_MIN, rbS_MAX, rbS_MIN, rbV_MAX, rbV_MIN};
	int gb_para[6] = {gbH_MAX, gbH_MIN, gbS_MAX, gbS_MIN, gbV_MAX, gbV_MIN};
	int yb_para[6] = {ybH_MAX, ybH_MIN, ybS_MAX, ybS_MIN, ybV_MAX, ybV_MIN};

	detectBuoy(HSV, gray, rb_x, rb_y, rb_para,imgShow);
	detectBuoy(HSV, gray, gb_x, gb_y, gb_para,imgShow);
	detectBuoy(HSV, gray, yb_x, yb_y, yb_para,imgShow);
			std::cout << "Buoys: R" << rb_x << ", " << rb_y 
		       << "; G " << gb_x << ", " << gb_y 
		      << "; Y " << yb_x << ", " << yb_y << endl;
	
}

void detectRedBar(Mat &HSV, int &x, int &y, Mat &imgShow){
	Mat rbr;
	hsvThreshold(HSV, rbr, rbrH_MAX, rbrH_MIN, rbrS_MAX, rbrS_MIN, rbrV_MAX, rbrV_MIN);
	imshow("rbrHSV", rbr);
	Morphology_Operations(rbr);
	cv::GaussianBlur(rbr, rbr, Size(2*blur_size+1, 2*blur_size+1), 0, 0, BORDER_DEFAULT );
	imshow("rbrMorph", rbr);

	vector<ConnectObj> cO;
	connectedComponents(rbr,cO);
} 

int main()
{
	string img_path1 = "./dataset/_Recordings_/Front/";
	string image_name = img_path1 + "Image" + intToString(id) + ".png";
    Mat image, imgShow;
	Mat HSV, gray;
	Mat threshold;
	int rb_x, rb_y, gb_x, gb_y, yb_x, yb_y, xt, yt; 
	int xrbr, yrbr;
	rb_x = rb_y = gb_x = gb_y = yb_x = yb_y = xt = yt= xrbr = yrbr = -1;

	createTrackbars();

	while(key != 'q' && id <= 1180){

		image_name = img_path1 + "Image" + intToString(id) + ".png";
		image = imread(image_name, IMREAD_COLOR);   // Read the file

		if(image.empty()){                      // Check for invalid input
			cout <<  "Could not open or find the image" << endl ;
			return 0;
		}
		cvtColor(image, HSV, COLOR_BGR2HSV);
		cvtColor(image, gray, COLOR_BGR2GRAY);
		image.copyTo(imgShow); // if using "imgShow = image", text will appear in the processed images.
		//namedWindow( img_path1 + "Image", WINDOW_AUTOSIZE );// Create a window for display.
		putText(imgShow,"Image "+intToString(id), Point(100,100),1,2,Scalar(0,0,255),2);

		//detect buoys
		detectBuoys(HSV, gray, rb_x, rb_y, gb_x, gb_y, yb_x, yb_y, imgShow);

		//detect red bar
		//detectRedBar(HSV, xrbr, yrbr, imgShow);

		cv::imshow(img_path1 + "Image", imgShow); 

		if(pause == 1)	
			waitTime = -1;
		else  
			waitTime = 10;

		key = (char)waitKey(waitTime); 
		keyboardControl(key);

	}


    return 0;

}

int connectedComponents(Mat src,vector<ConnectObj> &cO)
{
	Mat srcCanny;
	Canny(src,srcCanny,50,100,3);
	vector<vector<Point> >contours, contours1;
	vector<Vec4i> hierarchy;
	findContours(srcCanny,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,Point(0,0));
	vector<vector<Point> >contoursPoly(contours.size());
		
	//cout<<"Contours1: "<<contours.size()<<endl;
	Mat srcContour=Mat::zeros(src.size(),CV_8UC3);
	//Mat srcContourFit=Mat::zeros(src,size(),CV_8UC3);

	float perimscale=64; // used for contour length threshold
	
	//vector<ConnectObj>::iterator iter=cO.begin();
	//filer the contours and smooth it
	for(int i=0,j=0; i<contours.size(); i++)
	{
		double len=arcLength(contours[i],1);
		double lenThreshold=(src.size().height+src.size().width) / perimscale;
		if(len<lenThreshold)
		{	
			cout<<"contour size "<< i <<": "<<contours[i].size()<<endl;
			contours.erase(contours.begin()+i);
			
			cout<<"remove the small contour"<<endl;
			contours[i].clear();
			cout<<contours[i].size()<<endl;
		}
		else //smooth it
		{
			Scalar color=Scalar(255,0,0);
			drawContours(srcContour,contours,i,color,2,8,hierarchy,0,Point());
			
			contours1.resize(contours1.size()+1);
			//approxPolyDP(Mat(contours[i]),contours[i],3,true);
			approxPolyDP(Mat(contours[i]),contours1.back(),3,true);
			//cout<<contours1.size()<<endl;

			ConnectObj c1;
			vector<Point>::iterator iter2;
			iter2=contours1.back().begin();
			for(; iter2!=contours1.back().end(); iter2++)
			{
				c1.contour.push_back(*iter2);
			}
  		        
			cO.push_back(c1);
			//color.val[0]=0;color.val[1]=255;color.val[2]=0;
			drawContours(srcContour,contours1,contours1.size()-1,Scalar(0,255,0),2,8,hierarchy,0,Point());
		}
		//end for contours
	}
		
	if(contours1.size()!=0)
	{	
		//Calculate the center of mass and the exterior rectangle
		vector<Moments> mu(contours1.size());
		for(int i=0; i<contours1.size(); i++)
		{
			mu[i]=moments(contours1[i],false);
		}

		vector<Point> mc(contours1.size());
		//vector<ConnectObj>::iterator iter;
		//iter=cO.begin();
		for(int i=0;i<contours1.size();i++/*,iter++*/)
		{
			mc[i]=Point(mu[i].m10/mu[i].m00,mu[i].m01/mu[i].m00);
			cO[i].center=mc[i];
			circle(srcContour,mc[i],4,Scalar(0,0,255),-1);
		}	
		
		vector<RotatedRect>minRect(contours1.size());
		vector<Rect>boundRect(contours1.size());
		for(int i=0;i<contours1.size();i++)
		{
			//minRect[i]=minAreaRect(Mat(contours[i]));
			boundRect[i]=boundingRect(Mat(contours1[i]));
			cO[i].bound=boundRect[i];
			rectangle(srcContour,boundRect[i].tl(),boundRect[i].br(),Scalar(0,0,255),1);
		}
		
	}
	else
	{
		return -1;
	}

	//cout<<"Contours2: "<<contours.size()<<endl;
	
	imshow("contour",srcContour);
	return 0;
} // need .h
