// use nonfree library
// only on OpenCV 2.xx

#include "vector"  
#include "iostream"  
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>        
#include <opencv2\features2d\features2d.hpp>
#include "vector"  
#include "iostream"  
#include <opencv2/nonfree/nonfree.hpp>  

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	//Load Image   
	Mat c_src1 = imread("test.jpg");
	Mat c_src2 = imread("test0.jpg");
	Mat src1 = imread("test.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat src2 = imread("test0.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	if (!src1.data || !src2.data)
	{
		std::cout << " --(!) Error reading images " << std::endl; return -1;
	}

	//sift feature detect  
	SiftFeatureDetector detector;
	std::vector<KeyPoint> kp1, kp2;
	detector.detect(src1, kp1);
	detector.detect(src2, kp2);

	//sift descriptor
	SiftDescriptorExtractor extractor;
	Mat des1, des2;//descriptor  
	extractor.compute(src1, kp1, des1);
	extractor.compute(src2, kp2, des2);
	
	//draw circle on keypoints
	Mat res1, res2;
	int drawmode = DrawMatchesFlags::DRAW_RICH_KEYPOINTS;
	drawKeypoints(c_src1, kp1, res1, Scalar::all(-1), drawmode);//在内存中画出特征点  
	drawKeypoints(c_src2, kp2, res2, Scalar::all(-1), drawmode);
	cout << "size of description of Img1: " << kp1.size() << endl;
	cout << "size of description of Img2: " << kp2.size() << endl;

	//write the size of features on picture  
	CvFont font;
	double hScale = 1;
	double vScale = 1;
	int lineWidth = 2;// 相当于写字的线条      
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, hScale, vScale, 0, lineWidth);//初始化字体，准备写到图片上的     
	
	// cvPoint 为起笔的x，y坐标     
	char str1[20], str2[20];
	sprintf(str1, "%d", kp1.size());
	sprintf(str2, "%d", kp2.size());
	cvPutText(&(IplImage)res1, str1, cvPoint(20, 20), &font, CV_RGB(255, 0, 0));//在图片中输出字符   
	cvPutText(&(IplImage)res2, str2, cvPoint(20, 20), &font, CV_RGB(255, 0, 0));//在图片中输出字符   
	cvShowImage("descriptor1", &(IplImage)res1);
	cvShowImage("descriptor2", &(IplImage)res2);

	//find match using Brute force
	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(des1, des2, matches);
	//keep N matches for drawing
	int N = 10;
	matches.erase(matches.begin() + N, matches.end());

	//draw matching
	Mat img_match;
	drawMatches(src1, kp1, src2, kp2, matches, img_match);//,Scalar::all(-1),Scalar::all(-1),vector<char>(),drawmode);  
	imshow("matches", img_match);
	
	cout << "number of matched points: " << matches.size() << endl;
	cvWaitKey();
	cvDestroyAllWindows();

	return 0;
}
