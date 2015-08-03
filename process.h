#ifndef PERPROCESS_H
#define PERPROCESS_H
#include<opencv2/opencv.hpp>

#define EYE_SX 0.16
#define EYE_SY 0.26
#define EYE_SW 0.30
#define EYE_SH 0.28

using namespace cv;
using namespace std;

extern Mat frame;
extern char key;
extern CascadeClassifier faceDetector;
extern CascadeClassifier eyeDetector1;
extern CascadeClassifier eyeDetector2;

void preProcess(Mat &frameProc);
void faceDetect(Mat &frameProc,vector<Rect> &faces);
int findEyes(vector<Rect> &faces,vector<Rect> &leftEyeRect, vector<Rect> &rightEyeRect);
Mat faceProcess(Point leftEye,Point rightEye,vector<Rect> &facesRect);

#endif // PERPROCESS_H

