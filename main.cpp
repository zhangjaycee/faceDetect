#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
//#include <opencv2/objdetect/objdetect.hpp>
#include "process.h"

using namespace cv;
using namespace std;

Mat frame;
char key;
CascadeClassifier faceDetector;
CascadeClassifier eyeDetector1;
CascadeClassifier eyeDetector2;

int main(int argc, char *argv[])
{
    //load xml
    //char * faceCascadeFilename="/usr/local/share/OpenCV/lbpcascades/lbpcascade_frontalface.xml";
    char * faceCascadeFilename="/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml";
    char * eyeCascadeFilename1="/usr/share/opencv/haarcascades/haarcascade_eye.xml";
    char * eyeCascadeFilename2="/usr/share/opencv/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
    try{
        faceDetector.load(faceCascadeFilename);
        eyeDetector1.load(eyeCascadeFilename1);
        eyeDetector2.load(eyeCascadeFilename2);
    }catch(cv::Exception e){}
    if(faceDetector.empty()){
        cerr<<"ERROR:Couldn't load Face Detector(";
        cerr<< faceCascadeFilename<<")!"<<endl;
    }

    //init camera
    int camNumber=1;
    if(argc>1){
        camNumber=atoi(argv[1]);
    }
    VideoCapture webcam;
    webcam.open(camNumber);

    if(!webcam.isOpened()){
        printf("cam open error!\n");
        exit(1);
    }
    webcam.set(CV_CAP_PROP_FRAME_WIDTH,640);
    webcam.set(CV_CAP_PROP_FRAME_HEIGHT,480);

    while(webcam.read(frame)){

        Mat frameProc;
        //perProcess
        preProcess(frameProc);

        //detect face
        vector<Rect> facesRect;
        faceDetect(frameProc,facesRect);

        //acquire eye
        vector<Rect> leftEyeRect;
        vector<Rect> rightEyeRect;
        Point leftEye=Point(-1,-1);
        Point rightEye=Point(-1,-1);
        int eyenum=0;
        if((int)facesRect.size()>0){
            eyenum=findEyes(facesRect,leftEyeRect,rightEyeRect);
        }
        if(eyenum==2){
            leftEye.x=leftEyeRect[0].x+leftEyeRect[0].width/2;
            leftEye.y=leftEyeRect[0].y+leftEyeRect[0].height/2;
            rightEye.x=rightEyeRect[0].x+rightEyeRect[0].width/2;
            rightEye.y=rightEyeRect[0].y+rightEyeRect[0].height/2;
        //face process
        Mat faceProcessed;
        faceProcessed=faceProcess(leftEye,rightEye,facesRect);

        //show Rects
            rectangle(frame, facesRect[0], Scalar(255,0,0));//show faceRect
            rectangle(frame,leftEyeRect[0], Scalar(0,255,0));//show eyeRect
            rectangle(frame,rightEyeRect[0], Scalar(0,255,0)); //show eyeRect
        }
        imshow("perProcessed",frame);
        key=waitKey(20);
        if(key==27){
            break;
        }
    }

    return 0;

}
