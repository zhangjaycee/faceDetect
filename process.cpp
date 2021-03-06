#include<opencv2/opencv.hpp>
#include"process.h"
#include<stdio.h>


using namespace cv;

void preProcess(Mat &frameProc)
{
    frameProc.setTo(0);
    //change to gary
    if(frame.channels()==3){
        cvtColor(frame,frameProc,CV_BGR2GRAY);
    }else if(frame.channels()==4){
        cvtColor(frame,frameProc,CV_BGRA2GRAY);
    }else{
        frameProc=frame;
    }

    //to smaller size
    const int DETECTION_WIDTH=320;
    Size bigSize=frameProc.size();
    float scale = frameProc.cols/(float)DETECTION_WIDTH;
    if(frameProc.cols>DETECTION_WIDTH){
        Size smallSize;
        smallSize.width=DETECTION_WIDTH;
        smallSize.height=cvRound(frameProc.rows/scale);
        resize(frameProc,frameProc,smallSize);
    }

    //equlize
    equalizeHist(frameProc,frameProc);

}



void faceDetect(Mat &frameProc, std::vector<Rect> &faces)
{
    int flags=CASCADE_FIND_BIGGEST_OBJECT;
    //int flags=CASCADE_DO_ROUGH_SEARCH;
    Size miniFaturesize(20,20);
    float serchScaleFactor=1.1f;
    int minNeighbors=4;

    faceDetector.detectMultiScale(frameProc,faces,serchScaleFactor,minNeighbors,flags,miniFaturesize);
    if(frame.cols>frameProc.cols){
        int scale=cvRound(frame.cols/frameProc.cols);
        for(int i=0;i<(int)faces.size();i++){
            faces[i].x=cvRound(faces[i].x*scale);
            faces[i].y=cvRound(faces[i].y*scale);
            faces[i].width=cvRound(faces[i].width*scale);
            faces[i].height=cvRound(faces[i].height*scale);
        }
    }
    //printf("faces.size()=%d\n",(int)faces.size());//print face number
    for(int i=0;i<(int)faces.size();i++){
        if(faces[i].x<0) faces[i].x=0;
        if(faces[i].y<0) faces[i].y=0;
        if(faces[i].x+faces[i].width>frame.cols) faces[i].x=frame.cols-faces[i].width;
        if(faces[i].y+faces[i].height>frame.rows) faces[i].y=frame.rows-faces[i].height;
    }
    //display
//    for(int i=0;i<(int)faces.size();i++){
//        rectangle(frame, faces[i], Scalar(255,0,0));
//    }
}

int findEyes(std::vector<Rect> &faces,std::vector<Rect> &leftEyeRect, std::vector<Rect> &rightEyeRect)
{
    int eyenum=0;
    Mat face=frame(faces[0]);
    int leftX=cvRound(face.cols * EYE_SX);
    int topY=cvRound(face.rows * EYE_SY);
    int widthX=cvRound(face.cols*EYE_SW);
    int heightY=cvRound(face.rows*EYE_SH);
    int rightX=cvRound(face.cols*(1.0-EYE_SX-EYE_SW));

    Mat righteyeArea = face(Rect(leftX,topY,widthX,heightY));
    Mat lefteyeArea = face(Rect(rightX,topY,widthX,heightY));

    int flags=CASCADE_FIND_BIGGEST_OBJECT;
    //int flags=CASCADE_DO_ROUGH_SEARCH;
    Size miniFaturesize(10,10);
    float searchScaleFactor=1.1f;
    int minNeighbors=4;
    //left eye
    eyeDetector1.detectMultiScale(lefteyeArea,leftEyeRect,searchScaleFactor,minNeighbors,flags,miniFaturesize);
    if((int)leftEyeRect.size()==0){
        eyeDetector2.detectMultiScale(lefteyeArea,leftEyeRect,searchScaleFactor,minNeighbors,flags,miniFaturesize);
    }

    if((int)leftEyeRect.size()==1){
        eyenum++;
        leftEyeRect[0].x=leftEyeRect[0].x+leftX+faces[0].x;
        leftEyeRect[0].y=leftEyeRect[0].y+topY+faces[0].y;
    }

    //right eye
    eyeDetector1.detectMultiScale(righteyeArea,rightEyeRect,searchScaleFactor,minNeighbors,flags,miniFaturesize);
     if((int)rightEyeRect.size()==0){
        eyeDetector2.detectMultiScale(righteyeArea,rightEyeRect,searchScaleFactor,minNeighbors,flags,miniFaturesize);
    }
    Point rightEye=Point(-1,-1);
    if((int)rightEyeRect.size()==1){
        eyenum++;
        rightEyeRect[0].x=rightEyeRect[0].x+rightX+faces[0].x;
        rightEyeRect[0].y=rightEyeRect[0].y+topY+faces[0].y;
    }
    return eyenum;
}


Mat faceProcess(Point leftEye,Point rightEye,vector<Rect> &facesRect)
{
    Point2f eyesCenter;
    eyesCenter.x=(leftEye.x+rightEye.x)*0.5f-facesRect[0].x;
    eyesCenter.y=(leftEye.y+rightEye.y)*0.5f-facesRect[0].y;

    double dy= rightEye.y-leftEye.y;
    double dx= rightEye.x-leftEye.x;
    double len=sqrt(dx*dx+dy*dy);
    double angle=atan2(dy,dx)*180.0/CV_PI;

    const double DESIRED_LEFT_EYE_X=0.16;
    const double DESIRED_RIGHT_EYE_X=(1.0f-0.16);
    const double DESIRED_LEFT_EYE_Y=0.14;
    const double DESIRED_RIGHT_EYE_Y=0.14;
    const int DESIRED_FACE_WIDTH=70;
    const int DESIRED_FACE_HEIGHT=70;

    double desiredLen=(1.0f-0.32);
    double scale = desiredLen*DESIRED_FACE_WIDTH/len;

    //get transformation matrix for the desired angle and size
    Mat rotMat=getRotationMatrix2D(eyesCenter,angle,scale);
    //shift center of eyes to be desired center
    double ex=DESIRED_FACE_WIDTH*0.5f - eyesCenter.x;
    double ey=DESIRED_FACE_HEIGHT*DESIRED_LEFT_EYE_Y-eyesCenter.y;
    rotMat.at<double>(0,2)+=ex;
    rotMat.at<double>(1,2)+=ey;

    Mat gray=frame(facesRect[0]);
    cvtColor(gray,gray,CV_RGB2GRAY);
    imshow("faceGray",gray);
    Mat warped=Mat(DESIRED_FACE_HEIGHT,DESIRED_FACE_WIDTH,CV_8U,Scalar(128));
    warpAffine(gray,warped,rotMat,warped.size());
    imshow("rotedGray",warped);


    return warped;
}
