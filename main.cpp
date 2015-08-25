#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <unistd.h>
//#include <opencv2/objdetect/objdetect.hpp>
#include "process.h"


using namespace cv;
using namespace std;

Mat frame;
char key;
struct timespec timeStart,t0,t1;
long timediff;
CascadeClassifier faceDetector;
CascadeClassifier eyeDetector1;
CascadeClassifier eyeDetector2;

vector<Mat> preProcessedFaces;
vector<int> faceLabels;

int person=0;

int main(int argc, char *argv[])
{
    int firstFaceFlag=1;
    int collectFlag=0;
    int trainedFlag=0;
    clock_gettime(CLOCK_REALTIME,&t0);
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

    //load contrib module
    bool haveContribModule= initModule_contrib();
    if(!haveContribModule){
        printf("ERROR: contrib cant be loaded\n");
        exit(1);
    }
    //string faceRecAlgorithm="FaceRecognizer.Fisherfaces";
    //string faceRecAlgorithm="FaceRecognizer.Eigenfaces";
    string faceRecAlgorithm="FaceRecognizer.LBPH";
    Ptr<FaceRecognizer> model;
    model= Algorithm::create<FaceRecognizer>(faceRecAlgorithm);
    if(model.empty()){
        printf("ERROR: FaceRecognizer not available!\n");
        exit(1);
    }
    //load .yml train data
    Mat labels;

    try{
        model->load("trainedModel.yml");
        //labels=model->get<Mat>("labels");
    }catch(cv::Exception &e){}
    //if(labels.rows<=0){
    //    cerr<<"ERROR:couldnt load trained data"<<endl;
    //    trainedFlag=0;
    //}else{
        //printf("data loaded,%d faces alredy there\n",(int)preProcessedFaces.size());
        printf("data loaded\n ");
        trainedFlag=1;
    //}
    //init camera
    int camNumber=0;
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
    Mat faceProcessed;
    Mat oldFaceProcessed;
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
            clock_gettime(CLOCK_REALTIME,&t1);
            //printf("t0_sec=%ld\tt1_sec=%ld\t",t0.tv_sec,t1.tv_sec);
            timediff=(t1.tv_sec-t0.tv_sec);
            //printf("timediff=%ld\n",timediff);
            if(firstFaceFlag||timediff>1){
                    collectFlag=1;
            }
            //get eye point
            leftEye.x=leftEyeRect[0].x+leftEyeRect[0].width/2;
            leftEye.y=leftEyeRect[0].y+leftEyeRect[0].height/2;
            rightEye.x=rightEyeRect[0].x+rightEyeRect[0].width/2;
            rightEye.y=rightEyeRect[0].y+rightEyeRect[0].height/2;
            //face process
            faceProcessed=faceProcess(leftEye,rightEye,facesRect);
            //collcet or not
            if(firstFaceFlag){
                oldFaceProcessed=faceProcessed;
                firstFaceFlag=0;
            }else{
                double similarity=getSimilarity(faceProcessed,oldFaceProcessed);
                if(similarity<0.3){
                    collectFlag=0;
                }
            }
            if(collectFlag){//collect it
                //collect
                Mat mirroredFace;
                flip(faceProcessed,mirroredFace,1);
                preProcessedFaces.push_back(faceProcessed);
                preProcessedFaces.push_back(mirroredFace);
                faceLabels.push_back(person);
                faceLabels.push_back(person);
                //now-->old
                oldFaceProcessed=faceProcessed;
                t0.tv_nsec=t1.tv_nsec;
                t0.tv_sec=t1.tv_sec;
                //flag set to 0
                collectFlag=0;
                printf("collect count=%d\n",(int)preProcessedFaces.size());
                if((trainedFlag==0)&&((int)preProcessedFaces.size()>=30)){
                    printf("30 faces collect finished\n");
                    printf("train start\n");
                    model->train(preProcessedFaces,faceLabels);
                    printf("train finished\n");
                    trainedFlag=1;
                }else if(trainedFlag==1){
                    printf("colleton more face,updating model..\n");
                    model->update(preProcessedFaces,faceLabels);
                    printf("%d faces in the model now\n",(int)preProcessedFaces.size());
                }
            }
            if(trainedFlag){
               //start predict
                printf("[predict mode]\t");
                int identify =model->predict(faceProcessed);
                /***/
                Mat eigenvectors=model->get<Mat>("eigenvectors");
                Mat averageFaceRow=model->get<Mat>("mean");
                Mat projection = subspaceProject(eigenvectors,averageFaceRow,faceProcessed.reshape(1,1));
                Mat reconstructionRow=subspaceReconstruct(eigenvectors,averageFaceRow,projection);
                Mat reconstructionMat = reconstructionRow.reshape(1,faceProcessed.rows);
                Mat reconstructionFace=Mat(reconstructionMat.size(),CV_8U);
                reconstructionMat.convertTo(reconstructionFace,CV_8U,1,0);
                double similaity=getSimilarity(faceProcessed,reconstructionFace);
                if(similaity>0.7){
                    identify=-1;
                }
                /***/
                if(identify==-1){
                    printf("not zjc\n");
                }else if(identify==0){
                    printf("hello jaycee!\n");
                }
                //printf("person %d\n",identify);
            }
            //show Rects
            rectangle(frame, facesRect[0], Scalar(255,0,0));//show faceRect
            rectangle(frame,leftEyeRect[0], Scalar(0,255,0));//show eyeRect
            rectangle(frame,rightEyeRect[0], Scalar(0,255,0)); //show eyeRect
            imshow("faceProcessed",faceProcessed);
        }
        imshow("frame",frame);
        key=waitKey(20);
        if(key==27){
            model->save("trainedModel.yml");
            printf("save trained data ok,%d faces in it now.\nbye!~\n",(int)preProcessedFaces.size());
            break;
        }
    }



    return 0;

}
