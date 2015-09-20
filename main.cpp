

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <unistd.h>
//#include <opencv2/objdetect/objdetect.hpp>
#include "process.h"
#include "gprs.h"


using namespace cv;
using namespace std;

int socketfd;



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
    //init_tcp();
    int firstFaceFlag=1;
    int collectFlag=0;
    int trainedFlag=0;\
    clock_gettime(CLOCK_REALTIME,&t0);
    //load xml
    //char * faceCascadeFilename="/usr/local/share/OpenCV/lbpcascades/lbpcascade_frontalface.xml";
    char * faceCascadeFilename="/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml";
    char * eyeCascadeFilename1="/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml";
    char * eyeCascadeFilename2="/usr/local/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
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
    string faceRecAlgorithm="FaceRecognizer.Eigenfaces";
    Ptr<FaceRecognizer> model_smile;
    Ptr<FaceRecognizer> model_sad;
    model_smile= Algorithm::create<FaceRecognizer>(faceRecAlgorithm);
    model_sad= Algorithm::create<FaceRecognizer>(faceRecAlgorithm);
    //load yml
    Mat labels;
    try{
        model_smile->load("smile.yml");
        labels=model_smile->get<Mat>("labels");
        model_sad->load("sad.yml");
        labels=model_sad->get<Mat>("labels");
    }catch(cv::Exception &e){}
    if(labels.rows<=0){
        cout<<"ERROR:couldnt load trained data,start train"<<endl;
    }else{
        trainedFlag=1;
    }
    if(model_smile.empty()){
        printf("ERROR: FaceRecognizer smile not available!\n");
        exit(1);
    }
    if(model_sad.empty()){
        printf("ERROR: FaceRecognizer sad not available!\n");
        exit(1);
    }
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
            if(trainedFlag==0&&(firstFaceFlag||timediff>1)){
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
                if((int)preProcessedFaces.size()>=30){
                    printf("collect finished\n");
                    printf("train start\n");
                    model_smile->train(preProcessedFaces,faceLabels);
                    printf("train finished\n");
                    trainedFlag=1;
                }
            }else if(trainedFlag){
                //start predict
                printf("[predict mode]\t");
                //int identify =model_smile->predict(faceProcessed);
                int identify;
                /***/
                Mat eigenvectors_smile=model_smile->get<Mat>("eigenvectors");
                Mat averageFaceRow_smile=model_smile->get<Mat>("mean");
                Mat projection_smile = subspaceProject(eigenvectors_smile,averageFaceRow_smile,faceProcessed.reshape(1,1));
                Mat reconstructionRow_smile=subspaceReconstruct(eigenvectors_smile,averageFaceRow_smile,projection_smile);
                Mat reconstructionMat_smile = reconstructionRow_smile.reshape(1,faceProcessed.rows);
                Mat reconstructionFace_smile=Mat(reconstructionMat_smile.size(),CV_8U);
                reconstructionMat_smile.convertTo(reconstructionFace_smile,CV_8U,1,0);


                Mat eigenvectors_sad=model_sad->get<Mat>("eigenvectors");
                Mat averageFaceRow_sad=model_sad->get<Mat>("mean");
                Mat projection_sad = subspaceProject(eigenvectors_sad,averageFaceRow_sad,faceProcessed.reshape(1,1));
                Mat reconstructionRow_sad=subspaceReconstruct(eigenvectors_sad,averageFaceRow_sad,projection_sad);
                Mat reconstructionMat_sad = reconstructionRow_sad.reshape(1,faceProcessed.rows);
                Mat reconstructionFace_sad=Mat(reconstructionMat_sad.size(),CV_8U);
                reconstructionMat_sad.convertTo(reconstructionFace_sad,CV_8U,1,0);
                /*
                Mat eigenvectors=model_smile->get<Mat>("eigenvectors");
                Mat averageFaceRow=model_smile->get<Mat>("mean");
                Mat projection = subspaceProject(eigenvectors,averageFaceRow,faceProcessed.reshape(1,1));
                Mat reconstructionRow=subspaceReconstruct(eigenvectors,averageFaceRow,projection);
                Mat reconstructionMat = reconstructionRow.reshape(1,faceProcessed.rows);
                Mat reconstructionFace=Mat(reconstructionMat.size(),CV_8U);
                */
                double similaity_smile=getSimilarity(faceProcessed,reconstructionFace_smile);
                double similaity_sad=getSimilarity(faceProcessed,reconstructionFace_sad);
                printf("Similarity: smile=%f   sad=%f \n",similaity_smile,similaity_sad);
                if(similaity_smile>=similaity_sad){
                    //sad:1;smile:0  other:-1
                    identify=1;
                }else{
                    identify=0;

                }
                 if((similaity_sad>0.8)&&(similaity_smile>0.8)){
                     identify=-1;
                 }

                /*
                if(similaity>0.6){
                    identify=-1;
                }
                */
                /***/
                //printf("person %d\n",identify);
                switch(identify){
                    case -1:
                        printf("other\n");
                        break;
                    case 0:
                        printf("smile\n");
                        break;
                    case 1:
                        printf("sad\n");
                }
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
            model_smile->save("smile.yml");
            break;
        }
    }


    stop_tcp();
    return 0;

}
