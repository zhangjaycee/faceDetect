#-------------------------------------------------
#
# Project created by QtCreator 2015-08-02T12:42:53
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = faceDetect
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp \
    process.cpp \
    gprs.cpp

HEADERS += \
    process.h \
    gprs.h \
    jc_err.h \
    jc_wr.h
 INCLUDEPATH += /usr/local/include \
                 /usr/local/include/opencv \
                 /usr/local/include/opencv2

 #LIBS += /usr/local/lib/libopencv_highgui.so \
  #       /usr/local/lib/libopencv_core.so\
   #      /usr/local/lib/libopencv_imgproc.so\
    #     /usr/local/lib/libopencv_objdetect.so\
     #  /usr/local/lib/libopencv_contrib.so



 LIBS +=/usr/local/lib/libopencv_calib3d.a \
 /usr/local/lib/libopencv_nonfree.a\
/usr/local/lib/libopencv_contrib.a  \
/usr/local/lib/libopencv_objdetect.a\
/usr/local/lib/libopencv_ocl.a\
/usr/local/lib/libopencv_features2d.a \
 /usr/local/lib/libopencv_photo.a\
/usr/local/lib/libopencv_flann.a     \
/usr/local/lib/libopencv_stitching.a\
/usr/local/lib/libopencv_gpu.a       \
/usr/local/lib/libopencv_superres.a\
/usr/local/lib/libopencv_highgui.a   \
 /usr/local/lib/libopencv_ts.a\
/usr/local/lib/libopencv_imgproc.a   \
 /usr/local/lib/libopencv_video.a\
/usr/local/lib/libopencv_legacy.a    \
 /usr/local/lib/libopencv_videostab.a\
/usr/local/lib/libopencv_ml.a\
/usr/local/share/OpenCV/3rdparty/lib/libIlmImf.a \
/usr/local/share/OpenCV/3rdparty/lib/liblibjasper.a \
/usr/local/share/OpenCV/3rdparty/lib/liblibjpeg.a \
/usr/local/share/OpenCV/3rdparty/lib/liblibpng.a \
/usr/local/share/OpenCV/3rdparty/lib/liblibtiff.a \
/usr/local/lib/libopencv_core.a\
/usr/local/share/OpenCV/3rdparty/lib/libzlib.a
