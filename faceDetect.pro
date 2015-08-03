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
    process.cpp

HEADERS += \
    process.h
 INCLUDEPATH += /usr/local/include \
                 /usr/local/include/opencv \
                 /usr/local/include/opencv2

 LIBS += /usr/local/lib/libopencv_highgui.so \
         /usr/local/lib/libopencv_core.so\
         /usr/local/lib/libopencv_imgproc.so\
         /usr/local/lib/libopencv_objdetect.so\
        /usr/local/lib/libopencv_contrib.so
