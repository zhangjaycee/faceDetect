#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<string.h>
#include<termios.h>
#include<sys/ioctl.h>
#include<signal.h>
#include<netinet/in.h>
#include<sys/socket.h>

#include<time.h>

#ifndef GPRS_H
#define GPRS_H
//#include"jc_wr.h"
//#include"jc_err.h"
#endif



extern int socketfd;

void init_tcp();
void stop_tcp();

