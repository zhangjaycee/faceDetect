#define MAXLINE 50
#define SA struct sockaddr
#define SERV_PORT 9877
#define LISTENQ 5

#include "gprs.h"

void init_tcp()
{
    int connfd;
    int listenfd;
    socklen_t clilen;
    pid_t childpid;
    struct sockaddr_in cliaddr,servaddr;
    listenfd=socket(AF_INET,SOCK_STREAM,0);//AF_INET:ipv4  SOCK_STREAM:字节流
    bzero(&servaddr,sizeof(servaddr));//将servaddr置零
    servaddr.sin_family = AF_INET;
    //INADDR_ANY本机地址0.0.0.0,可以代表任意所有网卡的地址
    //htonl函数把本机序转为网络序,用于long型；htons函数功能类似，用于short型
    servaddr.sin_addr.s_addr=htonl(INADDR_ANY);
    servaddr.sin_port=htons(SERV_PORT);
    bind(listenfd,(SA *)&servaddr,sizeof(servaddr));
    listen(listenfd,LISTENQ);
   // for(;;){
    clilen = sizeof(cliaddr);
    connfd=accept(listenfd,(SA *)&cliaddr,&clilen);
    close(listenfd);
    socketfd=connfd;
    /*  if((childpid=fork())==0){//child
            close(listenfd);
            //str_send(connfd);
            exit(0);
    }else{
            //str_recv(connfd);
            exit(0);
    }
    close(connfd);
    //}
    */
}

void stop_tcp()
{
    close(socketfd);
}
