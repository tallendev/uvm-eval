#include <stdlib.h>
#include <stdio.h>
#include<sys/types.h>
#include<sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <assert.h>

#define BUF_SIZE (1024ul * 1024ul * 1024ul * 16ul)


int cont = 1;
void sighandler(int signum, siginfo_t* info, void* ptr)
{
    cont = 0;
    fprintf(stderr, "kill sig recvd, cleaning up\n");
    __sync_synchronize();
}

static __inline__ void dump(char** beg, char** end, int ofd)
{
    write(ofd, *beg, *end - *beg);
    *end = *beg;
    /*
    *next = '\0';
    printf("%s", dat);
    fflush(stdout);
    next = dat;
    */
}

#define likely(x)      __builtin_expect(!!(x), 1) 
#define unlikely(x)    __builtin_expect(!!(x), 0) 

#define MAX_LINE (1024 * 1024)
int main(int argc, char* argv[])
{
    char* next = NULL;
    int errout = 0;
    ssize_t ret;
    char* dat;
    //char dat[BUF_SIZE];
    struct sigaction sig = {0};
    int fd = open("/dev/kmsg", O_RDONLY | O_NONBLOCK);
    int ofd = open(argv[1], O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd < 0)
    {
        fprintf (stderr, "Failed to open /dev/kmsg\n");
        exit(1);
    }
    if (ofd < 0)
    {
        fprintf (stderr, "Failed to open output file %s\n", argv[1]);
        exit(1);
    }
    
    sig.sa_sigaction = sighandler;
    sig.sa_flags = SA_SIGINFO;
    sigaction(SIGTERM, &sig, NULL);
    sigaction(SIGINT, &sig, NULL);
    
    dat = calloc(BUF_SIZE, sizeof(char));
    lseek(fd, 0, SEEK_END);
    next = dat;
    fprintf(stderr, "log init finished\n");
    while(cont || errout != EAGAIN)
    {
        //errout = 0;
        ret = read(fd, next, MAX_LINE);
        errout = errno;
        if (likely(ret >= 0))
        {
            next += ret;
            if (unlikely((size_t)(next - dat) >= BUF_SIZE - MAX_LINE))
            {
                dump(&dat, &next, ofd);
            }
        }
        else
        {
            assert (errout == EPIPE || errout == EAGAIN || errout == 0);
            if (next != dat)
            {
                dump(&dat, &next, ofd);
            }
        }
    }
    fsync(ofd);
    free(dat);
    close(fd);
    close(ofd);
    return 0;
}
