#ifndef __WSP_MPI__
#define __WSP_MPI__

#define SYSEXPECT(expr) do { if(!(expr)) { perror(__func__); exit(1); } } while(0)
#define error_exit(fmt, ...) do { fprintf(stderr, "%s error: " fmt, __func__, ##__VA_ARGS__); exit(1); } while(0);

typedef char city_t;

typedef struct path_struct_t {
  int cost;         // path cost.
  city_t *path;     // order of city visits (you may start from any city).
} path_t;

void parseArgs(int argc, char **argv);
void wsp_print_result(int procID, double time);
void wspStart(int procID, int procNum);
void freePath();

#endif