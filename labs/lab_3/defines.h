
#define TEN 0
#define HUNDRED 1
#define THOUSAND 0
#define TEN_THOUSAND 0

#if TEN
#define DIM 10
#define TILE 2
#endif

#if HUNDRED
#define DIM 100
#define TILE 4
#endif 

#if THOUSAND
#define DIM 1000
#define TILE 16
#endif 

#if TEN_THOUSAND
#define DIM 10000
#define TILE 32
#endif 