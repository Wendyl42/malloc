/*
 * mm.c
 *
 * - Using Segregated free lists, and every list is a Explicit free list
 *   We have 10 lists: {16}, {17~32}, ..., {2049~4096}, {4097~INF}.
 *   We don't set lists for block with size smaller than 16, because the
 *   minimum of size is 16 bytes.
 * - The hole heap is an Implicit free list
 * - Every block has header and footer like textbook
 * - Every free block has 2 "pointers", which is actually 32 bytes 
 *   bias from heap pointer heap_listp. So we have to macros to help
 *   the bias and the pointer to transform
 * - For the first node of everylist, we put their bias at the start of heap, 
 *   and use seg_lists to find them
 * 
 * Blocks must be aligned to doubleword (8 byte) boundaries.
 * 
 * Minimum block size is 16 bytes. 
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "memlib.h"
#include "mm.h"

/* If you want debugging output, use the following macro.  When you hand
 * in, remove the #define DEBUG line. */
//#define DEBUG
#ifdef DEBUG
#define dbg_printf(...) printf(__VA_ARGS__)
#else
#define dbg_printf(...)
#endif

/* do not change the following! */
#ifdef DRIVER
/* create aliases for driver tests */
#define malloc mm_malloc
#define free mm_free
#define realloc mm_realloc
#define calloc mm_calloc
#endif /* def DRIVER */

#define u_32 unsigned int

/* single word (4) or double word (8) alignment */
#define ALIGNMENT 8

/* rounds up to the nearest multiple of ALIGNMENT */
#define ALIGN(p) (((size_t)(p) + (ALIGNMENT - 1)) & ~0x7)
#define SIZE_T_SIZE (ALIGN(sizeof(size_t)))

/* Basic constants and macros */
#define WSIZE 4             /* Word and header/footer size (bytes) */ 
#define DSIZE 8             /* Double word size (bytes) */
#define CHUNKSIZE (1 << 12) /* Extend heap by this amount (bytes) */ 
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) > (y) ? (y) : (x))

/* Pack a size and allocated bit into a word */
#define PACK(size, alloc) ((size) | (alloc))

/* Read and write a word at address p */
#define GET(p) (*(u_32 *)(p))
#define PUT(p, val) (*(u_32 *)(p) = (val))

/* Read the size and allocated fields from address p */
#define GET_SIZE(p) (GET(p) & ~0x7)
#define GET_ALLOC(p) (GET(p) & 0x1)

/* Given block ptr bp, compute address of its header and footer */
#define HDRP(bp) ((char *)(bp)-WSIZE)
#define FTRP(bp) ((char *)(bp) + GET_SIZE(HDRP(bp)) - DSIZE)

/* Given block ptr bp, compute address of next and previous blocks */
#define NEXT_BLKP(bp) ((char *)(bp) + GET_SIZE(((char *)(bp)-WSIZE)))
#define PREV_BLKP(bp) ((char *)(bp)-GET_SIZE(((char *)(bp)-DSIZE)))

/* Given free block ptr bp, compute address of "pointer"(bias) to next and previous free block in the list */
#define PTR2PREVBIAS(bp) ((char *)(bp))
#define PTR2NEXTBIAS(bp) ((char *)(bp) + WSIZE)

/* Given free block ptr bp, compute "pointer"(bias) to next and previous free block in the list */
#define PREV_FBP_BIAS(bp) (*((u_32 *)(PTR2PREVBIAS(bp))))
#define NEXT_FBP_BIAS(bp) (*((u_32 *)(PTR2NEXTBIAS(bp))))

/* 
 * Get pointer from bias
 * if the bias is zero, it means the pointer is NULL, not head_listp!
 */
#define B2P(base, bias) (((bias) > 0) ? ((char *)(((size_t)(base)) + ((size_t)(bias)))) : NULL)

/* 
 * Get bias from pointer, and set the bias for a block that p points to
 * if the pointer is NULL, we don't compute the bias because it may overflow. Instead, we set bias to 0
 */
#define P2B(base, ptr) (u_32)((size_t)(ptr) - (size_t)(base))
#define SET_BIAS(p, bias) (GET(p) = (u_32)(bias)) // 将偏移量存储到p指向的区域

/* 
 * 10 lists:
 * {16},{17~32}, {33~64},{65~128},{129~256},{257~512},
 * {513~1024},{1025~2048},{2049~4096},{4097~INF}
 */
#define LISTNUM 10

/* Global variables */
static char *heap_listp = NULL; /* Pointer to first block */
static u_32 *seg_lists = NULL;  /* bias of first nodes in lists */

static void *extend_heap(size_t size);
static void *coalesce(void *bp);
static void *place(void *bp, size_t size);
static void add_to_list(void *bp, size_t asize);
static void remove_from_list(void *bp, size_t asize);

static int in_heap(const void *p);
static int aligned(const void *p);

int mm_init(void);
void *malloc(size_t size);
void free(void *bp);
void *realloc(void *oldptr, size_t size);
void *calloc(size_t nmemb, size_t size);
void mm_checkheap(int lineno);

/*
 * mm_init - Initialize the memory manager 
 * return -1 on error, 0 on success.
 */
int mm_init(void)
{
    /* initialize the heap first */
    if ((heap_listp = mem_sbrk(14 * WSIZE)) == (void *)-1)
        return -1;
    /*initialize the array of bias of lists' first nodes*/
    PUT(heap_listp + (0 * WSIZE), 0);  // {16}
    PUT(heap_listp + (1 * WSIZE), 0);  // {17~32}
    PUT(heap_listp + (2 * WSIZE), 0);  // {33~64}
    PUT(heap_listp + (3 * WSIZE), 0);  // {65~128}
    PUT(heap_listp + (4 * WSIZE), 0);  // {129~256}
    PUT(heap_listp + (5 * WSIZE), 0);  // {257~512}
    PUT(heap_listp + (6 * WSIZE), 0);  // {513~1024}
    PUT(heap_listp + (7 * WSIZE), 0);  // {1025~2048}
    PUT(heap_listp + (8 * WSIZE), 0);  // {2049~4096}
    PUT(heap_listp + (9 * WSIZE), 0);  // {4097~INF}
    PUT(heap_listp + (10 * WSIZE), 0); // just for align
    /* Prologue and Epilogue */
    PUT(heap_listp + (11 * WSIZE), PACK(DSIZE, 1));
    PUT(heap_listp + (12 * WSIZE), PACK(DSIZE, 1));
    PUT(heap_listp + (13 * WSIZE), PACK(0, 1));

    seg_lists = heap_listp;
    heap_listp += (12 * WSIZE);

    if (extend_heap(CHUNKSIZE) == NULL)
        return -1;
#ifdef DEBUG
    mm_checkheap(__LINE__);
#endif
    return 0;
}

/*
 * malloc - Allocate a block with at least size bytes of payload 
 * return NULL if failed
 */
void *malloc(size_t size)
{
    size_t asize; // actual size after align
    size_t extendsize;
    char *bp = NULL;
    u_32 list_index; // which list?

    if (heap_listp == NULL)
        mm_init();

    if (!size)
        return NULL;

    asize = (size <= DSIZE) ? (DSIZE << 1) : (ALIGN(size + DSIZE));

    /* Get which list should we get the free block */
    for (list_index = 0; list_index < (LISTNUM - 1); ++list_index)
    {
        if (((asize <= (16 << list_index)) && (seg_lists[list_index] != 0)))
        {
            bp = B2P(heap_listp, seg_lists[list_index]);
            /* search fit free block for malloc */
            while ((bp != NULL) && ((asize > GET_SIZE(HDRP(bp)))))
                bp = B2P(heap_listp, NEXT_FBP_BIAS(bp));
            if (bp != NULL)
                break;
        }
    }
    if (bp == NULL) /* can't find the block */
    {
        /* search the last list: {4097 ~ INF} */
        bp = B2P(heap_listp, seg_lists[list_index]);
        while ((bp != NULL) && ((asize > GET_SIZE(HDRP(bp)))))
            bp = B2P(heap_listp, NEXT_FBP_BIAS(bp));
    }

    /* have to extend heap */
    if (bp == NULL)
    {
        if ((bp = extend_heap(MAX(asize, CHUNKSIZE))) == NULL)
            return NULL;
    }
    /* place the block */
    bp = place(bp, asize);

#ifdef DEBUG
    mm_checkheap(__LINE__);
#endif
    return bp;
}

/*
 * free - Free a block 
 */
void free(void *bp)
{
    if (heap_listp == 0)
        mm_init();
    if (bp == NULL)
        return;
    size_t asize = GET_SIZE(HDRP(bp));
    /* refresh footer and header */
    PUT(HDRP(bp), PACK(asize, 0));
    PUT(FTRP(bp), PACK(asize, 0));
    /* refresh the lists */
    add_to_list(bp, asize);
    /* after free, check coalesce immediately */
    coalesce(bp);

#ifdef DEBUG
    mm_checkheap(__LINE__);
#endif
}

/*
 * realloc - reallocate the block with new size
 * return NULL if failed
 */
void *realloc(void *oldptr, size_t size)
{
    size_t oldsize;
    void *newptr;
    /* new size is 0, just free */
    if (size == 0)
    {
        mm_free(oldptr);
        return 0;
    }
    /* old block don't exist, just malloc */
    if (oldptr == NULL)
        return mm_malloc(size);
    
    /* first malloc a block */
    newptr = mm_malloc(size); 
    if (!newptr)
        return 0;

    /* get new and old black's payload size */
    size = GET_SIZE(HDRP(newptr)) - DSIZE;
    oldsize = GET_SIZE(HDRP(oldptr)) - DSIZE;

    if (size < oldsize)
        oldsize = size;
    memcpy(newptr, oldptr, oldsize);
    /* free old block after copy */
    mm_free(oldptr);

    return newptr;

#ifdef DEBUG
    mm_checkheap(__LINE__);
#endif
}

/*
 * calloc - simple, just malloc and memset
 */
void *calloc(size_t nmemb, size_t size)
{
    size_t asize = nmemb * size;
    void *newptr;

    newptr = mm_malloc(asize);
    memset(newptr, 0, GET_SIZE(HDRP(newptr)));

#ifdef DEBUG
    mm_checkheap(__LINE__);
#endif
    return newptr;
}

/*
 * Return whether the pointer is in the heap.
 * May be useful for debugging.
 */
static int in_heap(const void *p)
{
    return p <= mem_heap_hi() && p >= mem_heap_lo();
}

/*
 * Return whether the pointer is aligned.
 * May be useful for debugging.
 */
static int aligned(const void *p)
{
    return (size_t)ALIGN(p) == (size_t)p;
}

/*
 * mm_checkheap - check correctness
 * first check gloable pointers
 * then check the blocks in heap -- The header and footer, the alignment, and in_heap check
 * finally check the list -- neighbour free blocks should be coalesced and size should fit the list
 */
void mm_checkheap(int lineno)
{
    int error_found = 0;

    /* first check gloable pointers */
    if (heap_listp == NULL)
    {
        dbg_printf("line %d: heap not initialized\n", lineno);
        error_found = 1;
    }
    if (seg_lists == NULL)
    {
        dbg_printf("line %d: segregated lists not initialized\n", lineno);
        error_found = 1;
    }

    /* then check the heap */
    void *heap_bp = heap_listp;
    while (GET_SIZE(HDRP(heap_bp)) || !GET_ALLOC(HDRP(heap_bp)))
    {
        /* hearder should be consistent with footer*/
        if (GET_SIZE(HDRP(heap_bp)) != GET_SIZE(FTRP(heap_bp)))
        {
            dbg_printf("line %d: different size in header and footer\n", lineno);
            error_found = 1;
        }
        if (GET_ALLOC(HDRP(heap_bp)) != GET_ALLOC(FTRP(heap_bp)))
        {
            dbg_printf("line %d: different alloc bit in header and footer\n", lineno);
            error_found = 1;
        }

        /* block should be aligned */
        if (!aligned(heap_bp))
        {
            dbg_printf("line %d: block not aligned\n", lineno);
            error_found = 1;
        }

        /* block should be in the heap */
        if (!in_heap(heap_bp))
        {
            dbg_printf("line %d: block not in heap\n", lineno);
            error_found = 1;
        }
        heap_bp = NEXT_BLKP(heap_bp);
    }

    /* finally check every lists */
    for (int list_index = 0; list_index < LISTNUM; ++list_index)
    {
        void *free_bp = B2P(heap_listp, seg_lists[list_index]);

        while (free_bp != NULL)
        {
            /* first check if theres uncoalesced blocks */
            if (!GET_ALLOC(HDRP(PREV_BLKP(free_bp))) || !GET_ALLOC(HDRP(NEXT_BLKP(free_bp))))
            {
                dbg_printf("line %d: free block not coalesced\n", lineno);
                error_found = 1;
            }

            /* then check if the size fits */
            size_t size = GET_SIZE(HDRP(free_bp));
            if (!((size > (8 << list_index)) && (size <= (16 << list_index))))
            {
                dbg_printf("line %d: block in wrong list\n");
                error_found = 1;
            }
            free_bp = B2P(heap_listp, NEXT_FBP_BIAS(free_bp));
        }
    }
    if (error_found)
        exit(0);
}

/*
 * extend_heap - extend heap of at least size bytes
 * return pointer to block
 */
static void *extend_heap(size_t size)
{
    char *bp;
    size_t asize = ALIGN(size);

    /* extend the heap */
    if ((long)(bp = mem_sbrk(asize)) == -1)
        return NULL;

    /* initialize free block header/footer and the epilogue header */

    PUT(HDRP(bp), PACK(asize, 0));
    PUT(FTRP(bp), PACK(asize, 0));
    PUT(HDRP(NEXT_BLKP(bp)), PACK(0, 1));
    /* add the new block to the list */
    add_to_list(bp, asize);

    /* coalesce if the previous block was free */
    return coalesce(bp);
}

/*
 * coalesce - coalesce according to tag bit. 
 * Return ptr to coalesced block
 */
static void *coalesce(void *bp)
{
    short prev_alloc = GET_ALLOC(HDRP(PREV_BLKP(bp)));
    short next_alloc = GET_ALLOC(HDRP(NEXT_BLKP(bp)));
    size_t asize = GET_SIZE(HDRP(bp));
    size_t next_size, prev_size;

    if (prev_alloc && next_alloc)
        return bp;
    else if (prev_alloc && !next_alloc) /* just coalesce the next block */
    {
        next_size = GET_SIZE(HDRP(NEXT_BLKP(bp)));

        remove_from_list(bp, asize);
        remove_from_list(NEXT_BLKP(bp), next_size);
        asize += next_size;
        PUT(HDRP(bp), PACK(asize, 0));
        PUT(FTRP(bp), PACK(asize, 0));
    }
    else if (!prev_alloc && next_alloc) /* just coalesce the prev block */
    {
        prev_size = GET_SIZE(HDRP(PREV_BLKP(bp)));

        remove_from_list(bp, asize);
        remove_from_list(PREV_BLKP(bp), prev_size);
        asize += prev_size;
        bp = PREV_BLKP(bp);
        PUT(HDRP(bp), PACK(asize, 0));
        PUT(FTRP(bp), PACK(asize, 0));
    }
    else /* coalesce both the prev and next block */
    {
        next_size = GET_SIZE(HDRP(NEXT_BLKP(bp)));
        prev_size = GET_SIZE(HDRP(PREV_BLKP(bp)));
        remove_from_list(bp, asize);
        remove_from_list(PREV_BLKP(bp), prev_size);
        remove_from_list(NEXT_BLKP(bp), next_size);
        asize += (next_size + prev_size);
        bp = PREV_BLKP(bp);
        PUT(HDRP(bp), PACK(asize, 0));
        PUT(FTRP(bp), PACK(asize, 0));
    }
    add_to_list(bp, asize);
    return bp;
}

/* 
 * place - given free block pointer bp, allocate at least size bytes
 * split a new free block if block_size - size > 16
 */
static void *place(void *bp, size_t size)
{
    size_t blk_size = GET_SIZE(HDRP(bp));
    size_t delta = blk_size - size;

    remove_from_list(bp, blk_size);

    if (delta < (2 * DSIZE))
    {
        PUT(HDRP(bp), PACK(blk_size, 1));
        PUT(FTRP(bp), PACK(blk_size, 1));
    }
    else
    {
        PUT(HDRP(bp), PACK(size, 1));
        PUT(FTRP(bp), PACK(size, 1));
        add_to_list(NEXT_BLKP(bp), delta);
        PUT(HDRP(NEXT_BLKP(bp)), PACK(delta, 0));
        PUT(FTRP(NEXT_BLKP(bp)), PACK(delta, 0));
    }
    return bp;
}

/* 
 * add_to_list - find the list that the block fits, and insert it in
 */
static void add_to_list(void *bp, size_t asize)
{
    u_32 list_index = 0;
    void *next_ptr = NULL;
    void *prev_ptr = NULL;

    while ((list_index < LISTNUM - 1) && (asize > (16 << list_index)))
        ++list_index;

    next_ptr = B2P(heap_listp, seg_lists[list_index]); /* initialize it as the first node */

    if (next_ptr == NULL) /* empty list, just insert */
    {
        SET_BIAS(PTR2NEXTBIAS(bp), 0);
        SET_BIAS(PTR2PREVBIAS(bp), 0);
        seg_lists[list_index] = P2B(heap_listp, bp);
    }
    else if (asize <= GET_SIZE(HDRP(next_ptr))) /* not empty list, insert as the new first node */
    {
        SET_BIAS(PTR2NEXTBIAS(bp), P2B(heap_listp, next_ptr));
        SET_BIAS(PTR2PREVBIAS(next_ptr), P2B(heap_listp, bp));
        SET_BIAS(PTR2PREVBIAS(bp), 0);
        seg_lists[list_index] = P2B(heap_listp, bp);
    }
    else /* not empty list and not the first node */
    {
        while ((next_ptr != NULL) && (asize > GET_SIZE(HDRP(next_ptr))))
        {
            prev_ptr = next_ptr;
            next_ptr = B2P(heap_listp, NEXT_FBP_BIAS(next_ptr));
        }
        if (next_ptr != NULL) /* inset in the middle */
        {
            SET_BIAS(PTR2NEXTBIAS(bp), P2B(heap_listp, next_ptr));
            SET_BIAS(PTR2PREVBIAS(next_ptr), P2B(heap_listp, bp));
            SET_BIAS(PTR2NEXTBIAS(prev_ptr), P2B(heap_listp, bp));
            SET_BIAS(PTR2PREVBIAS(bp), P2B(heap_listp, prev_ptr));
        }
        else /* insert as the last node */
        {
            SET_BIAS(PTR2NEXTBIAS(bp), 0);
            SET_BIAS(PTR2NEXTBIAS(prev_ptr), P2B(heap_listp, bp));
            SET_BIAS(PTR2PREVBIAS(bp), P2B(heap_listp, prev_ptr));
        }
    }
}

/* 
 * add_to_list - find the block in which list and remove it from the list
 */
static void remove_from_list(void *bp, size_t asize)
{
    u_32 list_index = 0;
    void *next_ptr = NULL;
    void *prev_ptr = NULL;

    while ((list_index < LISTNUM - 1) && (asize > (16 << list_index)))
        ++list_index;

    next_ptr = B2P(heap_listp, NEXT_FBP_BIAS(bp));
    prev_ptr = B2P(heap_listp, PREV_FBP_BIAS(bp));

    if ((prev_ptr == NULL) && (next_ptr == NULL)) /* remove the only node in the list */
        seg_lists[list_index] = 0;
    else if ((prev_ptr == NULL) && (next_ptr != NULL)) /* remove the first node in the list */
    {
        SET_BIAS(PTR2PREVBIAS(next_ptr), 0);
        seg_lists[list_index] = P2B(heap_listp, next_ptr);
    }
    else if ((prev_ptr != NULL) && (next_ptr == NULL)) /* remove the last node in the list */
        SET_BIAS(PTR2NEXTBIAS(prev_ptr), 0);
    else /* remove a middle node in the list */
    {
        SET_BIAS(PTR2NEXTBIAS(prev_ptr), P2B(heap_listp, next_ptr));
        SET_BIAS(PTR2PREVBIAS(next_ptr), P2B(heap_listp, prev_ptr));
    }
}
