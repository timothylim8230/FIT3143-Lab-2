/**
 * FIT3143 Lab #2 - Task 2
 * Prime search using a hybrid of Open MPI (distributed memory) and
 * OpenMP (shared memory).
 *
 * Build: mpicc -O2 -Wall -fopenmp -o task2 task2.c -lm
 * Run:   mpirun -np <numProcs> ./task2 <n> <threadsPerProc>
 *        e.g. mpirun -np 4 ./task2 100000000 2   (4 processes x 2 threads = 8 workers)
 *
 * Timothy Lim  33111472 tlim0034@student.monash.edu
 * Scott Nguyen 33879095 sngu0065@student.monash.edu
 *
 * ---------------------------------------------------------------------------
 * Parallel partitioning scheme (two levels)
 *
 *   Level 1 (MPI, coarse grained): the search range [2, n) is split into one
 *   equal contiguous block per rank. Each rank allocates and sieves only its
 *   own block, so the memory footprint per machine is n/numProcs bytes rather
 *   than n bytes - this is what lets the hybrid version reach an n the pure
 *   OpenMP version (Week 4 Task 3) cannot fit in one machine's RAM.
 *
 *   Level 2 (OpenMP, fine grained): a rank's block is cut into cache-sized
 *   chunks that its threads pull off a shared queue with schedule(dynamic),
 *   so threads self-balance inside the rank.
 *
 * Why equal-sized blocks are already balanced: in a segmented sieve the cost
 * of a block is sum over base primes p of (blockLength / p), which is
 * proportional to blockLength and (near enough) independent of where the block
 * sits. This is unlike trial division, where large numbers cost far more than
 * small ones and equal blocks would be badly skewed.
 * ---------------------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

#define ROOT 0
#define OUTPUT_FILE "primes_hybrid.txt"

/* One chunk of the mark array is this many bytes (one byte per candidate).
 * Sized to sit comfortably in L1 so the marking loop stays cache resident. */
#define SIEVE_CHUNK 32768

/* Output staging. The file write runs only on the root, so by Amdahl's Law it
 * is the hard limit on the overall speedup. Splitting it up:
 *   - turning the primes into text is CPU work, so the root's OpenMP threads
 *     do it in parallel, using the same count-then-place pattern as the sieve;
 *   - the fwrite itself is disk bound and stays sequential.
 * WRITE_BATCH caps the staging buffer at WRITE_BATCH * DIGITS_MAX bytes so the
 * root's memory use does not grow with n. */
#define WRITE_BATCH (1 << 22) /* primes formatted per fwrite */
#define FORMAT_BLOCK 65536    /* primes handed to one thread at a time */
#define DIGITS_MAX 11         /* "2147483647" plus the newline */

/* Write value as decimal digits into dst (no terminator) and return the number
 * of characters written. Replaces fprintf("%d\n") in the output loop, which is
 * far too slow once there are millions of primes to emit. */
static size_t writeInt(char *dst, int value)
{
    char tmp[12];
    size_t len = 0;

    if (value == 0)
    {
        dst[0] = '0';
        return 1;
    }

    while (value > 0)
    {
        tmp[len++] = (char)('0' + (value % 10));
        value /= 10;
    }

    for (size_t i = 0; i < len; i++)
    {
        dst[i] = tmp[len - 1 - i];
    }

    return len;
}

/* Decimal width of a positive value. A comparison ladder rather than a divide
 * loop, because this runs once per prime purely to reserve buffer space. */
static int digitCount(int value)
{
    if (value < 10)
        return 1;
    if (value < 100)
        return 2;
    if (value < 1000)
        return 3;
    if (value < 10000)
        return 4;
    if (value < 100000)
        return 5;
    if (value < 1000000)
        return 6;
    if (value < 10000000)
        return 7;
    if (value < 100000000)
        return 8;
    if (value < 1000000000)
        return 9;
    return 10;
}

/* Format count primes as one-per-line text and write them to out.
 * Returns 0 if the staging buffer could not be allocated or a write failed. */
static int writePrimes(FILE *out, const int *primes, int count)
{
    int maxBlocks = (WRITE_BATCH + FORMAT_BLOCK - 1) / FORMAT_BLOCK;

    char *buf = malloc((size_t)WRITE_BATCH * DIGITS_MAX);
    long long *blockOffset = malloc((size_t)maxBlocks * sizeof(long long));

    if (buf == NULL || blockOffset == NULL)
    {
        free(buf);
        free(blockOffset);
        return 0;
    }

    for (int base = 0; base < count; base += WRITE_BATCH)
    {
        int batch = count - base;
        if (batch > WRITE_BATCH)
        {
            batch = WRITE_BATCH;
        }

        int blocks = (batch + FORMAT_BLOCK - 1) / FORMAT_BLOCK;

        /* Pass 1 - measure how many bytes each block of primes will occupy. */
#pragma omp parallel for schedule(static)
        for (int b = 0; b < blocks; b++)
        {
            int from = b * FORMAT_BLOCK;
            int to = from + FORMAT_BLOCK;
            if (to > batch)
            {
                to = batch;
            }

            long long bytes = 0;
            for (int i = from; i < to; i++)
            {
                bytes += digitCount(primes[base + i]) + 1; /* +1 for the newline */
            }

            blockOffset[b] = bytes;
        }

        /* Running sum turns the sizes into each block's start position. */
        long long used = 0;
        for (int b = 0; b < blocks; b++)
        {
            long long bytes = blockOffset[b];
            blockOffset[b] = used;
            used += bytes;
        }

        /* Pass 2 - every thread formats into the slice reserved for its own
         * block, so the threads never overlap and the lines stay in order. */
#pragma omp parallel for schedule(static)
        for (int b = 0; b < blocks; b++)
        {
            int from = b * FORMAT_BLOCK;
            int to = from + FORMAT_BLOCK;
            if (to > batch)
            {
                to = batch;
            }

            size_t pos = (size_t)blockOffset[b];
            for (int i = from; i < to; i++)
            {
                pos += writeInt(buf + pos, primes[base + i]);
                buf[pos++] = '\n';
            }
        }

        if (fwrite(buf, 1, (size_t)used, out) != (size_t)used)
        {
            free(buf);
            free(blockOffset);
            return 0;
        }
    }

    free(buf);
    free(blockOffset);
    return 1;
}

/* Plain sieve of Eratosthenes over [2, limit]. Every rank runs this on its own
 * copy: it is only O(sqrt(n)) work, so recomputing it everywhere is cheaper
 * than broadcasting the result. Returns the array, or NULL if there are none. */
static int *buildBasePrimes(long long limit, int *baseCount)
{
    *baseCount = 0;

    if (limit < 2)
    {
        return NULL;
    }

    char *composite = calloc((size_t)limit + 1, 1);
    if (composite == NULL)
    {
        return NULL;
    }

    for (long long i = 2; i * i <= limit; i++)
    {
        if (!composite[i])
        {
            for (long long j = i * i; j <= limit; j += i)
            {
                composite[j] = 1;
            }
        }
    }

    int count = 0;
    for (long long i = 2; i <= limit; i++)
    {
        if (!composite[i])
        {
            count++;
        }
    }

    int *primes = malloc((size_t)count * sizeof(int));
    if (primes == NULL)
    {
        free(composite);
        return NULL;
    }

    int index = 0;
    for (long long i = 2; i <= limit; i++)
    {
        if (!composite[i])
        {
            primes[index++] = (int)i;
        }
    }

    free(composite);
    *baseCount = count;
    return primes;
}

int main(int argc, char *argv[])
{
    int myRank, numProcs, provided;

    /* MPI_THREAD_FUNNELED: this process is multi-threaded but only the thread
     * that called MPI_Init_thread (the main thread) ever calls MPI. Every MPI
     * call below sits outside an omp parallel region, so that promise holds. */
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    if (provided < MPI_THREAD_FUNNELED && myRank == ROOT)
    {
        printf("Warning: MPI library only provides thread level %d.\n", provided);
        fflush(stdout);
    }

    /* ---- Read n on the root process only, as a command line argument ---- */
    long long config[2] = {0, 0}; /* {n, threadsPerProc}, packed into one message */

    if (myRank == ROOT)
    {
        if (argc != 3)
        {
            printf("Usage: mpirun -np <numProcs> %s <n> <threadsPerProc>\n", argv[0]);
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        config[0] = atoll(argv[1]);
        config[1] = atoll(argv[2]);

        if (config[0] < 2 || config[0] > 2147483647LL)
        {
            printf("n must be in the range [2, 2147483647].\n");
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (config[1] < 1)
        {
            printf("threadsPerProc must be at least 1.\n");
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* Everything is timed from here, so the broadcast, the gather and the file
     * write all count towards the overall speedup - not just the sieve. The
     * barrier lines the ranks up first so the root's clock is a fair
     * wall-clock measurement of the whole job. */
    MPI_Barrier(MPI_COMM_WORLD);
    double tStart = MPI_Wtime();

    /* ---- Serial section: disseminate n, then build the base primes ---- */
    double tSetup = MPI_Wtime();

    /* One broadcast carries both values instead of two round trips. */
    MPI_Bcast(config, 2, MPI_LONG_LONG, ROOT, MPI_COMM_WORLD);

    /* n and threadsPerProc are ordinary locals of the main thread. The OpenMP
     * regions below declare no private copy of them, so every thread in this
     * process reads the same broadcast value straight out of shared memory. */
    long long n = config[0];
    int threadsPerProc = (int)config[1];

    omp_set_num_threads(threadsPerProc);

    /* Largest number whose multiples still need crossing off: floor(sqrt(n-1)).
     * Computed with sqrt() then corrected, because sqrt() on a large double can
     * land one either side of the true integer root. */
    long long limit = (long long)sqrt((double)(n - 1));
    while ((limit + 1) * (limit + 1) <= n - 1)
    {
        limit++;
    }
    while (limit > 0 && limit * limit > n - 1)
    {
        limit--;
    }

    int baseCount = 0;
    int *basePrimes = buildBasePrimes(limit, &baseCount);
    if (limit >= 2 && basePrimes == NULL)
    {
        printf("Rank %d: out of memory building base primes.\n", myRank);
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    tSetup = MPI_Wtime() - tSetup;

    /* ---- Parallel section: this rank's block of the segmented sieve ---- */
    double tSieve = MPI_Wtime();

    /* Block decomposition of [2, n). The long long arithmetic keeps
     * span * myRank from overflowing, and the shared expression guarantees
     * rank r's hi is exactly rank r+1's lo, so no candidate is lost or done
     * twice. Ranks beyond the number of candidates simply get an empty block. */
    long long span = n - 2;
    long long lo = 2 + (span * myRank) / numProcs;
    long long hi = 2 + (span * (myRank + 1)) / numProcs;
    long long segLen = hi - lo;

    long long numChunks = (segLen + SIEVE_CHUNK - 1) / SIEVE_CHUNK;

    /* mark[i] corresponds to the number lo + i. 0 means "still a candidate". */
    char *mark = NULL;
    long long *chunkOffset = NULL;

    if (segLen > 0)
    {
        mark = calloc((size_t)segLen, 1);
        chunkOffset = malloc((size_t)numChunks * sizeof(long long));

        if (mark == NULL || chunkOffset == NULL)
        {
            printf("Rank %d: out of memory allocating a segment of %lld bytes.\n",
                   myRank, segLen);
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* Pass 1 - cross off composites. Each chunk is owned by exactly one thread
     * for the whole pass, and chunks never overlap, so no two threads ever
     * write the same byte: the marking needs no lock and no critical section. */
#pragma omp parallel for schedule(dynamic)
    for (long long c = 0; c < numChunks; c++)
    {
        long long chunkLo = lo + c * SIEVE_CHUNK;
        long long chunkHi = chunkLo + SIEVE_CHUNK;
        if (chunkHi > hi)
        {
            chunkHi = hi;
        }

        for (int b = 0; b < baseCount; b++)
        {
            long long p = basePrimes[b];
            long long start = p * p;

            /* Base primes ascend, so once p*p is past this chunk every later
             * base prime is too - nothing left to cross off here. */
            if (start >= chunkHi)
            {
                break;
            }

            /* Never start below p*p: a smaller multiple of p has a smaller
             * prime factor and was already crossed off by that factor. */
            if (start < chunkLo)
            {
                start = (chunkLo / p) * p; /* round chunkLo down to a multiple of p */
                if (start < chunkLo)
                {
                    start += p; /* then step up to the first one inside the chunk */
                }
            }

            for (long long j = start; j < chunkHi; j += p)
            {
                mark[j - lo] = 1;
            }
        }
    }

    /* Pass 2 - count the survivors per chunk, in parallel. */
#pragma omp parallel for schedule(dynamic)
    for (long long c = 0; c < numChunks; c++)
    {
        long long chunkLo = lo + c * SIEVE_CHUNK;
        long long chunkHi = chunkLo + SIEVE_CHUNK;
        if (chunkHi > hi)
        {
            chunkHi = hi;
        }

        long long found = 0;
        for (long long j = chunkLo; j < chunkHi; j++)
        {
            if (!mark[j - lo])
            {
                found++;
            }
        }

        chunkOffset[c] = found;
    }

    /* Running sum turns the per-chunk counts into per-chunk write positions.
     * Serial, but it is one pass over numChunks (a few hundred entries), not
     * over n, so it costs nothing measurable. */
    long long localCount = 0;
    for (long long c = 0; c < numChunks; c++)
    {
        long long found = chunkOffset[c];
        chunkOffset[c] = localCount;
        localCount += found;
    }

    int *localPrimes = NULL;
    if (localCount > 0)
    {
        localPrimes = malloc((size_t)localCount * sizeof(int));
        if (localPrimes == NULL)
        {
            printf("Rank %d: out of memory collecting %lld primes.\n", myRank, localCount);
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* Pass 3 - write each chunk's primes at its own precomputed offset. Threads
     * write into disjoint slices of localPrimes, so again no synchronisation is
     * needed, and the result comes out in ascending order for free. */
#pragma omp parallel for schedule(dynamic)
    for (long long c = 0; c < numChunks; c++)
    {
        long long chunkLo = lo + c * SIEVE_CHUNK;
        long long chunkHi = chunkLo + SIEVE_CHUNK;
        if (chunkHi > hi)
        {
            chunkHi = hi;
        }

        long long out = chunkOffset[c];
        for (long long j = chunkLo; j < chunkHi; j++)
        {
            if (!mark[j - lo])
            {
                localPrimes[out++] = (int)j;
            }
        }
    }

    tSieve = MPI_Wtime() - tSieve;

    /* The mark array is the biggest allocation on each rank and is finished
     * with, so release it before the root grows a buffer for every prime. */
    free(mark);
    free(chunkOffset);
    mark = NULL;
    chunkOffset = NULL;

    /* ---- Communication: gather every rank's primes into the root ---- */
    double tGather = MPI_Wtime();

    int localCountInt = (int)localCount;
    int *counts = NULL;
    int *displs = NULL;
    int *allPrimes = NULL;
    int totalPrimes = 0;

    if (myRank == ROOT)
    {
        counts = malloc((size_t)numProcs * sizeof(int));
        displs = malloc((size_t)numProcs * sizeof(int));
        if (counts == NULL || displs == NULL)
        {
            printf("Root: out of memory building the gather descriptors.\n");
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* Root needs the per-rank counts before it can size the receive buffer. */
    MPI_Gather(&localCountInt, 1, MPI_INT, counts, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    if (myRank == ROOT)
    {
        for (int r = 0; r < numProcs; r++)
        {
            displs[r] = totalPrimes;
            totalPrimes += counts[r];
        }

        if (totalPrimes > 0)
        {
            allPrimes = malloc((size_t)totalPrimes * sizeof(int));
            if (allPrimes == NULL)
            {
                printf("Root: out of memory gathering %d primes.\n", totalPrimes);
                fflush(stdout);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    /* Rank r owns the block immediately above rank r-1's, so laying the blocks
     * down in rank order already produces a globally sorted list. No merge and
     * no sort is performed - that work has been designed out. */
    MPI_Gatherv(localPrimes, localCountInt, MPI_INT,
                allPrimes, counts, displs, MPI_INT, ROOT, MPI_COMM_WORLD);

    tGather = MPI_Wtime() - tGather;

    /* ---- Output: the main thread of the root process writes the result ---- */
    double tWrite = MPI_Wtime();

    if (myRank == ROOT)
    {
        if (n < 100)
        {
            for (int i = 0; i < totalPrimes; i++)
            {
                printf("%d ", allPrimes[i]);
            }
            printf("\n");
        }
        else
        {
            FILE *out = fopen(OUTPUT_FILE, "wb");
            if (out == NULL)
            {
                printf("Could not open %s for writing.\n", OUTPUT_FILE);
                fflush(stdout);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            if (!writePrimes(out, allPrimes, totalPrimes))
            {
                printf("Root: failed to write %s.\n", OUTPUT_FILE);
                fflush(stdout);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            fclose(out);
        }
    }

    tWrite = MPI_Wtime() - tWrite;
    double tTotal = MPI_Wtime() - tStart;

    /* Slowest and fastest sieve across the ranks: the gap between them is the
     * load imbalance, which is what the gather time on the root absorbs. */
    double sieveMax = 0.0, sieveMin = 0.0;
    MPI_Reduce(&tSieve, &sieveMax, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);
    MPI_Reduce(&tSieve, &sieveMin, 1, MPI_DOUBLE, MPI_MIN, ROOT, MPI_COMM_WORLD);

    if (myRank == ROOT)
    {
        /* One line per run. setup/sieve/gather/write are consecutive phases of
         * the root's timeline and add up to total, which is what the Amdahl /
         * Gustafson serial and parallel fractions are derived from:
         *   parallel part = sieve,  serial part = setup + gather + write. */
        printf("n=%lld procs=%d threads=%d primes=%d "
               "setup=%.4f sieve=%.4f gather=%.4f write=%.4f total=%.4f "
               "sieveMax=%.4f sieveMin=%.4f\n",
               n, numProcs, threadsPerProc, totalPrimes,
               tSetup, tSieve, tGather, tWrite, tTotal,
               sieveMax, sieveMin);
        fflush(stdout);
    }

    /* mark and chunkOffset were already released above, right after the sieve. */
    free(basePrimes);
    free(localPrimes);
    free(counts);
    free(displs);
    free(allPrimes);

    MPI_Finalize();
    return 0;
}
