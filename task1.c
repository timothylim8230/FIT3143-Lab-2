// Build: mpicc -O2 -Wall -o task1_mpi.exe task1_mpi.c -lm
// Run (single machine):  mpirun -np <numProcesses> ./task1_mpi.exe <n>
// Run (cluster/CAAS):    mpirun --hostfile hosts.txt -np <numProcesses> ./task1_mpi.exe <n>
 
/**
 * Timothy Lim  33111472 tlim0034@student.monash.edu
 * Scott Nguyen 33879095 sngu0065@student.monash.edu
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <mpi.h>
 
#define OUTPUT_FILE "primes_mpi.txt"
 
int main(int argc, char **argv)
{
    // initialise execution environment, passes command line args to processes
    MPI_Init(&argc, &argv);
 
    int rank, numProcs;

    // details rank of processes in the communicator
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // determines number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
 
    
    int n = 0;
    if (rank == 0) {
        if (argc != 2) {
            printf("Must be run using mpirun -np <numProcesses> ./task1_mpi.exe <n>\n");
            n = -1;             
        } else {
            // n must be larger than or equal to 2
            n = atoi(argv[1]);
            if (n < 2) {
                printf("Need n >= 2.\n");
                n = -1;
            }
        }
    }
    
    // root sends msg to other processors 
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // n = -1 only happens when program not executed correctly, terminates
    if (n == -1) {
        MPI_Finalize();
        return 1;
    }
    
    // starts timing execution
    double startTime = MPI_Wtime();
 
    // CALCULATION STARTS HERE (SIEVE OF ERATOSTHENES)
    
    // gets square root of n
    int sqrtN = (int)sqrt((double)n);
    
    // initialise array to store base primes (primes below sqrt n)
    bool *baseIsPrime = malloc((sqrtN + 1) * sizeof(bool));

    // safety check if not enough memory, terminates all processes
    if (baseIsPrime == NULL) {
        printf("Rank %d could not allocate the base sieve array.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // assume all numbers are prime first
    for (int i = 0; i <= sqrtN; i++) {
        baseIsPrime[i] = true;
    }


    // marks all composite numbers in baseIsPrime using the sieve
    for (int i = 2; i <= sqrtN; i++) {
        if (baseIsPrime[i]) {
            for (int j = i * i; j <= sqrtN; j += i) {
                baseIsPrime[j] = false;
            }
        }
    }
    
    // counts total primes so far below sqrtN
    int baseCount = 0;
    for (int i = 2; i <= sqrtN; i++) {
        if (baseIsPrime[i]) {
            baseCount++;
        }
    }
 
    // PARALLEL SECTION

    // logistics on what needs to be processed next after base primes
    int first = sqrtN + 1;
    int total = n - first;
    // edge case - if n = 2 then negative
    if (total < 0) {
        total = 0;
    }
 
    // computes segment size for process to sieve
    int myStart   = first + (int)((long long)total * rank / numProcs);
    int myEnd     = first + (int)((long long)total * (rank + 1) / numProcs);
    int sliceSize = myEnd - myStart;
    
    // record time start search
    double searchStart = MPI_Wtime();
 
    // START COMPUTING PRIMES ABOVE BASE

    // allocate array to store which numbers are prime
    bool *localIsPrime = malloc((sliceSize > 0 ? sliceSize : 1) * sizeof(bool));
    // catches memory issues
    if (localIsPrime == NULL) {
        printf("Rank %d could not allocate its slice (%d numbers).\n", rank, sliceSize);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // assume all numbers prime initially
    for (int k = 0; k < sliceSize; k++) {
        localIsPrime[k] = true;
    }
 
    // cross off multiples of every base prime that land inside this slice
    for (int i = 2; i <= sqrtN; i++) {
        if (!baseIsPrime[i]) {
            continue;           // i is composite, its multiples are already gone
        }
 
        // first multiple of i at or above myStart, never below i*i: any
        // smaller multiple of i has a smaller prime factor and was already
        // crossed off by an earlier value of i
        int j = i * i;
        if (j < myStart) {
            j = (myStart / i) * i;
            if (j < myStart) {
                j += i;
            }
        }
 
        for (; j < myEnd; j += i) {
            // mark multipes of base prime as composite in local array (map slice index -> local index)
            localIsPrime[j - myStart] = false;
        }
    }
 
    // count all primes in this processes slice
    int myCount = 0;
    for (int k = 0; k < sliceSize; k++) {
        if (localIsPrime[k]) {
            myCount++;
        }
    }
 
    // collects all primes in the processes slice
    int *myPrimes = malloc((myCount > 0 ? myCount : 1) * sizeof(int));
    if (myPrimes == NULL) {
        printf("Rank %d could not allocate its primes buffer.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int idx = 0;
    for (int k = 0; k < sliceSize; k++) {
        if (localIsPrime[k]) {
            myPrimes[idx++] = myStart + k;
        }
    }

    free(localIsPrime);
 
    // prevents continuing until all processes finish, blocks all here already
    MPI_Barrier(MPI_COMM_WORLD);
    // total search time computed from slowest process (the one that reached barrier last)
    double searchTime = MPI_Wtime() - searchStart;
 

    // accumulators that root will use to gather all info from processes 
    int *recvCounts = NULL; // prime count
    int *displs = NULL; // helps order process information (proc 1 before 2 etc)
    if (rank == 0) {
        recvCounts = malloc(numProcs * sizeof(int));
        displs     = malloc(numProcs * sizeof(int));
    }
    
    // gathers all process counts 
    MPI_Gather(&myCount, 1, MPI_INT, recvCounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    int totalPrimes = 0;
    int *allPrimes = NULL;
    
    // gets total count of primes 
    if (rank == 0) {
        displs[0] = baseCount;      // base primes (2..sqrtN) go first in the output
        for (int r = 1; r < numProcs; r++) {
            displs[r] = displs[r - 1] + recvCounts[r - 1];
        }
        totalPrimes = displs[numProcs - 1] + recvCounts[numProcs - 1];
        
        // initialise array to store all primes
        allPrimes = malloc((totalPrimes > 0 ? totalPrimes : 1) * sizeof(int));
        if (allPrimes == NULL) {
            printf("Could not allocate the combined primes array.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
 
        // fill in the base primes directly; MPI_Gatherv below fills the rest
        int idx2 = 0;
        for (int i = 2; i <= sqrtN; i++) {
            if (baseIsPrime[i]) {
                allPrimes[idx2++] = i;
            }
        }
    }

    // uses variable gather to combine all process myPrimes into one
    MPI_Gatherv(myPrimes, myCount, MPI_INT,
                allPrimes, recvCounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);
        
    // time keep ends here
    double totalTime = MPI_Wtime() - startTime;
 
    // root writes the final, sorted result ---
    if (rank == 0) {
        if (n < 100) {
            printf("Primes below %d:\n", n);
            for (int i = 0; i < totalPrimes; i++) {
                printf("%d ", allPrimes[i]);
            }
            printf("\n");
        } else {
            FILE *out = fopen(OUTPUT_FILE, "w");
            if (out == NULL) {
                printf("Could not open %s for writing.\n", OUTPUT_FILE);
            } else {
                for (int i = 0; i < totalPrimes; i++) {
                    fprintf(out, "%d\n", allPrimes[i]);
                }
                fclose(out);
                printf("Primes below n written to %s\n", OUTPUT_FILE);
            }
        }
 
        printf("n=%d processes=%d primes=%d search=%.4f s total=%.4f s\n",
               n, numProcs, totalPrimes, searchTime, totalTime);
 
        free(recvCounts);
        free(displs);
        free(allPrimes);
    }
 
    free(myPrimes);
    free(baseIsPrime);
 
    MPI_Finalize();
    return 0;
}
