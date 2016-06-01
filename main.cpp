
#include<stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define MATRIX_SIZE 5

class EquationSystemInit{
public:
	/// <summary>
	///     Генерация случайного уравнения
	/// </summary>
	/// <param name="coefficients">Матрица коэффициентов</param>
	/// <param name="free">Вектор свободных членов</param>
	/// <param name="range">Случайные числа будут в диапазоне {-range..+range}</param>
	static void InitializeRandom(double coefficients[MATRIX_SIZE][MATRIX_SIZE], double free[MATRIX_SIZE], double range){
		#pragma omp parallel for
        for (int i=0; i < MATRIX_SIZE; i++){
        	#pragma omp parallel for
            for (int j=0; j < MATRIX_SIZE; j++){
                coefficients[i][j] = range * (1.0-2.0*(double)rand()/RAND_MAX);
			}
            free[i] = range * (1.0 - 2.0 * (double)rand()/RAND_MAX);
        }
	}
};

class EquationSystemSolve{
public:
	/// <summary>
	///     Стадия декомпозиции
	/// </summary>
	/// <param name="coefficients">Матрица коэффициентов</param>
	/// <param name="free">Вектор свободных членов</param>
	/// <param name="rank">Ранг текущего процесса</param>
	/// <param name="processesCount">Количество процессов</param>
	static void ParallelDecomposition(double coefficients[MATRIX_SIZE][MATRIX_SIZE], double free[MATRIX_SIZE], int rank, int processesCount){
		int map[MATRIX_SIZE];
		int temp[MATRIX_SIZE];
		#pragma omp parallel for  
		for(int i=0; i<MATRIX_SIZE; i++){
			map[i]= i % processesCount;
		}

		for(int k = 0; k < MATRIX_SIZE; k++){
			MPI_Bcast (&coefficients[k][k],MATRIX_SIZE-k,MPI_DOUBLE,map[k],MPI_COMM_WORLD);
			MPI_Bcast (&free[k],1,MPI_DOUBLE,map[k],MPI_COMM_WORLD);
			#pragma omp parallel for
			for(int i = k + 1; i<MATRIX_SIZE; i++){
				if(map[i] == rank){
					temp[i] = coefficients[i][k] / coefficients[k][k];
					#pragma omp parallel for
					for(int j = 0; j < MATRIX_SIZE; j++){
						coefficients[i][j] -= temp[i] * coefficients[k][j];
					}
					free[i] -= temp[i] * free[k];
				}
			}
		}
	}
	
	/// <summary>
	///     Стадия обратного хода
	/// </summary>
	/// <param name="coefficients">Матрица коэффициентов</param>
	/// <param name="free">Вектор свободных членов</param>
	/// <param name="result">Вектор результата</param>
	static void BackSubstitution(double coefficients[MATRIX_SIZE][MATRIX_SIZE], double free[MATRIX_SIZE], double result[MATRIX_SIZE]){
		result[MATRIX_SIZE-1] = free[MATRIX_SIZE-1] / coefficients[MATRIX_SIZE-1][MATRIX_SIZE-1];
        for(int i = MATRIX_SIZE - 2; i >= 0; i--){
            double sum = 0.0;
            for(int j = i + 1; j < MATRIX_SIZE; j++){
                sum = sum + coefficients[i][j] * result[j];
            }
            result[i] = (free[i] - sum) / coefficients[i][i];
        }
	}
};


class EquationOutput{
public:
	/// <summary>
	///     Стадия обратного хода
	/// </summary>
	/// <param name="coefficients">Матрица коэффициентов</param>
	/// <param name="free">Вектор свободных членов</param>
	/// <param name="result">Вектор результата</param>
	static void WriteEquation(double coefficients[MATRIX_SIZE][MATRIX_SIZE], double free[MATRIX_SIZE]){
		for (int i = 0; i < MATRIX_SIZE; i++){
			for (int j = 0; j < MATRIX_SIZE; j++){
				printf("%10f ", coefficients[i][j]);
			}
			printf(" == %10f\n", free[i]);
		}
	}
};

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);

    double A[MATRIX_SIZE][MATRIX_SIZE];
    double b[MATRIX_SIZE];
    double x[MATRIX_SIZE];
    double range=1.0;
    int rank, nprocs;
    clock_t begin1, end1, begin2, end2;
    MPI_Status status;
	
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   /* get current process id */
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs); /* get number of processes */
	
	// Стадия 1: Инициализация
    if (rank==0){
    	EquationSystemInit::InitializeRandom(A, b, range);
    	EquationOutput::WriteEquation(A, b);
    }

	// Стадия 2: Декомпозиция
    begin1 =clock();
    MPI_Bcast (&A[0][0],MATRIX_SIZE*MATRIX_SIZE,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast (b,MATRIX_SIZE,MPI_DOUBLE,0,MPI_COMM_WORLD);
    EquationSystemSolve::ParallelDecomposition(A, b, rank, nprocs);
    end1 = clock();

	// Стадия 3: Обратный ход
    begin2 =clock();
    if (rank==0){ 
        EquationSystemSolve::BackSubstitution(A, b, x);
        end2 = clock();
    }
	
	// Вывод результата
    if (rank==0){ 
        printf("\nThe solution is:\n");
        for(int i = 0; i < MATRIX_SIZE; i++)
        	printf("\tx%d  =  %10f\n",i,x[i]);
        printf("Decomposition time: %f\n", (double)(end1 - begin1) / CLOCKS_PER_SEC);
        printf("Back substitution time: %f\n", (double)(end2 - begin2) / CLOCKS_PER_SEC);
    }
    MPI_Finalize();
    return(0);
}


