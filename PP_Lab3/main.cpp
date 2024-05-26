#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <mpi.h>
#include <string>

using namespace std;
using namespace std::chrono;

void writeMatrixToFile(const vector<vector<int>>& matrix, const string& filename) {
    ofstream file(filename);
    for (const auto& row : matrix) {
        for (const auto& element : row) {
            file << element << " ";
        }
        file << "\n";
    }
    file.close();
}

vector<vector<int>> generateMatrix(int size) {
    vector<vector<int>> matrix(size, vector<int>(size));
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(-100, 100);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = distrib(gen);
        }
    }

    return matrix;
}

vector<vector<int>> multiplyMatrices(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int n = A.size();
    vector<vector<int>> C(n, vector<int>(n, 0));
    int i, j, k;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            int sum = 0;
            for (k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    return C;
}

void parallelMatrixMultiplication(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int rank, int size) {
    int n = A.size();
    int rowsPerProcess = n / size;
    int extraRows = n % size;

    int startRow = rank * rowsPerProcess + min(rank, extraRows);
    int endRow = startRow + rowsPerProcess + (rank < extraRows ? 1 : 0);

    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < n; ++j) {
            int sum = 0;
            for (int k = 0; k < n; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ofstream resultFile("dataMy/ResultExperimentMPI_4thread.txt");

    vector<int> sizes = { 10, 50, 100,  500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800 };
    for (int size : sizes) {
        vector<vector<int>> A, B, C;
        if (rank == 0) {
            A = generateMatrix(size);
            B = generateMatrix(size);
            C.resize(size, vector<int>(size, 0));

            writeMatrixToFile(A, "matrix_" + to_string(size) + "_A.txt");
            writeMatrixToFile(B, "matrix_" + to_string(size) + "_B.txt");
        }

        auto start = high_resolution_clock::now();
        if (rank == 0) {
            cout << "Выполняются расчеты для матриц размером " << size << "x" << size << endl;
        }

        if (rank == 0) {
            MPI_Bcast(&A[0][0], size * size, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&B[0][0], size * size, MPI_INT, 0, MPI_COMM_WORLD);
        }
        else {
            A.resize(size, vector<int>(size));
            B.resize(size, vector<int>(size));
            C.resize(size, vector<int>(size, 0));
            MPI_Bcast(&A[0][0], size * size, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&B[0][0], size * size, MPI_INT, 0, MPI_COMM_WORLD);
        }

        parallelMatrixMultiplication(A, B, C, rank, size);

        if (rank == 0) {
            for (int i = 1; i < size; ++i) {
                MPI_Recv(&C[0][0] + i * (size / size + (i < (size % size) ? 1 : 0)), (size / size + (i < (size % size) ? 1 : 0)) * size, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        else {
            MPI_Send(&C[0][0] + rank * (size / size + (rank < (size % size) ? 1 : 0)), (size / size + (rank < (size % size) ? 1 : 0)) * size, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }

        if (rank == 0) {
            auto stop = high_resolution_clock::now();
            writeMatrixToFile(C, "matrixRes_" + to_string(size) + ".txt");

            auto duration = duration_cast<milliseconds>(stop - start);
            resultFile << "For matrix size" << size << "x" << size << " time -  " << duration.count() << " ms" << endl;
            cout << "Для матрицы размером " << size << "x" << size << " расчеты выполнены за " << duration.count() << " ms" << endl;
        }
    }

    MPI_Finalize();
    return 0;
}
