#include <iostream>
#include <string>
#include "CSVRow.h"

#define N_VARIABLES 4

int main()
{
	std::string filename = std::to_string(N_VARIABLES) + "d.txt";
    std::ifstream       file(filename);

    CSVRow				variable;

    // data_table[i][j] corresponds to the ith data point and jth variable. If j = N_VARIABLES, j
    // is the output of the ith data point
    CSVRow              data_table[N_VARIABLES + 1];
    int row = 0;
    while(file >> variable)
    {
        data_table[row] = variable;
    	std::cout << "4th Element(" << data_table[row][3] << ")\n";
        row++;
    }
}	