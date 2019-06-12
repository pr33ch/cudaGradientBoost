#include <iostream>
#include <string>
#include "CSVRow.h"

int main()
{
    std::ifstream       file("plop.csv");

    CSVRow              row;
    while(file >> row)
    {
        std::cout << "4th Element(" << row[3] << ")\n";
    }
}	