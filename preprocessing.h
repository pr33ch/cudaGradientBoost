#include "CSVRow.h"

// run this on CPU. Initialize the array of predictions
void preprocessing(float * actual, float * predicted_array)
{
	float runningSum = 0;
	//  take the average of all elements in the output's row of data_table
	for (int i = 0; i < actual.size(); i++)
	{
			runningSum += data_table[outputIndex][i];
	}
	float average = runningSum/data_table[outputIndex].size();

	// place the average into each spot in predicted_array
	for (int i = 0; i < predicted_array.size(); i++)
	{
			memcpy(predicted_array[i], average, sizeof(float));
	}
}