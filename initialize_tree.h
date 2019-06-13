#include "CSVRow.h"

void initialize_tree(CSVRow data_table[], float * tree)
{
	for (int nth_variable = 0; nth_variable < data_table[0].size()-1; nth_variable++)
	{
		float average = 0
		for (int nth_sample = 0; nth_sample < data_table.size(); nth_sample ++)
		{
			average += data_table[nth_sample][nth_variable]/data_table.size();
		}
		memcpy(tree[nth_variable], average, sizeof(float));
	}
}