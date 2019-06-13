#include "CSVRow.h"
#include <math.h>

// naive CPU implementation of leaf assignment of each datapoint

// the structure of tree is a balanced binary decision tree, where the decisions at each depth i are
// whether or not a given data point's value for variable i is <= the average for variable i. The 
// depth of the tree is the dimensionality of our dataset

void leaf_assign(CSVRow data_table[], float * tree, std::vector<int> * leaf_bins, int * leafAssignment)
{
	for (int nth_sample = 0; nth_sample < data_table.size(); nth_sample ++)
	{
		int upper = pow(2, tree.size()) - 1;
		int lower = 0;
		// perform binary search to classify sample
		for(int nth_variable = 0; nth_variable < tree.size(); nth_variable ++)
		{
			if (nth_variable == tree.size() - 1) // if we've reached the last decision node
			{
				if (data_table[nth_sample][nth_variable] <= tree[nth_variable])
				{
					leafAssignment[nth_sample] = lower;
					leaf_bins[lower].push_back(nth_sample);
				}
				else
				{
					leafAssignment[nth_sample] = upper;
					leaf_bins[upper].push_back(nth_sample);
				}
			}
			if (data_table[nth_sample][nth_variable] <= tree[nth_variable])
			{
				upper = upper/2;
			}
			else
			{
				lower = upper/2;
			}
		}
	}
}

__global__ void cuda_leaf_assign()
{
	int nth_sample = blockDim.x * blockIdx.x + threadIdx.x;
	int upper = pow(2, tree.size()) - 1;
	int lower = 0;
	// perform binary search to classify sample
	for(int nth_variable = 0; nth_variable < tree.size(); nth_variable ++)
	{
		if (nth_variable == tree.size() - 1) // if we've reached the last decision node
		{
			if (data_table[nth_sample][nth_variable] <= tree[nth_variable])
			{
				leafAssignment[nth_sample] = lower;
				leaf_bins[lower].push_back(nth_sample);
			}
			else
			{
				leafAssignment[nth_sample] = upper;
				leaf_bins[upper].push_back(nth_sample);
			}
		}
		if (data_table[nth_sample][nth_variable] <= tree[nth_variable])
		{
			upper = upper/2;
		}
		else
		{
			lower = upper/2;
		}
	}
}