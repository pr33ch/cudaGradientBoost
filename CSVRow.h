#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <math.h>

class CSVRow
{
    public:
        float const& operator[](std::size_t index) const
        {
            return m_data[index];
        }
        std::size_t size() const
        {
            return m_data.size();
        }
        void readNextRow(std::istream& str)
        {
            std::string         line;
            std::getline(str, line);

            std::stringstream   lineStream(line);
            std::string         cell;

            m_data.clear();
            while(std::getline(lineStream, cell, ','))
            {
                m_data.push_back(stof(cell));
            }
            // // This checks for a trailing comma with no data after it.
            // if (!lineStream && cell.empty())
            // {
            //     // If there was a trailing comma then add an empty element.
            //     m_data.push_back("");
            // }
        }
    private:
        std::vector<float>    m_data;
};

std::istream& operator>>(std::istream& str, CSVRow& data)
{
    data.readNextRow(str);
    return str;
}

void initialize_tree(CSVRow data_table[], float * tree)
{
	for (int nth_variable = 0; nth_variable < data_table[0].size()-1; nth_variable++)
	{
		float average = 0;
		for (int nth_sample = 0; nth_sample < data_table.size(); nth_sample ++)
		{
			average += data_table[nth_sample][nth_variable]/data_table.size();
		}
		memcpy(tree[nth_variable], average, sizeof(float));
	}
}

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