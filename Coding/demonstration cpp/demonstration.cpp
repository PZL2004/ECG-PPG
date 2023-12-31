#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>

float average(std::vector<double> const& v){
    if(v.empty()){
        return 0;
    }

    auto const count = static_cast<double>(v.size());
    return std::reduce(v.begin(), v.end()) / count;
}

float stdev(std::vector<double> const& v) {
//calculates sample standard deviation
  float sum = 0.0, avg, stdev = 0.0;
  int i;

  avg = average(v);

  for(i = 0; i < v.size(); ++i) {
    stdev += pow(v[i] - avg, 2);
  }

  return sqrt(stdev / (v.size() - 1));
}

//Peak finder code starts here. See license in folder
class peak_detection {

	void diff(std::vector<float> in, std::vector<float>& out)
	{
		out = std::vector<float>(in.size()-1);

		for(int i=1; i<in.size(); ++i)
			out[i-1] = in[i] - in[i-1];
	}

	void vectorElementsProduct(std::vector<float> a, std::vector<float> b, std::vector<float>& out)
	{
		out = std::vector<float>(a.size());

		for(int i=0; i<a.size(); ++i)
			out[i] = a[i] * b[i];
	}

	void findIndicesLessThan(std::vector<float> in, float threshold, std::vector<int>& indices)
	{
		for(int i=0; i<in.size(); ++i)
			if(in[i]<threshold)
				indices.push_back(i+1);
	}

	void selectElementsFromIndices(std::vector<float> in, std::vector<int> indices, std::vector<float>& out)
	{
		for(int i=0; i<indices.size(); ++i)
			out.push_back(in[indices[i]]);
	}

	void selectElementsFromIndices(std::vector<int> in, std::vector<int> indices, std::vector<int>& out)
	{
		for(int i=0; i<indices.size(); ++i)
			out.push_back(in[indices[i]]);
	}

	void signVector(std::vector<float> in, std::vector<int>& out)
	{
		out = std::vector<int>(in.size());

		for(int i=0; i<in.size(); ++i)
		{
			if(in[i]>0)
				out[i]=1;
			else if(in[i]<0)
				out[i]=-1;
			else
				out[i]=0;
		}
	}

	void scalarProduct(float scalar, std::vector<float> in, std::vector<float>& out)
	{
		out = std::vector<float>(in.size());

		for(int i=0; i<in.size(); ++i)
			out[i] = scalar * in[i];
	}
	public:
		void findPeaks(std::vector<float> x0, std::vector<int>& peakInds, bool includeEndpoints, float extrema)
		{
			const float EPS = 2.2204e-16f;
			int minIdx = distance(x0.begin(), min_element(x0.begin(), x0.end()));
			int maxIdx = distance(x0.begin(), max_element(x0.begin(), x0.end()));

			float sel = (x0[maxIdx]-x0[minIdx])/4.0;
			int len0 = x0.size();

			scalarProduct(extrema, x0, x0);

			std::vector<float> dx;
			diff(x0, dx);
			replace(dx.begin(), dx.end(), 0.0f, -EPS);
			std::vector<float> dx0(dx.begin(), dx.end()-1);
			std::vector<float> dx0_1(dx.begin()+1, dx.end());
			std::vector<float> dx0_2;

			vectorElementsProduct(dx0, dx0_1, dx0_2);

			std::vector<int> ind;
			findIndicesLessThan(dx0_2, 0, ind); // Find where the derivative changes sign	
			std::vector<float> x;
			float leftMin;
			int minMagIdx;
			float minMag;
			
			if(includeEndpoints)
			{
				//x = [x0(1);x0(ind);x0(end)];	
				selectElementsFromIndices(x0, ind, x);		
				x.insert(x.begin(), x0[0]);
				x.insert(x.end(), x0[x0.size()-1]);
				//ind = [1;ind;len0];
				ind.insert(ind.begin(), 1);
				ind.insert(ind.end(), len0);
				minMagIdx = distance(x.begin(), std::min_element(x.begin(), x.end()));
				minMag = x[minMagIdx];		
				//std::cout<<"Hola"<<std::endl;
				leftMin = minMag;
			}
			else
			{
				selectElementsFromIndices(x0, ind, x);
				if(x.size()>2)
				{
					minMagIdx = distance(x.begin(), std::min_element(x.begin(), x.end()));		
					minMag = x[minMagIdx];				
					leftMin = x[0]<x0[0]?x[0]:x0[0];
				}
			}

			int len = x.size();

			if(len>2)
			{
				float tempMag = minMag;
				bool foundPeak = false;
				int ii;

				if(includeEndpoints)
				{
					// Deal with first point a little differently since tacked it on
					// Calculate the sign of the derivative since we tacked the first
					//  point on it does not neccessarily alternate like the rest.
					std::vector<float> xSub0(x.begin(), x.begin()+3);//tener cuidado subvector
					std::vector<float> xDiff;//tener cuidado subvector
					diff(xSub0, xDiff);

					std::vector<int> signDx;
					signVector(xDiff, signDx);

					if (signDx[0] <= 0) // The first point is larger or equal to the second
					{
						if (signDx[0] == signDx[1]) // Want alternating signs
						{
							x.erase(x.begin()+1);
							ind.erase(ind.begin()+1);
							len = len-1;
						}
					}
					else // First point is smaller than the second
					{
						if (signDx[0] == signDx[1]) // Want alternating signs
						{
							x.erase(x.begin());
							ind.erase(ind.begin());
							len = len-1;
						}
					}
				}

				//Skip the first point if it is smaller so we always start on maxima
				if ( x[0] >= x[1] )
					ii = 0;
				else
					ii = 1;

				//Preallocate max number of maxima
				float maxPeaks = ceil((float)len/2.0);
				std::vector<int> peakLoc(maxPeaks,0);
				std::vector<float> peakMag(maxPeaks,0.0);
				int cInd = 1;
				int tempLoc;		
			
				while(ii < len)
				{
					ii = ii+1;//This is a peak
					//Reset peak finding if we had a peak and the next peak is bigger
					//than the last or the left min was small enough to reset.
					if(foundPeak)
					{
						tempMag = minMag;
						foundPeak = false;
					}
				
					//Found new peak that was lager than temp mag and selectivity larger
					//than the minimum to its left.
				
					if( x[ii-1] > tempMag && x[ii-1] > leftMin + sel )
					{
						tempLoc = ii-1;
						tempMag = x[ii-1];
					}

					//Make sure we don't iterate past the length of our vector
					if(ii == len)
						break; //We assign the last point differently out of the loop

					ii = ii+1; // Move onto the valley
					
					//Come down at least sel from peak
					if(!foundPeak && tempMag > sel + x[ii-1])
					{            	
						foundPeak = true; //We have found a peak
						leftMin = x[ii-1];
						peakLoc[cInd-1] = tempLoc; // Add peak to index
						peakMag[cInd-1] = tempMag;
						cInd = cInd+1;
					}
					else if(x[ii-1] < leftMin) // New left minima
						leftMin = x[ii-1];
					
				}

				// Check end point
				if(includeEndpoints)
				{        
					if ( x[x.size()-1] > tempMag && x[x.size()-1] > leftMin + sel )
					{
						peakLoc[cInd-1] = len-1;
						peakMag[cInd-1] = x[x.size()-1];
						cInd = cInd + 1;
					}
					else if( !foundPeak && tempMag > minMag )// Check if we still need to add the last point
					{
						peakLoc[cInd-1] = tempLoc;
						peakMag[cInd-1] = tempMag;
						cInd = cInd + 1;
					}
				}
				else if(!foundPeak)
				{
					float minAux = x0[x0.size()-1]<x[x.size()-1]?x0[x0.size()-1]:x[x.size()-1];
					if ( x[x.size()-1] > tempMag && x[x.size()-1] > leftMin + sel )
					{
						peakLoc[cInd-1] = len-1;
						peakMag[cInd-1] = x[x.size()-1];
						cInd = cInd + 1;
					}
					else if( !tempMag >  minAux + sel)// Check if we still need to add the last point
					{
						peakLoc[cInd-1] = tempLoc;
						peakMag[cInd-1] = tempMag;
						cInd = cInd + 1;
					}
				}

				//Create output
				if( cInd > 0 )
				{        	
					std::vector<int> peakLocTmp(peakLoc.begin(), peakLoc.begin()+cInd-1);
					selectElementsFromIndices(ind, peakLocTmp, peakInds);        	
				}		

			}
			//else
			//{
				//input signal length <= 2
			//}
		}
};

double convertStringToUnixTime(const std::string& dateString) {
    std::tm tm = {};
    std::istringstream ss(dateString);
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");

    // Handling milliseconds
    char dot;
    double milliseconds = 0.0;

    if (ss >> dot && dot == '.' && ss >> milliseconds) {
        // Convert milliseconds to seconds
        milliseconds /= 1000.0;
    }

    auto tp = std::chrono::system_clock::from_time_t(std::mktime(&tm)) + std::chrono::duration<double>(milliseconds);
    return std::chrono::duration<double>(tp.time_since_epoch()).count();
}

std::vector<double> find_peak_times(const std::vector<double>& times,
    const std::vector<int>& peak_indices) {
    std::vector<double> peak_unix_times;

    // Iterate through peak indices and get corresponding Unix time values
    for (int index : peak_indices) {
        peak_unix_times.push_back(times[index]);
    }

    return peak_unix_times;
}

int main()
{
    std::vector<double> time;
    std::vector<float> ir;

    std::ifstream inFile("ECPPG_2023-11-10_13-32-13.txt");
    if (inFile.is_open())
    {
        std::string line;
        bool header = true;
        while(std::getline(inFile,line))
        {
            if (header)
            {
                header = false;
                continue;
            }
            std::stringstream ss(line);

            std::string Time, Sample_Count, IR_Count, Red_Count, Raw_ECG, Raw_ECG_mV, Filtered_ECG, Filtered_ECG_mV;
            std::getline(ss,Time,',');    //std::cout<<"\""<<Time<<"\"";
            std::getline(ss,Sample_Count,','); //std::cout<<", \""<<Sample_Count<<"\"";
            std::getline(ss,IR_Count,','); //std::cout<<", \""<<IR_Count<<"\"";
            std::getline(ss,Red_Count,','); //std::cout<<", \""<<Red_Count<<"\"";
            std::getline(ss,Raw_ECG,','); //std::cout<<", \""<<Raw_ECG<<"\"";
            std::getline(ss,Raw_ECG_mV,','); //std::cout<<", \""<<Raw_ECG_mV<<"\"";
            std::getline(ss,Filtered_ECG,','); //std::cout<<", \""<<Filtered_ECG<<"\"";
            std::getline(ss,Filtered_ECG_mV,','); //std::cout<<", \""<<Filtered_ECG_mV<<"\"";

            
            time.push_back(convertStringToUnixTime(Time));
            ir.push_back(std::stof(IR_Count));

            std::cout << '\n';
        }
    }

    std::vector<int> peaks_loc;
	peak_detection obj;
    obj.findPeaks(ir, peaks_loc, false, 1);
    std::vector<double> peak_times = find_peak_times(time, peaks_loc);
    std::vector<double> diffs;
    for (int i = 1; i < peak_times.size(); ++i)
    {
        double diff = peak_times[i] - peak_times[i-1];
        if (diff != 0 && diff >= 0.3)
        {
            diffs.push_back(diff);
            // std::cout << diff << std::endl;
        }
    }

    std::vector<double> heartrates;
    for (int i = 0; i < diffs.size(); ++i)
    {
        double heartrate = 60/diffs[i];
        heartrates.push_back(heartrate);
    }
	/*
	//printing each calculated heart rate value in vector
    for (int i = 0; i < heartrates.size(); ++i)
    {
        std::cout << heartrates[i] << std::endl;
    }
	*/
    float avg_BPM = average(heartrates);
	float BPM_stdev = stdev(heartrates);
	float coeff_of_variation = BPM_stdev/avg_BPM;
    std::cout << "AVG HR: " << avg_BPM << std::endl;
	std::cout << "Standard Deviation of HR: " << BPM_stdev << std::endl;
	std::cout << "Coefficient of Variation (<1 is good): " << coeff_of_variation << std::endl;

    return 0;
}