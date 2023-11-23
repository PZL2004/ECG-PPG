#include "mbed.h"
#include <cstdio>
#include <iostream>
#include <vector>
#include <ctime>

/*
This would probably not run properly if flashed onto the board in this state.
Some things would have to be drastically changed. There are some things I understand,
and others not so much. Tried my best to mimic the format of an actual project that 
could be flashed onto it no problem.

- Pablo Zurita Lozano
*/
BufferedSerial pc(USBTX, USBRX);

int get_time()
{
    std::time_t t = std::time(0);
    return t;
}

double moving_average(const std::vector<double> &vec)
{
    int size = vec.size();
    int l_size = 300; //moving average lookback
    double sum = 0;
    double mAvg = 0;

    for (int i = 0; i <= size - l_size; i++)
    {
        sum = 0;
        
        for (int j = i; j < i + l_size; j++)
        {
            sum += vec[j];
        }
        mAvg = sum / l_size;
    }
    return mAvg;
}

int NumberOfPeaks(const std::vector<double> &vec)
{
    int peaks_num = 0;
    int n = vec.size();
    for (int i = 1; i  < n - 1; i++)
    {
        if (vec[i-1] < vec[i] && vec[i] >= vec[i+1]) // peak?
        {
            while(i < n - 1 && vec[i] == vec[i+1]) i++; //several of same value in a row
            if ( i < n - 1 && vec[i] > vec[i+1]) // check if actually peak
            {
                if (n > 300 && (moving_average(vec) + 1000 <= vec[i])){ // 1000 is the prominence for IR count PPG data.
                    peaks_num++;
                }
                else {
                    continue;
                }
            }
        }
    }
    return peaks_num;
}

// main() runs in its own thread in the OS
int main()
{
    int init_time = get_time();
    double time;
    double peaks_num;
    double heartrate;
    char* buffer = new char[18];
    
    /*
    Vector of PPG Data. Involves reading in of data in real time.
    Theoretically uses the same method the tool uses to plot the data in real time. - PZL
    */
    std::vector<double> vec;

    while (true) {
        if (pc.readable() > 0) // do not know if this will work? Best guess it will, but idk
        {
            int num = pc.read(buffer, 18); //amount of bits in an IR Count value is 18
            vec.push_back(num);
            delete[] buffer;
        }
        time = get_time() - init_time;
        peaks_num = NumberOfPeaks(vec);

        //This would be an average over the time frame. Would be checking the amount of peaks in the data, then dividing by time elapsed since start.
        heartrate = 60*(peaks_num / time);
        printf("%.2f", heartrate);
        ThisThread::sleep_for(500ms);
    }
}

