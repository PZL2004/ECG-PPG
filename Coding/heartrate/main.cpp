#include "mbed.h"
#include <cstdio>
#include <iostream>
#include <vector>
#include <ctime>
/*
This would probably not run properly if flashed onto the board in this state.
Some things would have to be drastically changed. There are some things I understand,
and others not so much. I only know C++; not JSON, mbed OS, etc. Tried my best to mimic the
format of an actual project that could be flashed onto it no problem.

- Pablo Zurita Lozano
*/

int get_time()
{
    std::time_t t = std::time(0);
    return t;
}

int NumberOfPeaks(const std::vector<double> &vec)
{
    // Will probably still get the dicrotic notches in data. I don't know how to take prominence into account.
    // Especially if variable
    int peaks_num = 0;
    int n = vec.size();
    for (int i = 1; i  < n - 1; i++)
    {
        if (vec[i-1] < vec[i] && vec[i] >= vec[i+1]) // peak?
        {
            while(i < n - 1 && vec[i] == vec[i+1]) i++; //several of same value in a row
            if ( i < n - 1 && vec[i] > vec[i+1]) peaks_num++; // check if actually peak
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
    
    /*
    Vector of PPG Data. Involves reading in of data in real time.
    Theoretically uses the same method the tool uses to plot the data in real time. Don't know
    where to even start. I assume copy-pasting relevant code from the tool provided for the MAX32630FTHR
    could be done. - PZL
    */
    std::vector<double> vec;

    while (true) {
        time = get_time() - init_time;
        peaks_num = NumberOfPeaks(vec);

        //This would be an average over the time frame. Would be checking the amount of peaks in the data, then dividing by time elapsed since start.
        heartrate = 60*(peaks_num / time);
        std::cout << heartrate << std::endl; // this is for VSCode; replace line with whatever version of printf mbed OS wants you to use
    }
}

