#include <iostream>
#include <algorithm>
#include <thread>
#include <array>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <numeric>
#include <iomanip>
#include <cmath>
#include "CL.hpp"
#include "MathCode.h"


using namespace std;

const int DATA_SIZE = 20*1024*1024;
const int TESTS_NUMBER = 200;
float *pInputVector1 = nullptr;
float *pInputVector2 = nullptr;
float *pOutputVector = nullptr;
float *pOutputVectorHost = nullptr;
double hostPerformanceTimeMS = 0;
std::vector<double> timeValues;

void PrintTimeStatistic() {
    std::sort(timeValues.begin(), timeValues.end());
    double totalTime = std::accumulate(timeValues.begin(), timeValues.end(), 0.0);
    double averageTime = totalTime/timeValues.size();
    double minTime = timeValues[0];
    double maxTime = timeValues[timeValues.size()-1];
    double medianTime = timeValues[timeValues.size()/2];
    cout << "Calculation time statistic: (" << timeValues.size() << " runs)" << endl;
    cout << "Med: " << medianTime << " ms (" << hostPerformanceTimeMS/medianTime << "X faster then host)" << endl;
    cout << "Avg: " << averageTime << " ms" << endl;
    cout << "Min: " << minTime << " ms" << endl;
    cout << "Max: " << maxTime << " ms" << endl << endl;
}

void GenerateTestData(){
    pInputVector1 = new float[DATA_SIZE];
    pInputVector2 = new float[DATA_SIZE];
    pOutputVector = new float[DATA_SIZE];
    pOutputVectorHost = new float[DATA_SIZE];
    
    srand ((unsigned int)time(NULL));
    for (int i=0; i<DATA_SIZE; i++)
    {
        pInputVector1[i] = rand() * 1000.0f / RAND_MAX;
        pInputVector2[i] = rand() * 1000.0f / RAND_MAX;
    }
}

void STDThreadCalculationFunction(int start, int end) {
    for(int iJob=start; iJob<end; iJob++) {
        //Perform calculations
        pOutputVector[iJob] = MathCalculations(pInputVector1[iJob], pInputVector2[iJob]);
    }
}

void PerformCalculationsOnHostSeparateFunction()
{
    cout << endl << "-------------------------------------------------" << endl;
    cout << "Device: Host separate function" << endl << endl;
    
    //Some performance measurement
    timeValues.clear();
    
    for(int iTest = 0; iTest < (TESTS_NUMBER / 5); iTest++)
    {
        // Начало вычислений
        clock_t beginTime = clock();
        
        STDThreadCalculationFunction(0, DATA_SIZE);
        
        // Время завершения вычислений
        clock_t endTime = clock();
        double timeSpent = (double)(endTime - beginTime) / CLOCKS_PER_SEC;
        
        timeValues.push_back(timeSpent);
    }
    
    PrintTimeStatistic();
}

void PerformCalculationsOnHost()
{
    cout << "Device: Host" << endl << endl;
    
    //Some performance measurement
    timeValues.clear();
    
    for(int iTest=0; iTest<(TESTS_NUMBER/10); iTest++)
    {
        // Начало вычислений
        clock_t beginTime = clock();
        
        for(int iJob=0; iJob<DATA_SIZE; iJob++)
        {
            //Perform calculations
            pOutputVectorHost[iJob] = MathCalculations(pInputVector1[iJob], pInputVector2[iJob]);
        }
        
        // Время завершения вычислений
        clock_t endTime = clock();
        double timeSpent = (double)(endTime - beginTime) / CLOCKS_PER_SEC;
        
        timeValues.push_back(timeSpent);
    }
    hostPerformanceTimeMS = std::accumulate(timeValues.begin(), timeValues.end(), 0.0)/timeValues.size();
    
    PrintTimeStatistic();
}

void PerformCalculationsOnHostSTDThread()
{
    cout << endl << "-------------------------------------------------" << endl;
    cout << "Device: Host std::thread" << endl << endl;
    
    //Some performance measurement
    timeValues.clear();
    
    int threadsNumber = std::max(unsigned(1), std::thread::hardware_concurrency());
    cout << "Threads number: " << threadsNumber << endl << endl;
    int jobsPerThread = DATA_SIZE/threadsNumber;
    
    for(int iTest=0; iTest<(TESTS_NUMBER/5); iTest++)
    {
        // Начало вычислений
        clock_t beginTime = clock();
        
        int curStartJob = 0;
        std::vector<std::thread> threadVector;
        for(int iThread=0; iThread<threadsNumber; iThread++)
        {
            threadVector.push_back(std::thread(STDThreadCalculationFunction, curStartJob, min(curStartJob+jobsPerThread, DATA_SIZE)));
            curStartJob += jobsPerThread;
        }
        
        for(auto thread=threadVector.begin(); thread!=threadVector.end(); thread++)
            thread->join();
        
        // Время завершения вычислений
        clock_t endTime = clock();
        double timeSpent = (double)(endTime - beginTime) / CLOCKS_PER_SEC;
        
        timeValues.push_back(timeSpent);
    }
    
    PrintTimeStatistic();
}

void PerformCalculationsOnHostSTDThread1()
{
    cout << endl << "-------------------------------------------------" << endl;
    cout << "Device: Host std::thread 1 " << endl << endl;
    
    //Some performance measurement
    timeValues.clear();
    
    int threadsNumber = 1;
    cout << "Threads number: " << threadsNumber << endl << endl;
    int jobsPerThread = DATA_SIZE/threadsNumber;
    
    for(int iTest=0; iTest<(TESTS_NUMBER/5); iTest++)
    {
        // Начало вычислений
        clock_t beginTime = clock();
        
        int curStartJob = 0;
        std::vector<std::thread> threadVector;
        for(int iThread=0; iThread<threadsNumber; iThread++)
        {
            threadVector.push_back(std::thread(STDThreadCalculationFunction, curStartJob, min(curStartJob+jobsPerThread, DATA_SIZE)));
            curStartJob += jobsPerThread;
        }
        
        for(auto thread=threadVector.begin(); thread!=threadVector.end(); thread++)
            thread->join();
        
        // Время завершения вычислений
        clock_t endTime = clock();
        double timeSpent = (double)(endTime - beginTime) / CLOCKS_PER_SEC;
        
        timeValues.push_back(timeSpent);
    }
    
    PrintTimeStatistic();
}

void PerformTestOnDevice(cl::Device device)
{
    cout << endl << "-------------------------------------------------" << endl;
    cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << endl << endl;
    
    //For the selected device create a context
    vector<cl::Device> contextDevices;
    contextDevices.push_back(device);
    cl::Context context(contextDevices);
    
    //For the selected device create a context and command queue
    cl::CommandQueue queue(context, device);
    
    //Clean output buffers
    fill_n(pOutputVector, DATA_SIZE, 0);
    
    //Create memory buffers
    cl::Buffer clmInputVector1 = cl::Buffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, DATA_SIZE * sizeof(float), pInputVector1);
    cl::Buffer clmInputVector2 = cl::Buffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, DATA_SIZE * sizeof(float), pInputVector2);
    cl::Buffer clmOutputVector = cl::Buffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, DATA_SIZE * sizeof(float), pOutputVector);
    
    //Load OpenCL source code
    std::ifstream sourceFile("./Kernel.cl");
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile),(std::istreambuf_iterator<char>()));
    
    //Build OpenCL program and make the kernel
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));
    cl::Program program = cl::Program(context, source);
    program.build(contextDevices);
    cl::Kernel kernel(program, "TestKernel");
    
    //Set arguments to kernel
    int iArg = 0;
    kernel.setArg(iArg++, clmInputVector1);
    kernel.setArg(iArg++, clmInputVector2);
    kernel.setArg(iArg++, clmOutputVector);
    kernel.setArg(iArg++, DATA_SIZE);
    
    //Some performance measurement
    timeValues.clear();
    
    //Run the kernel on specific ND range
    for(int iTest=0; iTest<TESTS_NUMBER; iTest++)
    {
        // Начало вычислений
        clock_t beginTime = clock();
        
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(DATA_SIZE), cl::NDRange(128));
        queue.finish();
        
        // Время завершения вычислений
        clock_t endTime = clock();
        double timeSpent = (double)(endTime - beginTime) / CLOCKS_PER_SEC;
        
        timeValues.push_back(timeSpent);
    }
    
    PrintTimeStatistic();
    
    // Read buffer C into a local list
    queue.enqueueReadBuffer(clmOutputVector, CL_TRUE, 0, DATA_SIZE * sizeof(float), pOutputVector);
}

void CheckResults() {
    double avgRelAbsDiff = 0;
    double maxRelAbsDiff = 0;
    for(int iJob=0; iJob<DATA_SIZE; iJob++){
        double absDif = std::abs(pOutputVectorHost[iJob] - pOutputVector[iJob]);
        double relAbsDif = std::abs(absDif/pOutputVectorHost[iJob]);
        avgRelAbsDiff += relAbsDif;
        maxRelAbsDiff = max(maxRelAbsDiff, relAbsDif);
        pOutputVector[iJob] = 0;
    }
    avgRelAbsDiff /= DATA_SIZE;
    
    cout << "Errors:" << endl;
    cout << "avgRelAbsDiff = " << avgRelAbsDiff << endl;
    cout << "maxRelAbsDiff = " << maxRelAbsDiff << endl;
}

int main(int argc, char* argv[]) {
    GenerateTestData();
    
    PerformCalculationsOnHost();
    
    PerformCalculationsOnHostSeparateFunction();
    CheckResults();
    
    PerformCalculationsOnHostSTDThread();
    CheckResults();
    
    PerformCalculationsOnHostSTDThread1();
    CheckResults();
    
    //Get all available platforms
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    
    for (unsigned int iPlatform=0; iPlatform < platforms.size(); iPlatform++) {
        //Get all available devices on selected platform
        std::vector<cl::Device> devices;
        platforms[iPlatform].getDevices(CL_DEVICE_TYPE_ALL, &devices);
        
        //Perform test on each device
        for (unsigned int iDevice=0; iDevice<devices.size(); iDevice++){
            try{
                PerformTestOnDevice(devices[iDevice]);
            }catch(std::exception error){
                std::cout << error.what() << std::endl;
            }
            CheckResults();
        }
    }
    
    //Clean buffers
    delete[](pInputVector1);
    delete[](pInputVector2);
    delete[](pOutputVector);
    delete[](pOutputVectorHost);
    
    return 0;
}
