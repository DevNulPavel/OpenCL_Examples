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


const int DATA_SIZE = 1024*1024/sizeof(float) * 32; // 32 мегабайта
const int OPEN_CL_TESTS_NUMBER = 200;
const int CPU_TESTS_NUMBER = 20;
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
    cout << "Calculation time statistic (" << timeValues.size() << " runs):" << endl;
    cout << "- Avg: " << averageTime << " ms (" << hostPerformanceTimeMS/averageTime << "X faster then host)" << endl;
    cout << "- Min: " << minTime << " ms" << endl;
    cout << "- Max: " << maxTime << " ms" << endl;
    cout << "- Med: " << medianTime << " ms" << endl << endl;
}


void CheckResults() {
    double avgRelAbsDiff = 0;
    double maxRelAbsDiff = 0;
    for(int iJob = 0; iJob < DATA_SIZE; iJob++){
        double absDif = std::abs(pOutputVectorHost[iJob] - pOutputVector[iJob]);
        double relAbsDif = std::abs(absDif/pOutputVectorHost[iJob]);
        avgRelAbsDiff += relAbsDif;
        maxRelAbsDiff = max(maxRelAbsDiff, relAbsDif);
        pOutputVector[iJob] = 0;
    }
    avgRelAbsDiff /= DATA_SIZE;
    
    cout << "Errors:" << endl;
    cout << "- avgRelAbsDiff = " << avgRelAbsDiff << endl;
    cout << "- maxRelAbsDiff = " << maxRelAbsDiff << endl;
}

void GenerateTestData(){
    // Буфферы данных
    pInputVector1 = new float[DATA_SIZE];
    pInputVector2 = new float[DATA_SIZE];
    pOutputVector = new float[DATA_SIZE];
    pOutputVectorHost = new float[DATA_SIZE];
    
    // Заполнение случайными значениями входных буфферов
    srand((unsigned int)time(NULL));
    for (int i = 0; i < DATA_SIZE; i++){
        pInputVector1[i] = rand() * 1000.0f / RAND_MAX;
        pInputVector2[i] = rand() * 1000.0f / RAND_MAX;
    }
    
    // Обнуление выходных буффером
    memset(pOutputVector, 0, sizeof(float) * DATA_SIZE);
    memset(pOutputVectorHost, 0, sizeof(float) * DATA_SIZE);
}

// Выполнение вычислений на центральном проце
void PerformCalculationsOnHost() {
    cout << "-------------------------------------------------" << endl;
    cout << "Device: Host" << endl << endl;
    
    // Обнуляем счетчики
    timeValues.clear();
    
    for(int iTest = 0; iTest < CPU_TESTS_NUMBER; iTest++){
        // Начало вычислений
        clock_t beginTime = clock();
        
        for(int iJob=0; iJob<DATA_SIZE; iJob++){
            //Perform calculations
            pOutputVectorHost[iJob] = MathCalculations(pInputVector1[iJob], pInputVector2[iJob]);
        }
        
        // Время завершения вычислений
        clock_t endTime = clock();
        double timeSpent = (double)(endTime - beginTime) / CLOCKS_PER_SEC;
        
        timeValues.push_back(timeSpent);
    }
    
    // Сохраняем среднее значение времени вычислений на хосте как опорное
    hostPerformanceTimeMS = std::accumulate(timeValues.begin(), timeValues.end(), 0.0)/timeValues.size();
    
    PrintTimeStatistic();
}

// Функция для вычисления данных в потоке в диапазоне значений от и до
void STDThreadCalculationFunction(int start, int end) {
    for(int i = start; i < end; i++) {
        pOutputVector[i] = MathCalculations(pInputVector1[i], pInputVector2[i]);
    }
}

// Выполняем вычисление на центральном процессоре c помощью отдельной функции
void PerformCalculationsOnHostSeparateFunction() {
    cout << endl << endl << "-------------------------------------------------" << endl;
    cout << "Device: Host separate function" << endl << endl;
    
    // Обнуляем счетчики
    timeValues.clear();
    
    for(int iTest = 0; iTest < CPU_TESTS_NUMBER; iTest++) {
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

// Выполнение вычислений в отдельном потоке
void PerformCalculationsOnHostSTDThread() {
    cout << endl << endl << "-------------------------------------------------" << endl;
    cout << "Device: Host std::thread" << endl << endl;
    
    // Обнуляем счетчики
    timeValues.clear();
    
#ifdef _MSC_VER
    int threadsNumber = max(unsigned(1), std::thread::hardware_concurrency());
#else
    int threadsNumber = std::max(unsigned(1), std::thread::hardware_concurrency());
#endif // _MS
    cout << "Threads number: " << threadsNumber << endl << endl;
    int jobsPerThread = DATA_SIZE/threadsNumber;
    
    for(int iTest=0; iTest<CPU_TESTS_NUMBER; iTest++){
        // Начало вычислений
        clock_t beginTime = clock();
        
        // Разделяем вычисления по отдельным потокам, каждый поток обрабатывает свой кусок данных
        int curStartJob = 0;
        std::vector<std::thread> threadVector;
        for(int iThread=0; iThread<threadsNumber; iThread++){
            std::thread thread = std::thread(STDThreadCalculationFunction, curStartJob, min(curStartJob+jobsPerThread, DATA_SIZE));
            threadVector.push_back(std::move(thread));
            curStartJob += jobsPerThread;
        }
        
        // Ждем завершения
        for(auto thread = threadVector.begin(); thread != threadVector.end(); thread++){
            thread->join();
        }
        
        // Время завершения вычислений
        clock_t endTime = clock();
        double timeSpent = (double)(endTime - beginTime) / CLOCKS_PER_SEC;
        
        timeValues.push_back(timeSpent);
    }
    
    PrintTimeStatistic();
}

// Вычисления в одном стороннем потоке, а не в нескольких
void PerformCalculationsOnHostSTDThread1() {
    cout << endl << endl << "-------------------------------------------------" << endl;
    cout << "Device: Host std::thread 1 " << endl << endl;
    
    // Обнуляем счетчики
    timeValues.clear();
    
    int threadsNumber = 1;
    cout << "Threads number: " << threadsNumber << endl << endl;
    int jobsPerThread = DATA_SIZE/threadsNumber;
    
    for(int iTest = 0; iTest < CPU_TESTS_NUMBER; iTest++){
        // Начало вычислений
        clock_t beginTime = clock();
        
        // Запускаем все вычисления в одном потоке
        int curStartJob = 0;
        std::vector<std::thread> threadVector;
        for(int iThread=0; iThread<threadsNumber; iThread++) {
            threadVector.push_back(std::thread(STDThreadCalculationFunction, curStartJob, min(curStartJob+jobsPerThread, DATA_SIZE)));
            curStartJob += jobsPerThread;
        }
        
        // Ждем завершения потоков
        for(auto thread=threadVector.begin(); thread!=threadVector.end(); thread++){
            thread->join();
        }
        
        // Время завершения вычислений
        clock_t endTime = clock();
        double timeSpent = (double)(endTime - beginTime) / CLOCKS_PER_SEC;
        
        timeValues.push_back(timeSpent);
    }
    
    PrintTimeStatistic();
}

// Пробуем вычисления на OpenCL устройстве
void PerformTestOnDevice(cl::Device device) {
    cout << endl << endl << "-------------------------------------------------" << endl;
    cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << endl << endl;
    
    // Создание контекста для данного устройства
    vector<cl::Device> contextDevices;
    contextDevices.push_back(device);
    cl::Context context(contextDevices);
    
    // Создание очереди
    cl::CommandQueue queue(context, device);
    
    // Обнуляем вектор выходных данных
    fill_n(pOutputVector, DATA_SIZE, 0.0f);
    
    // Создание буфферов входных и выходных данных сразу с данными
    //cl::Buffer clmInputVector1 = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, DATA_SIZE * sizeof(float), pInputVector1);
    //cl::Buffer clmInputVector2 = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, DATA_SIZE * sizeof(float), pInputVector2);
    //cl::Buffer clmOutputVector = cl::Buffer(context, CL_MEM_READ_WRITE, DATA_SIZE * sizeof(float), NULL);

    // Создание буфферов входных и выходных данных без данных
    cl::Buffer clmInputVector1 = cl::Buffer(context, CL_MEM_READ_ONLY, DATA_SIZE * sizeof(float), NULL);
    cl::Buffer clmInputVector2 = cl::Buffer(context, CL_MEM_READ_ONLY, DATA_SIZE * sizeof(float), NULL);
    cl::Buffer clmOutputVector = cl::Buffer(context, CL_MEM_READ_WRITE, DATA_SIZE * sizeof(float), NULL);
    
    // Загружаем вычислительное ядро OpenCL
    std::ifstream sourceFile("./Kernel.cl");
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()) );
    
    // Компилируем GPU программу
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));
    cl::Program program = cl::Program(context, source);
    program.build(contextDevices);
    
    // Получаем вычислительное ядро
    cl::Kernel kernel(program, "TestKernel");
    
    // Устанавливаем аргументы для вычислительного ядра
    int iArg = 0;
    kernel.setArg(iArg++, clmInputVector1);
    kernel.setArg(iArg++, clmInputVector2);
    kernel.setArg(iArg++, clmOutputVector);
    kernel.setArg(iArg++, DATA_SIZE);
    
    // Обнуляем счетчики
    timeValues.clear();
    
    for(int iTest = 0; iTest < OPEN_CL_TESTS_NUMBER; iTest++) {
        // Начало вычислений
        clock_t beginTime = clock();
        
        // Копируем входные данные в буффер
        queue.enqueueWriteBuffer(clmInputVector1, CL_TRUE, 0, DATA_SIZE * sizeof(float), pInputVector1);
        queue.enqueueWriteBuffer(clmInputVector2, CL_TRUE, 0, DATA_SIZE * sizeof(float), pInputVector2);
        
        // Закидываем задачу
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(DATA_SIZE), cl::NDRange(128));
        
        // Ждем завершения
        queue.finish();
        
        // Читаем данные в выходной буффер
        queue.enqueueReadBuffer(clmOutputVector, CL_TRUE, 0, DATA_SIZE * sizeof(float), pOutputVector);
        
        // Время завершения вычислений
        clock_t endTime = clock();
        double timeSpent = (double)(endTime - beginTime) / CLOCKS_PER_SEC;
        
        timeValues.push_back(timeSpent);
    }
    
    // Читаем данные в выходной буффер (не совсем корректно читать один раз в самом конце)
    // queue.enqueueReadBuffer(clmOutputVector, CL_TRUE, 0, DATA_SIZE * sizeof(float), pOutputVector);
    
    PrintTimeStatistic();
}

int main(int argc, char* argv[]) {
    // Создаем буффер с тестовыми данными
    GenerateTestData();
    
    PerformCalculationsOnHost();
    CheckResults();
    
    PerformCalculationsOnHostSeparateFunction();
    CheckResults();
    
    PerformCalculationsOnHostSTDThread();
    CheckResults();
    
    PerformCalculationsOnHostSTDThread1();
    CheckResults();
    
    // Получаем все доступные платформы
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    
    for (unsigned int iPlatform = 0; iPlatform < platforms.size(); iPlatform++) {
        // Получаем все доступные устройства для платформы
        std::vector<cl::Device> devices;
        platforms[iPlatform].getDevices(CL_DEVICE_TYPE_ALL, &devices);
        
        // Для каждого устройства выполняем тест
        for (unsigned int iDevice = 0; iDevice < devices.size(); iDevice++){
            try{
                PerformTestOnDevice(devices[iDevice]);
            }catch(std::exception error){
                std::cout << error.what() << std::endl;
            }
            CheckResults();
        }
    }
    
    // Удаляем буффер
    delete[](pInputVector1);
    delete[](pInputVector2);
    delete[](pOutputVector);
    delete[](pOutputVectorHost);
    
    return 0;
}
