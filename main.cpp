#include <iostream>
#include <vector>
#include <chrono>
#include "simplecwt.h"
#include <fcwt.h>



int f() {


    int sr = 44000;
    int size_cwt = 1024*64*2;
    int size_input = 100000;
    int nb_frequencies = 256;
    std::cout << "Sample rate: " << sr << std::endl;
    std::cout << "Size of CWT: " << size_cwt << std::endl;
    std::cout << "Size of input: " << size_input << std::endl;
    std::cout << "Number of frequencies: " << nb_frequencies << std::endl;

    // Mesurer le temps d'instanciation
    auto startInstantiation = std::chrono::high_resolution_clock::now();
    SimpleCWT cwt(44000, size_cwt, 20, 200, sr/2.1, nb_frequencies);
    auto endInstantiation = std::chrono::high_resolution_clock::now();
    auto instantiationTime = std::chrono::duration_cast<std::chrono::milliseconds>(endInstantiation - startInstantiation).count();

    std::cout << "Temps d'instanciation : " << instantiationTime << " millisecondes" << std::endl;

    // Mesurer le temps d'exécution
    auto startRun = std::chrono::high_resolution_clock::now();

    std::vector<float> s(size_input);
    s[42] = 1;

    SimpleCWT::result output;
    cwt.run(s.data(), s.size(), output);

    auto endRun = std::chrono::high_resolution_clock::now();
    auto runTime = std::chrono::duration_cast<std::chrono::milliseconds>(endRun - startRun).count();

    std::cout << "Temps d'exécution : " << runTime << " millisecondes" << std::endl;



    vector<std::complex<float>> out(nb_frequencies*s.size());

    auto fcwt_instance_start = std::chrono::high_resolution_clock::now();
    Morlet morlet(20);
    FCWT fcwt(&morlet,4,true,false);
    Scales scs(&morlet, FCWT_LOGSCALES, sr, 200 , sr/2.1, nb_frequencies);
    fcwt.create_FFT_optimization_plan(size_cwt,FFTW_ESTIMATE);
    auto fcwt_instance_end = std::chrono::high_resolution_clock::now();
    auto fcwt_instance_time = std::chrono::duration_cast<std::chrono::milliseconds>(fcwt_instance_end - fcwt_instance_start).count();
    std::cout << "Temps instanciation fcwt : " << fcwt_instance_time << " millisecondes" << std::endl;



    auto fcwt_run_start = std::chrono::high_resolution_clock::now();
    fcwt.cwt(s.data(),s.size(),out.data(),&scs);
    auto fcwt_run_end = std::chrono::high_resolution_clock::now();
    auto fcwt_run_time = std::chrono::duration_cast<std::chrono::milliseconds>(fcwt_run_end - fcwt_run_start).count();
    std::cout << "Temps d'exécution fcwt : " << fcwt_run_time << " millisecondes" << std::endl;



    return 0;
}

int main(){
    f();
    return 0;
}

#if 0
#include <iostream>
#include "simplecwt.h"

using namespace std;

int main()
{


    SimpleCWT cwt(44000, 1024*16, 20, 200,20000, 64);

    vector<float> s(1024*16);

    s[42]=1;

    SimpleCWT::result output;
    cwt.run(s.data(),s.size(),output);



    return 0;
}
#endif
