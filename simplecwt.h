#ifndef SIMPLECWT_H
#define SIMPLECWT_H
#include <cmath>
#include <cassert>
#include <vector>
#include <complex>
#include <fftw3.h>
using std::vector;

class MorletFrequencyDomain{
public: // TODO : check this class, not so sure about the maths.
    MorletFrequencyDomain();
    MorletFrequencyDomain(int sr, float frequency, int n, int size);
    inline float operator[](int i) const ;
    float value(int i) const;

    void multiply(const fftwf_complex * v, fftwf_complex * out) const ;

private:
    int frequency_idx;
    double sigma2;
    int size;
    vector<float> morlet_fd;
    int first_idx,last_idx;
};


class SimpleCWT{
public:
    SimpleCWT(int sr, int size,int n,  const vector<float> & frequencies);
    void init(int sr, int size,int n,  const vector<float> & frequencies);

    SimpleCWT(int sr, int size, int n, float begin_frequency, float end_frequency, int nb_frequencies);
    void init(int sr, int size, int n, float begin_frequency, float end_frequency, int nb_frequencies);

    ~SimpleCWT();

    struct result{
        result(){}
        void init(vector<float> & frequency, int size){
            this->res.resize(frequency.size());
            this->frequency = frequency;
            for(auto & i : res){
                i.resize(size);
            }
        }
        vector<vector<float>> res;
        vector<float> frequency;
    };
    void run(const fftwf_complex * input, int size_input, struct result& output);
    void run(float* begin, int size_input, struct result & output);
    void convolve(const MorletFrequencyDomain & morlet, const fftwf_complex * input, vector<float> & output);

private:
    vector<float> frequency;
    vector<MorletFrequencyDomain> morlets;
    int size;
    fftwf_plan forward,backward;

};

#endif // SIMPLECWT_H
