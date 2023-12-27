#include "simplecwt.h"

#include <omp.h>


using std::vector;

template<typename T>
T square(T i){
    return i*i;
};

template<typename T>
T * fftw_malloc(int size){
    return (T*) fftwf_malloc(sizeof(T) * size);
}
template<typename T>
void fftw_free(T * t){
    fftwf_free(t);
}


MorletFrequencyDomain::MorletFrequencyDomain():frequency_idx(-1),sigma2(nanf("not initialized")),size(-1){}
MorletFrequencyDomain::MorletFrequencyDomain(int sr, float frequency, int n, int size){
    double f_idx_1 =  sr/static_cast<double>(size);
    this->frequency_idx = frequency/f_idx_1;
    assert(this->frequency_idx < size / 2);
    this->sigma2 = static_cast<double>(n) / (double)this->frequency_idx;
    this->sigma2 = this->sigma2 * this->sigma2;
    this->size = size;
    morlet_fd.resize(size);
    first_idx = -1;
    for(int i = 0; i < size && last_idx != i - 1; i++){
        morlet_fd[i] = value(i);
        if(value(i)>1e-6){
            if(first_idx == -1){
                first_idx=i;
            }
            last_idx = i;
        }
    }
}

void MorletFrequencyDomain::multiply(const fftwf_complex * v, fftwf_complex * out) const {
    for(int i = 0; i < size; i++){
        if(i >= first_idx && i <= last_idx){
            const auto s = morlet_fd[i];
            out[i][0] = v[i][0] * s;
            out[i][1] = v[i][1] * s;
        } else {
            out[i][0] = out[i][1] = 0.0;
        }
    }
}

float MorletFrequencyDomain::operator[](int i) const {
    assert(i < this->size);
    assert(i >= 0);
    return morlet_fd[i];
}



float MorletFrequencyDomain::value(int i) const{
    return exp(-0.5 * square(static_cast<float>(this->frequency_idx - i)) * sigma2) / (2*M_PI);
}



SimpleCWT::SimpleCWT(int sr, int size,int n,  const vector<float> & frequencies){
    init(sr,size,n,frequencies);
}

void SimpleCWT::init(int sr, int size,int n,  const vector<float> & frequencies) {
    this->size = size;
    auto in = fftw_malloc<fftwf_complex>(size);
    auto out = fftw_malloc<fftwf_complex>(size);
    fftw_init_threads();
    fftw_plan_with_nthreads(4);
    forward = fftwf_plan_dft_1d(size,in,out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan_with_nthreads(1);
    backward = fftwf_plan_dft_1d(size,in,out,FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_free(in);
    fftwf_free(out);

    this->morlets.resize(frequencies.size());
    this->frequency = frequencies;
#pragma omp parallel for num_threads(4)
    for(int i = 0; i < frequencies.size(); i++){
        MorletFrequencyDomain m{sr,frequencies[i],n,size};
        this->morlets[i] = m;
    }
}



SimpleCWT::SimpleCWT(int sr, int size, int n, float begin_frequency, float end_frequency, int nb_frequencies){
    init(sr,size,n,begin_frequency,end_frequency, nb_frequencies);
}
void SimpleCWT::init(int sr, int size, int n, float begin_frequency, float end_frequency, int nb_frequencies)    {
    auto lb = log(begin_frequency);
    auto le = log(end_frequency);
    vector<float> f;
    for(int i = 0; i< nb_frequencies; i++){
        f.push_back(exp(lb + i*(le-lb)/nb_frequencies));
    }
    init(sr,size,n,f);
}

SimpleCWT::~SimpleCWT(){
    fftwf_destroy_plan(forward);
    fftwf_destroy_plan(backward);
}


void SimpleCWT::run(const fftwf_complex * input, int size_input, struct result& output){
    output.init(frequency,size_input);
#pragma omp parallel for num_threads(4)
    for(int i = 0; i < morlets.size(); i++){
        convolve(morlets[i], input, output.res[i]);
    }
}

void SimpleCWT::run(float* begin, int size_input, struct result & output){
    auto input = fftw_malloc<fftwf_complex>(size);
    assert(size_input <= size);
    for(int i = 0; i < size_input; i++){
        input[i][0] = begin[i];
        input[i][1] = 0.0;
    }
    for(int i = size_input; i < size; i++){
        input[i][0] = 0.0;
        input[i][1] = 0.0;
    }
    auto out = fftw_malloc<fftwf_complex>(size);
    fftwf_execute_dft(forward,input,out);
    fftw_free(input);
    run(out, size_input, output);
    fftw_free(out);
}

void SimpleCWT::convolve(const MorletFrequencyDomain & morlet, const fftwf_complex * input, vector<float> & output){
    auto in = fftw_malloc<fftwf_complex>(size);
    morlet.multiply(input, in);
    for(int i = 0; i < size; i++){
        auto s = morlet[i];
        in[i][0] = input[i][0] * s;
        in[i][1] = input[i][1] * s;
    }
    auto out = fftw_malloc<fftwf_complex>(size);
    fftwf_execute_dft(backward,in,out);
    fftw_free(in);
    for(int i = 0; i < output.size(); i++){
        output[i] = std::abs(*reinterpret_cast<std::complex<float>*>(&out[i]));
    }
    fftw_free(out);
}

