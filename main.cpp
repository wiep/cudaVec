#include "cudavec.h"

#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>

#include <iostream>
#include <fstream>
#include <time.h>

using namespace cudaVector;

    int main() {
        // set vector size
        const int n = 20;

        // initialize host vectors
        thrust::host_vector<float> h_vec_a(n);
        thrust::host_vector<float> h_vec_b(n);
        thrust::host_vector<float> h_vec_c(n);

        // generate random data on the host
        //thrust::generate(h_vec_a.begin(), h_vec_a.end(), rand);
        //thrust::generate(h_vec_b.begin(), h_vec_b.end(), rand);
        //thrust::generate(h_vec_c.begin(), h_vec_c.end(), rand);

        // generate data on the host
        for (int i = 0; i < n; i++) {
            h_vec_a[i] = 0.1*i;
            h_vec_b[i] = 0.1*i;
            h_vec_c[i] = 0.1*i;
        }

        ////  transfer host vector to device
        cudaVec a(h_vec_a);
        cudaVec b(h_vec_b);
        cudaVec c(h_vec_c);
        
        // some expressions. first ten elements are displayed after each evaluation
        c = sin(a);
        std::cout << "c = sin(a)\n";
        for (int i = 0; i < 10 && i < n; ++i)
            std::cout << c[i] << " = sin(" << a[i] << ")\n";
        
        c = a + 2.0f*b;
        std::cout << "c = a + 2.0f * b\n";
        for (int i = 0; i < 10 && i < n; ++i)
            std::cout << c[i] << " = " << a[i] << " + 2.0f * " << b[i] <<"\n";

        c = pow(4,a);
        std::cout << "c = pow(4,a)\n";
        for (int i = 0; i < 10 && i < n; ++i)
            std::cout << c[i] << " = pow(4, " << a[i] << ")\n";

        b = 1.2 < c && c < 2.1;
        std::cout << "b = 1.2 < c && c < 2.1\n";
        for (int i = 0; i < 10 && i < n; ++i)
            std::cout << b[i] << " = 1.2 < " << c[i] << " && " << c[i] << " < 2.1\n";

        float b_sum = sum(b);
        std::cout << "b_sum = sum(b)\n";
        std::cout << b_sum << " = sum(" << b << ")\n";

        return 0;
}
