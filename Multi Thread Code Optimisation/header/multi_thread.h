#include <iostream>
#include <pthread.h>

// Define a struct to hold the parameters for each thread
struct ThreadArgs
{
    int input_row;
    int input_col;
    int *input;
    int kernel_row;
    int kernel_col;
    int *kernel;
    int output_row;
    int output_col;
    long long unsigned int *output;
    int thread_id;
    int num_threads;
};

// Function that each thread will execute
void *threadFunction(void *threadArgs)
{
    ThreadArgs *args = static_cast<ThreadArgs *>(threadArgs);

    int chunk_size = args->output_row / args->num_threads;
    int start_row = args->thread_id * chunk_size;
    int end_row = (args->thread_id == args->num_threads - 1) ? args->output_row : (args->thread_id + 1) * chunk_size;

    int output_index;
    int kernel_index;
    int input_i, input_j, input_base;

    int akernel_row = args->kernel_row;
    int akernel_col = args->kernel_col;
    int kernel_row_ = akernel_row << 1;
    int kernel_col_ = akernel_col << 1;
    int aoutput_col = args->output_col;
    int aoutput_row = args->output_row;
    int ainput_row = args->input_row;
    int ainput_col = args->input_col;
    int *ainput = args->input;
    long long unsigned int *aoutput = args->output;
    int *akernel = args->kernel;

    for (int i = start_row; i < end_row; i++)
    {
        output_index = i * aoutput_col;
        for (int j = 0; j < aoutput_col; j += 8)
        {

            __m256i high_val = _mm256_setzero_si256();
            __m256i low_val = _mm256_setzero_si256();
            kernel_index = 0;
            for (int kernel_i = 0; kernel_i < kernel_row_; kernel_i += 2)
            {
                input_i = (i + kernel_i) % ainput_row;
                input_base = input_i * ainput_col;
                for (int kernel_j = 0; kernel_j < kernel_col_; kernel_j += 2)
                {
                    input_j = (j + kernel_j) % ainput_col;
                    if (input_j + 7 < ainput_col)
                    {

                        __m256i input_data = _mm256_loadu_si256((__m256i *)&ainput[input_base + input_j]);
                        __m256i kernel_v = _mm256_set1_epi32(akernel[kernel_index++]);
                        __m256i total = _mm256_mullo_epi32(input_data, kernel_v);
                        __m256i total_low = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(total));
                        __m256i total_high = _mm256_cvtepi32_epi64(_mm256_extractf128_si256(total, 1));
                        high_val = _mm256_add_epi64(total_high, high_val);
                        low_val = _mm256_add_epi64(total_low, low_val);
                    }
                    else
                    {
                        int input_j_7 = input_j + 7;
                        int input_j_6 = input_j + 6;
                        int input_j_5 = input_j + 5;
                        int input_j_4 = input_j + 4;
                        int input_j_3 = input_j + 3;
                        int input_j_2 = input_j + 2;
                        int input_j_1 = input_j + 1;
                        __m256i input_data = _mm256_set_epi32(
                            ainput[input_base + (input_j_7 >= ainput_col ? (input_j_7 - ainput_col) : input_j_7)],
                            ainput[input_base + (input_j_6 >= ainput_col ? (input_j_6 - ainput_col) : input_j_6)],
                            ainput[input_base + (input_j_5 >= ainput_col ? (input_j_5 - ainput_col) : input_j_5)],
                            ainput[input_base + (input_j_4 >= ainput_col ? (input_j_4 - ainput_col) : input_j_4)],
                            ainput[input_base + (input_j_3 >= ainput_col ? (input_j_3 - ainput_col) : input_j_3)],
                            ainput[input_base + (input_j_2 >= ainput_col ? (input_j_2 - ainput_col) : input_j_2)],
                            ainput[input_base + (input_j_1 >= ainput_col ? (input_j_1 - ainput_col) : input_j_1)],
                            ainput[(input_base + input_j)]);
                        __m256i kernel_v = _mm256_set1_epi32(akernel[kernel_index++]);
                        __m256i total = _mm256_mullo_epi32(input_data, kernel_v);
                        __m256i total_low = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(total));
                        __m256i total_high = _mm256_cvtepi32_epi64(_mm256_extractf128_si256(total, 1));
                        high_val = _mm256_add_epi64(total_high, high_val);
                        low_val = _mm256_add_epi64(total_low, low_val);
                    }
                }
            }
            if (j + 7 < aoutput_col)
            {
                _mm256_storeu_si256((__m256i *)&aoutput[output_index], low_val);
                _mm256_storeu_si256((__m256i *)&aoutput[output_index + 4], high_val);
            }

            else if (j + 3 < aoutput_col)
            {
                _mm256_storeu_si256((__m256i *)&aoutput[output_index], low_val);
                if (j + 4 < aoutput_col)
                    aoutput[output_index + 4] = high_val[0];
                if (j + 5 < aoutput_col)
                    aoutput[output_index + 5] = high_val[1];
                if (j + 6 < aoutput_col)
                    aoutput[output_index + 6] = high_val[2];
            }
            else
            {
                aoutput[output_index] = low_val[0];
                if (j + 1 < aoutput_col)
                    aoutput[output_index + 1] = low_val[1];
                if (j + 2 < aoutput_col)
                    aoutput[output_index + 2] = low_val[2];
            }
            output_index += 8;
        }
    }
    pthread_exit(nullptr);
}

void parallelThread(int input_row,
                    int input_col,
                    int *input,
                    int kernel_row,
                    int kernel_col,
                    int *kernel,
                    int output_row,
                    int output_col,
                    long long unsigned int *output,
                    int num_threads)
{
    pthread_t threads[num_threads];
    ThreadArgs threadArgs[num_threads];

    // Create threads
    for (int i = 0; i < num_threads; ++i)
    {
        threadArgs[i] = {input_row, input_col, input, kernel_row, kernel_col, kernel, output_row, output_col, output, i, num_threads};
        pthread_create(&threads[i], nullptr, threadFunction, &threadArgs[i]);
    }

    // Join threads
    for (int i = 0; i < num_threads; ++i)
    {
        pthread_join(threads[i], nullptr);
    }
}

void multiThread(
    int input_row,
    int input_col,
    int *input,
    int kernel_row,
    int kernel_col,
    int *kernel,
    int output_row,
    int output_col,
    long long unsigned int *output)
{   
    //Specify the number of threads to be launched
    const int num_threads = 14;
    std::cout << "Number of threads : " << num_threads << endl;

    // Call the parallelized function
    parallelThread(input_row, input_col, input, kernel_row, kernel_col, kernel, output_row, output_col, output, num_threads);
}