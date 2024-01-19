#include <immintrin.h>
#include <algorithm>
#include "cstring"
#include "cstdlib"

void singleThread(
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
    // Declaration of variables used
    int output_index;
    int k_index;
    int input_i, input_j, input_base;

    int kernel_row_ = kernel_row << 1;
    int kernel_col_ = kernel_col << 1;
    int op_col = output_col - output_col % 8;

    for (int i = 0; i < output_row; i++)
    {
        output_index = i * output_col;
        for (int j = 0; j < output_col; j += 8)
        {
            __m256i high_val = _mm256_setzero_si256();
            __m256i low_val = _mm256_setzero_si256();
            k_index = 0;
            for (int kernel_i = 0; kernel_i < kernel_row_; kernel_i += 2)
            {
                input_i = (i + kernel_i) % input_row;
                input_base = input_i * input_col;
                for (int kernel_j = 0; kernel_j < kernel_col_; kernel_j += 2)
                {
                    input_j = (j + kernel_j) % input_col;
                    if (input_j + 7 < input_col)
                    {
                        __m256i input_data = _mm256_loadu_si256((__m256i *)&input[input_base + input_j]);
                        __m256i kernel_data = _mm256_set1_epi32(kernel[k_index++]);
                        __m256i total = _mm256_mullo_epi32(input_data, kernel_data);
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
                            input[input_base + (input_j_7 >= input_col ? (input_j_7 - input_col) : input_j_7)],
                            input[input_base + (input_j_6 >= input_col ? (input_j_6 - input_col) : input_j_6)],
                            input[input_base + (input_j_5 >= input_col ? (input_j_5 - input_col) : input_j_5)],
                            input[input_base + (input_j_4 >= input_col ? (input_j_4 - input_col) : input_j_4)],
                            input[input_base + (input_j_3 >= input_col ? (input_j_3 - input_col) : input_j_3)],
                            input[input_base + (input_j_2 >= input_col ? (input_j_2 - input_col) : input_j_2)],
                            input[input_base + (input_j_1 >= input_col ? (input_j_1 - input_col) : input_j_1)],
                            input[(input_base + input_j)]);
                        __m256i kernel_data = _mm256_set1_epi32(kernel[k_index++]);
                        __m256i total = _mm256_mullo_epi32(input_data, kernel_data);
                        __m256i total_low = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(total));
                        __m256i total_high = _mm256_cvtepi32_epi64(_mm256_extractf128_si256(total, 1));
                        high_val = _mm256_add_epi64(total_high, high_val);
                        low_val = _mm256_add_epi64(total_low, low_val);
                    }
                }
            }
            if (j + 7 < output_col)
            {
                _mm256_storeu_si256((__m256i *)&output[output_index], low_val);
                _mm256_storeu_si256((__m256i *)&output[output_index + 4], high_val);
            }
            else if (j + 3 < output_col)
            {
                _mm256_storeu_si256((__m256i *)&output[output_index], low_val);
                if (j + 4 < output_col)
                    output[output_index + 4] = high_val[0];
                if (j + 5 < output_col)
                    output[output_index + 5] = high_val[1];
                if (j + 6 < output_col)
                    output[output_index + 6] = high_val[2];
            }
            else
            {
                output[output_index] = low_val[0];
                if (j + 1 < output_col)
                    output[output_index + 1] = low_val[1];
                if (j + 2 < output_col)
                    output[output_index + 2] = low_val[2];
            }
            output_index += 8;
        }
    }
}
