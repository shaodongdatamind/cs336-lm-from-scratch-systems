| model_name   |   d_model |   d_ff |   num_layers |   num_heads | forward_only   |   warmup_steps |   avg_ms_per_step |   std_ms_per_step |   tokens_per_step |   throughput_tokens_per_s |
|:-------------|----------:|-------:|-------------:|------------:|:---------------|---------------:|------------------:|------------------:|------------------:|--------------------------:|
| small        |       768 |   3072 |           12 |          12 | True           |              5 |           20.8058 |            0.9928 |               512 |                  24608.5  |
| medium       |      1024 |   4096 |           24 |          16 | True           |              5 |           54.8309 |            0.9197 |               512 |                   9337.8  |
| large        |      1280 |   5120 |           36 |          20 | True           |              5 |          115.118  |            1.115  |               512 |                   4447.62 |
| xl           |      1600 |   6400 |           48 |          25 | True           |              5 |         3588.65   |           71.6849 |               512 |                    142.67 |
| 2.7B         |      2560 |  10240 |           32 |          32 | True           |              5 |        14657.8    |          101.74   |               512 |                     34.93 |
| small        |       768 |   3072 |           12 |          12 | True           |              0 |           25.6031 |            6.1385 |               512 |                  19997.6  |
| medium       |      1024 |   4096 |           24 |          16 | True           |              0 |           60.0241 |            6.4227 |               512 |                   8529.91 |
| large        |      1280 |   5120 |           36 |          20 | True           |              0 |          145.531  |           75.7248 |               512 |                   3518.14 |
| xl           |      1600 |   6400 |           48 |          25 | True           |              0 |         3683.24   |          180.325  |               512 |                    139.01 |
| 2.7B         |      2560 |  10240 |           32 |          32 | True           |              0 |        15012.7    |          657.218  |               512 |                     34.1  |
| small        |       768 |   3072 |           12 |          12 | False          |              5 |           70.493  |            8.5629 |               512 |                   7263.13 |
| medium       |      1024 |   4096 |           24 |          16 | False          |              5 |          182.445  |            5.9119 |               512 |                   2806.32 |
| large        |      1280 |   5120 |           36 |          20 | False          |              5 |         2483.24   |           49.1963 |               512 |                    206.18 |
| xl           |      1600 |   6400 |           48 |          25 | False          |              5 |        45246.2    |         5612.65   |               512 |                     11.32 |
| 2.7B         |      2560 |  10240 |           32 |          32 | False          |              5 |        60128.2    |         2049.39   |               512 |                      8.52 |
| small        |       768 |   3072 |           12 |          12 | False          |              0 |           73.479  |            7.5395 |               512 |                   6967.98 |
| medium       |      1024 |   4096 |           24 |          16 | False          |              0 |          197.293  |           13.6348 |               512 |                   2595.13 |
| large        |      1280 |   5120 |           36 |          20 | False          |              0 |         2617.47   |          126.06   |               512 |                    195.61 |
| xl           |      1600 |   6400 |           48 |          25 | False          |              0 |        38739.2    |         1477.29   |               512 |                     13.22 |
| 2.7B         |      2560 |  10240 |           32 |          32 | False          |              0 |        56981.9    |         1381.42   |               512 |                      8.99 |


### Analysis
With 5 warm-up steps, forward passes range from about 21 ms (small) up to roughly 14.7 s (2.7B), while backward passes are slower, from about 70 ms (small) up to roughly 60 s (2.7B). The standard deviation after warm-up is small relative to the mean (generally a few percent or less), so the measurements look stable. Removing warm-up makes runs slower and much more variable (e.g., small forward 20.8 → 25.6 ms and std 1.0 → 6.1 ms), because the first iterations pay one-time costs like CUDA context setup, kernel autotuning, allocator/cache warm-up, and clocks ramping. 