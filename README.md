# cuda_upup
Enhance the skills of CUDA

## Supported Features
 * CMake build system to suport separately compiling source codes(.cu/.cpp) both on the host and the device, 
   and this is to say that using g++(or other standard c++ compilers) to compile .cpp files and using nvcc to compile .cu files.
 * Simple version of Google Test to test all test cases.
 * The Mat class makes it easy to do matrix operations on the host and the device.

## Modules
### matmuls
Using various methods to do matrix multiplication.

### reduce
Using various methods to do reduce operations.

### simple_gtest
The simple version of Google Test.

### utils
  * The implementation of Mat class.
  * Other utils.

## Build
```shell
bash scripts/build.sh
```
## Run
```shell
# 1. run all tests
./build/install/test/CUDAUpExe all

# 2. run a specific test case: mat_mul_test
./build/install/test/CUDAUpExe mat_mul_test
```

## Reference
[CUDATutorial](https://github.com/PaddleJitLab/CUDATutorial/tree/develop)
