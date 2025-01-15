# 矩阵乘法
当前实现方式 参考高通的矩阵乘OpenCL实现方式 未使用共享内存，但是使用了二维的分块操作

## 性能统计
### 第一版性能
![第一版性能](./../../../doc/png/mat_mul_block_and_no_local_memmat_mul_block_and_no_local_mem.png)