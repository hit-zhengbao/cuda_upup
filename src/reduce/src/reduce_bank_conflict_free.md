# reduce归约操作
当前实现参考[bank冲突优化](https://github.com/PaddleJitLab/CUDATutorial/blob/develop/docs/09_optimize_reduce/02_bank_conflict/README.md)

## bank冲突

由于前一版本[交错寻址方式](./reduce_interleaved_addressing.md)方式存在较大的bank冲突，即同一个warp中不同线程同时访问同一个bank的不同地址，因此需要对其进行优化。
### 交错寻址方式
统计了该版本的bank冲突情况，如下图所示：
| 图像大小            | bank冲突数量
|------------------- | -------------
|3*1024*1024         | 4,587,520
### 此版本优化bank冲突后
| 图像大小            | bank冲突数量
|------------------- | -------------
|3*1024*1024         | 0

使用的命令：
```shell
sudo /usr/local/cuda/bin/ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./build/install/test/CUDAUpExe reduce_test

# 示例结果-交错寻址方式
# l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                                                    4,587,520
```

## 性能统计
输入Mat为S32C1类型, 显卡类型为Quadro RTX 5000, 16G显存:

| 图像大小            | 性能(us)
|------------------- | -------------
|1024*1              | 22
|10000*1             | 30
|3*1024*1024         | 80766

对比[交错寻址方式](./reduce_interleaved_addressing.md)可知当尺寸较大(3*1024*1024)时，优化bank冲突方式的性能有明显提升。