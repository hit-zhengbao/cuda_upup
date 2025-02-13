# reduce归约操作
当前实现参考[IDLE线程](https://github.com/PaddleJitLab/CUDATutorial/blob/develop/docs/09_optimize_reduce/03_idle_threads_free/README.md)

## bank冲突

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
|1024*1              | 21
|3*1024*1024         | 78974

对比[Bank冲突优化](./reduce_bank_conflict_free.md), 可知当尺寸较大(3*1024*1024)时，优化空闲线程的性能有明显提升。