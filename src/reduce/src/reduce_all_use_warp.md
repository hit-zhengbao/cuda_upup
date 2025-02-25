# reduce归约操作
该方法使用warp的shuffle 来实现，未参考前几个版本借鉴的github作者的实现方式, 并且bank冲突为0

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
|1024*1              | 22
|3*1024*1024         | 80350

本版面对标[展开最后一个warp](./reduce_unroll_last_warp.md), 性能大致相当，也没有bank冲突，所以适合直接用这个版本。