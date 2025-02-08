# 矩阵乘法
当前实现方式参考[一维 Thread Tile 并行实现](https://github.com/PaddleJitLab/CUDATutorial/tree/develop/docs/07_optimize_matmul#%E4%B8%80%E7%BB%B4-thread-tile-%E5%B9%B6%E8%A1%8C%E4%BC%98%E5%8C%96)  

目前的实现方式只支持**宽高**是64的倍数.  

## 性能统计
输入Mat为F32C1类型, 显卡类型为Quadro RTX 5000, 16G显存:

| 图像大小            | 性能(us)
|------------------- | -------------
|256*256             | 75
|512*512             | 257
|1024*1024           | 1511
* 结论： 通过性能对比可知，相比单纯只使用[共享内存的实现方式](./mat_mul_shared.md), 当图像大于等于512*512时，这种每个线程计算多个元素，实现了数据的复用，即减少数据IO，同时也减少了线程块的数量，从而提高了性能。