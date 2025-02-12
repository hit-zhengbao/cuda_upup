# reduce归约操作
当前实现参考[初版实现](https://github.com/PaddleJitLab/CUDATutorial/blob/develop/docs/09_optimize_reduce/01_interleaved_addressing/README.md), 这个是初版, 后续会进行优化.

## 性能统计
输入Mat为S32C1类型, 显卡类型为Quadro RTX 5000, 16G显存:

| 图像大小            | 性能(us)
|------------------- | -------------
|1024*1              | 22
|10000*1             | 45
|3*1024*1024         | 82493

对比[原始实现](./reduce_native.md)可知当尺寸较小时，性能差距不大，但是当尺寸较大(3*1024*1024)时，交错寻址体现出优势。