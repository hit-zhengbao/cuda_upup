# reduce归约操作
当前实现参考[初版实现](https://github.com/PaddleJitLab/CUDATutorial/blob/develop/docs/08_impl_reduce/README.md), 这个是初版, 后续会进行优化.

## 性能统计
输入Mat为S32C1类型, 显卡类型为Quadro RTX 5000, 16G显存:

| 图像大小            | 性能(us)
|------------------- | -------------
|1024*1              | 22
|10000*1             | 33
|3*1024*1024         | 91339