# 矩阵乘法
当前实现方式参考[二维 Thread Tile 并行实现](https://github.com/PaddleJitLab/CUDATutorial/blob/develop/docs/11_gemm_optimize/01_tiled2d/README.md)  

目前的实现方式只支持**宽高**是64的倍数.  

## 性能统计
输入Mat为F32C1类型, 显卡类型为Quadro RTX 5000, 16G显存:

| 图像大小            | 性能(us)
|------------------- | -------------
|256*256             | 151
|512*512             | 314
|1024*1024           | 988
* 结论： 通过性能对比可知，相比使用[一维 Thread Tile 并行实现](./mat_mul_shared_tile_1d.md), 当图像是256和512时，比1维tile的慢，这是为什么？ 1024的比1维tile的快,符合预期。