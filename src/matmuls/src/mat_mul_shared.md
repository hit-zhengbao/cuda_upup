# 矩阵乘法
当前实现方式参考[共享内存的实现方式](https://github.com/PaddleJitLab/CUDATutorial/blob/develop/docs/07_optimize_matmul/README.md#%E5%85%B1%E4%BA%AB%E5%86%85%E5%AD%98%E7%9A%84%E4%BD%BF%E7%94%A8)  

目前的实现方式只支持**宽高**是BLOCK_SIZE的倍数.  

## 性能统计
输入Mat为F32C1类型, 显卡类型为Quadro RTX 5000, 16G显存:

| 图像大小            | 性能(us)
|------------------- | -------------
|32*32               | 9
|256*256             | 63
|512*512             | 345
|1024*1024           | 2880
* 结论： 通过性能对比可知，虽然该实现方式使用了共享内存，但性能还是弱于未使用共享内存的实现方式-参考[高通矩阵乘的实现方式](./mat_mul_block_and_no_local_mem.md), 由于高通的实现方式复用了数据