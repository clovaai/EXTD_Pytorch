## EXTD: Extremely Tiny Face Detector via Iterative Filter Reuse ##
A PyTorch Implementation of Extremely Tiny Face Detector via Iterative Filter Reuse

YoungJoon Yoo, Dongyoon Han, Sangdoo Yun

https://arxiv.org/abs/1906.06579

![extd_teaser](https://user-images.githubusercontent.com/12525981/62098369-f7aaa280-b2c4-11e9-80dc-7c21fbeda652.png)
![table](https://user-images.githubusercontent.com/12525981/62098372-faa59300-b2c4-11e9-8ed7-6ef302eace46.png)


### Requirement
* pytorch 1.0 (checked at 1.0) 
* opencv 
* numpy 
* easydict
* Python3

### Prepare data 
WIDER face dataset is used. see the S3FD.pytorch git for more detail.


### Train
You can use 
``` 
python train.py 
``` 

Refer the train.py files to check the arguement.
Our setting was

```
"--batch_size 16 --lr 0.001" 
```

### Evaluation on WIDER Dataset
You should complie the bounding box function. Type
```
python3 bbox_setup.py build_ext --inplace
```

Then run 
```
python3 wider_test.py
```

### Demo 
you can test your image from
```
python3 demo.py
```

### References
* [EXTD: Extremely Tiny Face Detector via Iterative Filter Reuse](https://arxiv.org/abs/1906.06579)
* [S3FD.pytorch](https://github.com/yxlijun/S3FD.pytorch)

### Citation
```
@article{yoo2019extd,
  title={EXTD: Extremely Tiny Face Detector via Iterative Filter Reuse},
  author={Yoo, YoungJoon and Han, Dongyoon and Yun, Sangdoo},
  journal={arXiv preprint arXiv:1906.06579},
  year={2019}
}
```

### License
```
Copyright (c) 2019-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE
```
