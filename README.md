# fasttext_pythonAPI
与新版本的c++版fasttext兼容
#### fastText的python API 接口由于c++版本的升级而不能相互兼容
#### 主要是不能载入c++版本训练的模型
#### 通过浏览相关网站，发现很多人都遇到同样的问题，并且也有些人给出了修改方法，
#### 但都没有完整的相关程序和修改方法，通过个人分析整合各种解决方案，相比于python API fasttext-0.8.3版本作如下修改：
#### 1.fasttext.cc文件loadModel函数增加部分代码（源程序已标注）
#### 2.增加了qmatrix.cc/qmatrix.h和productquantizer.cc/productquantizer.h

###### **在此对提供程序修改补丁和思路的网友表示感谢!!!**
