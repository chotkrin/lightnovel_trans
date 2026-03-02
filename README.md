## 使用方式

### ⚠️注意
- **data process**是训练数据集构建相关的，预测用不着
- 运行前修改脚本内的路径！！！

### 准备数据
- vol是卷编号
- 准备好要翻译的内容，放在data下，叫`j{vol}.epub`
- **如果有之前的中英对照翻译，将中文卷改为`c{vol}.epub`格式也放在data下**
- utils下的`dtw_transltion_map`和`generate_glossary`结合，生成已经翻译的名词对照关系，避免人名地名不一致

### 执行翻译
- 第一步：`vllm serve` 本地8000端口
- 第二步：运行`predict/chunk_predict.py`进行待翻译epub语义块切分
- 第三步：运行`predict/predict.py`进行最终翻译
- 第四步：运行`predict/fix.py`检查并修复漏翻问题
