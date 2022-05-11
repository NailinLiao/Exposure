## 数据验证模块

### A_GenerateValidationSet.py

- 功能：生成验证数据集记录文件用于与曝光检查日志做 指标评估
- 该脚本仅需要两个参数
    - 参数一：验证数据集路径 Validation_path = r'../DataSet'
        - 文件结构
            1. ../.../DataSet # 该文件为标注数据，二分类标注，noraml为曝光正常 overexposure 为过曝
                1. normal
                    1. xxx.png
                    2. xxx.png
                    3. ...
                2. overexposure
                    1. xxx.png
                    2. xxx.png
                    3. ...
    - 参数二：保存验证数据csv文件路径 save_path = r'./ValiaDataSet'

### B_evaluate.py

- 功能：对预测 日志 中的暴光检查结果进行评估
- 该脚本仅需要三个个参数
    - 参数一：pread_base_path = r'../log_file'
        - 预测时生成的记录文件根路径
    - 参数二：evaluate_base_path = r'../evaluate/ValiaDataSet'
        - 该文件由 A_GenerateValidationSet.py 生成的验证数据集记录文件
    - 参数三：save_path = r'../log_file' 
      - 该参数默认为None
      - 该参数为None时会自动在 USer 下创建路径 ~/DeepWay/overexposure_log/ 后并保存

