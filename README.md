# SMR-R1: Reinforcing Ability to Extract Structured Information From Medical Reports in Vision Language Models

[![GitHub](https://img.shields.io/badge/GitHub-开源项目-blue)](https://github.com/your-username/medical-report-structured-extraction)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)



鉴于grpo算法在deepseek-r1模型上的大放异彩，我们这里也将grpo应用在了医疗报告结构化抽取领域，我们在qwen2.5vl-7b模型上进行了强化训练，结果显示在评测集上比qwen2.5vl-7b模型高15pp，比在相同模型相同数据上sft训练高15pp, 比qwen2.5vl-72b高7pp。本项目旨在提供一套解决方案，用于医疗报告的结构化抽取。我们开源了以下内容：
- 医疗报告结构化数据评测集（已脱敏）
- 医疗报告结构化抽取模型

通过本项目，研究人员和开发者可以快速上手医疗报告的结构化抽取任务，并利用提供的数据和模型进行进一步的研究和应用开发。


## 数据集

我们开源了一个医疗报告结构化数据评测集，数据地址为：。该数据集包含以下特点：
- **多样性和代表性**：数据集涵盖了多种类型的医疗报告，如病历、检查报告、诊断报告等；覆盖了各种拍摄情况，如透视，斜视，光线明暗变化等；同时增加了非医疗报告相关图片（如药盒等），判断模型是否有过滤非医疗图片的能力。
- **高质量标注**：所有数据均经过专业医疗人员标注和审核，确保数据的准确性和可靠性。
- **数据脱敏**：为了保护患者隐私，数据集中的所有信息均已进行脱敏处理，确保不包含任何可识别个人身份的信息。


## 模型

我们开源了一个医疗报告结构化抽取模型，基于qwen2.5-vl-7b,模型地址为：


### 评估模型
修改evaluate.py中的MODEL_PATH， DATA_ROOT， IMAGE_ROOT为自己本机的目录，即可
运行以下命令评估模型性能：
```bash
python evaluate.py
```

### 指标
| 模型       | 训练数据       |性能       |
|:-----------|:-----------:|:-----------:|
| Qwen2.5-VL-7B-Instruct   | \    |54.71%    |
| Qwen2.5-VL-72B-Instruct   | \   | 62.89%|
| Qwen2.5-VL-7B-Instruct-sft   |  6k  | 55.21%|
| Qwen2.5-VL-7B-Instruct-rl(SMR-R1)   |  6k  | 70.25%|


### 训练曲线
![Image](./assets/train.png)


### 抽取展示
```md
<think>
根据图片内容，提取出以下信息：
- 姓名：未提供
- 性别：女
- 年龄：1岁
- 检查时间：2021/6/18 11:28
- 科室：内科门诊
- 检查名称：新冠抗体测定（含检测试剂B）
- 检查部位：未提供
- 指标：包括新冠病毒抗体IgM和新冠病毒抗体IgG
- 结果解释：提供了不同抗体组合的解释
</think>

<answer>
{
    "姓名": "",
    "性别": "女",
    "年龄": "1岁",
    "检查时间": "2021/6/18 11:28",
    "科室": "内科门诊",
    "检查名称": "新冠抗体测定(含检测试剂B)",
    "检查部位": "",
    "指标": [
        {
            "项目名称": "新冠病毒抗体IgM",
            "结果": "阴性(-)",
            "单位": "",
            "参考范围": "阴性(-)",
            "异常标记": "",
            "检测方法": "",
            "结果状态": "正常",
            "临床提示": "",
            "是否为危急值": ""
        },
        {
            "项目名称": "新冠病毒抗体IgG",
            "结果": "阴性(-)",
            "单位": "",
            "参考范围": "阴性(-)",
            "异常标记": "",
            "检测方法": "",
            "结果状态": "正常",
            "临床提示": "",
            "是否为危急值": ""
        }
    ],
    "检查描述": "",
    "诊断": "",
    "处理建议": "",
    "样本采集时间": "2021/6/18 11:47",
    "其他": ""
}
</answer>
```

## todo list

- [X] 在基座模型上直接进行强化学习
- [ ] 在基座模型上sft后再进行强化学习
- [ ] 对抽取字段不进行限制，自动结构化医疗报告
- [ ] 因为医疗报告有些数值或者用词识别错，对结果的影响很大，所以不能轻易用编辑距离作为reward，后续会优化reward方式为大模型判定
- [ ] ...


## Acknowledgements

We sincerely thank [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1), [QwenVL](https://github.com/QwenLM/Qwen2.5-VL), [vllm](https://github.com/vllm-project/vllm) (our initial codebase).


## 📚 Contributors and Citation

Contributors: Lijun Liu, Zhang Tao, Zhang Tao, Mingan Lin, Zenan Zhou, Weipeng Chen. 

If you find this work useful, please cite it as follows:
```bib
@misc{lijun2025SMR-R1,
  author       = {Lijun Liu, Zhang Tao, Zhang Tao, Mingan Lin, Zenan Zhou, Weipeng Chen},
  title        = {SMR-R1: Reinforcing Ability to Extract Structured Information From Medical Reports in Vision Language Models},
  howpublished = {\url{https://github.com/yingyukexiansheng/SMR-R1}},
  note         = {Accessed: 2025-03-26},
  year         = {2025}
}
```