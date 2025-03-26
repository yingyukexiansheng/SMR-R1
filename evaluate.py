from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
from pprint import pprint
import random


# 需要评测的模型名称
MODEL_PATH="mrlijun/SMR-R1" 

cuda = 3
use_api = False
BSZ=8

# 评测数据的目录
DATA_ROOT = ""
TEST_DATASETS = ['structure_rl_eval']
log_path = os.path.join(DATA_ROOT, "log_2.txt")

# 评测数据的图片存放目录
IMAGE_ROOT = f"{DATA_ROOT}/images"

random.seed(42)

def api_response(examples):
    responses = []
    
    import requests
    import json
    openai_api_base = ""
    for messages in examples:
        input_dict = dict(
            model="",
            stream=True,
            max_tokens = 16000,
            messages=messages,
            temperature=0.0,
        )
        headers = {'Content-Type': 'application/json'}
        headers['Authorization'] = f'Bearer empty'
        response = requests.post(url = openai_api_base, json = input_dict,stream=True)
        # print(response)
        ans = ''
        role = ''
        for byte_line in response.iter_lines():
            byte_line = byte_line.rstrip(b'\n')
            if byte_line == b'data: [DONE]':continue
            if byte_line.startswith(b'data:'):
                data = json.loads(byte_line.decode().lstrip('data:'))
                role_ = data['choices'][0]['delta'].get('role','')
                if role_ != '':
                    role = role_
                ans += data['choices'][0]['delta']['content']
        # print(ans)
        responses.append(ans)
    return responses

if not use_api:
    #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=f"cuda:{cuda}",
    )

    # default processer
    processor = AutoProcessor.from_pretrained(MODEL_PATH)


def extract_answer(predict_str, ground_truth):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    content_answer_match = re.search(answer_tag_pattern, predict_str, re.DOTALL)
    if content_answer_match:
        try_content = re.search(r'```json(.*?)```', content_answer_match.group(1).strip(), re.DOTALL)
        if try_content:
            content_answer_match = try_content
        content_answer = content_answer_match.group(1).strip()
        try:
            # 尝试解析字符串为 JSON
            parsed_json = json.loads(content_answer)
            if isinstance(ground_truth, dict):
                ground_truth = ground_truth
            else:
                ground_truth = json.loads(ground_truth)
            if "其他" in parsed_json: parsed_json.pop("其他")
            if "其他" in ground_truth: ground_truth.pop("其他")
            return parsed_json, ground_truth
        except:
            pass
    return None, ground_truth


def unique_items(items):
    res = []
    for item in items:
        new_item = {}
        for key, v in item.items():
            new_item[key] = v.strip()
        if new_item not in res:
            res.append(new_item)
    return res


def compute_ocr_score(pred_dict, ground_dict):
    pred_count = 0
    pred_true = 0
    ground_count = 0
    
    for key, value in pred_dict.items():
        if key != "指标":
            value = value.strip()
        if value != "":
            if key != "指标":
                pred_count += 1
                if key in ground_dict and re.sub("\s+", "", value, flags=re.DOTALL) == re.sub("\s+", "", ground_dict[key], flags=re.DOTALL):
                    pred_true += 1
            else:
                value = unique_items(value)
                for item in value:
                    pred_count += 1
                    if "指标" in ground_dict:
                        v_max = 0
                        for t in ground_dict["指标"]:
                            v = compute_ocr_score(item, t)
                            if v_max < v:
                                v_max = v
                        pred_true += v_max
    
    for key, item in ground_dict.items():
        if item != "":
            if key != "指标":
                ground_count += 1
            else:
                ground_count += len(ground_dict["指标"])
    if pred_count == 0 and ground_count == 0:
        return 1
    elif pred_count == 0 or ground_count == 0:
        return 0
    return 0.5 * (pred_true / pred_count + pred_true / ground_count)


for ds in TEST_DATASETS:
    print(f"Processing {ds}...")
    ds_path = os.path.join(DATA_ROOT, f"{ds}.json")
    data = json.load(open(ds_path, "r"))
    SYSTEM_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>" 
    messages = []
    QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
    import base64
    for x in data:
        image_path = os.path.join(IMAGE_ROOT, x['images'][0])
        image_base64 = base64.b64encode(open(image_path,'rb').read()).decode("utf-8")
        if use_api:
            image_key = "image_url"
            url = {"url": f"data:image/jpeg;base64,{image_base64}"}
        else:
            image_key = "image"
            url = f"data:image/jpeg;base64,{image_base64}"
        message = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
            "role": "user",
            "content": [
                {
                    "type": f"{image_key}", 
                    f"{image_key}": url
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=x['problem'])
                }
            ]
        }]
        messages.append(message)

    all_outputs = []  # List to store all answers
    if use_api == True:
        all_outputs = api_response(messages)
    else:
        # Process data
        for i in tqdm(range(0, len(messages), BSZ)):
            batch_messages = messages[i:i + BSZ]
        
            # Preparation for inference
            text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
            
            image_inputs, video_inputs = process_vision_info(batch_messages)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                padding_side='left',
            )
            inputs = inputs.to(f"cuda:{cuda}")

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=1024, temperature=None, top_p=None, top_k=None, do_sample=False)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            batch_output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            all_outputs.extend(batch_output_text)
            # print(f"Processed batch {i//BSZ + 1}/{(len(messages) + BSZ - 1)//BSZ}")

    final_output = []
    total_score = 0

    for input_example, model_output in zip(data, all_outputs):
        original_output = model_output
        ground_truth = input_example['answer']
        model_answer, ground_truth = extract_answer(original_output, ground_truth)
        
        # Count correct answers
        score = 0
        if model_answer is not None:
            score = compute_ocr_score(model_answer, ground_truth)
            
        total_score += score
        
        # Create a result dictionary for this example
        result = {
            'question': input_example['problem'],
            'ground_truth': ground_truth,
            'model_output': original_output,
            'extracted_answer': model_answer,
            'score': score
        }
        final_output.append(result)

    # Calculate and print accuracy
    accuracy = total_score / len(data) * 100
    print(f"\nAccuracy of {ds}: {accuracy:.2f}%")
with open(log_path, "w", encoding='utf-8') as f:
    for result in final_output:
        f.write(f"-------------  score : {result['score']} -------------\n")
        f.write(f"Content: {result['model_output']}\n")
        f.write(f"Answer: {result['ground_truth']}\n")