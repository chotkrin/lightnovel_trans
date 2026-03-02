import json
import re
import os
import glob
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from huggingface_hub import login

HF_TOKEN = "" 
HF_REPO_ID = "" 

DATA_DIR = "/home/zelin/code/lightnovel_trans/data/"
CLOSURES_DIR = os.path.join(DATA_DIR, "closures")

def is_valid_chunk(ja_text, zh_text):
    if not ja_text or not zh_text: return False
    ja_len, zh_len = len(ja_text), len(zh_text)
    if ja_len < 1 or zh_len < 1 or ja_len > 4000 or zh_len > 4000: return False
    
    ratio = ja_len / zh_len
    if ratio < 0.3 or ratio > 4.0: return False
    return True

def build_passage_sft_dataset():
    all_messages = []
    
    print("正在扫描全局闭包文件...")
    closure_files = glob.glob(os.path.join(CLOSURES_DIR, "*_closures.json"))
    
    for closure_file in closure_files:
        base_name_match = re.search(r'(aligned_book_dtw_\d+)_chap_(\d+)', os.path.basename(closure_file))
        if not base_name_match: continue
            
        base_name = base_name_match.group(1)
        chap_idx = int(base_name_match.group(2))
        
        dtw_file = os.path.join(DATA_DIR, f"{base_name}.jsonl")
        if not os.path.exists(dtw_file): continue
            
        chapter_pairs = []
        with open(dtw_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                if data.get("chapter_idx") == chap_idx:
                    chapter_pairs = data.get("pairs", [])
                    break
        if not chapter_pairs: continue
            
        with open(closure_file, 'r', encoding='utf-8') as f:
            closures = json.load(f)
            
        for cls in closures:
            start_id = max(0, cls["start_id"])
            end_id = min(len(chapter_pairs) - 1, cls["end_id"])
            
            # 提取整段日文和中文
            ja_chunk = "".join([chapter_pairs[i]['ja'].strip() for i in range(start_id, end_id + 1)])
            zh_chunk = "".join([chapter_pairs[i]['zh'].strip() for i in range(start_id, end_id + 1)])
            
            if not is_valid_chunk(ja_chunk, zh_chunk):
                continue
                
            system_instruction = (
                "你是一位精通中日文学的资深轻小说翻译家。\n"
                "请将 <text_to_translate> 标签内的整段日文翻译为符合轻小说文风的简体中文。\n"
                "请确保人物语气连贯，代词指代清晰，直接输出翻译结果，不要任何多余解释。"
            )
            
            user_content = f"<text_to_translate>\n{ja_chunk}\n</text_to_translate>"
            
            all_messages.append({
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": zh_chunk}
                ]
            })

    print(f"成功构建了 {len(all_messages)} 条【整段级别】高质量数据。")
    return pd.DataFrame(all_messages)

df_messages = build_passage_sft_dataset()
if not df_messages.empty:
    print("\n正在切分并上传到 Hugging Face...")
    train_df, test_df = train_test_split(df_messages, test_size=0.05, random_state=42)
    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "test": Dataset.from_pandas(test_df, preserve_index=False)
    })
    login(token=HF_TOKEN)
    dataset_dict.push_to_hub(HF_REPO_ID)
    print("Passage-Level 数据集上传完成！")