import asyncio
import json
import re
import glob
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
MODEL_NAME = "Qwen/Qwen3-30B-A3B"

# 你的对齐文件路径，可以用通配符匹配那三个 jsonl
ALIGNED_FILES = glob.glob("/home/zelin/code/lightnovel_trans/data_process/aligned_book_dtw_*.jsonl")
GLOSSARY_FILE = "/home/zelin/code/lightnovel_trans/data/glossary.json"

# 每 30 对句子打包成一个区块交给大模型处理
PAIRS_PER_CHUNK = 30
CONCURRENCY = 30 

async def extract_terms_from_pairs(sem, chunk_idx, pairs_chunk):
    async with sem:
        # 把句子对拼接成易于模型阅读的格式
        text_block = ""
        for i, pair in enumerate(pairs_chunk):
            text_block += f"[{i+1}]\n日文: {pair['ja']}\n中文: {pair['zh']}\n\n"
            
        sys_prompt = (
            "你是一个精准的双语术语对齐专家。\n"
            "请阅读以下提供的【日文原句】和对应的【中文标准译文】。你的任务是提取出其中的专有名词（人名、特定地名、组织名、专属技能名等）。\n"
            "严格要求：\n"
            "1. 中文翻译必须【完全一字不差】地来自我提供的中文译文，绝不能自己猜测或意译！\n"
            "2. 如果日文有专有名词，但中文译文中被省略或意译得面目全非，请不要提取该词条。\n"
            "3. 仅输出一个合法的 JSON 对象，格式为 {\"日文原词\": \"中文译词\"}。不要输出任何其他解释性文字，不要包含普通的日常词汇。"
        )

        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": text_block}
                ],
                temperature=0,  # 温度调到极低，确保它像机器一样死板地抠字眼
                max_tokens=1024,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            
            content = response.choices[0].message.content
            
            # 清理可能存在的 markdown code block 标记
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {}
            
        except Exception as e:
            print(f"[Chunk {chunk_idx} Failed]: {e}")
            return {}

async def build_ground_truth_glossary():
    all_pairs = []
    
    print("正在加载 DTW 对齐数据...")
    for file_path in ALIGNED_FILES:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                # 将每一章里的 pairs 扁平化收集起来
                if "pairs" in data:
                    all_pairs.extend(data["pairs"])
                    
    print(f"共加载了 {len(all_pairs)} 对平行句。")
    
    # 将句子对分块
    chunks = [all_pairs[i:i + PAIRS_PER_CHUNK] for i in range(0, len(all_pairs), PAIRS_PER_CHUNK)]
    print(f"打包成 {len(chunks)} 个处理区块，准备丢给大模型提取...")

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [extract_terms_from_pairs(sem, idx, chunk) for idx, chunk in enumerate(chunks)]
    
    print("开始并发提取专属名词...")
    results = await tqdm_asyncio.gather(*tasks)

    master_glossary = {}
    for res_dict in results:
        if isinstance(res_dict, dict):
            # 过滤掉一些可能被大模型误提取出来的单字或超长句子
            for ja_term, zh_term in res_dict.items():
                if 1 < len(ja_term) < 15 and 1 < len(zh_term) < 15:
                    master_glossary[ja_term] = zh_term

    # 保存
    with open(GLOSSARY_FILE, "w", encoding="utf-8") as f:
        json.dump(master_glossary, f, ensure_ascii=False, indent=2)
    
    print(f"提取了 {len(master_glossary)} 个精准词条，已保存至 {GLOSSARY_FILE}")

if __name__ == "__main__":
    asyncio.run(build_ground_truth_glossary())