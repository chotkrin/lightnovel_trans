import asyncio
import json
import re
import os
import glob
import networkx as nx
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

MODEL_NAME = "Qwen/Qwen3-30B-A3B" 
DATA_DIR = "/home/zelin/code/lightnovel_trans/data/"
OUTPUT_DIR = "/home/zelin/code/lightnovel_trans/data/closures/" 

WINDOW_SIZE_BLOCKS = 30 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

DEBUG_MODE = False
DEBUG_BOOK_KEYWORD = "dtw_1"
DEBUG_CHAPTER_IDX = 1

def build_atomic_blocks(sentences):
    G = nx.Graph()
    G.add_nodes_from(range(len(sentences)))

    for i in range(1, len(sentences)):
        prev = sentences[i-1].strip()
        curr = sentences[i].strip()

        if re.search(r'[？\?][）】」』]?$', prev):
            G.add_edge(i-1, i)
            continue
        if re.search(r'^[（【「『]?(不过|但是|可是|然而|因此|所以|于是|接着|然后|而且|并且|那么|不过|另外)', curr):
            G.add_edge(i-1, i)
            continue
        if re.search(r'(这|那|这件事|这种|那个|这些|那些|为此|与此同时|他|她)', curr[:15]):
            G.add_edge(i-1, i)
            continue
        if len(curr) <= 6:
            G.add_edge(i-1, i)
            continue

    atomic_blocks = []
    for component in nx.connected_components(G):
        atomic_blocks.append(sorted(list(component)))
    atomic_blocks.sort(key=lambda x: x[0])
    return atomic_blocks

async def merge_blocks_with_llm(blocks_chunk, block_start_offset, sentences):
    text_block = ""
    for i, block_sent_ids in enumerate(blocks_chunk):
        combined_text = "".join([sentences[sid] for sid in block_sent_ids])
        text_block += f'<block id="{block_start_offset + i}">{combined_text}</block>\n'

    system_prompt = (
        "你是一个顶级的自然语言处理专家，专精于小说的【宏观语篇分割】。\n"
        "请阅读 <text_to_segment> 标签内由 <block> 标签包裹的文本块。这些块本身已经是不可分割的最小单元。\n"
        "你的任务是：根据剧情场景、话题和叙事视角的连贯性，将这些相邻的 <block> 合并成更大的「宏观语义闭包」。\n\n"
        "<closure_definition>\n"
        "1. 同一个闭包内的 block 必须在描述同一个连贯的场景、同一场完整的对话或同一段连续的动作。\n"
        "2. 当场景发生空间转移、时间流逝，或者视角从一个角色切换到另一个角色时，必须切断，开启新的闭包。\n"
        "</closure_definition>\n\n"
        "<output_format>\n"
        "必须且只能输出一个合法的 JSON 对象。每个闭包必须连续且不能有遗漏！\n"
        "【极其重要】：严格按照以下键值对顺序生成 JSON！先输出闭包类型(type)和极简理由(reason)，最后输出起始和结束的 block ID。\n\n"
        "字段限制：\n"
        "- type: 必须是 ['dialogue', 'action', 'description', 'mixed'] 中的一个。\n"
        "- reason: 最多 10 个字概括核心剧情。\n"
        "{\n"
        "  \"closures\": [\n"
        "    {\"type\": \"mixed\", \"reason\": \"主角从梦中醒来并思考\", \"start_block_id\": 0, \"end_block_id\": 3},\n"
        "    {\"type\": \"dialogue\", \"reason\": \"与村长交谈\", \"start_block_id\": 4, \"end_block_id\": 8}\n"
        "  ]\n"
        "}\n"
        "</output_format>"
    )

    user_content = f"<text_to_segment>\n{text_block}</text_to_segment>"

    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.1, 
            max_tokens=2048,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        
        content = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())["closures"]
        return None
    except Exception as e:
        print(f"请求失败: {e}")
        return None

async def process_single_chapter(unique_id, pairs):
    zh_sentences = [pair['zh'] for pair in pairs]
    total_len = len(zh_sentences)
    print(f"\n开始处理章节 [{unique_id}]，共 {total_len} 句话。")
    
    # 检查是否已经处理过
    output_file = os.path.join(OUTPUT_DIR, f"{unique_id}_closures.json")
    if os.path.exists(output_file):
        print(f"发现已有缓存文件 {output_file}，跳过此章。")
        return []

    # 第一步：规则预绑定
    atomic_blocks = build_atomic_blocks(zh_sentences)
    total_blocks = len(atomic_blocks)
    print(f"正则图论预绑定完成：压缩为 {total_blocks} 个原子块。")

    # 第二步：LLM 合并
    master_closures = []
    current_block_idx = 0
    
    while current_block_idx < total_blocks:
        end_idx = min(current_block_idx + WINDOW_SIZE_BLOCKS, total_blocks)
        print(f"扫描块区间: [{current_block_idx} -> {end_idx}]")
        blocks_chunk = atomic_blocks[current_block_idx : end_idx]
        
        closures = await merge_blocks_with_llm(blocks_chunk, current_block_idx, zh_sentences)
        
        if not closures:
            print(f"区间 [{current_block_idx}] 解析失败，步进...")
            current_block_idx += 5 
            continue

        if current_block_idx + WINDOW_SIZE_BLOCKS >= total_blocks:
            master_closures.extend(closures)
            break
            
        if len(closures) == 1:
            master_closures.extend(closures)
            current_block_idx = closures[0]["end_block_id"] + 1
        else:
            valid_closures = closures[:-1]
            master_closures.extend(valid_closures)
            last_safe_block_id = valid_closures[-1]["end_block_id"]
            current_block_idx = last_safe_block_id + 1

    final_sentence_closures = []
    for cls in master_closures:
        start_blk = max(0, min(cls["start_block_id"], total_blocks - 1))
        end_blk = max(0, min(cls["end_block_id"], total_blocks - 1))
        if start_blk > end_blk:
            start_blk, end_blk = end_blk, start_blk
            
        real_start = atomic_blocks[start_blk][0]
        real_end = atomic_blocks[end_blk][-1]
        
        final_sentence_closures.append({
            "type": cls.get("type", "mixed"),
            "reason": cls.get("reason", ""),
            "start_id": real_start,
            "end_id": real_end
        })

    print(f"章节 [{unique_id}] 合并完成，生成了 {len(final_sentence_closures)} 个宏观闭包。")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_sentence_closures, f, ensure_ascii=False, indent=2)
    print(f"保存至: {output_file}")
    
    return final_sentence_closures

async def main():
    print("正在扫描全局 DTW 对齐数据...")
    aligned_files = sorted(glob.glob(os.path.join(DATA_DIR, "aligned_book_dtw_*.jsonl")))
    
    all_chapters = []
    
    for file_path in aligned_files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                chap_idx = data.get("chapter_idx", len(all_chapters))
                pairs = data.get("pairs", [])
                
                if pairs:
                    unique_id = f"{base_name}_chap_{chap_idx}"
                    all_chapters.append({
                        "unique_id": unique_id,
                        "base_name": base_name,
                        "chapter_idx": chap_idx,
                        "pairs": pairs
                    })

    if not all_chapters:
        print("没有找到任何章节数据，请检查路径！")
        return

    if DEBUG_MODE:
        print(f"\n[DEBUG 模式] 寻找包含关键字 '{DEBUG_BOOK_KEYWORD}' 且 chapter_idx 为 {DEBUG_CHAPTER_IDX} 的章节...")
        
        # 联合查找：文件名匹配 + 章节号匹配
        target_chapter = next(
            (c for c in all_chapters 
             if DEBUG_BOOK_KEYWORD in c["base_name"] and c["chapter_idx"] == DEBUG_CHAPTER_IDX), 
            None
        )
        
        if target_chapter:
            closures = await process_single_chapter(target_chapter["unique_id"], target_chapter["pairs"])
            
            if closures:
                print("\n抽查前 3 个宏观闭包的中日映射效果：")
                pairs = target_chapter["pairs"]
                for i in range(min(3, len(closures))):
                    cluster = closures[i]
                    start, end = cluster["start_id"], cluster["end_id"]
                    print(f"\n闭包 [{start} -> {end}] | 类型: {cluster['type']} | 理由: {cluster['reason']}")
                    print("-" * 40)
                    for idx in range(start, min(end + 1, len(pairs))):
                        print(f"日: {pairs[idx]['ja']}")
                        print(f"中: {pairs[idx]['zh']}")
                    print("-" * 40)
        else:
            print(f"找不到匹配的数据！请检查 DEBUG_BOOK_KEYWORD 和 DEBUG_CHAPTER_IDX。")
            
    else:
        print(f"\n[全量模式] 准备连续处理 {len(all_chapters)} 章数据！")
        for chapter_data in all_chapters:
            await process_single_chapter(chapter_data["unique_id"], chapter_data["pairs"])
            
        print("\n全书宏观语篇分割全部完成！")

if __name__ == "__main__":
    asyncio.run(main())