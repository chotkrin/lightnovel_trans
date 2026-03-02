import asyncio
import json
import re
import os
import networkx as nx
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
DATA_DIR = "/home/zelin/code/lightnovel_trans/data/"
BOOKS_TO_PROCESS = [4, 5, 6, 7]

WINDOW_SIZE_BLOCKS = 30 # LLM 每次观察的原子块数量
DEBUG_MODE = False

def build_atomic_blocks_jp(sentences):
    G = nx.Graph()
    G.add_nodes_from(range(len(sentences)))

    for i in range(1, len(sentences)):
        prev = sentences[i-1].strip()
        curr = sentences[i].strip()

        # 规则 A：问答与思考连贯 (日文问号或句尾省略号)
        if re.search(r'[？\?…][）】」』]?$', prev):
            G.add_edge(i-1, i)
            continue
            
        # 规则 B：强逻辑连词开头 (适配日文常见接续词)
        # 包含：但是、所以、然后、而且、那么 等
        if re.search(r'^[（【「『]?(しかし|だが|だけど|でも|だから|したがって|そして|さらに|また|ただし|それに|すると|なので|さて|では)', curr):
            G.add_edge(i-1, i)
            continue
            
        # 规则 C：显式指代词 (限制在句首前15个字符内，包含 こそあど 体系)
        if re.search(r'(これ|それ|あれ|この|その|あの|彼|彼女|彼ら|これら|それら|ここ|そこ|あそこ|そう|こう|ああ)', curr[:15]):
            G.add_edge(i-1, i)
            continue
            
        # 规则 D：极短的附和或拟声词 (日文包含平假名词缀，放宽到 8 个字符)
        if len(curr) <= 8:
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
        "请阅读 <text_to_segment> 标签内由 <block> 标签包裹的日文小说文本块。这些块本身已经是不可分割的最小单元。\n"
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
        "    {\"type\": \"mixed\", \"reason\": \"主角醒来并观察四周\", \"start_block_id\": 0, \"end_block_id\": 3},\n"
        "    {\"type\": \"dialogue\", \"reason\": \"两人讨论接下来的行动\", \"start_block_id\": 4, \"end_block_id\": 8}\n"
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

async def process_single_chapter(chapter_id, ja_sentences):
    total_len = len(ja_sentences)
    if total_len == 0:
        return []

    print(f"\n处理章节 [{chapter_id}]，共 {total_len} 个文本节点。")
    
    # 第一步：规则预绑定
    atomic_blocks = build_atomic_blocks_jp(ja_sentences)
    total_blocks = len(atomic_blocks)
    print(f"正则图论预绑定完成：压缩为 {total_blocks} 个原子块。")

    # 第二步：LLM 动态接力合并
    master_closures = []
    current_block_idx = 0
    
    while current_block_idx < total_blocks:
        end_idx = min(current_block_idx + WINDOW_SIZE_BLOCKS, total_blocks)
        print(f"扫描块区间: [{current_block_idx} -> {end_idx}]")
        blocks_chunk = atomic_blocks[current_block_idx : end_idx]
        
        closures = await merge_blocks_with_llm(blocks_chunk, current_block_idx, ja_sentences)
        
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
        
        # 将这个闭包内的所有日文合并，方便后续直接喂给翻译脚本
        merged_ja_text = "\n".join([ja_sentences[i] for i in range(real_start, real_end + 1)])
        
        final_sentence_closures.append({
            "type": cls.get("type", "mixed"),
            "reason": cls.get("reason", ""),
            "start_node_idx": real_start,
            "end_node_idx": real_end,
            "ja_text": merged_ja_text
        })

    return final_sentence_closures

async def main():
    for book_id in BOOKS_TO_PROCESS:
        epub_path = os.path.join(DATA_DIR, f"j{book_id}.epub")
        output_jsonl = os.path.join(DATA_DIR, f"predictbook{book_id}.jsonl")
        
        if not os.path.exists(epub_path):
            print(f"找不到文件: {epub_path}，跳过。")
            continue
            
        print(f"\n========================================")
        print(f"开始处理书籍: {epub_path}")
        print(f"========================================")
        
        book = epub.read_epub(epub_path)
        
        # 以追加模式打开 JSONL 文件（如果你中途中断了，记得手动删掉未完成的文件重跑）
        with open(output_jsonl, 'w', encoding='utf-8') as f_out:
            
            # 遍历 EPUB 的每一个章节 (HTML)
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                chapter_id = item.get_id()
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                
                chapter_texts = []
                for tag in soup.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
                    if tag.name == 'div' and tag.find(['p', 'div']):
                        continue
                    text = tag.get_text(strip=True)
                    if len(text) > 0:
                        chapter_texts.append(text)
                
                if not chapter_texts:
                    continue
                    
                if DEBUG_MODE:
                    chapter_texts = chapter_texts[:60]
                
                # 处理这一章的闭包
                closures = await process_single_chapter(chapter_id, chapter_texts)
                
                # 写入 JSONL (一行代表一个章节的完整切分结果)
                chapter_data = {
                    "chapter_id": chapter_id,
                    "total_nodes": len(chapter_texts),
                    "closures": closures
                }
                f_out.write(json.dumps(chapter_data, ensure_ascii=False) + "\n")
                f_out.flush() # 实时落盘
                
                if DEBUG_MODE:
                    print("\nDEBUG 模式结束，停止处理后续章节。")
                    break 
                    
        print(f"书籍 {book_id} 处理完毕保存至: {output_jsonl}")

if __name__ == "__main__":
    asyncio.run(main())