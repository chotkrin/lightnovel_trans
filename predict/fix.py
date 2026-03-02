import asyncio
import re
import os
import json
import uuid
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

MODEL_NAME = "Rinn000/LN-SFT-chunked" 
CONCURRENCY_LIMIT = 100 
DATA_DIR = "/home/zelin/code/lightnovel_trans/data/"
BOOKS_TO_PROCESS = [4, 5, 6, 7]

GLOSSARY_FILE = os.path.join(DATA_DIR, "glossary.json")
global_glossary = {}
if os.path.exists(GLOSSARY_FILE):
    with open(GLOSSARY_FILE, "r", encoding="utf-8") as f:
        global_glossary = json.load(f)

async def fetch_passage_translation(sem, task_id, ja_chunk_text):
    async with sem:
        matched_terms = {k: v for k, v in global_glossary.items() if k in ja_chunk_text}
        
        system_instruction = (
            "你是一位精通中日文学的资深轻小说翻译家。\n"
            "请严格遵循以下规则，将输入的日文翻译为符合轻小说文风的简体中文。\n\n"
            "<language_constraints>\n"
            "  <rule priority=\"critical\">【零容忍红线】：翻译结果中【绝对禁止】残留任何日文平假名（ひらがな）或片假名（カタカナ）！</rule>\n"
            "  <rule priority=\"high\">必须 100% 全量翻译！绝不允许只翻译前半段而漏翻后半段，也不允许中途变成复读机。</rule>\n"
            "  <rule priority=\"high\">如果遇到生僻的日文专有名词且词表中没有，请结合上下文意译为中文，绝不能直接复制粘贴日文原词。</rule>\n"
            "  <rule priority=\"medium\">确保人物语气连贯，代词指代清晰，直接输出纯中文翻译，不要任何解释说明。</rule>\n"
            "</language_constraints>"
        )
        
        user_content = ""
        if matched_terms:
            glossary_lines = [f"{k} -> {v}" for k, v in matched_terms.items()]
            user_content += f"<glossary>\n{chr(10).join(glossary_lines)}\n</glossary>\n"
            
        user_content += f"<text_to_translate>\n{ja_chunk_text}\n</text_to_translate>\n"
        user_content += "<output_translation_in_pure_chinese>\n"

        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1,
                max_tokens=2048, 
                extra_body={"chat_template_kwargs": {"enable_thinking": True}}
            )
            content = response.choices[0].message.content
            
            content = re.sub(r'</?text_to_translate>', '', content)
            if "<think>" in content:
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            return task_id, content.strip()
        except Exception as e:
            print(f"[Task ID: {task_id} Failed]: {e}")
            return task_id, None 

async def process_and_repair_book(book_id):
    epub_path = os.path.join(DATA_DIR, f"j{book_id}.epub")
    jsonl_path = os.path.join(DATA_DIR, f"predictbook{book_id}.jsonl")
    output_epub = os.path.join(DATA_DIR, f"c{book_id}_chunked.epub")
    cache_file = os.path.join(DATA_DIR, f"translation_cache_book{book_id}.json")
    
    if not os.path.exists(epub_path) or not os.path.exists(jsonl_path):
        return

    print("-" * 50)
    print(f"Starting Deep Inspection and Repair: Book {book_id}")
    print("-" * 50)

    book = epub.read_epub(epub_path)
    item_to_soup = {}
    chapter_nodes_map = {}

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        chapter_id = item.get_id()
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        item_to_soup[chapter_id] = soup
        nodes = []
        for tag in soup.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            if tag.name == 'div' and tag.find(['p', 'div']): continue
            if len(tag.get_text(strip=True)) > 0: nodes.append(tag)
        chapter_nodes_map[chapter_id] = nodes

    print("Step 1/3: Scanning blueprint for index continuity gaps...")
    patched_chapters = []
    gap_count = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            chapter_id = data["chapter_id"]
            total_nodes = data["total_nodes"]
            closures = sorted(data["closures"], key=lambda x: x["start_node_idx"])
            
            patched_closures = []
            expected_next_idx = 0
            
            for cls in closures:
                start_idx = cls["start_node_idx"]
                if start_idx > expected_next_idx:
                    nodes = chapter_nodes_map.get(chapter_id, [])
                    gap_ja_text = "\n".join([n.get_text(strip=True) for n in nodes[expected_next_idx : start_idx]])
                    patched_closures.append({
                        "type": "gap_patched",
                        "start_node_idx": expected_next_idx,
                        "end_node_idx": start_idx - 1,
                        "ja_text": gap_ja_text
                    })
                    gap_count += 1
                
                patched_closures.append(cls)
                expected_next_idx = cls["end_node_idx"] + 1
            
            if expected_next_idx < total_nodes:
                nodes = chapter_nodes_map.get(chapter_id, [])
                gap_ja_text = "\n".join([n.get_text(strip=True) for n in nodes[expected_next_idx : total_nodes]])
                patched_closures.append({
                    "type": "gap_patched_tail",
                    "start_node_idx": expected_next_idx,
                    "end_node_idx": total_nodes - 1,
                    "ja_text": gap_ja_text
                })
                gap_count += 1
                
            data["closures"] = patched_closures
            patched_chapters.append(data)

    print(f"Blueprints repaired. Patched {gap_count} gaps.")

    print("Step 2/3: Purging contaminated or lazy cache entries...")
    translated_dict = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            translated_dict = json.load(f)
            
        keys_to_purge = []
        for tid, text in translated_dict.items():
            if not text:
                keys_to_purge.append(tid)
                continue
            
            ja_chars = re.findall(r'[\u3040-\u309F\u30A0-\u30FF]', text)
            if len(ja_chars) > 3:
                keys_to_purge.append(tid)
                
        for tid in keys_to_purge:
            del translated_dict[tid]
            
        print(f"Purged {len(keys_to_purge)} faulty translations from cache.")

    print("Step 3/3: Re-translating missing/purged segments...")
    translation_tasks_info = []
    for ch_data in patched_chapters:
        chapter_id = ch_data["chapter_id"]
        nodes = chapter_nodes_map.get(chapter_id, [])
        for cls in ch_data["closures"]:
            tid = f"{chapter_id}_{cls['start_node_idx']}_{cls['end_node_idx']}"
            translation_tasks_info.append({
                "task_id": tid,
                "ja_text": cls["ja_text"],
                "soup": item_to_soup[chapter_id],
                "nodes": nodes[cls['start_node_idx'] : cls['end_node_idx'] + 1]
            })

    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    async_tasks = [
        fetch_passage_translation(sem, info["task_id"], info["ja_text"])
        for info in translation_tasks_info if info["task_id"] not in translated_dict
    ]

    if async_tasks:
        results = await tqdm_asyncio.gather(*async_tasks, desc=f"Patching Book {book_id}")
        for tid, text in results:
            if text: translated_dict[tid] = text
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(translated_dict, f, ensure_ascii=False, indent=2)

    for info in translation_tasks_info:
        tid = info["task_id"]
        translated_text = translated_dict.get(tid, "")
        if translated_text and translated_text != info["ja_text"]:
            nodes, soup = info["nodes"], info["soup"]
            first_node = nodes[0]
            first_node.clear()
            for line in translated_text.split('\n'):
                if line.strip():
                    first_node.append(soup.new_string(line.strip()))
                    first_node.append(soup.new_tag('br'))
            if first_node.contents and getattr(first_node.contents[-1], 'name', None) == 'br':
                first_node.contents[-1].extract()
            for node in nodes[1:]:
                if node.parent:
                    node.string = ""
                    node['style'] = "display: none;" 

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        item.set_content(str(item_to_soup[item.get_id()]).encode('utf-8'))
    
    for item in book.get_items():
        if not getattr(item, 'uid', None): item.uid = str(uuid.uuid4())
    
    def force_fix_uid(node):
        if isinstance(node, (list, tuple)):
            for sub_node in node: force_fix_uid(sub_node)
        elif hasattr(node, 'uid') and not node.uid:
            node.uid = str(uuid.uuid4())

    force_fix_uid(book.toc)

    try:
        epub.write_epub(output_epub, book, {})
        print(f"Success: Cleaned and repaired Book {book_id} saved.")
    except Exception as e:
        print(f"Failed to save Book {book_id}: {e}")

async def main():
    for book_id in BOOKS_TO_PROCESS:
        await process_and_repair_book(book_id)

if __name__ == "__main__":
    asyncio.run(main())