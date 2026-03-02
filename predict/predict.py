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
DEBUG_MODE = False

global_glossary = {}
if os.path.exists(GLOSSARY_FILE) :
    with open(GLOSSARY_FILE, "r", encoding="utf-8") as f:
        global_glossary = json.load(f)
    print(f"Glossary loaded: {len(global_glossary)} entries found.")

async def fetch_passage_translation(sem, task_id, ja_chunk_text):
    async with sem:
        # Match glossary terms present in current text chunk
        matched_terms = {k: v for k, v in global_glossary.items() if k in ja_chunk_text}
        
        system_instruction = (
            "你是一位精通中日文学的资深轻小说翻译家。\n"
            "请严格遵循以下规则，将 <text_to_translate> 标签内的日文翻译为符合轻小说文风的简体中文。\n\n"
            "<language_constraints>\n"
            "  <rule priority=\"critical\">【零容忍红线】：翻译结果中绝对禁止残留任何日文平假名或片假名！</rule>\n"
            "  <rule priority=\"high\">必须 100% 全量翻译，严禁漏翻或中途停止。</rule>\n"
            "  <rule priority=\"high\">遇到词表中未收录的生僻专有名词，请结合上下文意译，禁止直接复制日文原词。</rule>\n"
            "  <rule priority=\"medium\">确保人物语气连贯，代词指代清晰，直接输出翻译结果，不要任何解释说明。</rule>\n"
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
                top_p=0.85,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            
            content = response.choices[0].message.content
            
            # Post-processing: Remove potential residue tags or thought chains
            content = re.sub(r'</?text_to_translate>', '', content)
            content = re.sub(r'</?output_translation_in_pure_chinese>', '', content)
            if "<think>" in content:
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                
            return task_id, content.strip()
            
        except Exception as e:
            print(f"[Task ID: {task_id} Failed]: {e}")
            return task_id, ja_chunk_text 

async def process_book(book_id):
    epub_path = os.path.join(DATA_DIR, f"j{book_id}.epub")
    jsonl_path = os.path.join(DATA_DIR, f"predictbook{book_id}.jsonl")
    output_epub = os.path.join(DATA_DIR, f"c{book_id}_chunked.epub")
    cache_file = os.path.join(DATA_DIR, f"translation_cache_book{book_id}.json")
    
    if not os.path.exists(epub_path) or not os.path.exists(jsonl_path):
        print(f"Skipping Book {book_id}: Missing EPUB or JSONL files.")
        return

    print("-" * 40)
    print(f"Processing Book: {book_id}")
    print("-" * 40)

    chapter_closures_map = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            chapter_closures_map[data["chapter_id"]] = data["closures"]

    book = epub.read_epub(epub_path)
    item_to_soup = {}
    translation_tasks_info = [] 
    
    print("Aligning HTML nodes with closure blueprints...")
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        chapter_id = item.get_id()
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        item_to_soup[chapter_id] = soup
        
        if chapter_id not in chapter_closures_map:
            continue
            
        chapter_nodes = []
        for tag in soup.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            if tag.name == 'div' and tag.find(['p', 'div']):
                continue
            if len(tag.get_text(strip=True)) > 0:
                chapter_nodes.append(tag)
                
        for cls in chapter_closures_map[chapter_id]:
            start_idx = cls["start_node_idx"]
            end_idx = cls["end_node_idx"]
            
            task_id = f"{chapter_id}_{start_idx}_{end_idx}"
            translation_tasks_info.append({
                "task_id": task_id,
                "ja_text": cls["ja_text"],
                "chapter_id": chapter_id,
                "soup": soup,
                "nodes": chapter_nodes[start_idx : end_idx + 1]
            })

    if DEBUG_MODE:
        print("DEBUG_MODE: Processing first 10 items only.")
        translation_tasks_info = translation_tasks_info[:10]

    translated_dict = {}
    if os.path.exists(cache_file):
        print(f"Loading cache from {cache_file}...")
        with open(cache_file, "r", encoding="utf-8") as f:
            translated_dict = json.load(f)

    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    async_tasks = [
        fetch_passage_translation(sem, info["task_id"], info["ja_text"])
        for info in translation_tasks_info if info["task_id"] not in translated_dict
    ]

    if async_tasks:
        print(f"Dispatching LLM requests (Count: {len(async_tasks)})...")
        results = await tqdm_asyncio.gather(*async_tasks, desc=f"Book {book_id}")
        for tid, text in results:
            translated_dict[tid] = text
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(translated_dict, f, ensure_ascii=False, indent=2)

    print("Injecting translations and cleaning redundant nodes...")
    for info in translation_tasks_info:
        tid = info["task_id"]
        translated_text = translated_dict.get(tid, "")
        
        if translated_text and translated_text != info["ja_text"]:
            nodes = info["nodes"]
            soup = info["soup"]
            first_node = nodes[0]
            
            first_node.clear()
            lines = translated_text.split('\n')
            for line in lines:
                if line.strip():
                    first_node.append(soup.new_string(line.strip()))
                    first_node.append(soup.new_tag('br'))
            
            if first_node.contents and getattr(first_node.contents[-1], 'name', None) == 'br':
                first_node.contents[-1].extract()

            # Soft Delete
            for node in nodes[1:]:
                if node.parent is not None:
                    node.string = ""  
                    node['style'] = "display: none;" 

    # Finalize EPUB
    print("Finalizing EPUB structure...")
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        item.set_content(str(item_to_soup[item.get_id()]).encode('utf-8'))
    
    def ensure_node_uids(node):
        if isinstance(node, (list, tuple)):
            for sub_node in node:
                ensure_node_uids(sub_node)
        elif hasattr(node, 'uid') and not node.uid:
            node.uid = str(uuid.uuid4())

    ensure_node_uids(book.toc)

    try:
        epub.write_epub(output_epub, book, {})
        print(f"Success: Saved to {output_epub}")
    except Exception as e:
        print(f"Failed to save Book {book_id}: {e}")

async def main():
    for book_id in BOOKS_TO_PROCESS:
        await process_book(book_id)
    print("\nAll tasks completed.")

if __name__ == "__main__":
    asyncio.run(main())