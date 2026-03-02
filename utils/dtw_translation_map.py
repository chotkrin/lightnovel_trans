import json
import os
import ebooklib
import numpy as np
from ebooklib import epub
from bs4 import BeautifulSoup
from tqdm.auto import tqdm 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

vol = 1 # 卷号，根据实际情况调整

def extract_paragraphs_from_epub(epub_path):
    book = epub.read_epub(epub_path)
    chapters = []
    
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_body_content(), 'html.parser')
        paragraphs = [p.get_text(strip=True) for p in soup.find_all(['p', 'div'])]
        paragraphs = [p for p in paragraphs if len(p) > 0]
        
        if paragraphs:
            chapters.append(paragraphs)
            
    return chapters

print("parsing EPUB ...")
ja_chapters = extract_paragraphs_from_epub(f'/home/zelin/code/lightnovel_trans/data/j{vol}.epub')
zh_chapters = extract_paragraphs_from_epub(f'/home/zelin/code/lightnovel_trans/data/c{vol}.epub')

print(f"{len(ja_chapters)} -> {len(zh_chapters)}")

# 章节预处理，每个epub不一样，需要根据实际情况调整
ja_chapters_cuthead = ja_chapters[1:-1]
zh_chapters_cuttail = zh_chapters[:-2]
ja_head = ja_chapters[0]
ja_head = ja_head[2:]
short_ja_head = [c.replace('\u3000', '') for c in ja_head]
ja_head.extend(short_ja_head)
ja_head = set(ja_head)
ja_chapters_concat = [i for c in ja_chapters_cuthead for i in c]
ja_chapters_true = []
prev_i = 1
for i in range(2, len(ja_chapters_concat)):
    c = ja_chapters_concat[i]
    match_head = next((h for h in ja_head if c.startswith(h)), None)
    if match_head:
        ja_chapters_true.append(ja_chapters_concat[prev_i: i])
        prev_i = i + 1
ja_chapters_true.append(ja_chapters_concat[prev_i:])
# --------------------------

embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def align_paragraphs_traditional(ja_paras, zh_paras, gap_penalty=0.2):
    if not ja_paras or not zh_paras:
        return []

    ja_embs = embedder.encode(ja_paras, show_progress_bar=False)
    zh_embs = embedder.encode(zh_paras, show_progress_bar=False)

    sim_matrix = cosine_similarity(ja_embs, zh_embs)

    n, m = len(ja_paras), len(zh_paras)
    
    dp = np.zeros((n + 1, m + 1))
    pointers = np.zeros((n + 1, m + 1, 2), dtype=int)

    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] - gap_penalty
        pointers[i][0] = [i-1, 0]
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j-1] - gap_penalty
        pointers[0][j] = [0, j-1]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_score = sim_matrix[i-1][j-1]
            
            opt_match = dp[i-1][j-1] + match_score
            opt_skip_zh = dp[i-1][j] - gap_penalty
            opt_skip_ja = dp[i][j-1] - gap_penalty

            best_score = max(opt_match, opt_skip_zh, opt_skip_ja)
            dp[i][j] = best_score
            
            if best_score == opt_match:
                pointers[i][j] = [i-1, j-1]
            elif best_score == opt_skip_zh:
                pointers[i][j] = [i-1, j]
            else:
                pointers[i][j] = [i, j-1]

    path = []
    i, j = n, m
    while i > 0 or j > 0:
        prev_i, prev_j = pointers[i][j]
        if prev_i == i - 1 and prev_j == j - 1:
            path.append((prev_i, prev_j)) # 找到一对匹配
        i, j = prev_i, prev_j
        
    path.reverse()

    aligned_data = []
    current_ja_ids = []
    current_zh_ids = []
    
    for (ja_id, zh_id) in path:
        current_ja_ids.append(ja_id)
        current_zh_ids.append(zh_id)

    mapping = {}
    for ja_id, zh_id in path:
        if ja_id not in mapping:
            mapping[ja_id] = []
        mapping[ja_id].append(zh_id)

    final_pairs = []
    for ja_id in sorted(mapping.keys()):
        zh_ids = sorted(list(set(mapping[ja_id])))
        ja_str = ja_paras[ja_id]
        zh_str = "".join([zh_paras[z] for z in zh_ids])
        
        if ja_str.strip() and zh_str.strip():
            final_pairs.append({"ja": ja_str, "zh": zh_str})

    return final_pairs


test_chapter_idx = 1
ja_test = ja_chapters_true[test_chapter_idx]
zh_test = zh_chapters_cuttail[test_chapter_idx]

aligned_results = align_paragraphs_traditional(ja_test, zh_test)

for res in aligned_results[:3]:
    print(f"JA: {res['ja']}")
    print(f"ZH: {res['zh']}\n")


def process_and_save_full_book_dtw(ja_chapters, zh_chapters, output_jsonl="aligned_book_dtw.jsonl"):
    processed_chapters = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_chapters.add(data["chapter_idx"])
                except json.JSONDecodeError:
                    continue

    total_chapters = min(len(ja_chapters), len(zh_chapters))

    with open(output_jsonl, "a", encoding="utf-8") as f:
        for chapter_idx in tqdm(range(total_chapters)):
            if chapter_idx in processed_chapters:
                continue
                
            ja_paras = ja_chapters[chapter_idx]
            zh_paras = zh_chapters[chapter_idx]
            
            if not ja_paras or not zh_paras:
                continue
            
            aligned_pairs = align_paragraphs_traditional(ja_paras, zh_paras)
            
            chapter_data = {
                "chapter_idx": chapter_idx,
                "pairs": aligned_pairs
            }
            f.write(json.dumps(chapter_data, ensure_ascii=False) + "\n")
            f.flush() 
            
process_and_save_full_book_dtw(ja_chapters_true, zh_chapters_cuttail, output_jsonl=f"aligned_book_dtw_{vol}.jsonl")