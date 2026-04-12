import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import jieba
import pandas as pd
from tqdm import tqdm

from src.mda_topic_evolution.lda_step1_from_tokenized import (
    TokenizedSentenceMeta,
    train_lda_single_year_from_tokens,
)
from src.mda_topic_evolution.text_preprocess import (
    get_stopwords,
    preprocess_sentence_tokens,
    split_sentences_cn,
)


_YEAR_RE = re.compile(r"^\d{4}$")


def _read_text_with_fallback(path: Path, encodings: Sequence[str] = ("utf-8-sig", "utf-8", "gbk", "gb2312")) -> str:
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"Failed to read file: {path}, last_err={last_err}")


def _write_text(path: Path, content: str, encoding: str = "utf-8-sig") -> None:
    path.write_text(content, encoding=encoding)


def _infer_ticker_from_filename(path: Path) -> str:
    # 默认：用文件名（去扩展名）作为 ticker 标识
    return path.stem


def tokenize_and_overwrite_file(
    *,
    file_path: Path,
    year: int,
    stopwords,
    min_sentence_char_len: int,
    min_token_len: int,
) -> Tuple[List[List[str]], List[str]]:
    """
    读取原文 -> 按中文标点切句 -> jieba 分词/去停用词/过滤短句
    将分词后的句子（token 之间用空格拼接）按行写回原文件

    返回：
    - tokens_per_sentence: List[List[str]]
    - sentence_lines: List[str]  (每行就是 tokenized sentence text)
    """
    raw_text = _read_text_with_fallback(file_path)
    sentences = split_sentences_cn(raw_text)

    tokens_per_sentence: List[List[str]] = []
    sentence_lines: List[str] = []
    for s in sentences:
        if len(s.strip()) < min_sentence_char_len:
            continue
        toks = preprocess_sentence_tokens(
            s,
            stopwords=stopwords,
            min_char_len=min_sentence_char_len,
            min_token_len=min_token_len,
        )
        if not toks:
            continue
        tokens_per_sentence.append(toks)
        sentence_lines.append(" ".join(toks))

    # 覆盖写回
    new_content = "\n".join(sentence_lines)
    _write_text(file_path, new_content, encoding="utf-8-sig")

    return tokens_per_sentence, sentence_lines


def main():
    parser = argparse.ArgumentParser(description="Preprocess tokenization + Yearly Independent LDA (file-based Step1)")
    parser.add_argument("--data_root", required=True, help="根目录：包含 {year}/文本中/ 文件夹")
    parser.add_argument("--text_subdir", default="文本中", help="每年文本子目录名")
    parser.add_argument("--match_substr", default="12-31", help="只读取文件名包含该子串的文件")
    parser.add_argument("--year_start", type=int, default=2006)
    parser.add_argument("--year_end", type=int, default=2024)
    parser.add_argument("--out_dir", default="outputs/step1_from_files/yearly_lda")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_min", type=int, default=10)
    parser.add_argument("--k_max", type=int, default=50)
    parser.add_argument("--k_step", type=int, default=10)
    parser.add_argument("--coherence_sample_frac", type=float, default=1.0)
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)

    parser.add_argument("--min_sentence_char_len", type=int, default=5)
    parser.add_argument("--min_token_len", type=int, default=2)

    parser.add_argument("--dict_no_below", type=int, default=2)
    parser.add_argument("--dict_no_above", type=float, default=0.5)
    parser.add_argument("--dict_keep_n", type=int, default=None)

    parser.add_argument("--ticker_from_filename", action="store_true", help="ticker 默认用文件名（去扩展名）")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="关闭分词阶段的 tqdm 进度条（仍输出阶段文字）",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"--data_root not found: {data_root}")

    print(
        f"[Step1] 启动 | 数据根目录: {data_root} | 年份: {args.year_start}–{args.year_end} | "
        f"子目录: {args.text_subdir!r} | 文件名需包含: {args.match_substr!r}",
        flush=True,
    )
    print("[Step1] 阶段 0/3：初始化 jieba（下方几行来自 jieba，属正常）", flush=True)
    jieba.lcut("初始化")
    print("[Step1] jieba 就绪。阶段 1/3：按年读取文件、分词并写回原文路径", flush=True)

    stopwords = get_stopwords(None)

    year_to_bestk: Dict[int, int] = {}
    assignments_all: List[pd.DataFrame] = []

    for year in range(args.year_start, args.year_end + 1):
        year_dir = data_root / str(year)
        text_dir = year_dir / args.text_subdir
        if not text_dir.exists():
            print(f"[Step1] 跳过 {year}：目录不存在 {text_dir}", flush=True)
            continue

        tokens_per_sentence_all: List[List[str]] = []
        metas_all: List[TokenizedSentenceMeta] = []

        # 只读取文件名带 12-31 的文本文件
        files = sorted(p for p in text_dir.iterdir() if p.is_file() and (args.match_substr in p.name))
        if not files:
            print(
                f"[Step1] 跳过 {year}：{text_dir} 下无文件名包含 {args.match_substr!r} 的文件",
                flush=True,
            )
            continue

        print(f"[Step1] 年份 {year}：匹配到 {len(files)} 个文件，开始分词…", flush=True)
        file_iter: Iterable[Path] = files
        if not args.no_progress:
            file_iter = tqdm(
                files,
                desc=f"分词 {year}",
                unit="文件",
                file=sys.stdout,
                dynamic_ncols=True,
            )

        for file_path in file_iter:
            ticker = _infer_ticker_from_filename(file_path) if args.ticker_from_filename else file_path.stem
            tokens_per_sentence, _sentence_lines = tokenize_and_overwrite_file(
                file_path=file_path,
                year=year,
                stopwords=stopwords,
                min_sentence_char_len=args.min_sentence_char_len,
                min_token_len=args.min_token_len,
            )

            # tokenized 句子行就是我们存储的 sentence_text
            for toks in tokens_per_sentence:
                tokens_per_sentence_all.append(toks)
                metas_all.append(
                    TokenizedSentenceMeta(
                        ticker=ticker,
                        year=year,
                        sentence_text=" ".join(toks),
                    )
                )

        if len(tokens_per_sentence_all) == 0:
            print(f"[Step1] 跳过 {year}：分词后无任何有效句子", flush=True)
            continue

        print(
            f"[Step1] 年份 {year}：分词完成，有效句数 {len(tokens_per_sentence_all)}。"
            f"阶段 2/3：LDA 选 K 并训练（可能较慢）",
            flush=True,
        )
        assignments_df, best_k = train_lda_single_year_from_tokens(
            year=year,
            tokens_per_sentence=tokens_per_sentence_all,
            sentence_metas=metas_all,
            seed=args.seed,
            out_dir=args.out_dir,
            k_min=args.k_min,
            k_max=args.k_max,
            k_step=args.k_step,
            coherence_sample_frac=args.coherence_sample_frac,
            passes=args.passes,
            iterations=args.iterations,
            dict_no_below=args.dict_no_below,
            dict_no_above=args.dict_no_above,
            dict_keep_n=args.dict_keep_n,
            verbose=args.verbose,
        )
        print(f"[Step1] 年份 {year}：LDA 完成，best_k={best_k}", flush=True)
        assignments_all.append(assignments_df)
        year_to_bestk[year] = best_k

    print("[Step1] 阶段 3/3：写入汇总 CSV", flush=True)
    if assignments_all:
        assignments_all_df = pd.concat(assignments_all, ignore_index=True)
    else:
        assignments_all_df = pd.DataFrame()

    # 保存 year -> K_t 的汇总
    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = os.path.join(args.out_dir, "year_to_bestK.csv")
    pd.DataFrame({"year": list(year_to_bestk.keys()), "best_k": list(year_to_bestk.values())}).to_csv(
        meta_path, index=False, encoding="utf-8-sig"
    )

    print(f"[Step1] done. years_trained={len(year_to_bestk)} total_assignments={len(assignments_all_df)}")


if __name__ == "__main__":
    main()

