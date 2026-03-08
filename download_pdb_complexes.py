"""
从 PDB 批量下载蛋白质复合物

使用方法：
    # 下载 1000 个高质量复合物
    python download_pdb_complexes.py \
        --output_dir ./data/pdb_complexes \
        --max_resolution 2.5 \
        --download_limit 1000

    # 只搜索不下载
    python download_pdb_complexes.py --search_only

    # 从文件列表下载
    python download_pdb_complexes.py \
        --pdb_list pdb_ids.txt \
        --output_dir ./data/pdb_complexes
"""

import requests
import json
from pathlib import Path
from tqdm import tqdm
import time
import argparse


def search_pdb_complexes(min_chains=2, max_resolution=3.0, limit=5000):
    """
    搜索 PDB 中的蛋白质复合物

    Args:
        min_chains: 最少链数（2 表示至少是二聚体）
        max_resolution: 最大分辨率（Å）
        limit: 返回结果数量

    Returns:
        list of PDB IDs
    """
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "greater_or_equal",
                        "value": min_chains
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": max_resolution
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION"
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "pager": {
                "start": 0,
                "rows": limit
            }
        }
    }

    url = "https://search.rcsb.org/rcsbsearch/v2/query"

    try:
        response = requests.post(url, json=query, timeout=30)
        if response.status_code == 200:
            results = response.json()
            pdb_ids = [item['identifier'] for item in results.get('result_set', [])]
            return pdb_ids
        else:
            print(f"Error: HTTP {response.status_code}")
            return []
    except Exception as e:
        print(f"Error searching PDB: {e}")
        return []


def download_pdb_file(pdb_id, output_dir="./data/pdb_complexes", retry=3):
    """
    下载单个 PDB 文件

    Args:
        pdb_id: PDB ID (e.g., "1A2K")
        output_dir: 输出目录
        retry: 重试次数

    Returns:
        bool: 是否成功
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pdb_id = pdb_id.upper()
    output_path = f"{output_dir}/{pdb_id}.pdb"

    # 如果文件已存在，跳过
    if Path(output_path).exists():
        return True

    # 尝试多个镜像站点
    urls = [
        f"https://files.rcsb.org/download/{pdb_id}.pdb",
        f"https://files.wwpdb.org/pub/pdb/data/structures/all/pdb/pdb{pdb_id.lower()}.ent.gz",
    ]

    for attempt in range(retry):
        for url in urls:
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    # 处理 gzip 压缩文件
                    if url.endswith('.gz'):
                        import gzip
                        content = gzip.decompress(response.content).decode('utf-8')
                    else:
                        content = response.text

                    with open(output_path, 'w') as f:
                        f.write(content)
                    return True
            except Exception as e:
                if attempt == retry - 1:
                    print(f"  Error downloading {pdb_id}: {e}")
                continue

        time.sleep(1)  # 重试前等待

    return False


def download_from_list(pdb_list_file, output_dir):
    """从文件列表下载 PDB"""
    with open(pdb_list_file, 'r') as f:
        pdb_ids = [line.strip() for line in f if line.strip()]

    print(f"Found {len(pdb_ids)} PDB IDs in {pdb_list_file}")

    success = 0
    failed = []

    for pdb_id in tqdm(pdb_ids, desc="Downloading"):
        if download_pdb_file(pdb_id, output_dir):
            success += 1
        else:
            failed.append(pdb_id)

        # 避免请求过快
        if success % 50 == 0:
            time.sleep(1)

    return success, failed


def main():
    parser = argparse.ArgumentParser(
        description="Download protein complexes from PDB"
    )
    parser.add_argument('--output_dir', default='./data/pdb_complexes',
                       help='Output directory for PDB files')
    parser.add_argument('--min_chains', type=int, default=2,
                       help='Minimum number of protein chains (default: 2)')
    parser.add_argument('--max_resolution', type=float, default=2.5,
                       help='Maximum resolution in Angstroms (default: 2.5)')
    parser.add_argument('--search_limit', type=int, default=5000,
                       help='Maximum number of search results (default: 5000)')
    parser.add_argument('--download_limit', type=int, default=1000,
                       help='Maximum number of files to download (default: 1000)')
    parser.add_argument('--search_only', action='store_true',
                       help='Only search, do not download')
    parser.add_argument('--pdb_list', type=str, default=None,
                       help='Download from a list of PDB IDs (one per line)')
    parser.add_argument('--save_list', type=str, default=None,
                       help='Save search results to file')

    args = parser.parse_args()

    print("="*80)
    print("PDB Complex Downloader")
    print("="*80)

    # 从文件列表下载
    if args.pdb_list:
        print(f"Downloading from list: {args.pdb_list}")
        success, failed = download_from_list(args.pdb_list, args.output_dir)

        print(f"\nDownload complete:")
        print(f"  Success: {success}")
        print(f"  Failed: {len(failed)}")

        if failed:
            print(f"\nFailed PDB IDs:")
            for pdb_id in failed[:10]:
                print(f"  {pdb_id}")
            if len(failed) > 10:
                print(f"  ... and {len(failed)-10} more")

        return

    # 搜索 PDB
    print(f"\nSearching PDB database...")
    print(f"  Min chains: {args.min_chains}")
    print(f"  Max resolution: {args.max_resolution} Å")
    print(f"  Search limit: {args.search_limit}")

    pdb_ids = search_pdb_complexes(
        min_chains=args.min_chains,
        max_resolution=args.max_resolution,
        limit=args.search_limit
    )

    if not pdb_ids:
        print("No results found!")
        return

    print(f"Found {len(pdb_ids)} protein complexes")

    # 保存搜索结果
    if args.save_list:
        with open(args.save_list, 'w') as f:
            for pdb_id in pdb_ids:
                f.write(f"{pdb_id}\n")
        print(f"Search results saved to {args.save_list}")

    # 只搜索不下载
    if args.search_only:
        print("\nFirst 20 PDB IDs:")
        for pdb_id in pdb_ids[:20]:
            print(f"  {pdb_id}")
        if len(pdb_ids) > 20:
            print(f"  ... and {len(pdb_ids)-20} more")
        return

    # 下载
    download_count = min(len(pdb_ids), args.download_limit)
    print(f"\nDownloading {download_count} PDB files to {args.output_dir}...")

    success = 0
    failed = []

    for i, pdb_id in enumerate(tqdm(pdb_ids[:download_count]), 1):
        if download_pdb_file(pdb_id, args.output_dir):
            success += 1
        else:
            failed.append(pdb_id)

        # 避免请求过快
        if i % 50 == 0:
            time.sleep(1)

    print(f"\n{'='*80}")
    print("Download Summary")
    print(f"{'='*80}")
    print(f"Total attempted: {download_count}")
    print(f"Success: {success}")
    print(f"Failed: {len(failed)}")
    print(f"Output directory: {args.output_dir}")

    if failed:
        print(f"\nFailed PDB IDs (first 10):")
        for pdb_id in failed[:10]:
            print(f"  {pdb_id}")
        if len(failed) > 10:
            print(f"  ... and {len(failed)-10} more")

        # 保存失败列表
        failed_file = f"{args.output_dir}/failed_downloads.txt"
        with open(failed_file, 'w') as f:
            for pdb_id in failed:
                f.write(f"{pdb_id}\n")
        print(f"\nFailed IDs saved to {failed_file}")

    print(f"\n{'='*80}")
    print("Next step:")
    print(f"  python extract_binding_sites.py \\")
    print(f"      --input_dir {args.output_dir} \\")
    print(f"      --output_dir ./data/Dockground_{success}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
