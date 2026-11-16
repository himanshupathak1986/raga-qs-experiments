import hashlib
from pathlib import Path
from urllib.request import urlopen

ARTIFACT = 'https://raw.githubusercontent.com/SWE-bench/experiments'
ORIGINAL = '20260901_mini-v2.4.2_gemini-3-5-flash'
ALTERNATE = '20241028_agentless-1.5_gpt4o'
VERIFIED = 'https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified/resolve/78f471bf655a3137b2e8a75af1501690ec009ec3/data/test-00000-of-00001.parquet'
TARGET = Path(__file__).resolve().parents[2] / '.cache' / 'reviewer'
PIN = 'dda1f6e6338c69111516d86e95fa3c21c63e0f9e'
FILES = {
    'verified.parquet': (VERIFIED, '030cfd7f2a704c4c0226e7f104c725a3b41230b1d3517f9c915ad7ea5be3fa25'),
    'original_pred': (f'https://raw.githubusercontent.com/SWE-bench/{ORIGINAL}/2f6637a2557da1f7b5f2b583ab7090a077bbeed2/all_preds.jsonl', '74e15b9ceb4b4e2e53a92fc8b87465aaf594a7754a2f707bc02ab69487b212f9'),
    'original_result': (f'{ARTIFACT}/40f164d5b8f1d249bf95a6df8b74b577fd8e519d/evaluation/verified/{ORIGINAL}/results/results.json', '9fc1d88077468b2c655ba76d3de799094c565567ec175e389a3ab7a198949e57'),
    'pred': (f'{ARTIFACT}/{PIN}/evaluation/verified/{ALTERNATE}/all_preds.jsonl', '36e71807a34492f2e27303a9389ff14ce42c87972a5254309c37872eb478470e'),
    'result': (f'{ARTIFACT}/{PIN}/evaluation/verified/{ALTERNATE}/results/results.json', 'd4348eed42bb970040e057b353ddc04e94b9e0c0a1b3685242c9e1647a8c82a2'),
}

def main():
    TARGET.mkdir(exist_ok=True)
    for name, (url, expected) in FILES.items():
        path = TARGET / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            continue
        with urlopen(url, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise RuntimeError(f'Input checksum changed for {name}')
        path.write_bytes(payload)
        print(name, expected)

if __name__ == '__main__':
    main()
