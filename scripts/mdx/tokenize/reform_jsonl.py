import json
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            try:
                data = json.loads(line)
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
            except UnicodeDecodeError as e:
                print(e)
            except json.JSONDecodeError as e:
                print(e)
            except Exception as e:
                print(e)

def process_file(file, input_dir, output_dir):
    input_file = os.path.join(input_dir, file)
    output_file = os.path.join(output_dir, file)
    process_jsonl(input_file, output_file)

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with ThreadPoolExecutor() as executor:
        futures = []
        for file in os.listdir(input_dir):
            if file.endswith('.jsonl'):
                futures.append(executor.submit(process_file, file, input_dir, output_dir))

        # Wait for all the futures to complete
        for future in futures:
            future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process jsonl files in a directory")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing jsonl files")
    parser.add_argument("output_dir", type=str, help="Path to the output directory for processed files")

    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir)

