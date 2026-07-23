import os
import json
import csv
from datetime import datetime

def extract_info_from_log(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取实验参数
    args_start = content.find("Experiment Arguments:")
    args_end = content.find("----------------------------", args_start)
    args_str = content[args_start:args_end].split("Experiment Arguments:")[1]
    args_str = args_str.replace("'", '"')  # 转换单引号为双引号
    args = json.loads(args_str)

    # 提取最终准确率
    results_start = content.find('"all_clients":')
    results_str = "{" + content[results_start:]  # 补全JSON结构
    results = json.loads(results_str)

    return {
        "method": args["method"],
        "dataset_name": args["dataset"]["name"],
        "alpha": args["dataset"]["alpha"],
        "final_accuracy": float(results["all_clients"]["test"]["accuracy"].split("->")[-1].replace("%", "").strip())
    }

def main():
    root_dir = 'out'
    output_file = 'results.csv'

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Method', 'Dataset', 'Alpha', 'Final Accuracy (%)'])

        for root, dirs, files in os.walk(root_dir):
            if 'cifar100' in root.split(os.sep):
                # 找到最新实验目录
                exp_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
                print(exp_dirs)
                latest_dir = sorted(exp_dirs, key=lambda x: datetime.strptime(x, "%Y-%m-%d-%H-%M-%S"))[-1]

                log_path = os.path.join(root, latest_dir, 'main.log')
                if os.path.exists(log_path):
                    try:
                        info = extract_info_from_log(log_path)
                        writer.writerow([
                            info['method'],
                            info['dataset_name'],
                            info['alpha'],
                            info['final_accuracy']
                        ])
                    except Exception as e:
                        print(f"Error processing {log_path}: {str(e)}")

if __name__ == "__main__":
    main()
