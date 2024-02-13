import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='Name of the model')
    parser.add_argument('model_step', type=str, help='Step of the model')
    parser.add_argument('--poly', action='store_true', help='Options for Poly-Language')

    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"Unknown arguments: {unknown}")
        print("Usage: python main.py {model_name} {model_step} [--poly](Optional)")
        sys.exit(1)

    model_name = args.model_name
    model_step = args.model_step

    if args.poly:
        command = ["python", "./vits2/inference.py", model_name, model_step, "--poly"]
    else:
        command = ["python", "./vits2/inference.py", model_name, model_step]

    try:
        subprocess.run(command, check=True)
        
    except subprocess.CalledProcessError:
        print("Error occurred")

if __name__ == "__main__":
    main()
