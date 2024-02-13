import sys
import argparse
import subprocess

def main():
    if len(sys.argv) != 4:
        print("Usage: python main.py {model_name} {model_step} [--poly](Optional)")
        sys.exit(1)

    model_name = sys.argv[1]
    model_step = sys.argv[2]

    parser = argparse.ArgumentParser()
    parser.add_argument('--poly', action='store_true', help='Options for Poly-Language')
    args = parser.parse_args(sys.argv[3:])

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
