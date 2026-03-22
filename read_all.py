import sys

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        print(f.read())

if __name__ == "__main__":
    if len(sys.argv) > 1:
        read_file(sys.argv[1])
