import sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    text = f.read()
start = int(sys.argv[2])
end = int(sys.argv[3])
print(text[start:end])
