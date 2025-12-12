import json

file_path = ""

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)

except FileNotFoundError:
    print(f"错误: 文件 {file_path} 未找到。")
except json.JSONDecodeError:
    print(f"错误: 文件 {file_path} 不是有效的JSON格式。")
except Exception as e:
    print(f"发生其他错误: {e}")

layer1_hit = [0] * 256
layer2_hit = [0] * 256
layer3_hit = [0] * 256
for key, value in data_dict.items():
    for idx, code in enumerate(value):
        if code is None or code[0] != '<':
            print("warning")

        code_num = code.split('_')[1].rstrip('>')
        code_num = int(code_num)
        if idx == 0:
            layer1_hit[code_num] = 1
        elif idx == 1:
            layer2_hit[code_num] = 1
        else:
            layer3_hit[code_num] = 1

layer1_num = sum(layer1_hit)
layer2_num = sum(layer2_hit)
layer3_num = sum(layer3_hit)
print(layer1_num/256, layer2_num/256, layer3_num/256)

print((layer1_hit + layer2_hit + layer3_hit) / 256 * 3)

