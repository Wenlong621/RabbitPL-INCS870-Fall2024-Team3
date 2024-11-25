import numpy as np

# 替换为你的 .npz 文件路径
file_path = "./tedge.npz"
data = np.load(file_path)

# # 将内容保存到字符串
# output_string = ""

# for key in data.files:
#     output_string += f"Key: {key}\n"
#     output_string += f"Data:\n{data[key]}\n\n"
for key in data.files:
    print(f"Key: {key}")
    print(f"Data:\n{data[key]}\n")
# 打印或保存字符串
print(data)

# 打
