def lcg(seed, a=1664525, c=1013904223, m=2**32):
    value = seed
    while True:
        value = (a * value + c) % m
        print("Value:", value)  # ✅ 正確縮排
        yield value

# 自定義參數
min_val = -10
max_val = 10
seed = 42
n = 10  # 產生幾個

# 建立 LCG 產生器
gen = lcg(seed)
print('\n=========')
print('Next gen % (max - min + 1):', next(gen) % (max_val - min_val + 1))
print('Next gen % (max - min + 1):', next(gen) % (max_val - min_val + 1))
print('Next gen % (max - min + 1):', next(gen) % (max_val - min_val + 1))

# 將產生的數映射到 [-10, 10]
random_numbers = [(next(gen) % (max_val - min_val + 1)) + min_val for _ in range(n)]

print(random_numbers)

random_numbers = []
for _ in range(n):
    raw = next(gen)
    print ("Raw", raw, "Reminder: ", raw % (max_val - min_val + 1))
    mapped = (raw % (max_val - min_val + 1)) + min_val
    # print(f"Mapped to Range: {mapped}")
    random_numbers.append(mapped)

print(random_numbers)
