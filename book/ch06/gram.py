import numpy as np


def gram(_a, _b):
    v = [_a, _b]
    v_len = len(v)

    gram_matrix = np.zeros((v_len, v_len))

    for i in range(v_len):
        for j in range(v_len):
            gram_matrix[i,j] = sum(v[i] * v[j])
    
    return gram_matrix

def gram_tripl(_a, _b, _c):
    v = [_a, _b, _c]

    v_len = len(v)
    gram_matrix = np.zeros((v_len, v_len))

    for i in range(v_len):
        for j in range(v_len):
            gram_matrix[i, j] = sum(v[i] * v[j])

    return gram_matrix
    
def convert_t(a):
    return [[_a] for _a in a]


a = np.array([1, 1, 1, 0, 0])
a_t = convert_t(a) 

b = np.array([1, 1, 0, 1, 0])
b_t = convert_t(b)

c = np.array([-2, -2, 0, -1, 0])

d = np.array([2, 3])
d_t = convert_t(d)

# 计算两个相同向量的gram 矩阵
print(a * a_t)
print('*' * 20)

# 计算两个相同向量的内积
print(sum(a * a))
print('*' * 20)

# 计算两个不同向量的gram矩阵
print(gram(a, b)) 
print('*' * 20)

# 计算两个不同向量的外积 
print(a * b_t)
print('*' * 20)

# 计算两个不同向量的内积
print(sum(a * b))

print(sum(a * b * c))
print(gram_tripl(a, b, c))
#print(gram(a, c))

