import numpy as np

a = np.array([598, 196])
b = np.array([2, 3])
print(a)
print(b)


print('同为行向量:')
print(sum(a * b))
print('')


print('转置后向量:')
b_t = [[_b] for _b in b]
print(b_t)

r = 0
for i, _a in enumerate(a):
    r += _a * b_t[i][0]
print(r)
print('')

a_t = [[_a] for _a in a]
print(a_t)

print('同为列向量:')
r = 0
for i, _a in enumerate(a_t):
    r += _a[0] * b_t[i][0]
print(r)
