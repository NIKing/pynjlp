import sys
import math
sys.path.append('/pynjlp')

import numpy as np

from nlp.model.hmm.FirstOrderHiddenMarkovModel import FirstOrderHiddenMarkovModel

# 定义隐状态和观察状态，隐藏状态是健康状态，观察状态是身体表现状态；
# 健康状态 = （健康，发烧）身体表现状态 = （正常，发冷，头晕）
states = ('Healthy', 'Fever')
observations = ('normal', 'cold', 'dizzy')

# 以下这些参数是根据医生经验得出来的数据，可根据这些数据生成初始概率向量，转移概率矩阵和发射概率矩阵
start_probability = {'Healthy': 0.6, 'Fever': 0.4}
transition_probability = {
    'Healthy': {'Healthy': 0.7, 'Fever': 0.3},
    'Fever': {'Healthy': 0.4, 'Fever': 0.6}
}
emission_probability = {
    'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
    'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
}

def generate_index_map(lables):
    """生成标签和编号的映射关系"""
    index_label, label_index = {}, {}

    i = 0
    for l in lables:
        index_label[i] = l
        label_index[l] = i

        i += 1

    return label_index, index_label


def convert_observations_to_index(observations, label_index):
    """转换观察状态为索引编号"""
    _list = []
    for o in observations:
        _list.append(label_index[o])

    return _list

def convert_map_to_matrix(_map, label_index1, label_index2):
    """
    转换映射关系为矩阵, 这个很巧妙，在后面为张量赋值的时候，经常用到。
    在这里是根据状态的标签位置，把两个状态概率值，都列出来。
    -param _map 所有状态及其各自的概率值
    -param label_index1 第一个状态，所有标签的位置
    -param lable_index2 第二个状态，所有标签的位置
    """
    m = np.empty((len(label_index1), len(label_index2)), dtype = float)
    
    for line in _map:
        for col in _map[line]:
            m[label_index1[line]][label_index2[col]] = _map[line][col]

    return m.tolist()


def convert_map_to_vector(_map, label_index):
    """转换映射关系到向量"""
    v = np.empty(len(_map), dtype = float)
    for e in _map:
        v[label_index[e]] = _map[e]

    return v.tolist()

def generate_samples(pi, A, B):
    print(pi)
    print(A)
    print(B)

    model = FirstOrderHiddenMarkovModel(pi, A, B)
    for O, S in model.generateSamples(3, 5, 1):
        print(f'O={O}')
        print(f'身体状态（显）：{[observations[i] for i in O]}')
        
        print(f'S={S}')
        print(f'健康状态（隐）：{[states[i] for i in S]}')
        print(' ')


def train(pi, A, B):
    """
    通过设定的参数，来生成样本数据
    通过生成的样本数据，来训练（编码）出新的模型参数
    若两者对比，每个参数项差值不超过 0.01 ，就代表生成的模型样本是有效的
    这是只有隐马尔可夫模型才能生成样本还是其他模型也可以？
    """
    given_model = FirstOrderHiddenMarkovModel(pi, A, B)
    trained_model = FirstOrderHiddenMarkovModel()
    
    print('--------老模型参数-------')
    print(pi)
    print(A)
    print(B)

    # 最后的生成数量设置大一些，否则会造成取对数时参数为0的情况，而报错
    # 设置越大，两个模型的相似度越高
    samples = given_model.generateSamples(3, 10, 100000)
    trained_model.train(samples)
    
    assert trained_model.similar(given_model)

    trained_model.unLog()

    print('-----新模型参数---------')
    print(trained_model.start_probability)
    print(trained_model.transition_probability)
    print(trained_model.emission_probability)

def predict(pi, A, B, observations_index, observations_index_label, states_index_label):
    
    given_model = FirstOrderHiddenMarkovModel(pi, A, B)

    pred = [0, 0, 0]
    prod = given_model.predict(observations_index, pred)

    res = " ".join([(observations_index_label[o] + '/' + states_index_label[s]) for o, s in zip(observations_index, pred)])

    print(res, "{:.3f}".format(math.exp(prod)))

if __name__ == '__main__':
    states_label_index, states_index_label = generate_index_map(states)
    observations_label_index, observations_index_label = generate_index_map(observations)

    # 观察状态索引
    observations_index = convert_observations_to_index(observations, observations_label_index)
    
    # 初始化状态概率向量
    pi = convert_map_to_vector(start_probability, states_label_index)

    # 状态转移概率矩阵—A  发射概率矩阵—B
    A = convert_map_to_matrix(transition_probability, states_label_index, states_label_index)
    B = convert_map_to_matrix(emission_probability, states_label_index, observations_label_index)
    
    print(pi)
    print(A)
    print(B)

    #generate_samples(pi, A, B)
    #train(pi, A, B)
    #predict(pi, A, B, observations_index, observations_index_label, states_index_label)
