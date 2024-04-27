import numpy as np
import pickle

def float_to_bytes():
    _list = [3, 2, 0.0]

    _arr  = np.array(_list)
    _arr_bytes = _arr.tobytes()

    print(_arr)
    print(_arr_bytes)

    res = np.frombuffer(_arr_bytes, dtype=np.float64, count=1)
    print(res)

def uint8_to_bytes():
    _list = [3, 2, b'\xe7\x94\xb7', 0.0]

    _arr  = np.array(_list, dtype=np.uint32)
    _arr_bytes = _arr.tobytes()

    print(_arr)
    print(_arr_bytes)

    res = np.frombuffer(_arr_bytes, dtype=np.uint8, count=1)
    print(res)

def object_pickle_to_bytes():
    _list = [3, 2, '男', 0.0]

    _arr  = np.array(_list, dtype=object)
    #_arr_bytes = _arr.tobytes()
    _arr_bytes = pickle.dumps(_arr)
    print(_arr)
    print(_arr_bytes)

    #one = np.frombuffer(_arr_bytes, dtype=object, count=0)
    res = pickle.loads(_arr_bytes)
    print(res)

def object_to_bytes():
    _list = [3, 2, '男', 0.0]

    _arr  = np.array(_list, dtype=object)
    _arr_bytes = _arr.tobytes()
    print(_arr)
    print(_arr_bytes)

    res = np.frombuffer(_arr_bytes, count=0)
    print(res)

if __name__ == '__main__':
    
    #int_to_bytes()
    #float_to_bytes()
    #uint8_to_bytes()
    
    object_to_bytes()


