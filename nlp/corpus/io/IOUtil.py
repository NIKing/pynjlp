import os
import numpy as np
import pickle

import pandas as pd
import xlsxwriter

from nlp.corpus.document.sentence.Sentence import Sentence

def loadDictionary(path, splitter = '\t', defaultNature = None):
    if not path:
        return []

    line_list = readlinesTxt(path)
    storage = {}

    for line in line_list:
        param = line.rstrip('\n').split(splitter)
        
        # 每个关键词存在多组属性（以词性和词频为组），计算关键词存在特征总数
        natureCount = int((len(param) - 1) / 2)
        key = str(param[0])
        if natureCount == 0:
            storage[key] = defaultNature
            continue
        
        nature, frequency, totalFrequency = [], [], 0
        for i in range(natureCount):
            nature.append(param[(1 + 2 * i)])
            frequency.append(param[(2 + 2 * i)])
            totalFrequency += int(frequency[i])
        
        storage[key] = { 'nature': nature, 'frequency': frequency, 'totalFrequency': totalFrequency }
    
    return storage

def readlinesTxt(filePath):
    if not filePath:
        return []
    
    line_list = []
    try:
        with open(filePath, 'r', encoding='utf8') as file:
            line_list = file.readlines()
    except Exception as e:
        print('读取失败', e)
    
    #return line_list
    return [w.rstrip('\n') for w in line_list]

def writeTxtByList(filePath, dataList): 
    if not filePath or len(dataList) <= 0:
        return

    try:
        with open(filePath, "w", encoding='utf8') as file:
           file.writelines('\n'.join([str(d) for d in dataList]))
    except IOError as e:
        print('写入失败', e)


def getFileList(folderPath):
    if not folderPath:
        return []
    
    if os.path.isfile(folderPath):
        return [folderPath]

    fileList = []
    try:
        fileList = os.listdir(folderPath)
        fileList = [folderPath +'/'+ file for file in fileList]
    except IOError as e:
        print(f'{e}')

    return fileList


def lineIterator(filePath):
    if not filePath:
        return iter([None])

    line_list = readlinesTxt(filePath)
    return iter(line_list)

def loadInstance(path):
    lineList = readlinesTxt(path)
    
    sentenceList = []
    for line in lineList:
        line = line.strip()

        if len(line) <= 0:
            continue

        sentenceList.append(Sentence.create(line))

    return sentenceList

def writeListToBin(path, listData):
    """保存list数据到.bin文件"""
    array = np.array(listData, dtype=object)
    
    with open(path, 'wb') as f:
        f.write(pickle.dumps(array))

def readBinToList(path):
    """读取bin文件""" 
    arr_bytes = []
    
    with open(path, 'rb') as f:
        arr_bytes = f.read()
        arr_bytes = pickle.loads(arr_bytes)
    
    print('readBin', arr_bytes[:20])
    return arr_bytes

def toExcel(path, data, columns):
    """
    写入数据到文件
    :param data 保存的源数据
    :param columns 数据的列项
    :param path 保存的文件名称
    """
    if not data or not path:
        return

    data_frame = pd.DataFrame(data, columns = columns)
    data_frame.to_excel(path, engine='xlsxwriter', index=False)

def readExcel(path, columns) -> list:
    """
    读取excel文件
    -param path 读取路径
    -param columns 需要读取的文件列
    return list
    """
    if not path or not columns:
        return []

    data = pd.read_excel(path, keep_default_na=False).loc[:,columns].values

    return data
