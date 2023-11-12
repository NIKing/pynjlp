

def createTinyDictionary():
    return {
        "自然": "nature",
        "自然人": "human",
        "自然语言": "language",
        "自语": "talk to oneself",
        "入门": "introduction"
    }

if __name__ == '__main__':

    tinyDictionary = createTinyDictionary()
    db = DoubleArrayTrie(tinyDictionary)
