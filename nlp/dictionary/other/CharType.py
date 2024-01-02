class CharType:
    # 单字节
    CT_SINGLE = 5
    
    # 分隔符"!,.?()[]{}+=
    CT_DELIMITER = CT_SINGLE + 1

    #  中文字符
    CT_CHINESE = CT_SINGLE + 2

    # 字母
    CT_LETTER = CT_SINGLE + 3

    # 数字
    CT_NUM = CT_SINGLE + 4
    
    # 序号
    CT_INDEX = CT_SINGLE + 5

    # 中文数字
    CT_CNUM = CT_SINGLE + 6

    # 其他
    CT_OTHER = CT_SINGLE + 12
