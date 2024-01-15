"""
基础边，不允许构造
"""
class Edge:
   weight = 0

   name = ""

   def __init__(self, weight, name):
       self.weight = weight
       self.name = name
