import random
from nlp.mining.cluster.SparseVector import SparseVector

class Cluster():
    def __init__(self, documents = None):
        self.documents = []
        self.sectioned_clusters = []
        self.sectioned_gain = 0.0

        self.composite = SparseVector()

        if documents:
            self.documents = documents
    
    def size(self):
        return len(self.documents)

    def get_clusters(self):
        return self.sectioned_clusters
    
    def get_documents(self):
        return self.documents

    def add_document(self, doc):
        """注意在这里对文档的向量进行了归一化处理"""
        doc.feature().normalize()
        
        self.documents.append(doc)
        
        # 复合向量, 将文档向量添加到复合向量
        self.composite.add_vector(doc.feature())
    
    def remove_document(self, index):
        """不能真正的移除，因为在分簇过程中要保证文档数量的正确，否则refine_clusters()会报错"""
        doc = self.documents[index]
        
        self.documents[index] = None
        
        # 复合向量, 将文档向量移除复合向量
        self.composite.sub_vector(doc.feature())

    def refresh(self):
        self.documents = list(filter(lambda x: x != None, self.documents))

    def set_sectioned_gain(self):
        """设置分段增益"""
        gain = 0.0

        if self.sectioned_gain == 0.0 and len(self.sectioned_clusters) > 1:
            
            for cluster in self.sectioned_clusters:
                gain += cluster.composite_vector().norm()

            gain -= self.composite.norm()

        self.sectioned_gain = gain

    def get_sectioned_gain(self):
        return self.sectioned_gain

    def composite_vector(self):
        """获取复合向量"""
        return self.composite

    def section(self, nclusters):
        """将本簇划分为nclusters个簇"""

        if self.size() < nclusters:
            raise Exception("簇数目小于文档数目")
        
        # 选取质心
        centroids = [] * nclusters             
        self.choose_smartly(nclusters, centroids)
        
        for i in range(len(centroids)):
            cluster = Cluster()
            self.sectioned_clusters.append(cluster)
        
        # 计算所有文档与质心的距离
        for d in self.documents:
            max_similarity = -1.0
            max_index = 0
            
            # 找出文档在所有质心中，最大相似度的进行聚类
            for j in range(len(centroids)):
                similarity = SparseVector.inner_product(d.feature(), centroids[j].feature())

                if max_similarity < similarity:
                    max_similarity = similarity
                    max_index = j

            self.sectioned_clusters[max_index].add_document(d)

        return self.sectioned_clusters

    def choose_smartly(self, ndocs = int, docs = list):
        """选取初始质心"""
        size = self.size()

        if size < ndocs:
            ndocs = size

        index, count = 0, 0
    
        # 随机选取质心 index
        index = random.randrange(size)
        docs.append(self.documents[index])

        count += 1
        
        # 保存计算所有文档与初始质心的距离，并对距离求和
        potential, closest = 0.0, []
        for i in range(len(self.documents)):
            dist = 1.0 - SparseVector.inner_product(self.documents[i].feature(), self.documents[index].feature())
            #print(f'{i}', self.documents[i].feature().toString())
            #print('dist', dist)
            #print(' ')

            potential += dist
            closest.append(dist)

        # 选取质心, 按照给定的数量分簇选取
        while count < ndocs:
            # 加权选择，将随机浮点数([0,1]之间)与potential相乘，得到一个介于 0 和 potential 之间的随机数
            randval = random.random() * potential
            
            # 注意，这里index的值，跳出循环后，继续使用
            for index in range(len(self.documents)):
                dist = closest[index]
                if randval <= dist:
                    break

                randval -= dist

            if index == len(self.documents):
                index -= 1

            docs.append(self.documents[index])
            count += 1
            
            # index 发生了改变，需要重新计算每个文档与质心的距离
            # 特别注意，这里会把文档之前的准则数与新质心之间的准则数进行对比，保留最小值
            new_potential = 0.0
            for i in range(len(self.documents)):
                dist = 1.0 - SparseVector.inner_product(self.documents[i].feature(), self.documents[index].feature())
                min_ = closest[i]
                
                if dist < min_:
                    closest[i] = dist
                    min_ = dist

                new_potential += min_

            potential = new_potential

    def clear(self):
        self.documents = []
        self.sectioned_clusters = []
        self.sectioned_gain = 0.0

        self.composite.clear()


