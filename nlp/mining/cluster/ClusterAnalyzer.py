import math
import random

from nlp.mining.cluster.Document import Document
from nlp.mining.cluster.Cluster import Cluster

from collections import deque, defaultdict

class ClusterAnalyzer():
    NUM_PEFINE_LOOP = 10

    def __init__(self):
        self.documents = defaultdict(Document)

    def size(self):
        return len(self.documents.keys())
    
    def kmeans(self, nclusters):
        """
        K-means聚类
        """
        if nclusters > self.size():
            print("传入聚类数目大于文档数量")
            nclusters = self.size()

        # 所有文档纳入簇内
        cluster = Cluster()
        for document in self.documents.values():
            cluster.add_document(document)

        cluster.section(nclusters)
        
        self.refine_clusters(cluster.get_clusters())
        
        # 上面进行重新分簇后，需要重新获取簇内数据
        clusters = cluster.get_clusters()
        
        return self.toResult(clusters)


    def repeatedBisection(self, nclusters = 0, limit_eval = 0):
        """
        二分法聚类
        -param nclusters 簇的数量
        -param limit_eval 准则函数增幅阈值
        return 指定数量的簇构成的集合（Set)
        """
        if nclusters > self.size():
            print("传入聚类数目大于文档数量")
            nclusters = self.size()
        
        # 所有文档纳入簇内
        cluster = Cluster()
        for document in self.documents.values():
            cluster.add_document(document)

        que = deque()
        
        # 切分两部分
        sectioned_clusters = cluster.section(2)
        
        # 根据K-means算法迭代优化聚类, 在这里会改变簇的数据分布
        self.refine_clusters(sectioned_clusters)
        
        # 设置分段增益
        cluster.set_sectioned_gain()
        
        # 清空复合向量的数据？？？
        cluster.composite_vector().clear()

        que.append(cluster)
        
        # 通过队列的方式，广度优先遍历所有簇进行不断的聚类
        while que:
            if nclusters > 0 and len(que) >= nclusters:
                break

            cluster = que[0]
            if len(cluster.get_clusters()) < 1:
                break

            #print('gain', cluster.get_sectioned_gain())
            if limit_eval > 0 and cluster.get_sectioned_gain() < limit_eval:
                break

            cluster = que.popleft()
            sectioned = cluster.get_clusters()
            
            # 若簇的大小，大于2，继续二分
            for c in sectioned:
                #print('size', c.size())
                if c.size() >= 2:
                    sectioned_clusters = c.section(2)
                    self.refine_clusters(sectioned_clusters)
                    
                    c.set_sectioned_gain()

                    if c.get_sectioned_gain() < limit_eval:
                        for sub in c.get_clusters():
                            sub.clear()

                    c.composite_vector().clear()

                que.append(c)
        
        clusters = []
        while que:
            clusters.insert(0, que.popleft())

        return self.toResult(clusters)


    def refine_clusters(self, clusters = list):
        """
        根据K-means算法(K均值算法)迭代优化聚类, 使用复合向量的方式来替换之前的均值算法，目的就是找文档在所有簇中最大的增量值，之所以找最大化，是因为在两个向量相似度比较高的情况下，把它们相同的特征进行加权（*2）后，它们的增量就会增大（不同特征不会加权）。
        通过计算每个文档在每个簇的增量，获取最大化的增量，便将其文档移动到最大化增量的簇内
        -param clusters 簇
        return 准则函数的值
        """
        
        # 计算簇的质心(标量，合成向量的范数)
        norms = []
        for cluster in clusters:
            norms.append(cluster.composite_vector().norm())
        
        loop_count, eval_cluster = 0, 0.0
        while loop_count < ClusterAnalyzer.NUM_PEFINE_LOOP:
            
            loop_count += 1

            # 展开簇内的所有文档
            items = []
            for i in range(len(clusters)):
                documents_ = clusters[i].get_documents()
                for j in range(len(documents_)):
                    #print(i, j, documents_[j])
                    items.append([i, j])

            random.shuffle(items)
            
            changed = False
            for cluster_id, item_id in items:
                
                cluster = clusters[cluster_id]

                doc = cluster.documents[item_id]
                
                # 计算目标文档从其所属簇移除后的向量变化, composite_vector 是复合向量，一个簇内所有文档特征都包含在内
                value_base = self.refine_vector_value(cluster.composite_vector(), doc.feature(), -1)
                
                # 最终获得是当前簇内所有文档向量(复合向量)的范数，即doc在当前簇的增量
                # 当前簇质心的平方 + 当前文档与簇（复合向量）相似度，后开平方
                norm_base_moved = math.pow(norms[cluster_id], 2) + value_base
                norm_base_moved = math.sqrt(norm_base_moved) if norm_base_moved > 0 else 0.0
                
                # 找到doc移动到其他簇最大增量
                eval_max, norm_max, max_index = -1.0, 0.0, 0
                for j, other in enumerate(clusters):
                    if cluster_id == j:
                        continue

                    # 计算目标文档移动到新的簇的增幅, composite_vector 是复合向量，一个簇内所有文档特征都包含在内
                    value_target = self.refine_vector_value(other.composite_vector(), doc.feature(), 1)
                    
                    norm_target_moved = math.pow(norms[j], 2) + value_target
                    norm_target_moved = math.sqrt(norm_target_moved) if norm_base_moved > 0 else 0.0
                    
                    # 增幅计算
                    eval_moved = norm_base_moved + norm_target_moved - norms[cluster_id] - norms[j]
                    if eval_max < eval_moved:
                        eval_max = eval_moved
                        norm_max = norm_target_moved

                        max_index = j

                if eval_max > 0:
                    eval_cluster += eval_max
                    
                    # 在这里修改了所有簇的内部向量分布, 并没有真正的删除
                    clusters[max_index].add_document(doc)
                    clusters[cluster_id].remove_document(item_id)

                    norms[cluster_id] = norm_base_moved
                    norms[max_index]  = norm_max

                    changed = True

            if not changed:
                break
            
            # 真正的删除在这里
            for cluster in clusters:
                cluster.refresh()

        return eval_cluster


    def refine_vector_value(self, composite, vec, sign) -> float:
        """
        改善向量的值, c^2 - 2c(a + c) + d^2 - 2d(b + d)
        下面的代码，实现了 c^2 - 2c(a+c) ，并且进行了向量的累加 ：
            1，a + c 是复合向量中的一个数据，这是代表复合向量实际上是其他多个向量相加而得到的
            2，c^2 表示文档向量的欧几里得长度
            3，2c(a+c) 表示两个向量的加权内积，两个向量的相似性

        -param composite 复合向量 (a + c, b + d)
        -param vec 文档向量 (c, d)
        -param sign [1, -1]
        """
        sum_ = 0.0
        for key, val in vec.entrySet():
            sum_ += math.pow(val, 2) + sign * 2 * composite.get(key) * val
        
        return sum_
    
    def toResult(self, clusters):
        """返回的是文档id"""
        result = []
        
        for c in clusters:
            
            s = set()
            for d in c.documents:
                s.add(d.id)
            
            result.append(s)

        return result


