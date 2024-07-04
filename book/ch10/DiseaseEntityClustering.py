import sys
sys.path.append('/pynjlp')

from nlp.mining.cluster.CharacterClusterAnalyzer import CharacterClusterAnalyzer
from nlp.corpus.io.IOUtil import toExcel, readExcel

def get_object_entity_name_list():
    return ['胚胎型横纹肌肉', '间变大细胞淋巴瘤alk阳', '骨肉瘤', '淋巴瘤', '横纹肌肉', '经典型霍奇金林巴瘤', '滤泡淋巴瘤', '盆腔横纹肌肉', '左下肺恶性肿瘤', '盆腔内胚窦瘤', '右侧睾丸胚胎型横纹肌肉瘤', '前列腺胚胎型横纹肌肉瘤', 'b细胞性淋巴母细胞淋巴瘤', '甲状腺乳头状癌', '急性粒细胞白血病', '胚胎发育不良性神经上皮肿瘤', '急性髓细胞白血病', '间变大细胞淋巴瘤', '急性粒细胞性白血病', '结节硬化型霍奇金淋巴瘤', '滤泡性淋巴瘤', '肿瘤', '恶性混合生殖细胞肿瘤', '左侧股骨中下段骨肉瘤', '卵巢生殖细胞肿瘤', 'burkitt淋巴瘤', '卵巢未成熟性畸胎瘤', '急性pro-b淋巴细胞白血病', '急性髓系白血病', '结节硬化型经典霍奇金淋巴瘤', '右侧肩胛骨恶性肿瘤', '恶性横纹肌样瘤', '肾细胞癌', '中枢神经系统胚胎性肿瘤', '横纹肌肉瘤', '毛细胞星型细胞瘤', '神经胶质瘤', '急性淋巴细胞性白血病', '左肾尤文肉瘤', '脑膜瘤', '室管膜母细胞瘤', '脉络丛乳头状瘤', '肝母细胞瘤', '前体t细胞淋巴母细胞淋巴性淋巴瘤', '纵隔精原细胞瘤', '非典型畸胎瘤样/横纹肌样肿瘤', '肝脏恶性肿瘤', '结肠低分化腺癌', '急性pre-b淋巴细胞白血病', '急性淋巴细胞白细胞', 'nk/t细胞淋巴瘤', '横纹肌肉瘤（腺泡型）', '不典型畸胎样/横纹样肿瘤', '混合生殖细胞瘤', '伯基特淋巴瘤', '肝癌', '星形细胞瘤', '成熟b细胞淋巴瘤', '下颌骨尤文氏肉瘤', '左侧卵巢神经内分泌肿瘤', '卵黄囊瘤', '腹腔混合性生殖细胞瘤', '小细胞恶性肿瘤', '胶质神经元混合性肿瘤', '神经上皮肿瘤', '右侧颞叶中枢神经系统恶性肿瘤', '弥漫大b细胞性淋巴瘤', 'nk/t细胞淋巴瘤，鼻型', '盆腔恶性肿瘤', '神经母细胞瘤', '室管膜癌', '畸胎瘤', '急性非淋巴性白血病', '鼻咽非角化性鳞癌', '胸腺恶性肿瘤差分化癌', '左侧卵巢卵黄素', '内胚窦瘤', '右侧睾丸内胚窦瘤', '霍奇金病', '卵巢混合型恶性肿瘤', '卵巢粘液性癌', '毛细胞星形细胞瘤', '子宫内膜癌肉瘤', '生殖细胞瘤', '混合性生殖细胞瘤', '混合细胞型霍奇金淋巴瘤', '未成熟性畸胎瘤', '间变大细胞性淋巴瘤', '骶尾部恶性肿瘤', '典型畸胎瘤/横纹肌样瘤', '急性普通b淋巴细胞白血病', '非典型畸胎样/横纹肌样肿瘤', '急性t淋巴细胞白血病', '非霍奇金淋巴瘤', '急性髓系细胞白血病', '高侵袭性b细胞淋巴瘤', '胚胎发育不良性神经上皮性肿瘤', '右眼眶梭形细胞横纹肌肉瘤', '大b细胞淋巴瘤', '盆腔伯基特淋巴瘤', '卵巢恶性肿瘤', '脑胶质间变少突细胞瘤', '骶尾部恶性畸胎瘤', '急性非淋巴白血病', '毛细胞型星形细胞瘤', '滑膜肉瘤', '鼻咽非角化性癌', '肾母细胞瘤', '腺样囊性癌', '肾透明细胞肉瘤', '间皮瘤', '股骨骨肉瘤', '髓上皮瘤', '急性非淋巴细胞白血病', '盆腔恶性肿瘤：未成熟畸胎瘤', '急性单核细胞白血病', '胶质母细胞瘤', '鼻咽恶性肿瘤', '急性早幼粒细胞白血病', '睾丸内胚窦瘤', '间变型室管膜瘤', '直肠癌', '卵巢交界性浆液性肿瘤', '直肠腺癌', '胚胎性肿瘤', '脊膜瘤', '非典型脑膜瘤', '尤文氏肉瘤', '宫颈透明细胞癌', '甲状腺弥漫硬化性乳头状癌', '卵巢恶性生殖细胞肿瘤', '弥漫中线胶质瘤', '髓母细胞瘤', '肱骨骨肉瘤', '未成熟畸胎瘤', '经典型霍奇金淋巴瘤', '幼年型粒层细胞瘤', '膀胱胚胎型横纹肌肉瘤', '混合性生殖细胞肿瘤', '原始神经外胚瘤', '节细胞胶质瘤', '间变性室管膜瘤', '股骨近端尤文肉瘤', '卡波西肉瘤', '粘膜相关淋巴组织结外边缘区淋巴瘤', '胸膜肺母细胞瘤', '左肺下叶肺恶性肿瘤', '弥漫性胶质瘤', '急性淋巴细胞白血病', '急性前b淋巴细胞白血病', '非典型畸胎样/横纹肌样瘤', '混合型生殖细胞瘤', '胸腺瘤', '经典霍奇金淋巴瘤', '肺恶性肿瘤', '不成熟畸胎瘤', '肱骨近端骨肉瘤', '胸壁原始神经外胚瘤', '室管膜瘤', '视网膜母细胞瘤', '肝脏未分化肉瘤（胚胎型）', '绒毛膜癌', '骶尾部内胚窦瘤', '急性粒-单核细胞白血病', '少突胶质细胞瘤', '双侧颈部淋巴结继发恶性肿瘤', '随母细胞瘤', '卵巢未成熟畸胎瘤', '双侧股骨远端股肉瘤', '腹腔恶性肿瘤', '尤因肉瘤', '腺泡状横纹肌肉瘤', '卵巢内胚窦瘤', '纵膈精原细胞瘤', 'b细胞淋巴瘤', '高级别b细胞淋巴瘤', '睾丸卵黄囊瘤', '透明细胞肉瘤', '软骨肉瘤', '原始粒细胞白血病', '肝脏未分化肉瘤胚胎型', '腹壁内胚窦瘤', '左肺低分化鳞癌', '尤因氏肉瘤', '尤文肉瘤', '右侧大腿近端骨外尤因肉瘤', '高级别胶质瘤', '骨外尤因肉瘤', '双眼视网膜母细胞瘤', '卵巢无性细胞瘤', '结肠恶性肿瘤', '星型细胞瘤', '纵膈卵黄囊瘤', '差分化癌', '胚胎性横纹肌肉瘤', '颅内胚胎性肿瘤', '胚胎型横纹肌肉瘤', '肾癌', '间变性大细胞淋巴瘤', '右侧胫骨骨肉瘤', '颈胸椎尤因肉瘤', '卵巢卵黄囊瘤', '弥漫大b细胞淋巴瘤', 't淋巴母细胞淋巴瘤', '胸腺恶性肿瘤', '脑干胶质瘤', '膀胱胚胎性横纹肌肉瘤', '霍奇金淋巴瘤', '急性非淋巴细胞性白血病', '右侧卵巢无性生殖细胞瘤', '霍奇金淋巴瘤(混合细胞型)', 't淋巴母细胞白血病/淋巴瘤', '非成熟性畸胎瘤', '不成熟性畸胎瘤', '毛细胞星型胶质细胞瘤', '腰椎尤文氏肉瘤']

def get_subject_entity_name_list():
    return ['who4级', 'ic期', 'stjude为ⅳ期', '分期为iv期', 'e期', 'ⅰb期', '高危', 'm2', 'ivs期', '中危组', 'ⅲa期', 'm2型', 'who-4级', 'iv级', 'iii期', 'l2型', 'who-i级', 'c期', 'iv期', 'ia1期', 'iva期', 'who：4级', 'iea期', 'ⅲ期', 'i期', 'ic2期', '术后iv期', 'who，4级', 'ⅱ级', '1期', 'ib期', 'iie期', '超高危组', 'iiia期', '高危组', 'iiib期', 'iiieb期', 'ⅰ级', 'who2级', 'who，3级', 'd→e期', 'whoiii级', 'pretestiii期', 'ⅰc2期', 'whoⅰ级', 'd期', 'i级', 'ⅱ期', 'ⅳb期', '低危', 'who-iii级', 'who-1级', 'l1型', 'ii级', '低危组', 'who-ⅳ级', 'iib期', 'ic1期', 'ia期', 'iiic期', 'ⅲb期', '低度恶性', 'who-iv级', '3级', 'iii级', 'iia期', 'whoiv级', 'b期', 'who1级', 'm3型', 'ⅱa期', 'ii期', 'whoii级', 'm5型', 'm1型', 'whoⅳ级', 'ⅳ期', 'whoi级', 'l3型', 'ⅰ期', 'whoⅱ级', 'm4型', 'ivb期', 'ⅲc期', 'm5', '中危', 'l2', 'ⅲ级', 'ⅱb期', 'who分型：ⅳ级', 'm4', 'ⅳ级', 'whoⅲ级', 'who3级']
 

def cluster():
    words = get_object_entity_name_list()

    analyzer = CharacterClusterAnalyzer()
    analyzer.addWords(words)
    
    #print(words)

    for i in range(1):
        res = analyzer.repeatedBisection(limit_eval = 0.01)
        #res = analyzer.kmeans(50)

        values = []
        for ids in res:
            word_list = [words[id] for id in ids]
            values.append(','.join(word_list))
        
        print(values)
        print('分簇总数：', len(res))

        #toExcel('./cluster_object.xlsx', values, ['实体名称'])

def analyze_icc3():
    icc3_list = readExcel('./output/icc3.xlsx', [True])
    disease_cluster_list = readExcel('./output/cluster_subject_50.xlsx', [True])
    
    icc3_list = [icc3 for row in icc3_list for icc3 in row]
    
    results = []
    for row in disease_cluster_list:
        icc3 = []
        for disease in row[0].split(','):
            
            if disease in icc3_list:
                icc3.append(icc3_list[icc3_list.index(disease)])
        
        results.append((row[0], ','.join(icc3)))

    #toExcel('./cluster_subject_icc3_50.xlsx', results, ['实体名称', 'icc3名称'])
        
if __name__ == '__main__':
    cluster()
    #analyze_icc3()


