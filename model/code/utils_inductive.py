import numpy as np
import networkx
import torch
import os
from tqdm import tqdm
import pickle as pkl
from utils import dense_tensor_to_sparse


def get_related_nodes(adj_list, idx, k):
    def bfs_1_deep(adj, candidate):
        candidate = torch.LongTensor(list(candidate))
        ans = torch.sum(adj[candidate], dim=0)
        ans = ans.nonzero().squeeze()
        return set(ans.numpy())

    def bfs_k_deep(adj, candidate, k):
        candidate = torch.LongTensor(list(candidate))
        ans = set(candidate.numpy())
        next_candidate = candidate
        for i in range(k):
            next_candidate = bfs_1_deep(adj, next_candidate)
            next_candidate = next_candidate - ans
            ans.update(next_candidate)
        return ans

    adj = combine_adj_list(adj_list)
    involved_nodes = bfs_k_deep(adj, list(idx), k=k)
    ans = torch.LongTensor(list(involved_nodes), device=adj.device)
    return ans.sort()[0]


def combine_adj_list(adj_list):
    ans = []
    for i in adj_list:
        cache = []
        for j in i:
            cache.append(j.to_dense())
        ans.append(torch.cat(cache, dim=1))
    return torch.cat(ans, dim=0)


def transform_idx_for_adjList(adj_list, idx):
    N = len(adj_list)
    bias = 0
    ans = []
    for adj in adj_list[0]:
        shape = adj.shape
        idx_r = idx[(idx >= bias) & (idx < bias + shape[1])] - bias
        # idx_c = idx[(idx>=bias_c) & (idx<bias_c+shape[1])]
        bias += shape[1]
        ans.append(idx_r)
    return ans


def transform_idx_by_related_nodes(idx_mask, idx_query):
    a, b = idx_mask, idx_query
    c = [(a == t).nonzero()[0].item() for t in b if not (a == t).nonzero().nelement() == 0]
    return torch.LongTensor(c, device=idx_mask.device)


# 根据输入的related_idx, 返回新的对应的，adj, fea, idx
def transform_dataset_by_idx(adj_list, feature_list, related_idx, trans_idx, hop=1):
    related_nodes = get_related_nodes(adj_list, related_idx, hop)
    idx_new = transform_idx_for_adjList(adj_list, related_nodes)
    adj_list_new = []
    for r in range(len(adj_list)):
        adj_list_new.append([])
        for c in range(len(adj_list[r])):
            adj = adj_list[r][c]
            adj = adj.to_dense()[idx_new[r]].t()[idx_new[c]].t()
            adj_list_new[-1].append(dense_tensor_to_sparse(adj))

    feature_list_new = []
    for i in range(len(feature_list)):
        fea = feature_list[i]
        fea = fea.to_dense()[idx_new[i]]
        feature_list_new.append(dense_tensor_to_sparse(fea))

    output_idx = transform_idx_by_related_nodes(related_nodes, trans_idx)
    return adj_list_new, feature_list_new, related_nodes, output_idx


def get_entity(sentences):  # 不需要分词
    # sent_list(content) => entity_set( entity_name ), edge_list( (doc_idx, entity_name) )

    # path = "/home/ytc/GCN/整理的/HGAT/data/entity_recog/test.txt"
    rootpath = "../../data/entity_recog/"
    with open(rootpath + "test.txt", 'w') as f:
        f.write('\n'.join(sentences))
    origin_path = os.getcwd()
    os.chdir(rootpath)
    os.system("python ER.py --test=test.txt --output=output_file.txt")
    os.chdir(origin_path)

    edges = []
    entities = set()
    with open(rootpath + "output_file.txt", 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):
            l1, l2, l3 = lines[i:i + 3]
            entity_list = l2.strip('\n').split('\t')
            entities.update(set(entity_list))
            edges.extend([("test_{}".format(i//3), e) for e in entity_list])

    return edges


def get_topic(sentences, datapath):  # 需要分好词的
    # sent_list(content) => entity_set( topic_name: "topic_idx" ), edge_list( (doc_idx, topic_name) )
    TopK_for_Topics = 2

    def naive_arg_topK(matrix, K, axis=0):
        """
        perform topK based on np.argsort
        :param matrix: to be sorted
        :param K: select and sort the top K items
        :param axis: dimension to be sorted.
        :return:
        """
        full_sort = np.argsort(-matrix, axis=axis)
        return full_sort.take(np.arange(K), axis=axis)

    with open(datapath + "vectorizer_model.pkl", 'rb') as f:
        vectorizer = pkl.load(f)
    with open(datapath + "lda_model.pkl", 'rb') as f:
        lda = pkl.load(f)

    X = vectorizer.transform(sentences)
    doc_topic = lda.transform(X)
    topK_topics = naive_arg_topK(doc_topic, TopK_for_Topics, axis=1)

    topics = []
    for i in range(doc_topic.shape[1]):
        topicName = 'topic_' + str(i)
        topics.append(topicName)
    edges = []
    for i in range(topK_topics.shape[0]):
        for j in range(TopK_for_Topics):
            edges.append(("test_{}".format(i), topics[topK_topics[i, j]]))

    return edges


def build_subnetwork(datapath, entity_edges, topic_edges):
    graph_path = datapath + 'model_network_sampled.pkl'
    with open(graph_path, 'rb') as f:
        g = pkl.load(f)
    # g.add_edges_from(entity_edges)
    '''
    这个的目的是把新来的文本加入原图中，以便后续求交操作不会把他们剔除掉
    不加实体的边的原因是，会有新的实体引入，这部分新加的实体，由于没有对应的特征（和对应的参数矩阵），应当剔除
    可以加主题的边的原因是，主题没有新的，都是旧有的，不会引入新的主题节点
    '''
    g.add_edges_from(topic_edges)

    sub_g = networkx.Graph()
    sub_g.add_edges_from(entity_edges)
    sub_g.add_edges_from(topic_edges)
    return g, sub_g


def judge_node_type(node):
    if node[:5] == 'test_':
        return 'text'
    if node[:6] == 'topic_':
        return 'topic'
    return 'entity'



def release_feature(DATASET, node_list):
    # 实体的，最麻烦了…………
    mapindex = dict()
    with open('../data/{}/mapindex.txt'.format(DATASET), 'r') as f:
        for line in f:
            k, v = line.strip().split('\t')
            mapindex[k] = int(v)
    featuremap = dict()

    for node_type in ['entity', 'topic']:
        with open('../data/{0}/{0}.content.{1}'.format(DATASET, node_type), 'r') as f:
            for line in tqdm(f):
                cache = line.strip().split('\t')
                index = int(cache[0])
                feature = np.array(cache[1:-1], dtype=np.float32)
                featuremap[index] = feature

        with open('../data/{0}/{0}_inductive.content.{1}'.format(DATASET, node_type), 'w') as f:
            for n in range(len(node_list)):
                if judge_node_type(node_list[n]) == node_type:
                    f.write(
                        str(n) + '\t' + '\t'.join(map(str, featuremap[mapindex[node_list[n]]]))
                    + '\n')


def preprocess_inductive_text(sentences, DATASET):
    datapath = '../../data/{}/'.format(DATASET)

    entity_edges = get_entity(sentences)

    def tokenize(sen):
        from nltk.tokenize import WordPunctTokenizer
        return WordPunctTokenizer().tokenize(sen)
        # return jieba.cut(sen)
    sentences = [[word.lower() for word in tokenize(sentence)] for sentence in sentences]
    sentences = [' '.join(i) for i in sentences]
    topic_edges = get_topic(sentences, datapath)

    g, sub_g = build_subnetwork(DATASET, entity_edges, topic_edges)
    # 剔除不认识的实体
    sub_g.remove_nodes_from( set(sub_g.nodes()) - set(g.nodes()) )
    del g
    g = sub_g
    node_list = list(g.nodes())

    # build_feature
    release_feature(DATASET, node_list)
    with open(datapath + "vectorizer_model.pkl", 'rb') as f:
        vectorizer = pkl.load(f)
    with open(datapath + "transformer_model.pkl", 'rb') as f:
        tfidf_model = pkl.load(f)
    tfidf = tfidf_model.transform(vectorizer.transform(sentences))

    with open('../data/{0}/{0}_inductive.content.{1}'.format(DATASET, 'text'), 'w') as f:
        for n in range(len(node_list)):
            if judge_node_type(node_list[n]) == 'text':
                idx = int(node_list[n].split('_')[1])
                f.write(
                    str(n) + '\t' + '\t'.join(map(str, tfidf[idx, :].toarray()[0].tolist() ))
                + '\n')

    # build_adj
    adj_dict = g.adj
    with open('../data/{0}/{0}_inductive.cites'.format(DATASET), 'w') as f:
        for i in adj_dict:
            for j in adj_dict[i]:
                f.write('{}\t{}\n'.format(i, j))



if __name__ == '__main__':
    sent = [
        "VE子接口接入L3VPN转发，L3VE接口上下行均配置qos-profile模板，qos-profile中配置SQ加入GQ，然后进行主备倒换，查看配置恢复并且上下行流量按SQ参数进行队列调度和限速。查询和Reset SQ和GQ统计，检查统计和清除统计功能正常。",
        "整机启动时间48小时内、回退的文件不存在（大包文件不存在、或配置文件不在、或补丁文件不存在、或paf文件不存在），以上场景下，执行一键式版本回退，回退失败，提示对应Error信息",
        "全网1588时间同步;dot1q终结子接口终结多Q段接入VLL;IP FPM基本配置，丢包和时延使用默认值，测量周期1s;配置单向流，六元组匹配;MCP主备倒换3次",
        "集中式NAT444实例配置natserver，配置满规格日志服务器带不同私网vpn，反复配置去配置日志服务器，主备倒换设备无异常，检查反向建流和流表老化时，日志发送功能正常；修改部分日志服务器私网VPN与用户不一致，检查流表新建和老化时日志发送正常。",
        "2R native eth组网，配置基于物理口的PORT CC场景的1ag，使能eth-test测量。接收端设备ISSU升级过程中，发送端设备分别发送有bit错、CRC错、乱序的ETH-TEST报文，统计结果符合预期。配置周期为1s的，二层主接口的outward型的1ag，使能ETH_BN功能，测试仪模拟BNM报文，ISSU升级过程中，带宽通告可恢复，可产生。",
        "配置VPLS场景，接口类型为dot1q终结，设置检测接口为物理口，指定测试模式为丢包，手工触发平滑对账，查看测量结果符合规格预期",
        "NATIVE IP场景，测试仪模拟TWAMP LIGHT的发起端，设备作为TWAMP LIGHT的反射端，TWAMP LIGHT的时延统计功能正常，reset接口所在单板后，TWAMP LIGHT的时延统计功能能够恢复。",
        "接口上行配置qos-profile模板，模板中配置SQ，SQ关联的flow-queue模板中配置八个队列都为同种调度方式(lpq)，打不同优先级的流量，八种队列按照配置的调度方式进行正确调度",
        "满规格配置IPFPM逐跳多路径检测，中间节点设备主备倒换3次后，功能正常",
        "通道口上建立TE隧道，隧道能正常建立并UP，ping和tracert该lsp都通，流量通过TE转发无丢包，ping可设置发送的报文长度，长度覆盖65、4000、8100（覆盖E3 Serial口）",
        "双归双活EVPN场景，AC侧接口为eth-trunk，evc封装dot1q，流动作为pop single接入evpn，公网使用te隧道，配置ac侧mac本地化功能，本地动态学习mac后ac侧本地化学习，动态mac清除、ac侧本地化mac清除",
        "配置两条静态vxlan隧道场景，Dot1q子接口配置远程端口镜像,切换镜像实例指定pw隧道测试恢复vxlan隧道镜像,再子卡插拔测试",
        "单跳检测bdif，交换机接口shut down后BFD会话的状态变为down，接口重新up则BFD会话可以协商UP"
    ]
    preprocess_inductive_text(sent, 'hw')