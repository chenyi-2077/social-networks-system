#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>   // 用于time(NULL)
#include <float.h>  // 用于DBL_MAX（双精度浮点最大值）




// 邻接表节点结构（存储邻接用户和边权重）
typedef struct AdjNode {
    int vertex;               // 邻接用户ID（节点编号，如A对应0、B对应1...E对应4）
    double weight;            // 边权重（互动次数倒数，衡量"距离"）
    struct AdjNode* next;     // 下一个邻接节点
} AdjNode;

// 邻接表头结构（对应单个用户节点）
typedef struct AdjList {
    AdjNode* head;            // 邻接表头部指针
} AdjList;

// 图结构（包含所有用户节点的邻接表）
typedef struct Graph {
    int numVertices;          // 图中节点总数（用户数量）
    AdjList* array;           // 邻接表数组
} Graph;

// 创建邻接表节点
AdjNode* createAdjNode(int v, double weight) {
    AdjNode* newNode = (AdjNode*)malloc(sizeof(AdjNode));
    if (newNode == NULL) {     // 新增：内存分配失败判断
        printf("内存分配失败！\n");
        exit(1);
    }
    newNode->vertex = v;
    newNode->weight = weight;
    newNode->next = NULL;
    return newNode;
}

// 创建图（初始化邻接表）
Graph* createGraph(int vertices) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    if (graph == NULL) {       // 新增：内存分配失败判断
        printf("内存分配失败！\n");
        exit(1);
    }
    graph->numVertices = vertices;
    graph->array = (AdjList*)malloc(vertices * sizeof(AdjList));
    if (graph->array == NULL) { // 新增：内存分配失败判断
        printf("内存分配失败！\n");
        free(graph);
        exit(1);
    }

    for (int i = 0; i < vertices; i++) {
        graph->array[i].head = NULL;
    }
    return graph;
}

// 添加无向边（好友关系双向）
void addEdge(Graph* graph, int u, int v, double weight) {
    if (graph == NULL || u < 0 || v < 0 || u >= graph->numVertices || v >= graph->numVertices) {
        printf("无效的图或节点编号！\n"); // 新增：参数合法性判断
        return;
    }
    // 添加 u -> v 的边
    AdjNode* newNode = createAdjNode(v, weight);
    newNode->next = graph->array[u].head;
    graph->array[u].head = newNode;

    // 添加 v -> u 的边（无向图双向存储）
    newNode = createAdjNode(u, weight);
    newNode->next = graph->array[v].head;
    graph->array[v].head = newNode;
}

// Dijkstra算法结果存储结构
typedef struct DijkstraResult {
    double* dist;      // 源点到各节点的最短距离（d(E)即对应此数组的E节点索引值）
    int* prev;         // 各节点的前驱节点索引，用于回溯路径
} DijkstraResult;

// 初始化Dijkstra结果
DijkstraResult* initDijkstraResult(int vertices) {
    DijkstraResult* res = (DijkstraResult*)malloc(sizeof(DijkstraResult));
    if (res == NULL) {         // 新增：内存分配失败判断
        printf("内存分配失败！\n");
        exit(1);
    }
    res->dist = (double*)malloc(vertices * sizeof(double));
    res->prev = (int*)malloc(vertices * sizeof(int));
    if (res->dist == NULL || res->prev == NULL) { // 新增：内存分配失败判断
        printf("内存分配失败！\n");
        free(res->dist);
        free(res->prev);
        free(res);
        exit(1);
    }

    for (int i = 0; i < vertices; i++) {
        res->dist[i] = DBL_MAX;  // 修改：用DBL_MAX替代INT_MAX，类型更匹配
        res->prev[i] = -1;       // 前驱节点初始化为-1（无前驱）
    }
    return res;
}

// 标签传播算法结果结构
typedef struct LPAResult {
    int* labels;       // 节点-社区标签映射（labels[i]表示节点i的社区标签）
    double modularity; // 最终社区划分的模块度Q值
    int numCommunities;// 最终社区数量
    int iter;          // 最终迭代次数
    int isStable;      // 是否收敛
} LPAResult;

// ===================== 函数前置声明 =====================
// 释放图内存
void freeGraph(Graph* graph);
// 释放Dijkstra结果内存
void freeDijkstraResult(DijkstraResult* res);
// 释放LPA结果内存
void freeLPAResult(LPAResult* res);
// Dijkstra算法声明（若需也可添加，此处主要解决释放函数问题）
DijkstraResult* dijkstra(Graph* graph, int start);
// 标签传播算法声明
LPAResult* labelPropagation(Graph* graph, int maxIter);
// ============================================================

// 初始化标签传播结果
LPAResult* initLPAResult(int vertices) {
    LPAResult* res = (LPAResult*)malloc(sizeof(LPAResult));
    if (res == NULL) {         // 新增：内存分配失败判断
        printf("内存分配失败！\n");
        exit(1);
    }
    res->labels = (int*)malloc(vertices * sizeof(int));
    if (res->labels == NULL) { // 新增：内存分配失败判断
        printf("内存分配失败！\n");
        free(res);
        exit(1);
    }
    // 初始化：每个节点赋予唯一标签（标签值=节点ID）
    for (int i = 0; i < vertices; i++) {
        res->labels[i] = i;
    }
    res->modularity = 0.0;
    res->numCommunities = vertices;
    return res;
}

// 找到未访问节点中距离最小的节点
int minDistance(double dist[], int visited[], int vertices) {
    double min = DBL_MAX;      // 修改：用DBL_MAX替代INT_MAX，类型更匹配
    int min_index = -1;

    for (int v = 0; v < vertices; v++) {
        if (visited[v] == 0 && dist[v] <= min) {
            min = dist[v];
            min_index = v;
        }
    }
    return min_index;
}

// 回溯生成路径（递归打印，反向输出→正向输出）
void printPath(int prev[], int v,const char* nodeNames[]) {
    if (prev[v] == -1) { // 递归终止：源点
        printf("%s", nodeNames[v]);
        return;
    }
    printPath(prev, prev[v], nodeNames);
    printf("→%s", nodeNames[v]);
}

// Dijkstra算法：求解源点start到所有节点的最短路径
DijkstraResult* dijkstra(Graph* graph, int start) {
    if (graph == NULL || start < 0 || start >= graph->numVertices) {
        printf("无效的图或源点编号！\n"); // 新增：参数合法性判断
        return NULL;
    }
    int vertices = graph->numVertices;
    DijkstraResult* res = initDijkstraResult(vertices);
    int* visited = (int*)malloc(vertices * sizeof(int)); // 访问标记：1=已确定最短路径，0=未确定
    if (visited == NULL) {     // 新增：内存分配失败判断
        printf("内存分配失败！\n");
        freeDijkstraResult(res);
        exit(1);
    }

    // 初始化
    res->dist[start] = 0.0;
    memset(visited, 0, vertices * sizeof(int));

    for (int i = 0; i < vertices - 1; i++) {
        // 选择当前距离最小的未访问节点u
        int u = minDistance(res->dist, visited, vertices);
        if (u == -1) break; // 无可达节点，提前终止

        visited[u] = 1; // 标记u为已访问（最短路径确定）

        // 更新u的所有邻接节点的距离
        AdjNode* temp = graph->array[u].head;
        while (temp != NULL) {
            int v = temp->vertex;
            double weight = temp->weight;

            // 松弛操作：若经过u到v的距离更短，则更新（修改：用DBL_MAX判断）
            if (visited[v] == 0 && res->dist[u] != DBL_MAX && res->dist[u] + weight < res->dist[v]) {
                res->dist[v] = res->dist[u] + weight;
                res->prev[v] = u;
            }
            temp = temp->next;
        }
    }

    free(visited);
    return res;
}

// 统计邻居标签频率，返回最高频率的标签
int getMostFrequentLabel(Graph* graph, int node, int* labels) {
    if (graph == NULL || labels == NULL || node < 0 || node >= graph->numVertices) {
        printf("无效的图、标签数组或节点编号！\n");
        return -1;
    }
    int vertices = graph->numVertices;
    int* freq = (int*)calloc(vertices, sizeof(int));
    if (freq == NULL) {
        printf("内存分配失败！\n");
        exit(1);
    }
    int maxFreq = 0;
    // 1. 统计邻居标签频率，找到最大频率
    AdjNode* temp = graph->array[node].head;
    while (temp != NULL) {
        int neighbor = temp->vertex;
        freq[labels[neighbor]]++;
        if (freq[labels[neighbor]] > maxFreq) {
            maxFreq = freq[labels[neighbor]];
        }
        temp = temp->next;
    }

    // 2. 收集所有达到最大频率的标签
    int* maxFreqLabels = (int*)malloc(vertices * sizeof(int));
    if (maxFreqLabels == NULL) {
        printf("内存分配失败！\n");
        free(freq);
        exit(1);
    }
    int count = 0;
    for (int i = 0; i < vertices; i++) {
        if (freq[i] == maxFreq && maxFreq > 0) { // 只收集有效高频标签
            maxFreqLabels[count++] = i;
        }
    }

    int targetLabel = labels[node]; // 默认保留原标签
    // 3. 若存在高频标签，随机选择一个（解决平局问题）
    if (count > 0) {
        int randIdx = rand() % count;
        targetLabel = maxFreqLabels[randIdx];
    }

    // 释放内存
    free(freq);
    free(maxFreqLabels);
    return targetLabel;
}

// 计算模块度Q（衡量社区划分质量，值越大划分越好，范围[-1,1]）
double calculateModularity(Graph* graph, LPAResult* lpaRes) {
    if (graph == NULL || lpaRes == NULL || lpaRes->labels == NULL) {
        printf("无效的图或LPA结果！\n"); // 新增：参数合法性判断
        return 0.0;
    }
    int vertices = graph->numVertices;
    double totalEdges = 0.0; // 图中总边权重和（无向图需注意不重复计算，此处邻接表双向存储，需除2）
    double Q = 0.0;

    // 第一步：计算总边权重和
    for (int u = 0; u < vertices; u++) {
        AdjNode* temp = graph->array[u].head;
        while (temp != NULL) {
            if (u < temp->vertex) { // 避免重复计算（u < v 只统计一次）
                totalEdges += temp->weight;
            }
            temp = temp->next;
        }
    }

    if (totalEdges == 0.0) return 0.0; // 空图，模块度为0

    // 第二步：计算模块度Q
    // 关键优化1：预计算所有节点的k值，存储在数组中，仅计算一次
    double* k_array = (double*)malloc(vertices * sizeof(double));
    if (k_array == NULL) {
        printf("内存分配失败！\n");
        exit(1);
    }
    for (int i = 0; i < vertices; i++) {
        double k = 0.0;
        AdjNode* temp = graph->array[i].head;
        while (temp != NULL) {
            k += temp->weight;
            temp = temp->next;
        }
        k_array[i] = k; // 存储节点i的k值，后续直接使用
    }

    // 关键优化2：仅遍历u <= v的节点对，避免无向图重复计算
    for (int u = 0; u < vertices; u++) {
        for (int v = u; v < vertices; v++) { // v从u开始，不重复计算(u,v)和(v,u)
            // 优化A_uv查找：可保留原有逻辑，或进一步优化（见扩展）
            double A_uv = 0.0;
            AdjNode* temp = graph->array[u].head;
            while (temp != NULL) {
                if (temp->vertex == v) {
                    A_uv = temp->weight;
                    break;
                }
                temp = temp->next;
            }

            // 直接使用预计算的k值，无需重复遍历邻接表
            double k_u = k_array[u];
            double k_v = k_array[v];

            // 模块度核心累加项
            if (lpaRes->labels[u] == lpaRes->labels[v]) {
                // 若u != v，累加2倍（补偿只遍历u<=v，对应(u,v)和(v,u)两个节点对）
                double factor = (u == v) ? 1.0 : 2.0;
                Q += factor * (A_uv - (k_u * k_v) / (2 * totalEdges));
            }
        }
    }

    Q = Q / (2 * totalEdges);

    // 释放预计算数组的内存
    free(k_array);
    return Q;
}

// 统计社区数量
int countCommunities(int* labels, int vertices) {
    if (labels == NULL || vertices <= 0) { // 新增：参数合法性判断
        printf("无效的标签数组或节点数量！\n");
        return 0;
    }
    int count = 0;
    int* visitedLabel = (int*)calloc(vertices, sizeof(int));
    if (visitedLabel == NULL) {         // 新增：内存分配失败判断
        printf("内存分配失败！\n");
        exit(1);
    }

    for (int i = 0; i < vertices; i++) {
        if (visitedLabel[labels[i]] == 0) {
            count++;
            visitedLabel[labels[i]] = 1;
        }
    }

    free(visitedLabel);
    return count;
}

// 标签传播算法（LPA）实现社区发现
LPAResult* labelPropagation(Graph* graph, int maxIter) {
    if (graph == NULL || maxIter <= 0) { // 新增：参数合法性判断
        printf("无效的图或最大迭代次数过低！\n");
        return NULL;
    }
    int vertices = graph->numVertices;
    LPAResult* res = initLPAResult(vertices);
    int* newLabels = (int*)malloc(vertices * sizeof(int)); // 存储迭代后的新标签（避免实时更新影响结果）
    if (newLabels == NULL) {         // 新增：内存分配失败判断
        printf("内存分配失败！\n");
        freeLPAResult(res);
        exit(1);
    }
    int iter = 0;
    int isStable = 0; // 标签是否稳定：1=稳定，0=不稳定

    while (iter < maxIter && !isStable) {
        isStable = 1;
        memcpy(newLabels, res->labels, vertices * sizeof(int)); // 初始化新标签为当前标签

        // 随机遍历所有节点（打乱节点顺序，避免顺序影响）
        int* nodeOrder = (int*)malloc(vertices * sizeof(int));
        if (nodeOrder == NULL) {     // 新增：内存分配失败判断
            printf("内存分配失败！\n");
            free(newLabels);
            freeLPAResult(res);
            exit(1);
        }
        for (int i = 0; i < vertices; i++) {
            nodeOrder[i] = i;
        }
        // 随机洗牌（Fisher-Yates洗牌算法）
        for (int i = vertices - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = nodeOrder[i];
            nodeOrder[i] = nodeOrder[j];
            nodeOrder[j] = temp;
        }

        // 遍历所有节点，更新标签
        for (int idx = 0; idx < vertices; idx++) {
            int u = nodeOrder[idx];
            int newLabel = getMostFrequentLabel(graph, u, res->labels);
            if (newLabel == -1) {     // 新增：异常标签判断
                continue;
            }
            if (newLabels[u] != newLabel) {
                newLabels[u] = newLabel;
                isStable = 0; // 标签变化，标记为不稳定
            }
        }

        memcpy(res->labels, newLabels, vertices * sizeof(int)); // 更新标签
        free(nodeOrder);
        iter++;
        
    }
    // ========== 新增：初始化两个新字段 ==========
    res->iter = 0;        // 初始迭代次数为0
    res->isStable = 0;    // 初始标记为未收敛
    // ==========================================

    // 计算社区数量和模块度
    res->numCommunities = countCommunities(res->labels, vertices);
    res->modularity = calculateModularity(graph, res);
    // ========== 新增：给LPAResult的新字段赋值 ==========
    res->iter = iter;          // 实际执行的迭代次数（最终迭代轮次）
    res->isStable = isStable;  // 最终是否收敛（1=收敛，0=未收敛）
    // ==================================================

    free(newLabels);
    return res;
}

// 释放图内存
void freeGraph(Graph* graph) {
    if (graph == NULL) {
        return;
    }
    int vertices = graph->numVertices;
    for (int i = 0; i < vertices; i++) {
        AdjNode* temp = graph->array[i].head;
        while (temp != NULL) {
            AdjNode* next = temp->next;
            free(temp);
            temp = next;
        }
    }
    free(graph->array);
    free(graph);
}

// 释放Dijkstra结果内存
void freeDijkstraResult(DijkstraResult* res) {
    if (res == NULL) {
        return;
    }
    free(res->dist);
    free(res->prev);
    free(res);
}

// 释放LPA结果内存
void freeLPAResult(LPAResult* res) {
    if (res == NULL) {
        return;
    }
    free(res->labels);
    free(res);
}

int main() {
    srand(time(NULL)); // 优化：移至main开头，全局仅初始化一次，保证随机数有效性

    // ===================== 测试用例1：小规模图（5节点 A-E） =====================
    // 小规模测试用例（A-E）优化：明确划分2个社区，强化内部聚集性
    int nodeNum1 = 5;
    Graph* graph1 = createGraph(nodeNum1);
    const char* nodeNames1[] = {"A", "B", "C", "D", "E"};

    // 社区1：A/D/E（内部边密集，权重小，强化聚集性）
    addEdge(graph1, 0, 3, 0.1);  // A-D（内部边，权重最小）
    addEdge(graph1, 0, 4, 0.2);  // A-E（内部边，权重小）
    addEdge(graph1, 3, 4, 0.15); // D-E（内部边，权重小）

    // 社区2：B/C（内部边密集，权重小）
    addEdge(graph1, 1, 2, 0.1);  // B-C（内部边，权重最小）

    // 社区间外部边：稀疏且权重大（仅保留1条，弱化外部连接）
    addEdge(graph1, 0, 1, 0.75); // A-B（外部边，权重大于所有内部边）

    // 1. Dijkstra算法：求解A(0)到E(4)的最紧密路径
    int start = 0; // 源点A
    int target = 4; // 目标点E
    DijkstraResult* dijkRes1 = dijkstra(graph1, start);

    printf("===================== 小规模测试用例（A-E） =====================\n");
    printf("源点：%s，目标点：%s\n", nodeNames1[start], nodeNames1[target]);
    if (dijkRes1->dist[target] == DBL_MAX) { // 修改：用DBL_MAX判断无可达路径
        printf("无可达路径从%s到%s\n", nodeNames1[start], nodeNames1[target]);
    } else {
        printf("最短距离d(E)：%.3f（对应关系最紧密）\n", dijkRes1->dist[target]);
        printf("最紧密路径序列：");
        printPath(dijkRes1->prev, target, nodeNames1);
        printf("\n");
    }

    // 2. 标签传播算法：社区发现
    LPAResult* lpaRes1 = labelPropagation(graph1, 30); // 最大迭代30次
    printf("标签传播算法社区划分结果：\n");
    printf("社区数量：%d\n", lpaRes1->numCommunities);
    printf("模块度Q值：%.3f（值越大，社区划分质量越好）\n", lpaRes1->modularity);
    printf("实际迭代次数：%d\n", lpaRes1->iter); // 新增：打印最终迭代次数
    // 优化输出：将0/1转为文字，更直观
    const char* stableStr1 = lpaRes1->isStable ? "是（已收敛）" : "否（未收敛，达到最大迭代次数）";
    printf("是否收敛：%d（%s）\n", lpaRes1->isStable, stableStr1); // 新增：打印收敛状态
    printf("节点-社区标签映射：\n");
    for (int i = 0; i < nodeNum1; i++) {
        printf("用户%s → 社区标签%d\n", nodeNames1[i], lpaRes1->labels[i]);
    }

    // ===================== 测试用例2：中等规模图（10节点 A-J） =====================
    int nodeNum2 = 10;
    Graph* graph2 = createGraph(nodeNum2);
    const char* nodeNames2[] = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"};
    // 随机添加边（模拟中等规模社交网络，权重为0.1~1.0随机值）
    for (int i = 0; i < nodeNum2; i++) {
        // 每个节点连接2~3个邻接节点，避免孤立节点
        int edgeNum = 2 + rand() % 2;
        for (int j = 0; j < edgeNum; j++) {
            int v = rand() % nodeNum2;
            if (v == i) continue; // 避免自环
            double weight = 0.1 + (double)rand() / RAND_MAX * 0.9; // 0.1~1.0
            addEdge(graph2, i, v, weight);
        }
    }

    printf("\n===================== 中等规模测试用例（A-J） =====================\n");
    // 1. Dijkstra算法：求解A(0)到J(9)的最紧密路径
    start = 0;
    target = 9;
    DijkstraResult* dijkRes2 = dijkstra(graph2, start);
    printf("源点：%s，目标点：%s\n", nodeNames2[start], nodeNames2[target]);
    if (dijkRes2->dist[target] == DBL_MAX) { // 修改：用DBL_MAX判断无可达路径
        printf("无可达路径从%s到%s\n", nodeNames2[start], nodeNames2[target]);
    } else {
        printf("最短距离d(J)：%.3f\n", dijkRes2->dist[target]);
        printf("最紧密路径序列：");
        printPath(dijkRes2->prev, target, nodeNames2);
        printf("\n");
    }

    // 2. 标签传播算法：社区发现
    LPAResult* lpaRes2 = labelPropagation(graph2, 50);  // 迭代50次
    printf("标签传播算法社区划分结果：\n");
    printf("社区数量：%d\n", lpaRes2->numCommunities);
    printf("模块度Q值：%.3f\n", lpaRes2->modularity);
    printf("实际迭代次数：%d\n", lpaRes2->iter); // 新增：打印最终迭代次数
    // 优化输出：将0/1转为文字，更直观
    const char* stableStr2 = lpaRes2->isStable ? "是（已收敛）" : "否（未收敛，达到最大迭代次数）";
    printf("是否收敛：%d（%s）\n", lpaRes2->isStable, stableStr2); // 新增：打印收敛状态
    printf("节点-社区标签映射：\n");
    for (int i = 0; i < nodeNum2; i++) {
        printf("用户%s → 社区标签%d\n", nodeNames2[i], lpaRes2->labels[i]);
    }

    // 释放所有内存
    freeGraph(graph1);
    freeDijkstraResult(dijkRes1);
    freeLPAResult(lpaRes1);
    freeGraph(graph2);
    freeDijkstraResult(dijkRes2);
    freeLPAResult(lpaRes2);

    return 0;
}