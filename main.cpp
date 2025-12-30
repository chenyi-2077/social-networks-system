#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

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
    newNode->vertex = v;
    newNode->weight = weight;
    newNode->next = NULL;
    return newNode;
}

// 创建图（初始化邻接表）
Graph* createGraph(int vertices) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->numVertices = vertices;
    graph->array = (AdjList*)malloc(vertices * sizeof(AdjList));

    for (int i = 0; i < vertices; i++) {
        graph->array[i].head = NULL;
    }
    return graph;
}

// 添加无向边（好友关系双向）
void addEdge(Graph* graph, int u, int v, double weight) {
    // 添加 u -> v 的边
    AdjNode* newNode = createAdjNode(v, weight);
    newNode->next = graph->array[u].head;
    graph->array[u].head = newNode;

    // 添加 v -> u 的边（无向图双向存储）
    newNode = createAdjNode(u, weight);
    newNode->next = graph->array[v].head;
    graph->array[v].head = newNode;
}