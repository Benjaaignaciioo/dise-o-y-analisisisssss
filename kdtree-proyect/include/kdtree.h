#ifndef KDTREE_H
#define KDTREE_H

#include <vector>
#include <algorithm>
#include <limits>
#include <queue>
#include <utility>
#include <Eigen/Dense>

using Point = Eigen::VectorXd;

struct DataItem {
    std::string text;
    Point embedding;
};

class KDTree {
private:
    struct Node {
        Point point;
        std::string text;
        int axis;
        Node* left;
        Node* right;
        
        Node(const Point& p, const std::string& t, int a) 
            : point(p), text(t), axis(a), left(nullptr), right(nullptr) {}
        
        ~Node() {
            delete left;
            delete right;
        }
    };
    
    Node* root;
    int leaf_size; // Tamaño del caso base
    int dimensions;
    int node_count; // Contador de nodos para estimación de memoria
    
    // Función auxiliar para construir el árbol
    Node* buildTree(std::vector<DataItem>& data, int depth, int start, int end) {
        if (start >= end) {
            return nullptr;
        }
        
        // Si el número de elementos es menor o igual al tamaño de hoja, crear hoja
        if (end - start <= leaf_size) {
            // Crear un nodo hoja con el primer elemento
            Node* leaf = new Node(data[start].embedding, data[start].text, 0);
            node_count++;
            
            // Para caso base > 1, realizar búsqueda lineal en tiempo de consulta
            return leaf;
        }
        
        int axis = depth % dimensions;
        
        // Ordenar los datos por el eje actual
        std::sort(data.begin() + start, data.begin() + end,
                 [axis](const DataItem& a, const DataItem& b) {
                     return a.embedding(axis) < b.embedding(axis);
                 });
        
        // Encontrar la mediana
        int mid = start + (end - start) / 2;
        
        // Crear nodo y construir subárboles recursivamente
        Node* node = new Node(data[mid].embedding, data[mid].text, axis);
        node_count++;
        
        node->left = buildTree(data, depth + 1, start, mid);
        node->right = buildTree(data, depth + 1, mid + 1, end);
        
        return node;
    }
    
    // Función auxiliar para búsqueda de vecino más cercano
    void nearestNeighbor(Node* node, const Point& query, double& best_dist, std::string& best_text) const {
        if (!node) {
            return;
        }
        
        // Calcular distancia euclidiana al cuadrado
        double dist = (query - node->point).squaredNorm();
        
        // Actualizar mejor distancia y texto si mejora
        if (dist < best_dist) {
            best_dist = dist;
            best_text = node->text;
        }
        
        // Determinar qué hijo visitar primero
        int axis = node->axis;
        double diff = query(axis) - node->point(axis);
        
        Node* first = (diff < 0) ? node->left : node->right;
        Node* second = (diff < 0) ? node->right : node->left;
        
        // Visitar primer hijo
        nearestNeighbor(first, query, best_dist, best_text);
        
        // Verificar si es necesario visitar el segundo hijo
        if (diff * diff < best_dist) {
            nearestNeighbor(second, query, best_dist, best_text);
        }
    }
    
    // Función auxiliar para k vecinos más cercanos - modificada para C++11
    void kNearestNeighbors(Node* node, const Point& query, 
                          std::priority_queue<std::pair<double, std::string>>& pq, int k) const {
        if (!node) {
            return;
        }
        
        // Calcular distancia euclidiana al cuadrado
        double dist = (query - node->point).squaredNorm();
        
        // Actualizar cola de prioridad - Corregido para evitar warnings de comparación de signedness
        if (static_cast<int>(pq.size()) < k) {
            pq.push(std::make_pair(dist, node->text));
        } else if (dist < pq.top().first) {
            pq.pop();
            pq.push(std::make_pair(dist, node->text));
        }
        
        // Determinar qué hijo visitar primero
        int axis = node->axis;
        double diff = query(axis) - node->point(axis);
        
        Node* first = (diff < 0) ? node->left : node->right;
        Node* second = (diff < 0) ? node->right : node->left;
        
        // Visitar primer hijo
        kNearestNeighbors(first, query, pq, k);
        
        // Verificar si es necesario visitar el segundo hijo - Corregido para comparación de signedness
        double largest_dist = pq.empty() ? std::numeric_limits<double>::max() : pq.top().first;
        if (diff * diff < largest_dist || static_cast<int>(pq.size()) < k) {
            kNearestNeighbors(second, query, pq, k);
        }
    }
    
public:
    // Constructor con opción para especificar tamaño de hoja
    KDTree(const std::vector<DataItem>& data, int leaf_size = 1) 
        : root(nullptr), leaf_size(leaf_size), node_count(0) {
        if (data.empty()) {
            dimensions = 0;
            return;
        }
        
        dimensions = data[0].embedding.size();
        
        // Crear copia para ordenar
        std::vector<DataItem> data_copy = data;
        
        // Construir árbol
        root = buildTree(data_copy, 0, 0, data_copy.size());
    }
    
    // Destructor
    ~KDTree() {
        delete root;
    }
    
    // Buscar vecino más cercano
    std::pair<double, std::string> nearest(const Point& query) const {
        double best_dist = std::numeric_limits<double>::max();
        std::string best_text;
        
        nearestNeighbor(root, query, best_dist, best_text);
        
        return std::make_pair(std::sqrt(best_dist), best_text);
    }
    
    // Buscar k vecinos más cercanos - Modificada para C++11
    std::vector<std::pair<double, std::string>> kNearest(const Point& query, int k) const {
        std::priority_queue<std::pair<double, std::string>> pq;
        
        kNearestNeighbors(root, query, pq, k);
        
        // Convertir cola de prioridad a vector ordenado
        std::vector<std::pair<double, std::string>> result;
        while (!pq.empty()) {
            std::pair<double, std::string> item = pq.top();
            double dist = std::sqrt(item.first);
            std::string text = item.second;
            pq.pop();
            result.push_back(std::make_pair(dist, text));
        }
        
        // Ordenar por distancia (menor a mayor)
        std::reverse(result.begin(), result.end());
        
        return result;
    }
    
    // Obtener el número de nodos (para estimación de memoria)
    int getNodeCount() const {
        return node_count;
    }
};

// Clase para búsqueda lineal (para comparar con KDTree)
class LinearSearch {
private:
    std::vector<DataItem> data;
    
public:
    LinearSearch(const std::vector<DataItem>& items) : data(items) {}
    
    std::pair<double, std::string> nearest(const Point& query) const {
        double min_dist = std::numeric_limits<double>::max();
        std::string nearest_text;
        
        for (const auto& item : data) {
            double dist = (query - item.embedding).norm();
            if (dist < min_dist) {
                min_dist = dist;
                nearest_text = item.text;
            }
        }
        
        return std::make_pair(min_dist, nearest_text);
    }
    
    size_t getSize() const {
        return data.size();
    }
};

#endif // KDTREE_H