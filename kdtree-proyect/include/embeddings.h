#ifndef EMBEDDINGS_H
#define EMBEDDINGS_H

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cctype>
#include <sstream>
#include <Eigen/Dense>

class DeterministicEmbedder {
private:
    // Dimensión de los vectores
    int embedding_dim;
    
    // Semilla para generación determinista de embeddings
    unsigned int seed = 42;
    
    // Función hash para strings 
    size_t hashString(const std::string& str) const {
        size_t hash = 0;
        for (size_t i = 0; i < str.length(); ++i) {
            hash = hash * 31 + static_cast<unsigned char>(str[i]);
        }
        return hash;
    }

public:
    // Constructor
    DeterministicEmbedder(int dim = 384) : embedding_dim(dim) {}
    
    // Tokenizar texto
    std::vector<std::string> tokenize(const std::string& text) {
        std::vector<std::string> tokens;
        std::istringstream iss(text);
        std::string token;
        
        while (iss >> token) {
            // Convertir a minúsculas
            std::transform(token.begin(), token.end(), token.begin(),
                           [](unsigned char c){ return std::tolower(c); });
            
            // Eliminar puntuación
            token.erase(std::remove_if(token.begin(), token.end(), 
                                      [](unsigned char c){ return !std::isalnum(c); }),
                       token.end());
            
            if (!token.empty()) {
                tokens.push_back(token);
            }
        }
        
        return tokens;
    }
    
    // Obtener embedding para un texto - método puramente determinista
    Eigen::VectorXd getEmbedding(const std::string& text) {
        std::vector<std::string> tokens = tokenize(text);
        
        if (tokens.empty()) {
            // Devolver vector aleatorio pero determinista para texto vacío
            return getTokenEmbedding(text);
        }
        
        // Vector resultante
        Eigen::VectorXd result = Eigen::VectorXd::Zero(embedding_dim);
        
        // Para cada token, generar un vector determinista y sumarlo
        for (const auto& token : tokens) {
            result += getTokenEmbedding(token);
        }
        
        // Normalizar
        double norm = result.norm();
        if (norm > 0) {
            result /= norm;
        }
        
        return result;
    }
    
    // Generar embedding determinista para un token
    Eigen::VectorXd getTokenEmbedding(const std::string& token) {
        Eigen::VectorXd vec = Eigen::VectorXd::Zero(embedding_dim);
        
        // Usar la semilla + hash del token para inicializar el generador
        std::mt19937 rng(seed + hashString(token));
        std::normal_distribution<double> dist(0.0, 1.0);
        
        // Generar valores aleatorios pero deterministas
        for (int i = 0; i < embedding_dim; ++i) {
            vec(i) = dist(rng);
        }
        
        // Normalizar
        vec.normalize();
        
        return vec;
    }
    
    // Obtener dimensión de los embeddings
    int getDimension() const {
        return embedding_dim;
    }
};

// Instancia global
extern DeterministicEmbedder embedder;

#endif // EMBEDDINGS_H