#ifndef DATABASE_H
#define DATABASE_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <json/json.h>
#include <Eigen/Dense>
#include "../include/kdtree.h"
#include "embeddings.h"

// Definir la instancia global del embedder
DeterministicEmbedder embedder(384);

// Función para cargar base de datos desde JSONL
std::vector<DataItem> loadDatabaseFromJsonl(const std::string& filename, int max_lines = -1) {
    std::vector<DataItem> database;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir el archivo " << filename << std::endl;
        return database;
    }
    
    std::cout << "Leyendo archivo JSONL..." << std::endl;
    
    std::string line;
    int count = 0;
    
    // Crear base de datos con embeddings directamente
    while (std::getline(file, line) && (max_lines == -1 || count < max_lines)) {
        Json::Value jsonData;
        Json::Reader jsonReader;
        
        if (jsonReader.parse(line, jsonData)) {
            // Formato esperado: ["título", "contenido"]
            if (jsonData.isArray() && jsonData.size() >= 2) {
                DataItem item;
                item.text = jsonData[1].asString();
                
                // Generar embedding puramente determinista
                item.embedding = embedder.getEmbedding(item.text);
                
                database.push_back(item);
                
                if (count % 100 == 0) {
                    std::cout << "Procesados " << count << " elementos..." << std::endl;
                }
            }
        }
        
        count++;
    }
    
    file.close();
    std::cout << "Base de datos cargada con " << database.size() << " elementos" << std::endl;
    
    return database;
}

// Función para cargar base de datos desde archivo binario
std::vector<DataItem> loadDatabase(const std::string& filename) {
    std::vector<DataItem> database;
    std::ifstream infile(filename, std::ios::binary);
    
    if (!infile.is_open()) {
        std::cerr << "Error: No se pudo abrir " << filename << std::endl;
        return database;
    }
    
    try {
        // Leer número de líneas procesadas
        int processed_lines;
        infile.read(reinterpret_cast<char*>(&processed_lines), sizeof(processed_lines));
        
        // Leer tamaño de la base de datos
        int db_size;
        infile.read(reinterpret_cast<char*>(&db_size), sizeof(db_size));
        
        // Leer dimensión de los embeddings
        int embedding_dim;
        infile.read(reinterpret_cast<char*>(&embedding_dim), sizeof(embedding_dim));
        
        // Leer cada item
        database.reserve(db_size);
        
        for (int i = 0; i < db_size; i++) {
            DataItem item;
            
            // Leer texto
            int text_length;
            infile.read(reinterpret_cast<char*>(&text_length), sizeof(text_length));
            
            item.text.resize(text_length);
            infile.read(&item.text[0], text_length);
            
            // Leer embedding
            item.embedding.resize(embedding_dim);
            for (int j = 0; j < embedding_dim; j++) {
                double value;
                infile.read(reinterpret_cast<char*>(&value), sizeof(value));
                item.embedding(j) = value;
            }
            
            database.push_back(item);
            
            // Mostrar progreso
            if (i % 1000 == 0) {
                std::cout << "Cargados " << i << "/" << db_size << " elementos..." << std::endl;
            }
        }
        
        infile.close();
        std::cout << "Base de datos cargada con " << database.size() << " elementos" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error al cargar la base de datos: " << e.what() << std::endl;
    }
    
    return database;
}

// Generar base de datos de prueba con embeddings aleatorios
std::vector<DataItem> generateMockDatabase(int size, int dimensions) {
    std::vector<DataItem> database;
    database.reserve(size);
    
    for (int i = 0; i < size; i++) {
        DataItem item;
        item.text = "Texto de prueba " + std::to_string(i);
        
        // Usar el embedder para generar embeddings deterministicos
        item.embedding = embedder.getEmbedding(item.text);
        
        database.push_back(item);
        
        if (i % 1000 == 0) {
            std::cout << "Generados " << i << "/" << size << " elementos de prueba..." << std::endl;
        }
    }
    
    std::cout << "Base de datos de prueba generada con " << size 
              << " elementos de dimensión " << dimensions << std::endl;
    
    return database;
}

// Generar consultas aleatorias desde la base de datos
std::vector<Point> generateQueries(const std::vector<DataItem>& database, int num_queries) {
    std::vector<Point> queries;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, database.size() - 1);
    
    for (int i = 0; i < num_queries; i++) {
        int idx = dist(gen);
        queries.push_back(database[idx].embedding);
    }
    
    return queries;
}

// Guardar base de datos en archivo binario
bool saveDatabase(const std::vector<DataItem>& database, const std::string& filename, int processed_lines) {
    std::ofstream outfile(filename, std::ios::binary);
    
    if (!outfile.is_open()) {
        std::cerr << "Error: No se pudo abrir " << filename << " para escribir" << std::endl;
        return false;
    }
    
    try {
        // Escribir número de líneas procesadas
        outfile.write(reinterpret_cast<const char*>(&processed_lines), sizeof(processed_lines));
        
        // Escribir tamaño de la base de datos
        int db_size = database.size();
        outfile.write(reinterpret_cast<const char*>(&db_size), sizeof(db_size));
        
        // Escribir dimensión de los embeddings (asumiendo que todos tienen la misma)
        int embedding_dim = database.empty() ? 0 : database[0].embedding.size();
        outfile.write(reinterpret_cast<const char*>(&embedding_dim), sizeof(embedding_dim));
        
        // Escribir cada item
        for (int i = 0; i < db_size; i++) {
            const auto& item = database[i];
            
            // Escribir texto
            int text_length = item.text.length();
            outfile.write(reinterpret_cast<const char*>(&text_length), sizeof(text_length));
            outfile.write(item.text.c_str(), text_length);
            
            // Escribir embedding
            for (int j = 0; j < embedding_dim; j++) {
                double value = item.embedding(j);
                outfile.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
            
            // Mostrar progreso
            if (i % 1000 == 0) {
                std::cout << "Guardados " << i << "/" << db_size << " elementos..." << std::endl;
            }
        }
        
        outfile.close();
        std::cout << "Base de datos guardada en " << filename << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error al guardar la base de datos: " << e.what() << std::endl;
        return false;
    }
}

#endif // DATABASE_H