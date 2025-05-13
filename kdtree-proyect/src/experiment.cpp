#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <cmath>
#include "../include/kdtree.h"
#include "database.h"
#include "embeddings.h"

// Función para medir estadísticas de rendimiento
struct PerformanceStats {
    double mean_time;
    double stddev_time;
    double min_time;
    double max_time;
    double median_time;
    double p90_time;  // percentil 90
    size_t memory_usage_kb;
};

// Función para medir el uso de memoria
size_t estimateMemoryUsage(const KDTree& tree) {
    // Aproximación: cada nodo contiene un punto (384 doubles) y algunos punteros/valores
    size_t point_size = 384 * sizeof(double);
    size_t node_overhead = sizeof(int) * 3; // axis, left, right
    size_t total_nodes = tree.getNodeCount();
    
    return (point_size + node_overhead) * total_nodes / 1024; // KB
}

size_t estimateMemoryUsage(const LinearSearch& search) {
    // Cada elemento tiene un embedding (384 doubles) y texto
    size_t point_size = 384 * sizeof(double);
    size_t avg_text_size = 100; // estimación promedio de longitud de texto
    size_t total_items = search.getSize();
    
    return (point_size + avg_text_size) * total_items / 1024; // KB
}

// Experimento con diferentes tamaños de base de datos
void experimentDatabaseSize(const std::vector<DataItem>& full_database) {
    std::cout << "\n==== Experimento: Tamaño de Base de Datos ====\n";
    
    // Definir tamaños a evaluar
    std::vector<int> sizes = {100, 500, 1000, 5000, 10000};
    if (full_database.size() > 10000) {
        sizes.push_back(full_database.size());
    }
    
    // Limitar tamaños según la base disponible
    sizes.erase(std::remove_if(sizes.begin(), sizes.end(), 
                          [&](int s) { return s > static_cast<int>(full_database.size()); }),
           sizes.end());
    
    // Archivo para resultados
    std::ofstream results_file("results/database_size_results.csv");
    results_file << "Size,KDTree_Mean_Time,KDTree_StdDev,KDTree_Min,KDTree_Max,KDTree_Median,KDTree_P90,KDTree_Memory_KB,";
    results_file << "Linear_Mean_Time,Linear_StdDev,Linear_Min,Linear_Max,Linear_Median,Linear_P90,Linear_Memory_KB,Speedup\n";
    
    // Generar consultas para el experimento (usamos las mismas para todos los tamaños)
    const int num_queries = 100;
    const int num_runs = 10; // número de ejecuciones para medir variabilidad
    
    std::vector<Point> queries = generateQueries(full_database, num_queries);
    
    // Para cada tamaño
    for (int size : sizes) {
        std::cout << "Evaluando base de datos de tamaño " << size << "..." << std::endl;
        
        // Crear subconjunto de la base
        std::vector<DataItem> subset(full_database.begin(), full_database.begin() + size);
        
        // Construir árbol KD
        auto build_start = std::chrono::high_resolution_clock::now();
        KDTree tree(subset);
        auto build_end = std::chrono::high_resolution_clock::now();
        auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();
        
        // Construir búsqueda lineal
        LinearSearch linear(subset);
        
        // Medir memoria
        size_t kdtree_memory = estimateMemoryUsage(tree);
        size_t linear_memory = estimateMemoryUsage(linear);
        
        std::cout << "  Árbol KD construido en " << build_time << " ms (memoria estimada: " 
                  << kdtree_memory << " KB)" << std::endl;
        
        // Vectores para almacenar tiempos
        std::vector<double> kdtree_times;
        std::vector<double> linear_times;
        
        // Para cada consulta
        for (int q = 0; q < num_queries; q++) {
            const auto& query = queries[q];
            
            // Vectores de tiempos para múltiples ejecuciones
            std::vector<double> kd_times_run;
            std::vector<double> linear_times_run;
            
            // Realizar múltiples ejecuciones
            for (int run = 0; run < num_runs; run++) {
                // Medir tiempo para árbol KD
                auto kd_start = std::chrono::high_resolution_clock::now();
                auto kd_result = tree.nearest(query);
                auto kd_end = std::chrono::high_resolution_clock::now();
                auto kd_time = std::chrono::duration_cast<std::chrono::microseconds>(kd_end - kd_start).count();
                kd_times_run.push_back(kd_time);
                
                // Medir tiempo para búsqueda lineal
                auto linear_start = std::chrono::high_resolution_clock::now();
                auto linear_result = linear.nearest(query);
                auto linear_end = std::chrono::high_resolution_clock::now();
                auto linear_time = std::chrono::duration_cast<std::chrono::microseconds>(linear_end - linear_start).count();
                linear_times_run.push_back(linear_time);
            }
            
            // Calcular estadísticas para esta consulta (promedio de ejecuciones)
            double kd_avg = std::accumulate(kd_times_run.begin(), kd_times_run.end(), 0.0) / num_runs;
            double linear_avg = std::accumulate(linear_times_run.begin(), linear_times_run.end(), 0.0) / num_runs;
            
            kdtree_times.push_back(kd_avg);
            linear_times.push_back(linear_avg);
        }
        
        // Calcular estadísticas para KDTree
        std::sort(kdtree_times.begin(), kdtree_times.end());
        double kd_mean = std::accumulate(kdtree_times.begin(), kdtree_times.end(), 0.0) / kdtree_times.size();
        
        double kd_var = 0.0;
        for (const auto& time : kdtree_times) {
            kd_var += (time - kd_mean) * (time - kd_mean);
        }
        kd_var /= kdtree_times.size();
        double kd_stddev = std::sqrt(kd_var);
        
        double kd_min = kdtree_times.front();
        double kd_max = kdtree_times.back();
        double kd_median = kdtree_times[kdtree_times.size() / 2];
        double kd_p90 = kdtree_times[static_cast<int>(kdtree_times.size() * 0.9)];
        
        // Calcular estadísticas para búsqueda lineal
        std::sort(linear_times.begin(), linear_times.end());
        double linear_mean = std::accumulate(linear_times.begin(), linear_times.end(), 0.0) / linear_times.size();
        
        double linear_var = 0.0;
        for (const auto& time : linear_times) {
            linear_var += (time - linear_mean) * (time - linear_mean);
        }
        linear_var /= linear_times.size();
        double linear_stddev = std::sqrt(linear_var);
        
        double linear_min = linear_times.front();
        double linear_max = linear_times.back();
        double linear_median = linear_times[linear_times.size() / 2];
        double linear_p90 = linear_times[static_cast<int>(linear_times.size() * 0.9)];
        
        // Calcular aceleración (speedup)
        double speedup = linear_mean / kd_mean;
        
        // Guardar resultados
        results_file << size << "," 
                    << kd_mean << "," << kd_stddev << "," << kd_min << "," << kd_max << "," 
                    << kd_median << "," << kd_p90 << "," << kdtree_memory << ","
                    << linear_mean << "," << linear_stddev << "," << linear_min << "," << linear_max << ","
                    << linear_median << "," << linear_p90 << "," << linear_memory << ","
                    << speedup << "\n";
        
        // Imprimir resultados parciales
        std::cout << "  Resultados para tamaño " << size << ":" << std::endl;
        std::cout << "    KD Tree:   " << kd_mean << " µs (stddev: " << kd_stddev << " µs)" << std::endl;
        std::cout << "    Lineal:    " << linear_mean << " µs (stddev: " << linear_stddev << " µs)" << std::endl;
        std::cout << "    Speedup:   " << speedup << "x" << std::endl;
        std::cout << "    Memoria:   KD Tree: " << kdtree_memory << " KB, Lineal: " << linear_memory << " KB" << std::endl;
    }
    
    results_file.close();
    std::cout << "Resultados guardados en results/database_size_results.csv" << std::endl;
}

// Experimento con diferentes tamaños de caso base para el árbol KD
void experimentLeafSize(const std::vector<DataItem>& database) {
    std::cout << "\n==== Experimento: Tamaño del Caso Base (Leaf Size) ====\n";
    
    // Definir tamaños de hoja a evaluar
    std::vector<int> leaf_sizes = {1, 5, 10, 20, 50, 100};
    
    // Archivo para resultados
    std::ofstream results_file("results/leaf_size_results.csv");
    results_file << "LeafSize,Mean_Time,StdDev,Min,Max,Median,P90,Memory_KB,Build_Time_ms\n";
    
    // Generar consultas para el experimento
    const int num_queries = 100;
    const int num_runs = 10; // número de ejecuciones para medir variabilidad
    
    std::vector<Point> queries = generateQueries(database, num_queries);
    
    // Para cada tamaño de hoja
    for (int leaf_size : leaf_sizes) {
        std::cout << "Evaluando tamaño de caso base (leaf size) " << leaf_size << "..." << std::endl;
        
        // Construir árbol KD con el tamaño de hoja específico
        auto build_start = std::chrono::high_resolution_clock::now();
        KDTree tree(database, leaf_size);  // Asumimos que KDTree acepta leaf_size como segundo parámetro
        auto build_end = std::chrono::high_resolution_clock::now();
        auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();
        
        // Medir memoria
        size_t memory = estimateMemoryUsage(tree);
        
        std::cout << "  Árbol KD construido en " << build_time << " ms (memoria estimada: " 
                  << memory << " KB)" << std::endl;
        
        // Vector para almacenar tiempos
        std::vector<double> times;
        
        // Para cada consulta
        for (int q = 0; q < num_queries; q++) {
            const auto& query = queries[q];
            
            // Vector de tiempos para múltiples ejecuciones
            std::vector<double> times_run;
            
            // Realizar múltiples ejecuciones
            for (int run = 0; run < num_runs; run++) {
                // Medir tiempo para árbol KD
                auto start = std::chrono::high_resolution_clock::now();
                auto result = tree.nearest(query);
                auto end = std::chrono::high_resolution_clock::now();
                auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                times_run.push_back(time);
            }
            
            // Calcular estadísticas para esta consulta (promedio de ejecuciones)
            double avg = std::accumulate(times_run.begin(), times_run.end(), 0.0) / num_runs;
            times.push_back(avg);
        }
        
        // Calcular estadísticas
        std::sort(times.begin(), times.end());
        double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        
        double var = 0.0;
        for (const auto& time : times) {
            var += (time - mean) * (time - mean);
        }
        var /= times.size();
        double stddev = std::sqrt(var);
        
        double min = times.front();
        double max = times.back();
        double median = times[times.size() / 2];
        double p90 = times[static_cast<int>(times.size() * 0.9)];
        
        // Guardar resultados
        results_file << leaf_size << "," 
                    << mean << "," << stddev << "," << min << "," << max << "," 
                    << median << "," << p90 << "," << memory << "," << build_time << "\n";
        
        // Imprimir resultados parciales
        std::cout << "  Resultados para leaf size " << leaf_size << ":" << std::endl;
        std::cout << "    Tiempo medio: " << mean << " µs (stddev: " << stddev << " µs)" << std::endl;
        std::cout << "    Memoria:      " << memory << " KB" << std::endl;
        std::cout << "    Build time:   " << build_time << " ms" << std::endl;
    }
    
    results_file.close();
    std::cout << "Resultados guardados en results/leaf_size_results.csv" << std::endl;
}

// Prueba estadística para determinar si hay diferencias significativas
bool areSignificantlyDifferent(const std::vector<double>& times1, const std::vector<double>& times2) {
    // Implementación simple: comparar medias y desviaciones estándar
    // En una implementación completa, se debería usar una prueba t o similar
    
    double mean1 = std::accumulate(times1.begin(), times1.end(), 0.0) / times1.size();
    double mean2 = std::accumulate(times2.begin(), times2.end(), 0.0) / times2.size();
    
    double var1 = 0.0, var2 = 0.0;
    for (const auto& t : times1) var1 += (t - mean1) * (t - mean1);
    for (const auto& t : times2) var2 += (t - mean2) * (t - mean2);
    var1 /= times1.size();
    var2 /= times2.size();
    
    // Cálculo simplificado del estadístico t
    double t_stat = std::abs(mean1 - mean2) / std::sqrt((var1 / times1.size()) + (var2 / times2.size()));
    
    // Valor crítico aproximado para 95% de confianza
    double critical_value = 1.96;
    
    return t_stat > critical_value;
}

// Modo interactivo mejorado
void interactiveMode(const std::vector<DataItem>& database) {
    std::cout << "\n==== Modo Interactivo de Búsqueda Semántica ====\n";
    std::cout << "Base de datos cargada con " << database.size() << " elementos.\n";
    
    // Construir árbol KD (puedes ajustar el tamaño de hoja según tus experimentos)
    int leaf_size = 10; // Valor por defecto, podría ser ajustable
    
    std::cout << "Construyendo árbol KD (leaf_size = " << leaf_size << ")..." << std::endl;
    auto build_start = std::chrono::high_resolution_clock::now();
    KDTree tree(database, leaf_size);
    auto build_end = std::chrono::high_resolution_clock::now();
    auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();
    std::cout << "Árbol KD construido en " << build_time << " ms\n";
    
    // Inicializar búsqueda lineal
    LinearSearch linear(database);
    
    while (true) {
        std::cout << "\nIngrese su consulta (o 'salir' para terminar): ";
        std::string query;
        std::getline(std::cin, query);
        
        if (query == "salir" || query == "exit" || query == "q") {
            break;
        }
        
        if (query.empty()) {
            continue;
        }
        
        // Generar embedding para la consulta usando nuestro modelo
        Point query_embedding = embedder.getEmbedding(query);
        
        // Búsqueda con árbol KD
        auto kd_start = std::chrono::high_resolution_clock::now();
        auto kd_result = tree.nearest(query_embedding);
        auto kd_end = std::chrono::high_resolution_clock::now();
        auto kd_time = std::chrono::duration_cast<std::chrono::microseconds>(kd_end - kd_start).count();
        
        // Búsqueda lineal
        auto linear_start = std::chrono::high_resolution_clock::now();
        auto linear_result = linear.nearest(query_embedding);
        auto linear_end = std::chrono::high_resolution_clock::now();
        auto linear_time = std::chrono::duration_cast<std::chrono::microseconds>(linear_end - linear_start).count();
        
        // Mostrar resultados
        std::cout << "\n=== Resultados de la búsqueda ===\n";
        std::cout << "Consulta: \"" << query << "\"\n\n";
        
        // Resultado del árbol KD
        std::cout << "Resultado del árbol KD (tiempo: " << kd_time << " µs):\n";
        std::cout << "Distancia: " << kd_result.first << "\n";
        std::cout << "Texto: " << kd_result.second << "\n\n";
        
        // Resultado de búsqueda lineal
        std::cout << "Resultado de búsqueda lineal (tiempo: " << linear_time << " µs):\n";
        std::cout << "Distancia: " << linear_result.first << "\n";
        std::cout << "Texto: " << linear_result.second << "\n\n";
        
        // Comparación de rendimiento
        double speedup = static_cast<double>(linear_time) / kd_time;
        std::cout << "Comparación de rendimiento:\n";
        std::cout << "- Árbol KD: " << kd_time << " µs\n";
        std::cout << "- Búsqueda lineal: " << linear_time << " µs\n";
        std::cout << "- Aceleración: " << speedup << "x\n";
        
        // Mostrar top 3 resultados
        std::cout << "\nResultados adicionales (top 5):\n";
        auto top_results = tree.kNearest(query_embedding, 5);
        for (size_t i = 0; i < top_results.size(); i++) {
            std::cout << (i+1) << ". Distancia: " << top_results[i].first 
                      << "\n   Texto: " << top_results[i].second << "\n";
        }
    }
}

// Función principal con opciones de experimentos
int main(int argc, char* argv[]) {
    // Crear directorio de resultados si no existe
    if (system("mkdir -p results") != 0) {
        std::cerr << "Error al crear el directorio 'results'" << std::endl;
    }
    
    // Procesar argumentos de línea de comandos
    bool interactive = false;
    bool exp_db_size = false;
    bool exp_leaf_size = false;
    std::string filename = "";
    int max_lines = -1;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--interactive" || arg == "-i") {
            interactive = true;
        } 
        else if (arg == "--exp-db-size" || arg == "-d") {
            exp_db_size = true;
        }
        else if (arg == "--exp-leaf-size" || arg == "-l") {
            exp_leaf_size = true;
        }
        else if (arg == "--max-lines" || arg == "-m") {
            if (i + 1 < argc) {
                max_lines = std::stoi(argv[i + 1]);
                i++;
            }
        }
        else if (arg[0] != '-') {
            filename = arg;
        }
    }
    
    // Usado para generación determinista de embeddings
    std::cout << "Usando generador de embeddings determinístico..." << std::endl;
    
    // Cargar o generar la base de datos
    std::vector<DataItem> database;
    
    if (filename.empty()) {
        std::cout << "No se proporcionó archivo. Generando base de datos de prueba." << std::endl;
        database = generateMockDatabase(1000, 384);
    } else {
        // Determinar el tipo de archivo por extensión
        if (filename.substr(filename.find_last_of(".") + 1) == "jsonl") {
            std::cout << "Cargando archivo JSONL: " << filename << std::endl;
            database = loadDatabaseFromJsonl(filename, max_lines);
        } else {
            std::cout << "Intentando cargar archivo binario: " << filename << std::endl;
            database = loadDatabase(filename);
        }
        
        // Si no se pudo cargar, generar una base de datos de prueba
        if (database.empty()) {
            std::cout << "No se pudo cargar la base de datos. Generando base de datos de prueba." << std::endl;
            database = generateMockDatabase(1000, 384);
        }
    }
    
    // Guardar la base de datos procesada en formato binario si viene de JSONL
    if (!filename.empty() && filename.substr(filename.find_last_of(".") + 1) == "jsonl") {
        std::cout << "Guardando base de datos procesada en formato binario..." << std::endl;
        saveDatabase(database, "processed_database.bin", max_lines);
    }
    
    // Ejecutar experimentos o modo interactivo según se solicite
    if (exp_db_size) {
        experimentDatabaseSize(database);
    }
    
    if (exp_leaf_size) {
        experimentLeafSize(database);
    }
    
    if (interactive || (!exp_db_size && !exp_leaf_size)) {
        interactiveMode(database);
    }
    
    return 0;
}