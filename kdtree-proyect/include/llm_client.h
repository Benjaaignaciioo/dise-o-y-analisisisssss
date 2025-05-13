#ifndef LLM_CLIENT_H
#define LLM_CLIENT_H

#include <iostream>
#include <string>
#include <curl/curl.h>
#include <chrono>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Función callback para recibir datos de CURL
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* s) {
    size_t newLength = size * nmemb;
    try {
        s->append((char*)contents, newLength);
        return newLength;
    } catch(std::bad_alloc& e) {
        // Manejar error de memoria
        return 0;
    }
}

class LLMClient {
private:
    std::string server_url;
    CURL* curl;
    
public:
    LLMClient(const std::string& url = "http://localhost:8000/v1/completions") 
        : server_url(url), curl(nullptr) {
        // Inicializar CURL
        curl_global_init(CURL_GLOBAL_DEFAULT);
        curl = curl_easy_init();
        
        if (!curl) {
            std::cerr << "Error: No se pudo inicializar CURL" << std::endl;
        }
    }
    
    ~LLMClient() {
        if (curl) {
            curl_easy_cleanup(curl);
        }
        curl_global_cleanup();
    }
    
    // Método para enviar consultas al servidor LLM y medir tiempo
    std::pair<std::string, double> query(const std::string& prompt, int max_tokens = 100) {
        if (!curl) {
            return {"Error: CURL no inicializado", 0.0};
        }
        
        // Preparar datos JSON para la consulta
        json request_data = {
            {"prompt", prompt},
            {"max_tokens", max_tokens},
            {"temperature", 0.7}
        };
        
        std::string request_str = request_data.dump();
        
        // Configurar la solicitud HTTP
        curl_easy_setopt(curl, CURLOPT_URL, server_url.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_str.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, request_str.length());
        
        // Configurar headers
        struct curl_slist* headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        
        // Configurar callback para recibir respuesta
        std::string response_string;
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);
        
        // Medir tiempo de ejecución
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Realizar la solicitud
        CURLcode res = curl_easy_perform(curl);
        
        // Terminar medición de tiempo
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        // Liberar recursos
        curl_slist_free_all(headers);
        
        // Procesar respuesta
        if (res != CURLE_OK) {
            return {std::string("Error en CURL: ") + curl_easy_strerror(res), elapsed_time};
        }
        
        try {
            json response = json::parse(response_string);
            if (response.contains("choices") && !response["choices"].empty() && 
                response["choices"][0].contains("text")) {
                return {response["choices"][0]["text"], elapsed_time};
            } else {
                return {"Formato de respuesta inesperado", elapsed_time};
            }
        } catch (const std::exception& e) {
            return {std::string("Error al procesar JSON: ") + e.what(), elapsed_time};
        }
    }
};

#endif // LLM_CLIENT_H