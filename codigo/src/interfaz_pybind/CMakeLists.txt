cmake_minimum_required(VERSION 2.8.12)
project(interfaz)

set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -O3 -I ../../eigen ")

add_subdirectory(pybind11)
pybind11_add_module(interfaz interfaz.cpp)

# Copiar el archivo .so en el directorio src/ para que los demas archivos puedan usarlo