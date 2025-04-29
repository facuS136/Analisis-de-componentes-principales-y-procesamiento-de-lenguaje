#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <tuple>
#include <ctime>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

pybind11::array_t<double> metodo_de_la_potencia_def(pybind11::array_t<double> array, int iteraciones, float tolerancia) {
    /*-----------------convertimos el dato a una matriz de Eigen "a mano"-------------------*/
    pybind11::buffer_info buf = array.request();
    Eigen::MatrixXd A(buf.shape[0], buf.shape[1]);
    double* ptr = static_cast<double*>(buf.ptr);
    for (ssize_t i = 0; i < buf.shape[0]; i++) {
        for (ssize_t j = 0; j < buf.shape[1]; j++) {
            A(i, j) = ptr[i * buf.shape[1] + j];
        }
    }

    Eigen::MatrixXd res(A.cols(), A.cols() + 2);
    for (int i = 0; i < A.cols(); i++) {    // Calculamos cada autovalor delta_i y autovector v_i
        Eigen::VectorXd v = Eigen::VectorXd::Random(A.rows()).normalized(); //creamos un vector inicial aleatorio normalizado

        int pasos = 0;  // para ir contando cuantos pasos tarda en calcularse ese autovector
        /*--------------------------------METODO DE LA POTENCIA--------------------------------------------*/
        for (int j = 0; j < iteraciones; j++) {
            Eigen::VectorXd new_v = (A * v).normalized();
            pasos++;    
            if ((v - new_v).lpNorm<Eigen::Infinity>() < tolerancia) {     
                // si la mejora fue menor a la tolerancia, nos quedamos con ese v
                v = new_v;
                break;
            }
            v = new_v;
        /*-------------------------------------------------------------------------------------------------*/
        }

        res.col(i) = v; // guardamos el autovector estimado
        double delta = (v.transpose() * A * v).value() / (v.transpose() * v).value();   // aproximamos el autovalor
        A = A - delta * v * v.transpose();  // hacemos deflacion en A
        res(i, A.cols()) = delta;       // guardamos el autovalor
        res(i, A.cols() + 1) = pasos;   // guardamos la cantidad de pasos
    }

    // volvemos a convertir la matriz a una de python
    pybind11::array_t<double> result({res.rows(), res.cols()});
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);
    for (ssize_t i = 0; i < res.rows(); i++) {
        for (ssize_t j = 0; j < res.cols(); j++) {
            result_ptr[i * res.cols() + j] = res(i, j);
        }
    }
    return result;
}

PYBIND11_MODULE(interfaz, m) {
    m.def("metodo_de_la_potencia_def", &metodo_de_la_potencia_def, "Método de la potencia",
          pybind11::arg("A"), pybind11::arg("iteraciones"), pybind11::arg("tolerancia"));
}
