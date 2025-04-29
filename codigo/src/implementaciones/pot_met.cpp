#include <iostream>
#include <Eigen/Dense>
#include <vector>

using namespace std;

Eigen::MatrixXd metodo_de_la_potencia_def(const Eigen::MatrixXd &A, int iteraciones, float tolerancia){
    Eigen::MatrixXd A_c = A;
    Eigen::MatrixXd res(A.cols(), A.cols() + 2);
    for (int i = 0; i < A.cols(); i++) {
        Eigen::VectorXd v = Eigen::VectorXd::Random(A.rows()).normalized();
        int pasos = 0;
        for (int j = 0; j < iteraciones; j++) {
            Eigen::VectorXd new_v = (A_c * v).normalized();
            pasos++;
            if ((v - new_v).lpNorm<Eigen::Infinity>() < tolerancia) {
                v = new_v;
                break;
            }
            v = new_v;
        }
        res.col(i) = v;
        double delta = (v.transpose() * A * v).value() / (v.transpose() * v).value();
        A_c = A_c - delta * v * v.transpose();
        res(i, A.cols()) = delta;
        res(i, A.cols() + 1) = pasos;
    }
    return res;
}

int main() {
    std::vector<std::tuple<Eigen::MatrixXd, std::string>> tests = {
        std::make_tuple((Eigen::MatrixXd(3, 3) << 7, 2, 3, 
                                                  0, 2, 0, 
                                                  -6, -2, -2).finished(), "4, 2, 1"),
        std::make_tuple((Eigen::MatrixXd(3, 3) << 1, 0, 1, 
                                                  0, 1, 1, 
                                                  0, 0, 2).finished(), "2, 1, 1"),
        std::make_tuple((Eigen::MatrixXd(3, 3) << 1, 1, 1, 
                                                  0, 1, 1, 
                                                  0, 0, 2).finished(), "2, 1, 1"),
        std::make_tuple((Eigen::MatrixXd(4, 4) << 1, 0, 0, 0, 
                                                  0, 2, 0, 0, 
                                                  0, 0, 3, 0,
                                                  0, 0, 0, 4).finished(), "4, 3, 2, 1"),
        std::make_tuple((Eigen::MatrixXd(5, 5) << 4, 1, 2, 3, 1,
                                                  0, 3, 1, 2, 0, 
                                                  0, 0, 2, 1, 0,
                                                  0, 0, 0, 1, 0,
                                                  0, 0, 0, 0, 0).finished(), "4, 3, 2, 1, 0"),
        std::make_tuple((Eigen::MatrixXd(3, 3) << 1, 1, 1, 
                                                  0, 1, 1, 
                                                  0, 0, 1).finished(), "1, 1, 1")


    };

    for(auto t : tests){
        std::cout << "MATRIZ: " << get<1>(t) << "\n" << get<0>(t) << "\n" << metodo_de_la_potencia_def(get<0>(t), 10000, 0) << "\n" << std::endl;
    }
    return 0;
}
