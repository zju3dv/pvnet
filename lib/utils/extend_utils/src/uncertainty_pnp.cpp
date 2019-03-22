#include <iostream>
#include <ceres/rotation.h>
#include <ceres/ceres.h>
#include <random>

using namespace std;
struct ReprojectionErrorArray {
    ReprojectionErrorArray(
            double x2d,double y2d,
            double x3d,double y3d, double z3d,
            double wxx, double wxy, double wyy,
            double fx, double fy, double px, double py
    ): x2d(x2d), y2d(y2d), x3d(x3d), y3d(y3d), z3d(z3d),
       fx(fx), fy(fy), px(px), py(py), wxx(wxx), wxy(wxy), wyy(wyy) {}

    template <typename T>
    bool operator()(const T* const pose,T* residuals) const {
        T pts3d[]={T(x3d),T(y3d),T(z3d)};
        T trans_pts3d[3];
        ceres::AngleAxisRotatePoint(pose,pts3d,trans_pts3d);
        trans_pts3d[0]+=pose[3];
        trans_pts3d[1]+=pose[4];
        trans_pts3d[2]+=pose[5];

        T proj_x=T(fx)*trans_pts3d[0]/trans_pts3d[2]+T(px);
        T proj_y=T(fy)*trans_pts3d[1]/trans_pts3d[2]+T(py);
        T diff_x=proj_x-T(x2d);
        T diff_y=proj_y-T(y2d);

        residuals[0]=T(wxx)*diff_x+T(wxy)*diff_y;
        residuals[1]=T(wxy)*diff_x+T(wyy)*diff_y;
//        cout<<residuals[0]<<" "<<residuals[1]<<endl;
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(
        double x2d,double y2d,
        double x3d,double y3d, double z3d,
        double wxx, double wxy, double wyy,
        double fx, double fy, double px, double py
    )
    {
//        cout<<wxx<<" "<<wxy<<" "<<wyy<<endl;
        return (new ceres::AutoDiffCostFunction<ReprojectionErrorArray, 2, 6>(
                new ReprojectionErrorArray(x2d,y2d,x3d,y3d,z3d,wxx,wxy,wyy,fx,fy,px,py)));
    }

    double x2d,y2d;
    double x3d,y3d,z3d;
    double fx,fy;
    double px,py;
    double wxx,wxy,wyy;
};


#ifdef __cplusplus
extern "C" {
#endif
void uncertainty_pnp(
    double* pts2d,  // pn,2
    double* pts3d,  // pn,3
    double* wgt2d,  // pn,3 wxx,wxy,wyy
    double* K,      // 3,3
    double* init_rt,// 6
    double* result_rt,// 6
    int pn
)
{
    ceres::Problem problem;
    double solution[6];
    memcpy(solution, init_rt, 6*sizeof(double));
    for (int i = 0; i < pn; ++i)
    {
        ceres::CostFunction* cost_function = ReprojectionErrorArray::Create
                (pts2d[i*2],pts2d[i*2+1],
                 pts3d[i*3],pts3d[i*3+1],pts3d[i*3+2],
                 wgt2d[i*3+0],wgt2d[i*3+1],wgt2d[i*3+2],
                 K[0],K[4],K[2],K[5]);
        problem.AddResidualBlock(cost_function, NULL, solution);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
//    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
//    cout<<summary.FullReport()<<"\n";

    memcpy(result_rt,solution,sizeof(double)*6);
}

#ifdef __cplusplus
}
#endif

int main() {
    // generate rotation and translation
    double rt[6];
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    for(int k=0;k<6;k++) rt[k]=distribution(generator);
    double fx=400.,fy=400.,px=128.,py=128.;

    double pts3d[24],pts2d[16];
    for(int i=0;i<8;++i)
    {
        pts3d[i*3]=distribution(generator);
        pts3d[i*3+1]=distribution(generator);
        pts3d[i*3+2]=distribution(generator);
        double trans_pts[3];
        ceres::AngleAxisRotatePoint(rt,&pts3d[i*3],trans_pts);
        trans_pts[0]+=rt[3];trans_pts[1]+=rt[4];trans_pts[2]+=rt[5];

        pts2d[i*2]=fx*trans_pts[0]/trans_pts[2]+px;
        pts2d[i*2+1]=fy*trans_pts[1]/trans_pts[2]+py;
    }

    distribution=std::uniform_real_distribution<double>(0.0,1e-1);
    cout<<"init ";
    for(int k=0;k<6;k++) cout<<rt[k]<<" ";
    cout<<endl;

    for(int k=0;k<6;k++)
    {
        rt[k]+=distribution(generator);
    }

    cout<<"dirty ";
    for(int k=0;k<6;k++) cout<<rt[k]<<" ";
    cout<<endl;

    ceres::Problem problem;
    double solution[6];
    memcpy(solution,rt, 6*sizeof(double));
    for (int i = 0; i < 8; ++i)
    {
        ceres::CostFunction* cost_function = ReprojectionErrorArray::Create
                (pts2d[i*2],pts2d[i*2+1],pts3d[i*3],pts3d[i*3+1],pts3d[i*3+2],1.0,0.0,1.0,fx,fy,px,py);
        problem.AddResidualBlock(cost_function, NULL, solution);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    cout<<"clean ";
    for(int k=0;k<6;k++) cout<<solution[k]<<" ";
    cout<<endl;


    return 0;
}
