#include <cstdint>

namespace Glucose {

struct Solver;

}

extern "C" {

Glucose::Solver* Glucose_CreateSolver();
void Glucose_DestroySolver(Glucose::Solver* solver);
int32_t Glucose_NewVar(Glucose::Solver* solver);
int32_t Glucose_AddClause(Glucose::Solver* solver, int32_t* lits, int32_t n_lits);
int32_t Glucose_Solve(Glucose::Solver* solver);
int32_t Glucose_NumVar(const Glucose::Solver* solver);
int32_t Glucose_GetModelValueVar(const Glucose::Solver* solver, int32_t var);
int32_t Glucose_AddOrderEncodingLinear(Glucose::Solver* solver, int32_t n_terms, const int32_t* domain_size, const int32_t* lits, const int32_t* domain, const int32_t* coefs, int32_t constant);

}
