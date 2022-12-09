#include <cstdint>

namespace Glucose {

struct Solver;

}

extern "C" {

Glucose::Solver* Glucose_CreateSolver();
void Glucose_DestroySolver(Glucose::Solver* solver);
int32_t Glucose_NewVar(Glucose::Solver* solver);
int32_t Glucose_NewNamedVar(Glucose::Solver* solver, const char* name);
int32_t Glucose_AddClause(Glucose::Solver* solver, int32_t* lits, int32_t n_lits);
int32_t Glucose_Solve(Glucose::Solver* solver);
int32_t Glucose_NumVar(const Glucose::Solver* solver);
int32_t Glucose_GetModelValueVar(const Glucose::Solver* solver, int32_t var);
int32_t Glucose_AddOrderEncodingLinear(Glucose::Solver* solver, int32_t n_terms, const int32_t* domain_size, const int32_t* lits, const int32_t* domain, const int32_t* coefs, int32_t constant);
int32_t Glucose_AddActiveVerticesConnected(Glucose::Solver* solver, int32_t n_vertices, const int32_t* lits, int32_t n_edges, const int32_t* edges);
int32_t Glucose_AddDirectEncodingExtensionSupports(Glucose::Solver* solver, int32_t n_vars, const int32_t* domain_size, const int32_t* lits, int32_t n_supports, const int32_t* supports);
void Glucose_Set_random_seed(Glucose::Solver* solver, double random_seed);
void Glucose_Set_rnd_init_act(Glucose::Solver* solver, int32_t rnd_init_act);
void Glucose_Set_dump_analysis_info(Glucose::Solver* solver, int32_t value);

}
