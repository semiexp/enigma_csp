#include <cstdint>

#include "glucose/core/Constraint.h"

namespace Glucose {

struct Solver;

class RustExtraConstraint : public Constraint {
public:
    RustExtraConstraint(void* trait_object) : trait_object_(trait_object) {}

    ~RustExtraConstraint() override = default;

    virtual bool initialize(Solver& solver) override;
    virtual bool propagate(Solver& solver, Lit p) override;
    virtual void calcReason(Solver& solver, Lit p, Lit extra, vec<Lit>& out_reason) override;
    virtual void undo(Solver& solver, Lit p) override;

private:
    void* trait_object_;
};

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
int32_t Glucose_AddGraphDivision(Glucose::Solver* solver, int32_t n_vertices, const int32_t* dom_sizes, const int32_t* domains, const int32_t* dom_lits, int32_t n_edges, const int32_t* edges, const int32_t* edge_lits);
void Glucose_Set_random_seed(Glucose::Solver* solver, double random_seed);
void Glucose_Set_rnd_init_act(Glucose::Solver* solver, int32_t rnd_init_act);
void Glucose_Set_dump_analysis_info(Glucose::Solver* solver, int32_t value);

int32_t Glucose_AddRustExtraConstraint(Glucose::Solver* solver, void* trait_object);
void Glucose_CustomPropagatorCopyReason(void* reason_vec, int32_t n_lits, int32_t* lits);
int32_t Glucose_SolverValue(Glucose::Solver* solver, int32_t lit);
void Glucose_SolverAddWatch(Glucose::Solver* solver, int32_t lit, void* wrapper_object);
int32_t Glucose_SolverEnqueue(Glucose::Solver* solver, int32_t lit, void* wrapper_object);

// Implement functions below in Rust
int32_t Glucose_CallCustomPropagatorInitialize(Glucose::Solver* solver, void* wrapper_object, void* trait_object);
int32_t Glucose_CallCustomPropagatorPropagate(Glucose::Solver* solver, void* wrapper_object, void* trait_object, int32_t p);
void Glucose_CallCustomPropagatorCalcReason(Glucose::Solver* solver, void* trait_object, int32_t p, int32_t extra, void* out_reason);
void Glucose_CallCustomPropagatorUndo(Glucose::Solver* solver, void* trait_object, int32_t p);

}
