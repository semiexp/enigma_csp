#include "glucose_bridge.h"

#include "core/Solver.h"

extern "C" {

Glucose::Solver* Glucose_CreateSolver() {
    Glucose::Solver* solver = new Glucose::Solver();
    return solver;
}

void Glucose_DestroySolver(Glucose::Solver* solver) {
    delete solver;
}

int Glucose_NewVar(Glucose::Solver* solver) {
    return solver->newVar();
}

int32_t Glucose_AddClause(Glucose::Solver* solver, int32_t* lits, int32_t n_lits) {
    Glucose::vec<Glucose::Lit> lits_vec;
    for (int i = 0; i < n_lits; ++i) {
        lits_vec.push(Glucose::Lit{lits[i]});
    }
    return solver->addClause(lits_vec);
}

int32_t Glucose_Solve(Glucose::Solver* solver) {
    return solver->solve();
}

int32_t Glucose_NumVar(const Glucose::Solver* solver) {
    return solver->nVars();
}

int32_t Glucose_GetModelValueVar(const Glucose::Solver* solver, int32_t var) {
    return solver->modelValue(Glucose::Var(var)) == l_True ? 1 : 0;
}

}
