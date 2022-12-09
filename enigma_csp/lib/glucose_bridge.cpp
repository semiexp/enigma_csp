#include "glucose_bridge.h"

#include "core/Solver.h"
#include "constraints/DirectEncodingExtension.h"
#include "constraints/Graph.h"
#include "constraints/OrderEncodingLinear.h"

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

int32_t Glucose_NewNamedVar(Glucose::Solver* solver, const char* name) {
    std::string name_str(name);
    return solver->newNamedVar(name_str);
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

int32_t Glucose_AddOrderEncodingLinear(Glucose::Solver* solver, int32_t n_terms, const int32_t* domain_size, const int32_t* lits, const int32_t* domain, const int32_t* coefs, int32_t constant) {
    std::vector<Glucose::LinearTerm> terms;
    int lits_offset = 0, domain_offset = 0;
    for (int i = 0; i < n_terms; ++i) {
        std::vector<Glucose::Lit> term_lits;
        for (int j = 0; j < domain_size[i] - 1; ++j) {
            term_lits.push_back(Glucose::Lit{lits[lits_offset++]});
        }
        std::vector<int> term_domain;
        for (int j = 0; j < domain_size[i]; ++j) {
            term_domain.push_back(domain[domain_offset++]);
        }
        terms.push_back(Glucose::LinearTerm{ term_lits, term_domain, coefs[i] });
    }
    return solver->addConstraint(std::make_unique<Glucose::OrderEncodingLinear>(std::move(terms), constant)) ? 1 : 0;
}

int32_t Glucose_AddActiveVerticesConnected(Glucose::Solver* solver, int32_t n_vertices, const int32_t* lits, int32_t n_edges, const int32_t* edges) {
    std::vector<Glucose::Lit> g_lits;
    for (int i = 0; i < n_vertices; ++i) {
        g_lits.push_back(Glucose::Lit{lits[i]});
    }
    std::vector<std::pair<int, int>> g_edges;
    for (int i = 0; i < n_edges; ++i) {
        g_edges.push_back({edges[i * 2], edges[i * 2 + 1]});
    }
    return solver->addConstraint(std::make_unique<Glucose::ActiveVerticesConnected>(std::move(g_lits), std::move(g_edges))) ? 1 : 0;
}

int32_t Glucose_AddDirectEncodingExtensionSupports(Glucose::Solver* solver, int32_t n_vars, const int32_t* domain_size, const int32_t* lits, int32_t n_supports, const int32_t* supports) {
    std::vector<std::vector<Glucose::Lit>> g_lits;
    int lits_offset = 0;
    for (int i = 0; i < n_vars; ++i) {
        std::vector<Glucose::Lit> var_lits;
        for (int j = 0; j < domain_size[i]; ++j) {
            var_lits.push_back(Glucose::Lit{lits[lits_offset++]});
        }
        g_lits.push_back(var_lits);
    }
    std::vector<std::vector<int>> g_supports;
    for (int i = 0; i < n_supports; ++i) {
        std::vector<int> s;
        for (int j = 0; j < n_vars; ++j) {
            s.push_back(supports[i * n_vars + j]);
        }
        g_supports.push_back(s);
    }
    return solver->addConstraint(std::make_unique<Glucose::DirectEncodingExtensionSupports>(std::move(g_lits), std::move(g_supports)));
}

uint64_t Glucose_SolverStats_decisions(Glucose::Solver* solver) {
    return solver->decisions;
}

uint64_t Glucose_SolverStats_propagations(Glucose::Solver* solver) {
    return solver->propagations;
}

uint64_t Glucose_SolverStats_conflicts(Glucose::Solver* solver) {
    return solver->conflicts;
}

void Glucose_Set_random_seed(Glucose::Solver* solver, double random_seed) {
    solver->random_seed = random_seed;
}

void Glucose_Set_rnd_init_act(Glucose::Solver* solver, int32_t rnd_init_act) {
    solver->rnd_init_act = rnd_init_act != 0;
}

void Glucose_Set_dump_analysis_info(Glucose::Solver* solver, int32_t value) {
    solver->dump_analysis_info = value != 0;
}

}
