#include "cadical_bridge.h"

#include "cadical.hpp"
#include "ext_subgraph_connectivity.hpp"

int to_cadical_lit(int l) {
    if (l & 1) {
        return -((l >> 1) + 1);
    } else {
        return (l >> 1) + 1;
    }
}

extern "C" {

CaDiCaL::Solver* CaDiCaL_CreateSolver() {
    CaDiCaL::Solver* solver = new CaDiCaL::Solver();
    solver->set("chrono", 0);  // TODO: do this explicitly
    return solver;
}

void CaDiCaL_DestroySolver(CaDiCaL::Solver* solver) {
    delete solver;
}

void CaDiCaL_AddClause(CaDiCaL::Solver* solver, int32_t* lits, int32_t n_lits) {
    for (int i = 0; i < n_lits; ++i) {
        solver->add(to_cadical_lit(lits[i]));
    }
    solver->add(0);
}

int32_t CaDiCaL_Solve(CaDiCaL::Solver* solver) {
    int res = solver->solve();
    if (res == 10) return 1;
    return 0;
}

int32_t CaDiCaL_GetModelValueVar(CaDiCaL::Solver* solver, int32_t var) {
    int res = solver->val(var + 1);
    return (res > 0) ? 1 : 0;
}

void CaDiCaL_AddActiveVerticesConnected(CaDiCaL::Solver* solver, int32_t n_vertices, const int32_t* lits, int32_t n_edges, const int32_t* edges) {
    std::vector<int> c_lits;
    for (int i = 0; i < n_vertices; ++i) {
        c_lits.push_back(to_cadical_lit(lits[i]));
    }
    std::vector<std::pair<int, int>> g_edges;
    for (int i = 0; i < n_edges; ++i) {
        g_edges.push_back({edges[i * 2], edges[i * 2 + 1]});
    }
    solver->add_extra(std::make_unique<CaDiCaL::SubgraphConnectivity>(std::move(c_lits), std::move(g_edges)));
}

}
