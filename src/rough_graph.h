// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "shape_memory.h"

namespace h3
{
MemoryGraph make_rough_graph(std::string& error, const MemoryGraph& core);
std::string rough_graph_param(const MemoryGraph& graph);
}
