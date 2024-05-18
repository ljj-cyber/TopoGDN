#include "ATen/core/function_schema.h"
#include "unionfind.hh"
#include <ATen/Parallel.h>
#include <algorithm>
#include <iostream>
#include <torch/extension.h>

using namespace torch::indexing;

template <typename dtype> void set_to_one(at::TensorAccessor<dtype, 1> z) {
  at::parallel_for(0, z.size(0), 0, [&](int64_t begin, int64_t end) {
    for (auto i = begin; i < end; i++) {
      z[i] = 1.;
    }
  });
}

void set_to_one_tensor(torch::Tensor z) {
  AT_DISPATCH_FLOATING_TYPES(z.scalar_type(), "set_to_one", ([&] {
                               set_to_one<scalar_t>(z.accessor<scalar_t, 1>());
                             }));
}

torch::Tensor ones_tensor() {
  auto z = torch::zeros(100);
  AT_DISPATCH_FLOATING_TYPES(z.scalar_type(), "set_to_one", ([&] {
                               set_to_one<scalar_t>(z.accessor<scalar_t, 1>());
                             }));
  return z;
}

torch::Tensor uf_find(torch::Tensor parents, int u) {
  // Creating a single element tensor seems a bit hacky, but I didn't find an
  // alternative way to return a single element while supporting multiple
  // integer types.
  auto out = torch::empty(1, parents.options());
  AT_DISPATCH_INTEGRAL_TYPES(parents.scalar_type(), "uf_find", ([&] {
                               out[0] = UnionFind<scalar_t>::find(
                                   parents.accessor<scalar_t, 1>(),
                                   static_cast<scalar_t>(u));
                             }));
  return out[0];
}

void uf_merge(torch::Tensor parents, int u, int v) {
  AT_DISPATCH_INTEGRAL_TYPES(parents.scalar_type(), "uf_merge", ([&] {
                               UnionFind<scalar_t>::merge(
                                   parents.accessor<scalar_t, 1>(),
                                   static_cast<scalar_t>(u),
                                   static_cast<scalar_t>(v));
                             }));
}

std::tuple<torch::Tensor, torch::Tensor>
compute_persistence_homology(torch::Tensor filtered_v, torch::Tensor filtered_e,
                             torch::Tensor edge_index) {
  auto n_vertices = filtered_v.size(0);
  auto n_edges = filtered_e.size(0);
  auto parents = torch::arange(0, n_vertices);
  auto parents_data = parents.accessor<int64_t, 1>();
  auto persistence = torch::zeros({n_vertices, 2}, filtered_v.options());
  auto persistence1 = torch::zeros({n_edges, 2}, filtered_e.options());

  // Looks like the more eligant alternative is C++17 thus we will have to live
  // with this.
  auto sorted_out = filtered_e.sort();
  auto &&filtered_e_sorted = std::get<0>(sorted_out);
  auto &&sorted_indices = std::get<1>(sorted_out);

  persistence.index_put_({Ellipsis, 0}, filtered_v);
  // std::cout << "persistence:" << persistence << std::endl;
  auto unpaired_value = filtered_e_sorted.index({-1});

  for (auto i = 0; i < n_edges; i++) {
    auto cur_edge_index = sorted_indices[i].item<int64_t>();
    auto cur_edge_weight = filtered_e_sorted[i].item<float>();
    auto nodes = edge_index.index({Ellipsis, cur_edge_index});
    // std::cout << "nodes:" << nodes << std::endl;

    auto node1 = nodes[0].item<int64_t>();
    auto node2 = nodes[1].item<int64_t>();

    auto younger = UnionFind<int64_t>::find(parents_data, node1);
    auto older = UnionFind<int64_t>::find(parents_data, node2);
    if (younger == older) {
      persistence1.index_put_({cur_edge_index, 0}, cur_edge_weight);
      persistence1.index_put_({cur_edge_index, 1}, unpaired_value);
      continue;
    } else {
      if (filtered_v[younger].item<float>() < filtered_v[older].item<float>()) {
        // Flip older and younger, node1 and node 2
        auto tmp = younger;
        younger = older;
        older = tmp;
        tmp = node1;
        node1 = node2;
        node2 = tmp;
      }
    }

    persistence.index_put_({younger, 1}, cur_edge_weight);
    UnionFind<int64_t>::merge(parents_data, node1, node2);
  }
  // Collect roots
  auto is_root = parents == torch::arange(0, n_vertices, parents.options());
  auto root_values = filtered_v.index({is_root});
  persistence.index_put_({is_root, 0}, root_values);
  persistence.index_put_({is_root, 1}, unpaired_value);
  // persistence.index_put_({is_root, 1}, -1);
  return std::make_tuple(std::move(persistence), std::move(persistence1));
}

std::tuple<torch::Tensor, torch::Tensor> compute_persistence_homology_batched(
    torch::Tensor filtered_v, torch::Tensor filtered_e,
    torch::Tensor edge_index, torch::Tensor vertex_slices,
    torch::Tensor edge_slices) {

  auto batch_size = vertex_slices.size(0) - 1;
  auto n_nodes = filtered_v.size(0);
  auto n_edges = filtered_e.size(0);
  auto n_filtrations = filtered_v.size(1);

  auto parents = torch::arange(0, n_nodes, edge_index.options())
                     .unsqueeze(0)
                     .repeat({n_filtrations, 1});
  auto persistence =
      torch::zeros({n_filtrations, n_nodes, 2}, filtered_v.options());
  auto persistence1 =
      torch::zeros({n_filtrations, n_edges, 2}, filtered_v.options());
  auto vertex_slices_data = vertex_slices.accessor<int64_t, 1>();
  auto edge_slices_data = edge_slices.accessor<int64_t, 1>();
  for (auto i = 0; i < batch_size * n_filtrations; i++) {
    auto instance = i / n_filtrations;
    auto filtration = i % n_filtrations;
    auto vertex_slice =
        Slice(vertex_slices_data[instance], vertex_slices_data[instance + 1]);
    auto edge_slice =
        Slice(edge_slices_data[instance], edge_slices_data[instance + 1]);
    auto cur_vertices = filtered_v.index({vertex_slice, filtration});
    auto cur_edges = filtered_e.index({edge_slice, filtration});
    auto vertex_offset = vertex_slices_data[instance];
    auto cur_edge_indices =
        edge_index.index({Ellipsis, edge_slice}) - vertex_offset;
    auto cur_res =
        compute_persistence_homology(cur_vertices, cur_edges, cur_edge_indices);
    persistence.index_put_({filtration, vertex_slice}, std::get<0>(cur_res));
    persistence1.index_put_({filtration, edge_slice}, std::get<1>(cur_res));
  }
  // Below code does not work due to usage of at:Tensors in threads. Need to
  // rewrite inner part of loop to not use at:Tensor but only raw memory access.
  // at::parallel_for(
  //     0, batch_size * n_filtrations, 0, [&](int64_t begin, int64_t end) {
  //       for (auto i = begin; i < end; i++) {
  //         auto instance = i / n_filtrations;
  //         auto filtration = i % n_filtrations;
  //         auto cur_vertices = filtered_v.index({Slice(
  //             vertex_slices_data[instance], vertex_slices_data[instance +
  //             1])});
  //         auto vertex_offset = vertex_slices_data[instance];
  //         auto cur_edges = filtered_e.index({Slice(
  //             edge_slices_data[instance], edge_slices_data[instance +
  //             1])});
  //         auto cur_edge_indices =
  //             edge_index.index(
  //                 {Ellipsis, Slice(edge_slices_data[instance],
  //                                  edge_slices_data[instance + 1])}) -
  //             vertex_offset;
  //         persistence.index_put_(
  //             {filtration}, compute_persistence_homology(
  //                               cur_vertices, cur_edges,
  //                               cur_edge_indices));
  //       }
  //     });
  return std::make_tuple(persistence, persistence1);
}

/* We might need this later for a cuda implementation...

template <class BiDirIt, class Compare = std::less<>>
void inplace_merge_sort(BiDirIt first, BiDirIt last, Compare cmp = Compare{}) {
  auto const N = std::distance(first, last);
  if (N <= 1)
    return;
  auto const middle = std::next(first, N / 2);
  inplace_merge_sort(first, middle,
                     cmp); // assert(std::is_sorted(first, middle, cmp));
  inplace_merge_sort(middle, last,
                     cmp); // assert(std::is_sorted(middle, last, cmp));
  std::inplace_merge(first, middle, last,
                     cmp); // assert(std::is_sorted(first, last, cmp));
}
*/

template <typename float_t, typename int_t>
void compute_persistence_homology_raw(
    torch::TensorAccessor<float_t, 1> filtered_v,
    torch::TensorAccessor<float_t, 1> filtered_e,
    torch::TensorAccessor<int_t, 2> edge_index,
    torch::TensorAccessor<int_t, 1> parents,
    torch::TensorAccessor<int_t, 1> sorting_space,
    torch::TensorAccessor<int_t, 2> pers_indices,
    torch::TensorAccessor<int_t, 2> pers1_indices, int_t vertex_begin,
    int_t vertex_end, int_t edge_begin, int_t edge_end) {

  auto n_vertices = vertex_end - vertex_begin;
  auto n_edges = edge_end - edge_begin;

  // Argsort over constrained memory space
  // Hacky pointer magic to allow constrained inplace sort.
  // This assumes memory in contigous!!!
  int_t *sorting_begin = sorting_space.data() + edge_begin;
  int_t *sorting_end = sorting_space.data() + edge_end;
  std::stable_sort(sorting_begin, sorting_end, [&filtered_e](int_t i, int_t j) {
    return filtered_e[i] < filtered_e[j];
  });

  auto unpaired_index = *(sorting_end - 1);
  int_t unpaired_vertex_index;
  // TODO: This has assumptions on the edge filtration selected
  if (filtered_v[edge_index[unpaired_index][0]] <
      filtered_v[edge_index[unpaired_index][1]])
    unpaired_vertex_index = edge_index[unpaired_index][1];
  else
    unpaired_vertex_index = edge_index[unpaired_index][0];

  for (auto i = 0; i < n_edges; i++) {
    auto cur_edge_index = sorting_space[edge_begin + i];
    auto node1 = edge_index[cur_edge_index][0];
    auto node2 = edge_index[cur_edge_index][1];
    // TODO: This has some assumptions on the edge filtration
    int_t cur_vertex_index;
    if (filtered_v[node1] < filtered_v[node2])
      cur_vertex_index = node2;
    else
      cur_vertex_index = node1;
    auto younger = UnionFind<int_t>::find(parents, node1);
    auto older = UnionFind<int_t>::find(parents, node2);

    if (younger == older) {
      pers1_indices[cur_edge_index][0] = cur_vertex_index;
      pers1_indices[cur_edge_index][1] = unpaired_vertex_index;
      continue;
    } else {
      if (filtered_v[younger] < filtered_v[older]) {
        // Flip older and younger, node1 and node 2
        auto tmp = younger;
        younger = older;
        older = tmp;
        tmp = node1;
        node1 = node2;
        node2 = tmp;
      }
    }
    pers_indices[younger][1] = cur_vertex_index;
    UnionFind<int_t>::merge(parents, node1, node2);
  }
  // Handle roots, would make sense to do this outside as it can be
  // parallelized quite esily using torch operations.  Yet this would
  // require having access to the graph wise unpaired value, which we
  // usually dont have.
  //
  for (auto i = 0; i < n_vertices; i++) {
    auto vertex_index = vertex_begin + i;
    auto parent_value = parents[vertex_index];
    if (vertex_index == parent_value) {
      pers_indices[vertex_index][0] = vertex_index;
      pers_indices[vertex_index][1] = unpaired_vertex_index;
    }
  }
}

template <typename float_t, typename int_t>
void compute_persistence_homology_ptrs(
    torch::TensorAccessor<float_t, 2> filtered_v,
    torch::TensorAccessor<float_t, 2> filtered_e,
    torch::TensorAccessor<int_t, 2> edge_index,
    torch::TensorAccessor<int_t, 1> vertex_slices,
    torch::TensorAccessor<int_t, 1> edge_slices,
    torch::TensorAccessor<int_t, 2> parents,
    torch::TensorAccessor<int_t, 2> sorting_space,
    torch::TensorAccessor<int_t, 3> pers_ind,
    torch::TensorAccessor<int_t, 3> pers1_ind) {
  auto n_graphs = vertex_slices.size(0) - 1;
  auto n_filtrations = filtered_v.size(0);

  at::parallel_for(
      0, n_graphs * n_filtrations, 0, [&](int64_t begin, int64_t end) {
        for (auto i = begin; i < end; i++) {
          auto instance = i / n_filtrations;
          auto filtration = i % n_filtrations;
          compute_persistence_homology_raw<float_t, int_t>(
              filtered_v[filtration], filtered_e[filtration], edge_index,
              parents[filtration], sorting_space[filtration],
              pers_ind[filtration], pers1_ind[filtration],
              vertex_slices[instance], vertex_slices[instance + 1],
              edge_slices[instance], edge_slices[instance + 1]);
        }
      });
}

std::tuple<torch::Tensor, torch::Tensor>
compute_persistence_homology_batched_mt(torch::Tensor filtered_v,
                                        torch::Tensor filtered_e,
                                        torch::Tensor edge_index,
                                        torch::Tensor vertex_slices,
                                        torch::Tensor edge_slices) {
  // Changed index orders are required in order to allow slicing into
  // contingous memory regions Assumes shapes: filtered_v: [n_filtrations,
  // n_nodes] filtered_e: [n_filtrations, n_edges, 2] edge_index: [n_edges,
  // 2] vertex_slices: [n_graphs+1] edge_slices: [n_graphs+1]
  bool set_invalid_to_nan = false; // This might be relevant in the future when
                                   // we decide to handle cycles differently

  auto n_nodes = filtered_v.size(1);
  auto n_edges = filtered_e.size(1);
  auto n_filtrations = filtered_v.size(0);
  auto integer_no_grad = torch::TensorOptions();
  integer_no_grad = integer_no_grad.requires_grad(false);
  integer_no_grad = integer_no_grad.device(edge_index.options().device());
  integer_no_grad = integer_no_grad.dtype(edge_index.options().dtype());

  // Output indicators
  auto pers_ind = torch::full({n_filtrations, n_nodes, 2}, -1, integer_no_grad);
  // Already set the first part of the tuple
  pers_ind.index_put_({"...", 0}, torch::arange(0, n_nodes, integer_no_grad));
  auto pers1_ind =
      torch::full({n_filtrations, n_edges, 2}, -1, integer_no_grad);

  // Datastructure for UnionFind and sorting operations
  auto parents = torch::arange(0, n_nodes, integer_no_grad)
                     .unsqueeze(0)
                     .repeat({n_filtrations, 1});
  auto sorting_space = torch::arange(0, n_edges, integer_no_grad)
                           .unsqueeze(0)
                           .repeat({n_filtrations, 1})
                           .contiguous();

  // Double dispatch over int and float types
  AT_DISPATCH_FLOATING_TYPES(
      filtered_v.scalar_type(), "compute_persistence_batched_mt1", ([&] {
        using float_t = scalar_t;
        AT_DISPATCH_INTEGRAL_TYPES(
            edge_index.scalar_type(),
            "compute_persistence_batched_"
            "mt2",
            ([&] {
              using int_t = scalar_t;
              compute_persistence_homology_ptrs<float_t, int_t>(
                  filtered_v.accessor<float_t, 2>(),
                  filtered_e.accessor<float_t, 2>(),
                  edge_index.accessor<int_t, 2>(),
                  vertex_slices.accessor<int_t, 1>(),
                  edge_slices.accessor<int_t, 1>(),
                  parents.accessor<int_t, 2>(),
                  sorting_space.accessor<int_t, 2>(),
                  pers_ind.accessor<int_t, 3>(),
                  pers1_ind.accessor<int_t, 3>());
            }));
      }));

  // Construct tensors with values from the indicators in order to retain
  // gradient information

  // Gather the filtration values according to the indices definde in pers_ind.
  auto pers =
      filtered_v
          .index({torch::arange(0, n_filtrations, integer_no_grad).unsqueeze(1),
                  pers_ind.view({n_filtrations, -1})})
          .view({n_filtrations, n_nodes, 2});
  // Add fake value to filtered
  float_t invalid_fill_value;
  if (set_invalid_to_nan)
    invalid_fill_value = std::numeric_limits<float_t>::quiet_NaN();
  else
    invalid_fill_value = 0;

  // Gather filtration values according to the indices defined in pers_ind1.
  // Here we append a "fake value" to the filtration tensor. This value is
  // collected if no cycles were registered for the edge as the default value is
  // -1, i.e. the last element of the tensor.
  auto pers1 =
      torch::cat(
          {filtered_v, torch::full({n_filtrations, 1}, invalid_fill_value,
                                   filtered_v.options())},
          1)
          .index({torch::arange(n_filtrations, integer_no_grad).unsqueeze(1),
                  pers1_ind.view({n_filtrations, -1})})
          .view({n_filtrations, n_edges, 2});
  return std::make_tuple(std::move(pers), std::move(pers1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("set_to_one", &set_to_one_tensor, "Test inplace parallel set to one");
  m.def("ones_tensor", &ones_tensor, py::call_guard<py::gil_scoped_release>(),
        "Test parallel set to one");
  m.def("uf_find", &uf_find, py::call_guard<py::gil_scoped_release>(),
        "UnionFind find operation");
  m.def("uf_merge", &uf_merge, "UnionFind merge operation");
  m.def("compute_persistence_homology", &compute_persistence_homology,
        py::call_guard<py::gil_scoped_release>(), "Persistence routine");
  m.def("compute_persistence_homology_batched",
        &compute_persistence_homology_batched,
        py::call_guard<py::gil_scoped_release>(), "Persistence routine");
  m.def("compute_persistence_homology_batched_mt",
        &compute_persistence_homology_batched_mt,
        py::call_guard<py::gil_scoped_release>(),
        "Persistence routine multi threading");
}
