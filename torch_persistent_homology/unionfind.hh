#include <exception>
#include <torch/extension.h>

class OutOfBoundsException : public std::exception {
  virtual const char *what() const throw() { return "Input is out of bounds!"; }
};

template <typename int_type> class UnionFind {
public:
  static int_type find(at::TensorAccessor<int_type, 1> parents, int_type u) {
    if (u < 0 || u >= parents.size(0)) {
      throw OutOfBoundsException();
    }
    auto parent = parents[u];

    if (parent == u) {
      return u;
    }
    parents[u] = UnionFind::find(parents, parent);
    return parents[u];
  };
  static void merge(at::TensorAccessor<int_type, 1> parents, int_type u,
                    int_type v) {
    if (u != v) {
      auto root_u = UnionFind::find(parents, u);
      auto root_v = UnionFind::find(parents, v);
      parents[root_u] = root_v;
    }
  };
};
