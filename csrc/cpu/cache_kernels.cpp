#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/utils.h>

#include <c10/util/irange.h>

#include "../dispatch_utils.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

namespace {

using namespace at::native;

template <typename scalar_t>
static inline void copy_stub(
    const scalar_t* __restrict__ src,
    scalar_t* __restrict__ dst,
    int64_t size) {

  using Vec = at::vec::Vectorized<scalar_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(src + d);
    data_vec.store(dst + d);
  }
  for (; d < size; d++) {
    dst[d] = src[d];
  }
}

template <typename scalar_t>
void swap_blocks_kernel_cpu(
    const scalar_t* __restrict__ src,
    scalar_t* __restrict__ dst,
    const std::map<int64_t, int64_t>& block_mapping,
    const int numel_per_block) {

  // parallel on block_mapping
  const int num_block_mapping = block_mapping.size();
  at::parallel_for(0, num_block_mapping, 0, [&](int begin, int end) {
    auto local_iter = block_mapping.begin();
    std::advance(local_iter, begin);
    for (const auto i : c10::irange(begin, end)) {
      int64_t src_block_number = local_iter->first;
      int64_t dst_block_number = local_iter->second;
      int64_t src_offset = src_block_number * numel_per_block;
      int64_t dst_offset = dst_block_number * numel_per_block;

      copy_stub(
          src + src_offset,
          dst + dst_offset,
          numel_per_block);

      local_iter++;
    }
  });
}

template <typename scalar_t>
void copy_blocks_kernel_cpu(
    std::vector<torch::Tensor>& key_caches,
    std::vector<torch::Tensor>& value_caches,
    const std::vector<int>& block_mapping,
    const int num_layers,
    const int numel_per_block) {

  scalar_t* key_cache_ptrs[num_layers];
  scalar_t* value_cache_ptrs[num_layers];
  for (const auto layer_idx : c10::irange(num_layers)) {
    key_cache_ptrs[layer_idx] = key_caches[layer_idx].data_ptr<scalar_t>();
    value_cache_ptrs[layer_idx] = value_caches[layer_idx].data_ptr<scalar_t>();
  }

  // parallel on {num_layers, num_pairs}
  int num_pairs = block_mapping.size() / 2;
  at::parallel_for(0, num_layers * num_pairs, 0, [&](int begin, int end) {
    int layer_idx{0}, pair_idx{0};
    data_index_init(begin, layer_idx, num_layers, pair_idx, num_pairs);

    for (const auto i : c10::irange(begin, end)) {
      (void)i;

      scalar_t* key_cache = key_cache_ptrs[layer_idx];
      scalar_t* value_cache = value_cache_ptrs[layer_idx];
      int src_block_number = block_mapping[2 * pair_idx];
      int dst_block_number = block_mapping[2 * pair_idx + 1];

      copy_stub(
          key_cache + src_block_number * numel_per_block,
          key_cache + dst_block_number * numel_per_block,
          numel_per_block);
      copy_stub(
          value_cache + src_block_number * numel_per_block,
          value_cache + dst_block_number * numel_per_block,
          numel_per_block);

      data_index_step(layer_idx, num_layers, pair_idx, num_pairs);
    }
  });
}

template<typename scalar_t, bool is_reverse>
void reshape_and_cache_kernel_cpu(
    const scalar_t* __restrict__ key,     // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,   // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
    scalar_t* __restrict__ value_cache,   // [num_blocks, num_heads, head_size, block_size]
    const int* __restrict__ slot_mapping, // [num_tokens]
    const int key_stride,
    const int value_stride,
    const int num_tokens,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x) {

  // parallel on {num_tokens}
  at::parallel_for(0, num_tokens, 0, [&](int begin, int end) {
    for (const auto token_idx : c10::irange(begin, end)) {
      const int slot_idx = slot_mapping[token_idx];
      const int block_idx = slot_idx / block_size;
      const int block_offset = slot_idx % block_size;

      const int num_x = num_heads * head_size / x;
      for (const auto x_idx : c10::irange(num_x)) {
        // view src as {num_tokens, num_x, x}
        // view tgt as {num_blocks, num_x, block_size, x}
        // do shuffled copy per row of x
        int src_key_idx = token_idx * key_stride + x_idx * x;
        int tgt_key_idx = block_idx * num_x * block_size * x
            + x_idx * block_size * x
            + block_offset * x;
        if constexpr (is_reverse) {
          std::swap(src_key_idx, tgt_key_idx);
        }
        memcpy(
            key_cache + tgt_key_idx,
            key + src_key_idx,
            x * sizeof(scalar_t));
      }

      for (const auto i : c10::irange(num_heads * head_size)) {
        // view src as {num_tokens, num_heads * head_size}
        // view tgt as {num_blocks, num_heads * head_size, block_size}
        int src_value_idx = token_idx * value_stride + i;
        int tgt_value_idx = block_idx * num_heads * head_size * block_size
            + i * block_size
            + block_offset;
        if constexpr (is_reverse) {
          std::swap(src_value_idx, tgt_value_idx);
        }
        value_cache[tgt_value_idx] = value[src_value_idx];
      }
    }
  });
}

} // anonymous namespace

void swap_blocks(
  torch::Tensor& src,
  torch::Tensor& dst,
  const std::map<int64_t, int64_t>& block_mapping) {

  int numel_per_block = src.numel() / src.size(0);
  VLLM_DISPATCH_FLOATING_TYPES(src.scalar_type(), "swap_blocks_cpu", [&] {
    swap_blocks_kernel_cpu<scalar_t>(
        src.data_ptr<scalar_t>(),
        dst.data_ptr<scalar_t>(),
        block_mapping,
        numel_per_block);
  });
}

void copy_blocks(
    std::vector<torch::Tensor>& key_caches,
    std::vector<torch::Tensor>& value_caches,
    const std::map<int64_t, std::vector<int64_t>>& block_mapping) {

  int num_layers = key_caches.size();
  TORCH_CHECK(num_layers == static_cast<int>(value_caches.size()));
  if (num_layers == 0) {
    return;
  }

  // Create block mapping array.
  // TODO: evaluate if need to get rid of `push_back`
  std::vector<int> block_mapping_vec;
  for (const auto& pair : block_mapping) {
    int src_block_number = pair.first;
    for (int dst_block_number : pair.second) {
      block_mapping_vec.push_back(src_block_number);
      block_mapping_vec.push_back(dst_block_number);
    }
  }

  const auto first_key = key_caches[0];
  const auto scalar_type = key_caches[0].scalar_type();
  int numel_per_block = first_key.numel() / first_key.size(0);
  VLLM_DISPATCH_FLOATING_TYPES(scalar_type, "copy_blocks_cpu", [&] {
    copy_blocks_kernel_cpu<scalar_t>(
        key_caches,
        value_caches,
        block_mapping_vec,
        num_layers,
        numel_per_block);
  });
}

void reshape_and_cache(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping) {

  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  VLLM_DISPATCH_FLOATING_TYPES(key.scalar_type(), "reshape_and_cache_cpu", [&] {
    reshape_and_cache_kernel_cpu<scalar_t, /* is_reverse */false>(
        key.data_ptr<scalar_t>(),
        value.data_ptr<scalar_t>(),
        key_cache.data_ptr<scalar_t>(),
        value_cache.data_ptr<scalar_t>(),
        slot_mapping.data_ptr<int>(),
        key_stride,
        value_stride,
        num_tokens,
        num_heads,
        head_size,
        block_size,
        x);
  });
}

void gather_cached_kv(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping) {

  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  VLLM_DISPATCH_FLOATING_TYPES(key.scalar_type(), "gather_cached_kv_cpu", [&] {
    reshape_and_cache_kernel_cpu<scalar_t, /* is_reverse */true>(
        key.data_ptr<scalar_t>(),
        value.data_ptr<scalar_t>(),
        key_cache.data_ptr<scalar_t>(),
        value_cache.data_ptr<scalar_t>(),
        slot_mapping.data_ptr<int>(),
        key_stride,
        value_stride,
        num_tokens,
        num_heads,
        head_size,
        block_size,
        x);
  });
}
