# Hexagon-mlir-doc
# Hexagon-MLIR Compiler Internals

A source-level reference to the Hexagon-MLIR compiler: every dialect, every pass in the order the pass manager runs it, and every point where compiled code calls into the on-device runtime.

All paths are relative to `qcom_hexagon_backend/` unless they start with `docs/`, `test/` or `backend/`.

---

## Contents

1. [The big picture](#1-the-big-picture)
2. [Hardware concepts the compiler targets](#2-hardware-concepts-the-compiler-targets)
3. [Entry points and option plumbing](#3-entry-points-and-option-plumbing)
4. [Dialects](#4-dialects)
5. [The pass manager, in order](#5-the-pass-manager-in-order)
6. [Pass reference, phase by phase](#6-pass-reference-phase-by-phase)
7. [Passes outside the main pipeline](#7-passes-outside-the-main-pipeline)
8. [From LLVM dialect to object code](#8-from-llvm-dialect-to-object-code)
9. [Compiler ↔ runtime connection](#9-compiler--runtime-connection)
10. [Option reference](#10-option-reference)
11. [Debugging the compiler](#11-debugging-the-compiler)

---

## 1. The big picture

```
 ┌──────────────────────┐            ┌──────────────────────────┐
 │ Triton kernel (.py)  │            │ PyTorch model            │
 └──────────┬───────────┘            └────────────┬─────────────┘
            │ make_ttir (Triton passes)           │ torch-mlir
            ▼                                     │ (linalg + tm_tensor)
 ┌──────────────────────┐                         │
 │ Triton IR (tt.*)     │                         │
 └──────────┬───────────┘                         │
            │ triton-shared-opt                   │
            │ --triton-to-linalg-experimental     │
            ▼                                     ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ Linalg-on-tensors MLIR  (+ ttx, tptr, tm_tensor leftovers)  │
 └──────────────────────────────┬──────────────────────────────┘
                                │ LinalgToLLVMPass  (~100 passes, §5)
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ LLVM-dialect MLIR, with calls to hexagon_runtime_*, hexkl_*,│
 │ mlirAsyncRuntime*, qhmath_*, _hexagon_runtime_*             │
 └──────────────────────────────┬──────────────────────────────┘
                                │ translate to LLVM IR + O3          (§8)
                                │ link runtime bitcode on demand     (§9)
                                │ Hexagon TargetMachine codegen
                                ▼
                         kernel .o (bytes)
                                │ backend/hexagon_executor.py
                                │ hexagon-clang: wrapper.cpp + .o + static libs
                                ▼
                         lib<kernel>.so  → device / simulator
```

The compiler is a **single MLIR meta-pass**, `LinalgToLLVMPass`
(`lib/Conversion/LinalgToLLVM/LinalgToLLVMPass.cpp`). It builds a nested `PassManager` containing the whole pipeline and runs it. Everything before it belongs to a frontend (Triton or torch-mlir). Everything after it is LLVM-level work: translation, optimization, runtime linking, codegen.

---

## 2. Hardware concepts the compiler targets

Most of the passes only make sense in terms of these features:

| Concept | What it is | How the compiler models it |
|---|---|---|
| **DDR** | Main memory: large, high latency. | Default memref address space `0`. |
| **VTCM** | Vector Tightly Coupled Memory: fast on-chip scratchpad that HVX/HMX read efficiently. Its size depends on the device; the compiler assumes **2 MB** (`vtcmSizeInBytes` in `include/hexagon/Conversion/LinalgToLLVM/Common.h`). | Memref address space **`1`**, `!crouton.crouton<..., vtcm>`. Allocated by `hexagonmem.alloc`, which becomes a runtime call into a VTCM pool. |
| **HVX** | Hexagon Vector eXtensions: SIMD unit with **128-byte** vectors (`nativeVectorWidthInBytes = 128`). That is 32×f32, 64×f16 or 128×i8. | `arch_features = +hvxvNN,+hvx-length128b`. The tiling/vectorization passes target inner loops of exactly `128 / sizeof(elem)` elements. |
| **HMX** | Hexagon Matrix eXtensions: matrix-multiply engine working on 32×32 f16 tiles in special layouts. | Reached only through the **HexKL** library (`hexkl` dialect). |
| **Layouts: RM / AH / WH** | Row-Major, Activation-Horizontal (HMX activation layout), Weight-Horizontal (HMX weight layout). | `hexkl.*rm_to_ah*`, `*rm_to_wh*`, `*ah_to_rm*` ops; `PreprocessWeightsForHMX` converts constant weights to WH at compile time. |
| **Crouton** | Hexagon's tiled activation format: a tensor stored as a table of pointers to **2 KB** blocks (e.g. 8×4×32 f16 = 2048 B). | `!crouton.crouton<shape x elt, vtcm>` type; `linalg.pack` with crouton tile sizes. |
| **User DMA** | Hardware DMA engine that copies DDR↔VTCM asynchronously. | `memref.dma_start` / `memref.dma_wait`, which become `hexagon_runtime_dma_*`. |
| **QuRT threads** | Hardware threads under the QuRT RTOS. | `scf.forall` → `async.execute` → `mlirAsyncRuntime*` calls into a QuRT thread pool. |
| **32-bit target** | Pointers and `index` are 32-bit. | `ConvertIndexToLLVM{indexBitwidth=32}`, data layout `p:32:32:32`. |

---

## 3. Entry points and option plumbing

### 3.1 Triton flow (`backend/compiler.py`)

`HexagonBackend.add_stages()` registers these stages:

| Stage | Function | Work done |
|---|---|---|
| `ttir` | `make_ttir` | Upstream Triton passes: inliner, combine, canonicalize, CSE, LICM, symbol DCE. |
| `ttsharedir` | `ttir_to_ttsharedir` | Writes the IR to a temporary file and runs `triton-shared-opt --triton-to-linalg-experimental`. **This is where Triton IR becomes Linalg.** The output can still contain `tptr.*` and `tt.extern_elementwise`. |
| `o` | `ttsharedir_to_obj` | Regex-renames `tt.scan` to `ttx.scan`, parses the module with all Hexagon dialects loaded, extracts the kernel name and return types into metadata, then calls `qcom_hexagon_backend.translate_linalg_to_obj(mod, options_map)`. |
| `llir` *(HTP only)* | `ttsharedir_to_llir` | Same pipeline, but returns LLVM IR text. |
| `so` *(HTP only)* | `obj_to_so` | Links the `.o` into a `.so` with `HexagonExecutor.generate_shared_object`. |

`backend/htp.py:triton_htp_compile()` builds standalone kernels for QNN/HTP custom ops (`htp_kernel_gen=True`).

### 3.2 torch-mlir flow (`backend/torch_mlir_hexagon_launcher.py`)

`TorchMLIRHexagonLauncher.mlir_to_obj()` feeds the Linalg/TmTensor module from torch-mlir into the **same** `translate_linalg_to_obj`. After this point both frontends share the same compiler.

### 3.3 pybind layer (`python/`)

- `triton_qcom_hexagon_backend.cc` exposes `load_dialects`, `parse_mlir_module_from_str`, `extract_func_name_from_mlir_module`, `get_return_list`, `translate_linalg_to_obj` and `translate_linalg_to_llvmir`.
- `triton_qcom_hexagon_backend_api.cc:translateLinalgToObj()` is the C++ driver (§8).

### 3.4 How options flow

```
HexagonOptions (backend/hexagon_options.py, dataclass; per-kernel kwargs override)
   │  {k: str(v)}         ← every value is stringified ("True"/"False"/numbers)
   ▼
options_map : unordered_map<string,string>
   │  setLinalgToLLVMOptions()   (lib/Target/Linalg_MLLVMIR/MLLVMIRTranslation.cpp)
   ▼
LinalgToLLVMOptions      (generated from include/hexagon/Conversion/LinalgToLLVM/Passes.td)
   │  per-pass option structs built by small lambdas (setVTCMTiling, setFusion, ...)
   ▼
individual passes
```

`setLinalgToLLVMOptions` only copies the fields that Python sends. Any other field, such as `puntBuffer`, `rvo`, `addFastMath`, `expandBoolVec`, `enableHexagonRoutines` or `enableSlicing`, **keeps its TableGen default**. §10 has the full table.

---

## 4. Dialects

Six custom dialects live in `include/hexagon/Dialect/` (TableGen) and `lib/Dialect/` (C++). Each subsection below covers the dialect's role, its ops, which pass **creates** each op and which pass **consumes** it.

### 4.1 `hexagonmem`: Hexagon memory management

**Files:** `include/hexagon/Dialect/HexagonMem/IR/HexagonMemOps.td`, `lib/Dialect/HexagonMem/`
**Role:** This dialect is the compiler's model of Hexagon memory. It provides allocation, deallocation and copy ops that know whether a buffer lives in DDR or VTCM, and that work on both plain memrefs and crouton tables.
**Why it matters:** Upstream `memref.alloc` lowers to `malloc`, which cannot place a buffer in VTCM. Every VTCM buffer that VTCM tiling creates becomes a `hexagonmem.alloc` and then a runtime call into the VTCM pool, so without this dialect VTCM tiling can't do anything.

| Op | Semantics | Created by | Lowered by → runtime symbol |
|---|---|---|---|
| `hexagonmem.alloc` | Allocates a memref (one contiguous chunk) or a crouton (N×2 KB blocks) in VTCM (address space 1 / `vtcm`) or DDR. Takes dynamic sizes and an `alignment` attribute (default 128). | `ConvertToHexagonmem` (from `memref.alloc`), `MemoryOffsets` (fallback buffer) | `HexagonMemToLLVM` → `hexagon_runtime_alloc_1d_dsp(bytes, align, isVtcm)` / `hexagon_runtime_alloc_2d_dsp(nBlocks, blockSize, align, isVtcm)`. **Flat DDR allocs are turned back into `memref.alloc`**, which becomes `malloc`/`aligned_alloc`. |
| `hexagonmem.dealloc` | Frees a memref or crouton (all of its blocks). | `ConvertToHexagonmem` (from `memref.dealloc`) | `HexagonMemToLLVM` → `hexagon_runtime_free_1d_dsp` / `_2d_dsp` (DDR → `memref.dealloc` → `free`). The runtime functions are marked as having side effects so LLVM O3 cannot delete them. |
| `hexagonmem.copy` | Copies between memrefs or croutons that have the same shape and element type. Address spaces may differ (DDR↔VTCM). | `ConvertToHexagonmem` (from `memref.copy`) | `HexagonMemToLLVM` → `hexagon_runtime_copy_dsp(dst, src, nbytes, isDstVtcm, isSrcVtcm)`. If `enableHexagonmemCopyToDMA` is on, `HexmemCpyToDMA` first rewrites it into DMA ops. |
| `hexagonmem.convert_layout` | Destination-style copy into a new layout described by an affine map, with optional padding. | Frontend / tests | `ConvertLayout` pass → `linalg.generic` (opt-only, §7). |
| `hexagonmem.memref_to_crouton` | Reinterprets a contiguous memref as a crouton table. The last dimension is one crouton (2 KB) and the other dimensions form the table shape. | Layout lowering | `HexagonMemToLLVM` → `hexagon_runtime_build_crouton_dsp(src, nbytes)` (creates a buffer alias). |
| `hexagonmem.crouton_to_memref` | Gets a contiguous memref back from a crouton. Copies if the blocks aren't contiguous. | Layout lowering | `HexagonMemToLLVM` → `hexagon_runtime_get_contiguous_memref_dsp(src)`. |

### 4.2 `crouton`: the crouton type

**Files:** `include/hexagon/Dialect/Crouton/IR/CroutonTypes.td`, `lib/Dialect/Crouton/`
**Role:** A dialect that defines only a type, `!crouton.crouton<shape x elt[, vtcm]>`. It is a descriptor holding a statically shaped **table of pointers to 2 KB chunks**. The element type is `f16` or `i8`, and the `vtcm` flag gives the location. The type carries no layout, so the same blocks can be reinterpreted under different layouts.
**Why it matters:** It is the IR representation of Hexagon's indirect tensor format, used by HVX/HMX convolution kernels. `hexagonmem` ops accept it (`AnyMemRefOrCroutonType`), and the runtime implements it as `HexagonBufferAlias` (§9).

### 4.3 `hexkl`: HexKL / HMX matrix multiply

**Files:** `include/hexagon/Dialect/HexKL/IR/HexKLOps.td`, `lib/Dialect/HexKL/`
**Role:** Abstracts the **Hexagon Kernel Library** (HexKL), which is the only way this compiler uses the HMX matrix unit. There are three levels of ops:

**(a) High level**

| Op | Semantics | Created by | Consumed by |
|---|---|---|---|
| `hexkl.matmul ins(lhs, rhs) outs(out)` | f16 × f16 → f16/f32 matmul. Destination-style; works on tensors or memrefs. | `MatmulToHexKL` (from `linalg.matmul`) | `LowerHexKLMatmulToMacro` (macro mode), `DecomposeHexKLMatmul` (micro mode), or directly `HexKLToLLVM` → `hexkl_matmul_f16f16_f32(M, N, K, out, lhs, rhs)` (implemented in `bin/runtime/src/HexKLAPI.cpp`). |

**(b) Micro ops**: one 32×32-tile HMX primitive each. All are created by `DecomposeHexKLMatmul` and lowered by `HexKLToLLVM` to `hexkl_micro_hmx_*` (in `libhexkl_micro.a`). Most take an `hmx_block` memref, the VTCM work area. The HMX config region sits at the end of that block, at `vtcm_size - hexkl_micro_hmx_config_size()`.

| Op | Runtime function | Meaning |
|---|---|---|
| `micro_hmx_setup_acc_read_f16` | `hexkl_micro_hmx_setup_acc_read_f16` | Configure accumulator readback. |
| `micro_hmx_acc_clear_f16` | `hexkl_micro_hmx_acc_clear_f16` | Zero the accumulator. |
| `micro_hmx_copy_submatrix_to_f16` | `hexkl_micro_hmx_copy_submatrix_to_f16` | DDR flat f16 → 32×32 tile in VTCM. |
| `micro_hmx_rm_to_ah_f16` | `hexkl_micro_hmx_rm_to_ah_f16` | Tile: row-major → activation layout. |
| `micro_hmx_rm_to_wh_f16` | `hexkl_micro_hmx_rm_to_wh_f16` | DDR weight tile → weight layout in VTCM. |
| `micro_hmx_mm_f16` | `hexkl_micro_hmx_mm_f16` | Accumulating 32×32 HMX GEMM. |
| `micro_hmx_acc_read_f16` | `hexkl_micro_hmx_acc_read_f16` | Accumulator → VTCM. |
| `micro_hmx_ah_to_rm_f16` | `hexkl_micro_hmx_ah_to_rm_f16` | Tile: activation layout → row-major. |
| `micro_hmx_copy_f16_to_f32_submatrix` | `hexkl_micro_hmx_copy_f16_to_f32_submatrix` | VTCM f16 tile → DDR f32. |

**(c) Macro ops**: whole-matmul library calls (`libhexkl_macro.a`). They are modeled as ops instead of being lowered straight to calls, so that the IR can later hoist init/lock and remove redundant cache maintenance. All are created by `LowerHexKLMatmulToMacro` and lowered by `HexKLToLLVM`.

| Op | Runtime function |
|---|---|
| `macro_initialize` / `macro_finalize` | `hexkl_macro_initialize` / `hexkl_macro_finalize` |
| `macro_lock_hmx` / `macro_unlock_hmx` | `hexkl_macro_lock_hmx` / `hexkl_macro_unlock_hmx` |
| `macro_rm_to_ah_f16_inplace(m,k,ptr)` | `hexkl_macro_f16_rm_to_f16_ah_inplace` |
| `macro_ah_to_rm_f16_inplace(m,n,ptr)` | `hexkl_macro_f16_ah_to_f16_rm_inplace` |
| `macro_mm_f16(m,n,k,out,act,wgt)` | `hexkl_macro_mm_f16` |
| `qurt_mem_cache_clean(ptr,size,op,cache)` | `qurt_mem_cache_clean` (QuRT). Cache flush/invalidate so that HMX/DMA and the CPU agree on memory contents. |

**Why it matters:** HMX gives much higher matmul throughput than HVX. The pipeline enables this path with `enableHexKL=True`. HexKL is marked *experimental* in the README.

### 4.4 `ttx`: Triton extensions

**Files:** `include/hexagon/Dialect/TTX/IR/TTXOps.td`, `lib/Dialect/TTX/`
**Role:** Holds Triton ops that `triton-shared` does not lower to Linalg.

| Op | Semantics | Created by | Consumed by |
|---|---|---|---|
| `ttx.scan` (+ `ttx.scan.return`) | General prefix scan along `axis` with a user region; `reverse` flag. | Python regex `tt.scan → ttx.scan` in `ttsharedir_to_obj` | `LowerTTX` → `scf.for` with an accumulator. |
| `ttx.cumsum` | Cumulative sum, destination-style. | triton-shared output | `LowerTTX` → scalar-accumulation loops (optimized 1-D case, general n-D case). |

### 4.5 `tptr`: temporary pointer dialect

**Files:** `include/hexagon/Dialect/HexagonTPtr/IR/*.td`, `lib/Dialect/HexagonTPtr/`
**Role:** Supports raw pointer arithmetic coming out of triton-shared. It depends on the upstream `ptr` dialect and defines a `#tptr.default_memory_space` attribute.

| Op | Semantics | Lowered by `LowerTPtr` to |
|---|---|---|
| `tptr.type_offset <T> : iN` | `sizeof(T)` (folds to a constant) | Constant. |
| `tptr.from_memref` | memref → `!ptr.ptr` | Aligned-pointer extraction. |
| `tptr.ptradd base, off` | Pointer + offset | Index add → `llvm.add` + `inttoptr`. |
| `tptr.to_memref` | `!ptr.ptr` → static memref | `inttoptr` + a constructed `!llvm.struct` memref descriptor, cast back to memref. |

### 4.6 `tm_tensor`: torch-mlir TMTensor subset

**Files:** `include/hexagon/Dialect/TmTensor/IR/*.td`, `lib/Dialect/TmTensor/`
**Role:** Lets the compiler ingest torch-mlir's `tm_tensor` ops without depending on torch-mlir.

| Op | Semantics | Lowered by |
|---|---|---|
| `tm_tensor.attention ins(Q,K,V,mask) outs(out)` | Scaled dot-product attention: `softmax(QKᵀ/√d + mask)·V`, with Q:[B,Sq,D], K/V:[B,Skv,D], M:[B,Sq,Skv]. | `HexagonLowerTmTensor` → transpose, `batch_matmul`, scale, mask add, numerically stable softmax (max, exp(x−max), sum, div), `batch_matmul`, all as Linalg. |

### 4.7 Upstream dialects the pipeline relies on

`func, arith, math, linalg, tensor, affine, scf, async, cf, bufferization, vector, memref, LLVM, quant, ml_program, ub`. They are declared in `LinalgToLLVMPass::getDependentDialects`.

---

## 5. The pass manager, in order

The full pipeline, in the order built by `LinalgToLLVMPass::runOnOperation()` (`lib/Conversion/LinalgToLLVM/LinalgToLLVMPass.cpp`).

- **Kind:** **H** = Hexagon custom pass (in this repo), **U** = upstream MLIR pass.
- **Scope:** `M` = module pass, `F` = nested on each `func.func`.
- **Condition:** the option that must be on; "always" means unconditional.

Before any pass runs, `setTargetTriple` (`"hexagon"`) and `setDataLayout` are applied to the module.

### Phase A: Ingest and normalize

| # | Pass | Kind | Scope | Condition |
|---|---|---|---|---|
| A1 | `LLVMRequestCWrappers` | U | F | the function returns a value |
| A2 | `LowerTTX` | H | F | always |
| A3 | `LowerLibdevice` | H | M | always |
| A4 | `LowerTPtr` | H | F | always |
| A5 | `HexagonLowerTmTensor` | H | F | always |
| A6 | `ReduceContractionRank` | H | F | always |
| A7 | `LinalgFoldUnitExtentDims` | U | M | always |
| A8 | `Canonicalizer`, `CSE` | U | M | always |
| A9 | `HexagonPuntBuffer` | H | F | `puntBuffer` (default on) |
| A10 | `Canonicalizer` | U | M | always |
| A11 | `ConversionToFp16` | H | F | `enableConversionToFp16` |
| A12 | `Canonicalizer`, `CSE` | U | M | always |
| A13 | `OptimizeExtfTruncfOp` | H | F | always |
| A14 | `DivToMulOptimization` | H | F | always |
| A15 | `quant::LowerQuantOps` | U | F | always |
| A16 | `ConvertElementwiseToLinalg` | U | M | always |
| A17 | `CSE` | U | M | always |

### Phase B: Matmul and convolution specialization

| # | Pass | Kind | Scope | Condition |
|---|---|---|---|---|
| B1 | `FoldResourceTranspose` | H | F | `enableHexKL` |
| B2 | `FoldCastsIntoMatmul` | H | F | `enableHexKL` |
| B3 | `MatmulToHexKL{mode}` | H | F | `enableHexKL` |
| B4 | `PreprocessWeightsForHMX` | H | F | `enableHexKL && hexKLMode=="macro"` |
| B5 | `Canonicalizer` | U | M | `enableHexKL` |
| B6 | `MatmulToConv` | H | F | `enableMatmulToConv && enableSeedLayoutConversions` (unit tests) |
| B7 | `SeedLayoutConversions` | H | F | same as B6 |
| B8 | `HexagonExtendPack` | H | F | same as B6 |
| B9 | `CSE` | U | M | same as B6 |
| B10 | `ConvTiling` | H | F | `enableConvTiling` |
| B11 | `Canonicalizer` | U | M | `enableConvTiling` |
| B12 | `PreprocessTiledConv2D` | H | F | `enableSeedLayoutConversions` |
| B13 | `ScheduleMatmulForHVX` | H | F | always |
| B14 | `LinalgGeneralize` | H | F | always |
| B15 | `HexagonRVO` | H | F | `returnValueOptimization` (default on) |
| B16 | `Canonicalizer`, `CSE` | U | M | always |

### Phase C: Graph-level transforms, fusion and VTCM tiling (tensor level)

| # | Pass | Kind | Scope | Condition |
|---|---|---|---|---|
| C1 | `FormSCFThreads` | H | F | `enableSCFThreading` (asserts multithreading, VTCM tiling and scratch are off) |
| C2 | `HexagonFusion{far,fdmu}` | H | F | `fusion` (default on) |
| C3 | `EraseUnusedLinalgOperands` | H | M | always |
| C4 | `Canonicalizer`, `CSE` | U | M | always |
| C5 | `HexagonSlicing{factor}` | H | M | `enableSlicing` |
| C6 | `DecomposeTensorConcat` | H | F | always |
| C7 | `ForceHVXCrouton` | H | F | `forceHVXCroutonization` |
| C8 | `HexagonExtendPack{upper=false}` | H | F | `forceHVXCroutonization` |
| C9 | `LowerPack` | H | F | always |
| C10 | `CSE` | U | M | always |
| C11 | `VTCMTiling{tileSizes, vtcmBudget}` | H | F | `enableVTCMTiling \|\| scratch>0` (default on) |
| C12 | `Canonicalizer` | U | M | with C11 |
| C13 | `SplitReduceGeneric` | H | F | `enableSplitReduceGeneric` |
| C14 | `FormVirtualThreads` | H | F | `enableMultiThreading` |
| C15 | `RemoveMLProgram` | H | M | always |
| C16 | `LinalgFoldUnitExtentDims` | U | M | always |

### Phase D: HVX vectorization

| # | Pass | Kind | Scope | Condition |
|---|---|---|---|---|
| D1 | `HexagonTiling{interchange, splitRange, splitReduction}` | H | M | `enableVectorization` (default on) |
| D2 | `LinalgFoldUnitExtentDims`, `Canonicalizer`, `CSE` | U | M | always |
| D3 | `SmallExponentToMultiply` | H | M | always |
| D4 | `HoistScalarOps` | H | F | always |
| D5 | `EraseUnusedLinalgOperands` | H | M | always |
| D6 | `CSE` | U | M | always |
| D7 | `LoopInvariantCodeMotion` | U | F | always |
| D8 | `Canonicalizer`, `CSE` | U | M | always |
| D9 | `LoopInvariantCodeMotion` | U | F | always (second round) |
| D10 | `HexagonVectorization` | H | M | `enableVectorization` |
| D11 | `RewriteUBPoisonToZero` | H | M | always |
| D12 | `HexagonVectorLowering` | H | M | always |
| D13 | `Canonicalizer` | U | M | always |
| D14 | `HexagonAddFastMath` | H | M | `addFastMath` (default on) |
| D15 | `FoldMulFByZero` | H | F | `addFastMath` |
| D16 | `Canonicalizer` | U | M | `addFastMath` |
| D17 | `memref::ResolveShapedTypeResultDims` | U | M | always |

### Phase E: Bufferization (tensors → memrefs)

All of phase E runs only if `enableBufferization` (default on).

| # | Pass | Kind | Scope | Condition |
|---|---|---|---|---|
| E1 | `EmptyTensorElimination` | U | M | always |
| E2 | `EraseVectorToTensorWriteback` | H | F | always |
| E3 | `OneShotBufferize{bufferizeFunctionBoundaries, allowReturnAllocsFromLoops}` | U | M | always |
| E4 | `CSE`, `Canonicalizer` | U | M | always |
| E5 | `HexagonDoubleBufferGenericS1` | H | F | `enableDoubleBuffering` |
| E6 | `BufferLoopHoisting` | U | F | always |
| E7 | `CopyCanonicalization` | H | F | always |
| E8 | `Canonicalizer` | U | M | always |
| E9 | `buildBufferDeallocationPipeline` | U | — | always |
| E10 | `CSE` | U | M | always |
| E11 | `HexagonDoubleBufferGenericS2` | H | F | `enableDoubleBuffering` |
| E12 | `SCFLoopUnroll` | H | F | `enableSCFLoopUnroll` |
| E13 | `ConvertZeroSizeMemref` | H | F | always |
| E14 | `ConvertBufferizationToMemRef` | U | M | always |

### Phase F: Hexagon memory, library calls, loops and threads

| # | Pass | Kind | Scope | Condition |
|---|---|---|---|---|
| F1 | `ConvertToHexagonmem` | H | F | `enableConvertToHexagonmem` (default on) |
| F2 | `InsertScratchArg{scratch}` | H | F | `scratch > 0` |
| F3 | `MemoryOffsets` | H | F | `scratch > 0` |
| F4 | `LowerHexKLMatmulToMacro` | H | F | `enableHexKL && mode=="macro"` |
| F4' | `DecomposeHexKLMatmul` | H | F | `enableHexKL && mode=="micro"` |
| F5 | `HexagonReplaceWithLibraryCalls` | H | M | always |
| F6 | `HexmemCpyToDMA` | H | F | `enableHexagonmemCopyToDMA` |
| F7 | `CSE`, `Canonicalizer` | U | M | always |
| F8 | `ConvertLinalgToLoops` | U | M | always |
| F9 | `FormAsyncThreads` | H | F | always (no-op without `scf.forall`) |
| F10 | `AsyncFuncToAsyncRuntime` | U | M | always |
| F11 | `AsyncToAsyncRuntime` | U | M | always |
| F12 | `ConvertVectorToSCF` | U | F | always |
| F13 | `HexagonLWP{loop, depth}` | H | F | `enableLWP` |

### Phase G: Lowering to the LLVM dialect

| # | Pass | Kind | Scope | Condition |
|---|---|---|---|---|
| G1 | `SCFToControlFlow` | U | M | always |
| G2 | `memref::ExpandStridedMetadata` | U | M | always |
| G3 | `LowerAffine` | U | M | always |
| G4 | `SCFToControlFlow` | U | M | always |
| G5 | `ConvertMathToLLVM` | U | M | always |
| G6 | `ExpandMathOps` | H | F | always |
| G7 | `ExpandBoolVec` | H | F | `expandBoolVec` (default on) |
| G8 | `FastInverse` | H | M | always |
| G9 | `ConvertVectorToLLVM` | U | M | always |
| G10 | `ConvertIndexToLLVM{indexBitwidth=32}` | U | M | always |
| G11 | `ConvertAsyncToLLVM` | U | M | always |
| G12 | `ConvertFuncToLLVM` | U | M | always |
| G13 | `DMAToLLVM` | H | M | always |
| G14 | `HexagonMemToLLVM{device_type}` | H | M | always |
| G15 | `HexKLToLLVM` | H | M | always |
| G16 | `CollapseAddressSpace` | H | M | `enableCollapseAddressSpace` (default on) |
| G17 | `ReconcileUnrealizedCasts` | U | M | with G16 |
| G18 | `FinalizeMemRefToLLVMConversion` | U | M | always |
| G19 | `ArithToLLVMConversion` | U | M | always |
| G20 | `ConvertControlFlowToLLVM` | U | M | always |
| G21 | `Canonicalizer`, `CSE`, `ReconcileUnrealizedCasts` | U | M | always |
| G22 | `HexagonLLVMEnableHexagonRoutines` | H | M | `enableHexagonRoutines` (default on) |

If `lowerConstantsInSeparateSharedObjects` is on, a separate `CustomPassManager` (`MLLVMIRTranslation.cpp`) runs **`LowerConstantsSeparately`** after `LinalgToLLVM` (§7).

---

## 6. Pass reference, phase by phase

Each entry gives:

- **File**: where the pass lives.
- **What**: the transformation.
- **Why**: its role in the pipeline and why it matters on Hexagon.
- **Notes / options**: when present.

Upstream passes get shorter entries.

### Phase A: Ingest and normalize

The goal of this phase is to turn whatever the frontend produced into clean, standard Linalg-on-tensors, with nothing left from `ttx`, `tptr`, `tm_tensor` or `tt.extern_elementwise`.

#### A1. `LLVMRequestCWrappers` (U)
- **What:** Adds `llvm.emit_c_interface` to functions that return values. `ConvertFuncToLLVM` then emits an extra `_mlir_ciface_<name>` wrapper that takes and returns memref descriptors **by pointer**.
- **Why:** The generated C++ wrapper (`backend/*_launcher.py`) calls the kernel through a C ABI. Returning structs by value across that boundary is fragile, so pointer-based wrappers are used instead. The long comment in `LinalgToLLVMPass.cpp` shows the shape of the generated wrapper.

#### A2. `LowerTTX` (H)
- **File:** `lib/Transforms/LowerTTX.cpp`
- **What:** `ttx.scan` → `scf.for` loop. The scan region is moved into the loop body, the accumulator init is extracted, and `scan.return` becomes `scf.yield` plus a tensor insert. `ttx.cumsum` → scalar-accumulation loops, with a specialized 1-D path and a general n-D path along `axis`.
- **Why:** triton-shared has no lowering for scans, so without this pass kernels using `tl.cumsum` or `tl.associative_scan` don't compile.

#### A3. `LowerLibdevice` (H)
- **File:** `lib/Transforms/LowerLibdevicePass.cpp`
- **What:** Rewrites each `tt.extern_elementwise` (produced by `backend/hexagon_extern/hexagon/libdevice.py`, e.g. `rsqrt → qhmath_hvx_rsqrt_ahf`) into a call that processes a whole buffer:
  - allocates the output memref;
  - gets the aligned pointers of input and output;
  - emits `llvm.call @<symbol>(in_ptr, out_ptr, i32 numElems) -> i32`;
  - wraps the output back into a tensor.
- **Why:** This is how Triton `libdevice` math reaches Qualcomm's hand-optimized **QHL HVX** library (`libqhmath_hvx.a`, linked when the `.so` is built). **Runtime connection:** `qhmath_hvx_*` symbols.
- **Notes:** Static shapes only, and input and output shapes must match.

#### A4. `LowerTPtr` (H)
- **File:** `lib/Transforms/LowerTPtr.cpp`
- **What:** Lowers `tptr.type_offset`, `from_memref`, `ptradd` and `to_memref` to index arithmetic and LLVM pointer/struct ops (see §4.5).
- **Why:** Removes the temporary pointer dialect so the rest of the pipeline sees only memrefs and integers.

#### A5. `HexagonLowerTmTensor` (H)
- **File:** `lib/Transforms/LowerTmTensor.cpp`
- **What:** Expands `tm_tensor.attention` into Linalg ops (§4.6).
- **Why:** Lets attention from PyTorch models go through the normal fusion, tiling and vectorization path.

#### A6. `ReduceContractionRank` (H)
- **File:** `lib/Transforms/ReduceContractionRankPass.cpp`
- **What:** Wraps the upstream rank-reduction patterns: `batch_matmul` with batch=1 becomes `matmul`, and `matvec` with unit dimensions becomes `dot`.
- **Why:** Lower-rank named ops match more downstream patterns (HexKL, HVX matmul scheduling, tiling).

#### A7 / C16 / D2. `LinalgFoldUnitExtentDims` (U)
- **What:** Removes size-1 loop dimensions from Linalg ops, inserting reshapes.
- **Why:** Unit dimensions confuse tiling heuristics and the "innermost loop == vector width" check used by vectorization. It runs several times because tiling can introduce new unit dimensions.

#### `Canonicalizer` / `CSE` (U, many times)
- **What:** Applies fold and canonicalization patterns, and eliminates common subexpressions.
- **Why:** Almost every custom pass leaves dead ops behind: allocs that are no longer used, reinterpret-casts, single-iteration loops. The comments in the pipeline say which cleanup each instance is for, e.g. "erase unstrung allocs".

#### A9. `HexagonPuntBuffer` (H)
- **File:** `lib/Transforms/PuntBufferPass.cpp`
- **What:** triton-shared copies each kernel input argument into a fresh local buffer: `reinterpret_cast(arg) → alloc → memref.copy → to_tensor`. This pass removes the alloc and copy and feeds the `reinterpret_cast` view directly into `to_tensor`. It also handles views that are carried through `scf.for` iter_args.
- **Why:** Avoids a full extra copy of every input tensor. Read-only inputs never need to be copied.

#### A11. `ConversionToFp16` (H, opt-in)
- **File:** `lib/Conversion/LinalgToLLVM/ConversionToFp16Pass.cpp`
- **What:** Rewrites f32/f64 ops to f16. It inserts `truncf` on operands, clones each op with f16 types (recursively through regions, updating block arguments) and inserts `extf` on results for consumers. An exclusion list keeps accuracy-sensitive ops in higher precision.
- **Why:** f16 doubles the HVX lane count (64 instead of 32) and halves memory traffic. This trades accuracy for speed, so it's off unless the user turns it on.

#### A13. `OptimizeExtfTruncfOp` (H)
- **File:** `lib/Conversion/LinalgToLLVM/OptimizeExtfTruncfOpPass.cpp`
- **What:** Three clean-ups:
  1. cancels back-to-back `truncf → extf` pairs;
  2. replaces `tensor.empty` followed by a cast chain with a `tensor.empty` of the target type (cached by shape and type);
  3. replaces `linalg.fill` followed by a cast chain with a fill of the target type.
- **Why:** Mixed-precision frontends and A11 generate many redundant casts. Each one becomes a whole elementwise pass over memory if it survives.

#### A14. `DivToMulOptimization` (H)
- **File:** `lib/Transforms/DivToMulOptimizationPass.cpp`
- **What:** When the divisor in a `linalg.generic` comes from `linalg.fill(y)` or a broadcasting generic, the pass computes `1/y` once **before** the broadcast and replaces `divf` with `mulf`. It also handles `extf → divf → truncf` chains.
- **Why:** HVX has no fast vector divide, while multiplies are cheap. Common in softmax and normalization (`x / sum`).

#### A15. `quant::LowerQuantOps` (U)
- **What:** Lowers `quant.qcast`/`dcast` to arith (scale and zero-point math).

#### A16. `ConvertElementwiseToLinalg` (U)
- **What:** Turns elementwise arith/math ops on tensors into `linalg.generic`.
- **Why:** After this pass, all compute is Linalg, so fusion and tiling see everything.

### Phase B: Matmul and convolution specialization

#### B1. `FoldResourceTranspose` (H)
- **File:** `lib/Transforms/FoldResourceTransposePass.cpp`
- **What:** Folds `linalg.transpose` of a rank-2 f16 `dense_resource` constant into a new pre-transposed constant blob.
- **Why:** Weights are often stored transposed (e.g. `nn.Linear`). Folding the transpose at compile time removes a runtime transpose.

#### B2. `FoldCastsIntoMatmul` (H)
- **File:** `lib/Transforms/FoldCastsIntoMatmul.cpp`
- **What:** `matmul(extf(a16), extf(b16)) -> f32` becomes `matmul(a16, b16) -> f32`, and the dead casts are removed.
- **Why:** HexKL's natural form is f16×f16→f32 accumulation. Without this fold, the matmul would look like an f32 matmul and HexKL would not apply.

#### B3. `MatmulToHexKL` (H)
- **File:** `lib/Transforms/MatmulToHexKLPass.cpp`
- **What:** `linalg.matmul` with f16 inputs becomes `hexkl.matmul`.
  - **micro** mode (default): f16 or f32 output, flexible shapes.
  - **macro** mode: requires constant weights, outputs f16 (upcast afterwards if f32 is needed), with limits M≤1600, N≤5120, K≤76030.
- **Why:** This is the entry point to HMX. Every matmul it doesn't match stays on the HVX path.

#### B4. `PreprocessWeightsForHMX` (H, macro mode)
- **File:** `lib/Transforms/PreprocessWeightsForHMXPass.cpp`
- **What:** For each constant used as the RHS of a matmul, the pass converts row-major data to the HMX **WH** layout **at compile time**. It does this with two block-linearize steps (2×1 blocks, then (N/2)×64 blocks) and writes a new 128-byte-aligned resource blob.
- **Why:** Removes weight layout conversion from runtime entirely. Macro mode assumes weights are already in WH layout.

#### B6–B8. `MatmulToConv`, `SeedLayoutConversions`, `HexagonExtendPack` (H, unit-test path)
- **Files:** `lib/Transforms/MatmulToConvPass.cpp`, `lib/Transforms/SeedLayoutConversions.cpp`, `lib/Conversion/LinalgToLLVM/ExtendPackUnpackFrontier.cpp`
- **`MatmulToConv`:** Rewrites `matmul(A[M,K], B[K,N])` (and `batch_matmul` with batch=1) as `conv_2d_nhwc_fhwc`. A is reshaped to `(1, hF, hQ, K)` using the largest factor of M, Bᵀ to `(N,1,1,K)`, and the result is collapsed back. i8 and f16 only. This puts matmuls on the HMX convolution path.
- **`SeedLayoutConversions`:** Wraps HMX-eligible convolutions with `linalg.pack`/`unpack` so that:
  - the activation is in crouton layout: f16 is packed in two stages, `[8,4,32]` and then `[2]` on dim 5;
  - the filter is padded, reshaped and transposed;
  - `unrealized_conversion_cast`s keep the IR type-correct.

  Per-axis-quantized i8 uses the zero-point as the pad value.
- **`HexagonExtendPack`:** A fork of upstream data-layout propagation. It pushes `pack` up and `unpack` down through parallel generics and `tensor.pad`, so that whole regions of the graph stay in crouton layout instead of converting around every op. Supports padding and has `upper`/`lower`/`parallels-only` flags. The file header lists its limitations (multi-result generics, padding through division, dominance).
- **Status:** The code comment says these are *for unit tests only, not for full models*.

#### B10. `ConvTiling` (H)
- **File:** `lib/Transforms/ConvTilingPass.cpp`
- **What:** Tiles `conv_2d_nhwc_fhwc` along user-specified dimensions (`conv-tile-sizes="1:8,3:32"`; H must be a multiple of 8, C a multiple of 32). It then canonicalizes, peels the last iteration so that the main loop works on full croutons, and inserts a pack/unpack pair.
- **Why:** HMX convolution kernels consume crouton-sized tiles.

#### B12. `PreprocessTiledConv2D` (H)
- **File:** `lib/Transforms/PreprocessTiledConv2D.cpp`
- **What:** Moves inverse-layout (unpack) ops from outside tiled convolution loops to inside them, so that the loops carry **packed** tensors and each tiled conv reads packed slices.
- **Why:** Prepares for a later lowering of each conv tile to HMX convolution calls.

#### B13. `ScheduleMatmulForHVX` (H)
- **File:** `lib/Transforms/ScheduleMatmulForHVXPass.cpp`
- **What:**
  1. Folds `transpose + matmul` into `matmul_transpose_a/b` (and the `batch_matmul` variants).
  2. Applies a loop **interchange** to `matmul` and `batch_matmul` that have no `library_call` attribute, so that the innermost loop runs over a contiguous dimension.
- **Why:** After generalization, the HVX vectorizer only vectorizes the innermost loop. The loop order chosen here decides whether the matmul vectorizes well.

#### B14. `LinalgGeneralize` (H)
- **File:** `lib/Transforms/GeneralizePass.cpp`
- **What:** Converts named Linalg ops into `linalg.generic`, except ops that have a `library_call` attribute (which must stay named for F5) and `transpose`, which is kept as-is to work around a Flash-Attention issue.
- **Why:** The custom fusion, tiling and vectorization passes all operate on `linalg.generic`.

#### B15. `HexagonRVO` (H)
- **File:** `lib/Transforms/RVO.cpp`
- **What:** Return-value optimization. It:
  - replaces unused `outs` with `tensor.empty`;
  - uses alias analysis to find `materialize_in_destination` ops that write into a view of a function-argument memref;
  - moves the defining `reinterpret_cast` as early as dominance allows.
- **Why:** Once the cast is visible early, `EmptyTensorElimination` and `OneShotBufferize` can compute results **directly in the output buffer**, so there is no temporary buffer and no final copy.

### Phase C: Graph-level transforms, fusion and VTCM tiling

#### C1. `FormSCFThreads` (H, opt-in)
- **File:** `lib/Conversion/LinalgToLLVM/FormSCFThreads.cpp`
- **What:** Converts a top-level `scf.for` with no iter_args into `scf.forall`. F9 later turns it into threads.
- **Why:** A coarse-grained alternative to C14. The user is responsible for the loop actually being parallel. It is mutually exclusive with VTCM tiling, linalg multithreading and scratch mode (the pipeline asserts this).

#### C2. `HexagonFusion` (H)
- **File:** `lib/Conversion/LinalgToLLVM/FusionPass.cpp`
- **What:** Producer→consumer elementwise fusion of `linalg.generic`, built on upstream elementwise-op fusion. It adds **multi-use fusion** (`fdmu`, on by default): a producer with several consumers is fused into the consumer that dominates the others, provided every use is elementwise. `far` controls whether recomputation is allowed. It avoids duplicating constants and finishes with canonicalization.
- **Why:** This is one of the most important passes for performance. Every unfused intermediate is a full DDR round trip, and later each fused generic becomes **one** VTCM-tiled, vectorized loop nest.

#### C3 / D5. `EraseUnusedLinalgOperands` (H)
- **File:** `lib/Transforms/EraseUnusedLinalgOperands.cpp`
- **What:** Removes generic operands and results left unused after fusion or hoisting, then removes dead code.
- **Why:** Unused operands would otherwise be tiled, copied to VTCM and vectorized for nothing.

#### C5. `HexagonSlicing` (H, opt-in)
- **File:** `lib/Conversion/LinalgToLLVM/SlicingPass.cpp`
- **What:** Splits an op along its outermost loop into K slices and concatenates the results. Slice sizes are chosen so that at least K−1 of them are powers of two.
- **Why:** Equal-sized slices that aren't powers of two would each be padded separately, which causes bufferization problems. This keeps at most one remainder slice.

#### C6. `DecomposeTensorConcat` (H)
- **File:** `lib/Transforms/DecomposeTensorConcatPass.cpp`
- **What:** Wraps upstream patterns that turn `tensor.concat` into `tensor.empty` plus `insert_slice`s.
- **Why:** Bufferization and the later passes don't handle `concat` directly.

#### C7–C8. `ForceHVXCrouton` + `HexagonExtendPack{upper=false}` (H, opt-in)
- **File:** `lib/Transforms/ForceHVXCrouton.cpp`
- **What:** For static 4-D f16 tensors consumed by parallel generics, inserts a `pack → (crouton) → unpack` round trip in front of the consumer, then extends the unpack frontier downward.
- **Why:** Experimental path to make HVX run on crouton-layout data.

#### C9. `LowerPack` (H)
- **File:** `lib/Conversion/LinalgToLLVM/LowerPack.cpp`
- **What:** Lowers `linalg.pack`/`unpack` to simpler tensor ops (pad, expand_shape, transpose, extract_slice).
- **Why:** The bufferizer can't handle pack and unpack, so they have to be gone before phase E.

#### C11. `VTCMTiling` (H): the core memory optimization
- **Files:** `lib/Conversion/LinalgToLLVM/VTCMTiling.cpp`, `lib/Conversion/LinalgToLLVM/VTCMTilingOptions.cpp`
- **What:** For each (fused) `linalg.generic` on DDR tensors:
  1. **Footprint model.** Builds an affine map from loop bounds to bytes touched, summing over operands. For example, `(d0,d1) -> d1*4 + d0*d1*4`. Broadcast dimensions count as 1.
  2. **Tile-size search.** If the footprint exceeds the budget (2 MB, or `scratch` if set), it computes a reduction factor `nextPowerOf2(footprint/budget)` and picks which dimensions to tile by comparing how much each choice reduces the footprint. The file has a worked example: 56 MB → factor 32, comparing tiling d0, d1 and d2. `tile-size-override` replaces the heuristic for debugging.
  3. **Rewrite.** Tiles into an `scf.for` nest. Inside it, each operand slice becomes a `bufferization.alloc_tensor() copy(slice) {memory_space = 1}`, so the slices land in VTCM, and the tiled generic runs on those VTCM tensors. Small tensors are prefetched whole before the loop.
- **After bufferization** (comment in the pass): one `memref.alloc ... : memref<TxT, 1>` hoisted outside the loops; inside them, a `memref.copy` DDR-subview→VTCM, the generic on VTCM, and a `memref.copy` back to DDR.
- **Why:** HVX on DDR data stalls on memory. Bringing tiles into VTCM is what makes kernels fast, and every later memory pass (F1–F3, F6, E5/E11) works on the allocs and copies created here.

#### C13. `SplitReduceGeneric` (H, opt-in)
- **File:** `lib/Conversion/LinalgToLLVM/SplitReduceGenericPass.cpp`
- **What:** Splits a single-loop static reduction `[reduction]` into `[parallel, reduction]` followed by a small final reduction, using upstream `splitReduction`. It works around an upstream bug by preserving the **original identity constant**, e.g. `-inf` for `maxnumf`.
- **Why:** The parallel partial reduction vectorizes across HVX lanes. Only the final horizontal reduction stays scalar.

#### C14. `FormVirtualThreads` (H, `enableMultiThreading`)
- **File:** `lib/Conversion/LinalgToLLVM/FormVirtualThreadsPass.cpp`
- **What:** Tiles each sufficiently large generic into an `scf.forall` over a chosen parallel loop:
  - skips generics with no parallel loop, dynamic shapes, or work below `MIN_POLYTOPE_SIZE`;
  - otherwise picks the outermost non-unit parallel loop;
  - splits its range evenly into `preferredNumThreads` (default 4) "virtual threads".

  `blockSize` overrides this for testing.
- **Why:** Gives intra-kernel thread-level parallelism on the DSP's hardware threads. Execution happens in F9–F11 and the async runtime (§9).

#### C15. `RemoveMLProgram` (H)
- **File:** `lib/Transforms/RemoveMLProgram.cpp`
- **What:** Erases `ml_program.global` ops (left over from torch-mlir), which have no lowering.

### Phase D: HVX vectorization

#### D1. `HexagonTiling` (H)
- **File:** `lib/Conversion/LinalgToLLVM/TilingPass.cpp`
- **What:** Tiles each Linalg op so that its **innermost** loop has exactly `dataTileSize = 128 / sizeof(smallest operand element)` iterations: 32 for f32, 64 for f16, 128 for i8. All other tile sizes are 1. If the innermost range isn't a multiple of that:
  - `splitTilingRange=true` (default) splits the op into a tileable part and a remainder;
  - `false` pads the range up to a multiple instead.

  `useInterchangeVector` can interchange loops to get a long-enough innermost loop. `enableSplitReduction` splits reductions to expose vectorization.
- **Why:** The vectorizer only handles the case "innermost loop == one HVX register", so this pass establishes that condition.

#### D3. `SmallExponentToMultiply` (H)
- **File:** `lib/Transforms/SmallExponentToMultiplyPass.cpp`
- **What:** `math.fpowi(x, n)` with a small constant `|n|` (≤ `max-exponent`, default 4) becomes a square-and-multiply chain. Negative exponents give `1/chain` (`allow-negative`).
- **Why:** `pow` would otherwise become a scalar library call. Typical cases are `x²` and `x³` in GELU and variance.

#### D4. `HoistScalarOps` (H)
- **File:** `lib/Transforms/HoistScalarOps.cpp`
- **What:** Moves loop-invariant scalar arith/math chains (including `extf`/`truncf`, and chains depending on rank-0 inputs) **out of** `linalg.generic` bodies. Each hoisted value is passed back in as a rank-0 tensor input.
- **Why:** Otherwise the vectorizer would broadcast and vectorize, for example, `rsqrt(scalar)` in every inner iteration. Hoisting sets up D7/D9.

#### D7 / D9. `LoopInvariantCodeMotion` (U, twice)
- **What:** Moves invariant ops out of loops. It runs twice because canonicalization in between exposes new opportunities.

#### D10. `HexagonVectorization` (H)
- **File:** `lib/Conversion/LinalgToLLVM/VectorizationPass.cpp`
- **What:** For each Linalg op without `library_call` whose innermost static range equals `dataTileSize`, calls upstream `linalg::vectorize` with vector sizes `[1, ..., 1, innerDim]`. That yields `vector.transfer_read`, arithmetic on `vector<64xf16>` (etc.) and `transfer_write`.
- **Why:** **This is the step that generates HVX code.** LLVM's Hexagon backend maps 128-byte vectors directly onto HVX registers.

#### D11. `RewriteUBPoisonToZero` (H)
- **File:** `lib/Conversion/LinalgToLLVM/RewriteUBPoisonToZeroPass.cpp`
- **What:** Replaces `ub.poison` padding values of vector type (produced by masked transfers in the vectorizer) with zero constants.
- **Why:** Gives deterministic padding and avoids poison propagating into Hexagon instruction selection.

#### D12. `HexagonVectorLowering` (H)
- **File:** `lib/Transforms/VectorLoweringPass.cpp`
- **What:** Applies upstream vector lowering patterns configured for Hexagon:
  - contractions → **outer products**;
  - multi-reductions → **inner reduction**;
  - transposes → **Shuffle1D**;

  plus transfer permutation/full-partial lowering and canonicalization.
- **Why:** Brings high-level vector ops down to shapes LLVM can turn into efficient HVX instructions.

#### D14. `HexagonAddFastMath` (H, default on)
- **File:** `lib/Transforms/AddFastMathPass.cpp`
- **What:** Sets `fastmath<fast>` on every op that implements the fast-math interface (and on LLVM ops), **except comparisons**, so that `x != x` NaN checks keep working. This happens at MLIR level because setting it in LLVM would be too late.
- **Why:** Lets reassociation, contraction and reciprocal rewrites happen. Required by D15.

#### D15. `FoldMulFByZero` (H)
- **File:** `lib/Transforms/FoldMulFByZeroPass.cpp`
- **What:** `x * 0.0 → 0.0` when the `nnan` and `nsz` flags are present.

#### D17. `memref::ResolveShapedTypeResultDims` (U)
- **What:** Resolves `dim` ops of op results to their producers' operand dimensions before bufferization.

### Phase E: Bufferization

#### E1. `EmptyTensorElimination` (U)
- **What:** Replaces `tensor.empty` with the destination it eventually gets written into, so no new allocation is needed. Works together with B15 (RVO).

#### E2. `EraseVectorToTensorWriteback` (H)
- **File:** `lib/Conversion/LinalgToLLVM/EraseVectorToTensorWritebackPass.cpp`
- **What:** In `scf.for` loops that `transfer_write` a vector into a tensor, yield it, and `transfer_read` it back after the loop, this pass makes the loop **yield the vector** directly (when the tensor has no other uses).
- **Why:** Otherwise bufferization would materialize a memory buffer for a loop-carried vector, such as a reduction accumulator, adding stores and loads to every iteration.

#### E3. `OneShotBufferize` (U)
- **What:** Converts the whole program from tensors to memrefs, using `bufferizeFunctionBoundaries=true` (function arguments become memrefs) and `allowReturnAllocsFromLoops=true`. The `memory_space = 1` on VTCMTiling's `alloc_tensor`s becomes `memref<..., 1>`.
- **Why:** From here on everything is buffers. In-place decisions made here (helped by A9, B15, E1) decide how many copies are left.

#### E5 / E11. `HexagonDoubleBufferGenericS1` / `S2` (H, opt-in)
- **Files:** `lib/Transforms/DoubleBufferGenericS1.cpp`, `lib/Transforms/DoubleBufferGenericS2.cpp`
- **S1** (runs after bufferization): rewrites a single-buffered tiled loop, i.e. the output of VTCMTiling, into a ping-pong software pipeline. It:
  - allocates two VTCM buffer sets and a toggle;
  - adds a prologue that prefetches iteration 0;
  - builds alternating ping and pong sub-kernels, each of which prefetches iteration i+1 while computing iteration i.

  Transfers are still `memref.copy` at this point, because deallocation hasn't run yet.
- **S2** (after loop hoisting and deallocation): parses that structure with a state machine, then:
  - replaces the prefetch copies with `memref.dma_start`;
  - replaces store-backs with DMA plus `dma_wait`;
  - inserts waits before compute;
  - allocates and frees DMA tag buffers.
- **Why:** Overlaps DDR↔VTCM transfer with HVX compute, so most of the memory latency is hidden. The DMA ops become runtime calls (G13).

#### E6. `BufferLoopHoisting` (U)
- **What:** Moves allocations out of loops when their size is loop-invariant. VTCM tile buffers end up allocated once per kernel instead of once per tile.

#### E7. `CopyCanonicalization` (H)
- **File:** `lib/Transforms/CopyCanonicalizationPass.cpp`
- **What:**
  1. **Intermediate buffer elimination.** `copy src→tmp; copy tmp→dst1; copy tmp→dst2` becomes `copy src→dst1; copy src→dst2`, and `tmp` is removed.
  2. Adds explicit copies where a loop's yielded value doesn't match its `init_arg`. This simplifies deallocation analysis, so later patterns can remove more allocs and copies.
- **Why:** Bufferization tends to create passthrough buffers, and on Hexagon each one costs VTCM or DDR bandwidth.

#### E9. `buildBufferDeallocationPipeline` (U)
- **What:** Upstream ownership-based buffer deallocation: inserts `memref.dealloc`, simplifies, and lowers `bufferization.dealloc`.
- **Why:** Without it, every VTCM alloc would leak and the 2 MB pool would fill up.

#### E12. `SCFLoopUnroll` (H, opt-in)
- **File:** `lib/Transforms/SCFLoopUnrollPass.cpp`
- **What:** Unrolls innermost constant-bound `scf.for` loops without iter_args by `unroll-factor` (default 2), but only when the trip count divides exactly.
- **Why:** Adds instruction-level parallelism for Hexagon's VLIW packets, on top of LLVM's own unrolling.

#### E13. `ConvertZeroSizeMemref` (H)
- **File:** `lib/Transforms/ConvertZeroSizeMemrefPass.cpp`
- **What:** `memref.alloc` of size 0 becomes size 1.
- **Why:** The deallocation pipeline can produce zero-size allocs (Flash Attention does this). They become `malloc(0)`, which crashes on the device runtime.

#### E14. `ConvertBufferizationToMemRef` (U)
- **What:** Lowers the remaining `bufferization.*` ops (`clone` etc.) to memref ops.

### Phase F: Hexagon memory, library calls, loops and threads

#### F1. `ConvertToHexagonmem` (H, default on)
- **File:** `lib/Transforms/ConvertToHexagonmem.cpp`
- **What:** For statically shaped ranked memrefs, converts `memref.alloc → hexagonmem.alloc`, `memref.dealloc → hexagonmem.dealloc` and `memref.copy → hexagonmem.copy`, keeping the address space (VTCM = 1).
- **Why:** **This is the point where the compiler commits to runtime-managed memory.** Without it, a VTCM alloc would have no correct lowering. `CollapseAddressSpace` would then fold it into a plain DDR pointer, which is safe but slow.

#### F2. `InsertScratchArg` (H, `scratch>0`)
- **File:** `lib/Transforms/InsertScratchArgPass.cpp`
- **What:** Adds a `memref<Nxi8, 1> {hexagon.scratch}` argument before the 6 trailing Triton program-info i32 arguments (`num_programs_x/y/z, pid_x/y/z`). N is the per-instance VTCM budget. It is idempotent and skips functions without that 6-i32 ABI.
- **Why:** Implements **external VTCM** mode. The C++ wrapper acquires `scratch × prod(grid)` bytes of VTCM once (via `HAP_compute_res_*`, see `backend/triton_hexagon_launcher.py`) and passes each program instance its own slice. The kernel then never calls the allocator.

#### F3. `MemoryOffsets` (H, `scratch>0`)
- **File:** `lib/Transforms/MemoryOffsetsPass.cpp`
- **What:** Collects every static, unconditional alloc in address space 1, in program order, and replaces each with a `memref.view` at a statically computed offset (aligned to 2048 bytes by default) into a single buffer. Their deallocs are removed.
  - **External mode:** the buffer is the `hexagon.scratch` argument, and the pass fails at compile time if the allocs don't fit in N.
  - **Fallback mode:** creates one `hexagonmem.alloc` of the required size, capped at 1 MB by default.
- **Limitation:** There is no liveness analysis, so buffers whose lifetimes don't overlap still don't share space. Dynamic allocs and allocs inside loops are skipped.
- **Why:** No allocator calls in the hot path, a compile-time guarantee that VTCM fits, and safe SPMD slicing.

#### F4. `LowerHexKLMatmulToMacro` (H, macro mode)
- **File:** `lib/Transforms/LowerHexKLMatmulToMacroPass.cpp`
- **What:** Expands each `hexkl.matmul` into:
  - an activation copied into a temporary buffer (because the conversion is in place), then `macro_rm_to_ah_f16_inplace`;
  - weights used directly, since they are already in WH layout from B4;
  - `macro_mm_f16(M, N, K, out, act, wgt)`;
  - `macro_ah_to_rm_f16_inplace` on the output;
  - `qurt_mem_cache_clean` around buffers as needed.

  HMX resources are managed with `macro_initialize`/`lock_hmx` before the first matmul in a group and `unlock_hmx`/`finalize` after the last, so consecutive matmuls share one lock.
- **Why:** Macro mode leaves tiling and DMA inside HexKL's optimized implementation.

#### F4'. `DecomposeHexKLMatmul` (H, micro mode)
- **File:** `lib/Transforms/DecomposeHexKLMatmulPass.cpp`
- **What:** Writes out HexKL's own `hexkl_matmul_f16f16_f32` tiling strategy as micro ops.
  - VTCM block size: `(numKTiles*2 + 2 + 3) * 4096` bytes, laid out as `[activation tiles | scratch tiles | flat out | acc read | extra]`, where `numKTiles = ceil(K/32)`.
  - Setup: `setup_acc_read`.
  - **For each 32-row tile:** copy and lay out every K-tile of the activation (`copy_submatrix_to_f16`, then `rm_to_ah`).
    - **For each 32-column tile:** `acc_clear`.
      - **For each K-tile:** `rm_to_wh` (weight tile), then `mm_f16` (accumulate).
    - Then `acc_read` → `ah_to_rm` → `copy_f16_to_f32_submatrix` to DDR.
- **Why:** Making the loops visible in IR means the compiler can schedule and optimize them, for example by hoisting the offset arithmetic.

#### F5. `HexagonReplaceWithLibraryCalls` (H)
- **File:** `lib/Transforms/ReplaceWithLibraryCallsPass.cpp`
- **What:** A version of upstream `LinalgToStandard` that only touches Linalg ops carrying a `library_call = "fn"` attribute. Each one becomes `func.call @fn(memrefs...)` with canonicalized memref types, a declaration is inserted, and `llvm.emit_c_interface` is set.
- **Why:** Lets a frontend or earlier pass hand specific ops to a hand-written library. B13, B14 and D10 deliberately skip these ops.

#### F6. `HexmemCpyToDMA` (H, opt-in)
- **File:** `lib/Conversion/LinalgToLLVM/HexmemCpyToDMA.cpp`
- **What:** Rewrites static `hexagonmem.copy`/`memref.copy` ops (contiguous, or strided 2-D DDR↔VTCM) into `memref.dma_start` + `dma_wait`. It allocates a tag buffer, creates loops over the dimensions beyond the last two, uses 2-D DMA for the inner two dimensions, and shares constants across replacements.
- **Why:** The DMA engine copies much faster than the CPU and frees HVX for compute.

#### F8. `ConvertLinalgToLoops` (U)
- **What:** Turns the remaining (non-vectorized) Linalg ops into `scf.for` loops of scalar code. Remainder tiles and anything that didn't vectorize end up here.

#### F9. `FormAsyncThreads` (H)
- **File:** `lib/Conversion/LinalgToLLVM/FormAsyncThreadsPass.cpp`
- **What:** Each `scf.forall` (from C14 or C1) becomes:
  ```
  %g = async.create_group %num_threads
  scf.for %t = ... {
    %tok = async.execute { <forall body for thread %t> }
    async.add_to_group %tok, %g
  }
  async.await_all %g
  ```
  Falls back to sequential execution if the constraints aren't met.
- **Why:** Converts virtual threads into real concurrent tasks.

#### F10–F11. `AsyncFuncToAsyncRuntime`, `AsyncToAsyncRuntime` (U)
- **What:** Outline each `async.execute` body into a coroutine function and turn the async ops into `async.runtime.*` ops (create token/group, resume, await).

#### F12. `ConvertVectorToSCF` (U)
- **What:** Lowers the remaining n-D vector transfers to loops of 1-D transfers.

#### F13. `HexagonLWP` (H, opt-in)
- **File:** `lib/Transforms/HexagonLWPInstrumentation.cpp`
- **What:** Lightweight profiling. It inserts calls to the LLVM intrinsic `llvm.hexagon.instrprof.custom("lwp_handler", id)` at function and loop boundaries, up to `depth` levels of loops (`loop=false` disables loop instrumentation). At compile time it also writes `/tmp/lwp_infodump.txt`, which maps each id to source locations and op info. `test/python/process_lwp.py` joins that file with the measured results.
- **Runtime connection:** The intrinsic jumps to `lwp_handler` (`bin/runtime/profiler/lwp_handler.S`), which is linked together with `test/utils/dsp/include/prof_utils.cc` when the `.so` is built with LWP enabled.

### Phase G: Lowering to the LLVM dialect

#### G1/G4. `SCFToControlFlow` (U)
- **What:** Structured control flow (scf) becomes `cf` branches. It runs twice because `LowerAffine` creates new scf ops.

#### G2. `memref::ExpandStridedMetadata` (U)
- **What:** Expands subviews and reinterpret-casts into explicit offset, size and stride arithmetic.

#### G3. `LowerAffine` (U)
- **What:** `affine.apply`, `min`, `max` and affine loops become arith and scf.

#### G5. `ConvertMathToLLVM` (U)
- **What:** Math ops with LLVM intrinsic equivalents become `llvm.intr.*` (exp, sin, sqrt, …). G22 picks these up later.

#### G6. `ExpandMathOps` (H)
- **File:** `lib/Transforms/ExpandMathOpsPass.cpp`
- **What:** Applies upstream `math::populateExpansionPatterns`, which rewrites math ops with no direct LLVM lowering in terms of simpler math and arith ops.

#### G7. `ExpandBoolVec` (H)
- **File:** `lib/Transforms/ExpandBoolVec.cpp`
- **What:** `arith.uitofp vector<Nxi1> → vector<Nxfp>` becomes `select(mask, splat(1.0), splat(0.0))`, with 1.0 encoded correctly for f16.
- **Why:** Hexagon LLVM instruction selection can't lower `uitofp` from i1 vectors, so without this masks from comparisons would crash the backend.

#### G8. `FastInverse` (H)
- **File:** `lib/Transforms/FastInversePass.cpp`
- **What:** For f32 scalars and vectors, rewrites `p / q` (and divisions by `sqrt`) as multiplication by a reciprocal computed with the **Quake III fast inverse square root**: `rsqrt(q·q)`, using magic constant `0x5f3759df` and two Newton-Raphson refinement steps.
- **Why:** HVX has no vector divide. This turns a scalarized divide into a few vector multiplies and integer ops.

#### G9. `ConvertVectorToLLVM` (U)
- **What:** Vector dialect → LLVM vector types and intrinsics. A 128-byte vector maps to one HVX register.

#### G10. `ConvertIndexToLLVM{indexBitwidth=32}` (U)
- **What:** Lowers `index` to **i32**, because Hexagon is a 32-bit target.

#### G11. `ConvertAsyncToLLVM` (U)
- **What:** Turns `async.runtime.*` ops into calls to `mlirAsyncRuntime*` C functions and LLVM coroutine intrinsics.
- **Runtime connection:** These calls resolve against `libhexagon_mlir_async_runtime.a` (§9.4).

#### G12. `ConvertFuncToLLVM` (U)
- **What:** `func.func` → `llvm.func`. Memref arguments are expanded into descriptor fields, and the `_mlir_ciface_` wrappers requested in A1 are generated.

#### G13. `DMAToLLVM` (H)
- **Files:** `lib/Conversion/DMAToLLVM/DMAToLLVMPass.cpp`, `lib/Conversion/DMAToLLVM/DMAExternalFnNames.cpp`
- **What:**
  - `memref.dma_start` becomes `hexagon_runtime_dma_start(src, srcAS, dst, dstAS, bytes, bypassSrc, bypassDst, status)` for contiguous transfers, or `hexagon_runtime_dma2d_start(src, srcAS, dst, dstAS, width, height, srcStride, dstStride, ..., isOrdered, cachePolicy, status)` for 2-D. The returned token is stored in the tag memref.
  - `memref.dma_wait` loads the token and becomes `hexagon_runtime_dma_wait(token)`.

  Byte sizes come from element count × element size, and values are truncated to i32.
- **Why:** This connects every DMA op in the IR (from E11 or F6) to the hardware DMA driver.

#### G14. `HexagonMemToLLVM` (H)
- **Files:** `lib/Conversion/HexagonMemToLLVM/HexagonMemToLLVMPass.cpp`, `lib/Conversion/HexagonMemToLLVM/HexagonMemExternalFnNames.cpp`
- **What:** Lowers every `hexagonmem` op to a call into the runtime, as listed in §4.1:
  - computes sizes, using `getelementptr` so alignment is accounted for;
  - for crouton types, uses a block count and 2 KB blocks;
  - builds memref descriptors around the returned pointer;
  - marks runtime declarations as having side effects so O3 can't remove them;
  - turns **flat DDR** allocs and deallocs back into `memref.alloc`/`dealloc` (plain heap).

  Afterwards no `hexagonmem` op remains.

#### G15. `HexKLToLLVM` (H)
- **Files:** `lib/Conversion/HexKLToLLVM/HexKLToLLVMPass.cpp`, `lib/Conversion/HexKLToLLVM/HexKLExternalFnNames.cpp`
- **What:** Lowers every `hexkl` op to its extern C call, as listed in §4.3:
  - extracts aligned pointers and sizes from memref descriptors;
  - for micro ops, computes the VTCM base, its size, and the HMX config offset (`vtcm_size − hexkl_micro_hmx_config_size()`);
  - lowers `hexkl.matmul` to `hexkl_matmul_f16f16_f32(M, N, K, out, lhs, rhs)`.

#### G16. `CollapseAddressSpace` (H)
- **File:** `lib/Transforms/CollapseAddressSpace.cpp`
- **What:** Replaces `!llvm.ptr<1>` with `!llvm.ptr` everywhere: inside structs, function signatures (the function is rebuilt and its passthrough attributes copied over), block arguments and results.
- **Why:** Hexagon LLVM uses one flat address space, so any VTCM pointer that didn't go through `hexagonmem` (e.g. when `enableConvertToHexagonmem=False`) must still compile.

#### G17/G21. `ReconcileUnrealizedCasts` (U)
- **What:** Removes the `unrealized_conversion_cast` pairs left by the conversions.

#### G18–G20. `FinalizeMemRefToLLVMConversion`, `ArithToLLVMConversion`, `ConvertControlFlowToLLVM` (U)
- **What:** The final lowering of memref (descriptors, `memref.alloc → malloc/aligned_alloc`, `memref.copy` → `memrefCopy` for non-contiguous copies), arith and cf to LLVM.

#### G22. `HexagonLLVMEnableHexagonRoutines` (H, default on)
- **File:** `lib/Transforms/UseHexagonRoutines.cpp`
- **What:** On LLVM-dialect code, replaces vector math intrinsics on native-width (or factor-of-width) f16/f32 vectors with calls to hand-optimized HVX routines named `_hexagon_runtime_<fn>__<suffix>`. The mapping is:
  - `exp→exp`, `exp2→pow` (as `2^x`), `sin`, `cos`, `tan`, `tanh`, `asin`, `acos`, `atan`, `sqrt`, `floor`, `ceil`;
  - suffix `vf` for f32 and `vhf` for f16;
  - operands are bitcast to `vector<32xi32>`, which is the HVX_Vector ABI.
- **Why:** Otherwise LLVM **scalarizes** vector `exp`/`sin`/…, which destroys performance in softmax and GELU.
- **Runtime connection:** These functions are defined in `bin/runtime/IntrinsicsHVX.c` as `always_inline` wrappers around `qhmath_hvx_*`, so after linking and inlining they become inline HVX code (§9.3).

---

## 7. Passes outside the main pipeline

These are registered (most can be run with `linalg-hexagon-opt`) but are **not** part of the default `LinalgToLLVMPass` sequence:

| Pass | File | What it does | Used where |
|---|---|---|---|
| `LowerConstantsSeparately` | `include/hexagon/Conversion/LinalgToLLVM/LowerConstantsSeparately.h` | Moves large constant tensors (weights) into separate modules, starting a new one each time a module reaches 0.5 GB. Each is compiled with **no** optimization so the constants survive, then linked as its own `.o`/`.so`. | Runs after `LinalgToLLVM` in `CustomPassManager` when `lowerConstantsInSeparateSharedObjects=True` (large torch models). |
| `ConvertLayout` | `lib/Transforms/ConvertLayoutPass.cpp` | `hexagonmem.convert_layout` → `linalg.generic`. | opt / tests |
| `FoldPackUnpackConstants` | `lib/Transforms/FoldPackUnpackConstantsPass.cpp` | Evaluates pack, unpack, transpose, reshape, expand/collapse and pad of constant tensors at compile time. | opt / tests |
| `EliminateRedundantUnpackPack` | `lib/Transforms/EliminateRedundantUnpackPackPass.cpp` | Removes `unpack → pack` pairs with the same tiling. **Doesn't check that padding values match**, so it is unsound in general (documented in the .td). | opt / tests |
| `AffineToLLVM` (+ `AffineTiling`, `AffinePipelineFusion`, `AffineVectorize`, `AffineTileMemory`) | `lib/Conversion/AffineToLLVM/` | An alternative, older pipeline based on affine loops: generalize → bufferize → `ConvertLinalgToAffineLoops` → cache-size-driven affine tiling (default 4096 KiB) → affine vectorize → … → LLVM. | Only registered in `linalg-hexagon-opt`. |
| `ConvertToHexagonmem`, `MatmulToConv`, etc. standalone | — | Every pass can be run on its own for lit tests. | `qcom_hexagon_backend/test/**` |

---

## 8. From LLVM dialect to object code

`translateLinalgToObj()` in `python/triton_qcom_hexagon_backend_api.cc`:

1. **Options.** `setLinalgToLLVMOptions(options_map)`.
2. **MLIR pipeline.**
   - Normal mode: `translateLinalgToLLVMMLIR()` runs a `PassManager` with `LinalgToLLVMPass` (honouring the IR-printing env vars, §11).
   - Constant-splitting mode: `translateLinalgToMultipleLLVMMLIRModules()` instead. Module 0 is the code and modules 1..n hold only constants.
3. For each module:
   1. **Translate to LLVM IR.** `translateHexagonMlirLlvmToLLVMIR()` (`lib/Target/HEX_LLVMIR/LLVMIRTranslation.cpp`) runs the MLIR→LLVM IR translation and then the LLVM `PassBuilder` pipeline at **O3** for module 0 (no optimization for constant modules, so the data isn't elided), using a Hexagon `TargetMachine`.
   2. **Link the runtime** (module 0 only). `linkRuntimeModules()` (`bin/LinkRuntimeModules.cpp`) links in only the runtime bitcode modules that are needed (§9.2).
   3. **Inline** (optional). `cond_run_inliner()` runs the always-inliner if `enableHVXInlining`.
   4. `LLVM_IR_DUMP_TO_FILE=<path>` dumps the final IR.
   5. **`fixAlignedAllocTypes()`** rewrites `aligned_alloc(i64, i64)` declarations and calls to `(i32, i32)`. This is a regex fix-up on the IR text, needed to match the 32-bit async runtime's `aligned_alloc`.
   6. `HEXAGON_ASM_DUMP=1` dumps the Hexagon assembly (requires `HEXAGON_ARCH_VERSION`).
   7. **Codegen.** `llvm_module_to_obj_string()` uses `TargetMachine::addPassesToEmitFile(ObjectFile)` and returns the `.o` bytes.
4. The bytes go back to Python. `HexagonExecutor.generate_shared_object()` compiles the generated C++ wrapper and links it with the `.o` and static libraries into `lib<kernel>.so` (§9.4).

---

## 9. Compiler ↔ runtime connection

### 9.1 Every runtime symbol the compiler can emit

| Symbol(s) | Emitted by pass | Defined in | What it does on device |
|---|---|---|---|
| `hexagon_runtime_alloc_1d_dsp(bytes, align, isVtcm)` | G14 (from `hexagonmem.alloc` memref) | `bin/runtime/src/HexagonCAPI.cpp` → `HexagonAPI::Alloc` | Allocates a `HexagonBuffer` through `BufferManager`. VTCM requests are served from `VtcmPool`, DDR requests from the heap. |
| `hexagon_runtime_alloc_2d_dsp(nBlocks, blockSize, align, isVtcm)` | G14 (crouton) | same | Allocates N separate blocks (indirect-tensor/crouton format) and returns the pointer table. |
| `hexagon_runtime_free_1d_dsp` / `_2d_dsp(ptr)` | G14 | same → `HexagonAPI::Free` | Frees through `BufferManager`. |
| `hexagon_runtime_copy_dsp(dst, src, n, dstVtcm, srcVtcm)` | G14 | same → `HexagonAPI::Copy` → `BufferManager::Copy` → `HexagonBuffer::CopyFrom/CopyTo` | CPU `memcpy`, with cache invalidate of the source and flush of the destination around it when they are tracked `HexagonBuffer`s. |
| `hexagon_runtime_build_crouton_dsp(src, n)` | G14 | same → `CreateBufferAlias` | Builds a crouton pointer table that aliases a contiguous buffer (`HexagonBufferAlias`). `n` must be a multiple of 2 KB. |
| `hexagon_runtime_get_contiguous_memref_dsp(src)` | G14 | same → `GetOrigBufferFromAlias` | Returns the contiguous buffer behind an alias. |
| `hexagon_runtime_dma_start`, `hexagon_runtime_dma2d_start`, `hexagon_runtime_dma_wait` | G13 | `bin/runtime/UserDMA/RuntimeDMA.cc` → `UserDMA` (static instance with `MAX_NUM_DMA_DESCRIPTORS`) | Writes 1-D/2-D DMA descriptors into a ring buffer (`UserDMADescriptors.h`, `RingBuffer.h`) and launches them with the user-DMA instructions (`UserDMAInstructions.h`). Returns a token; `wait` polls until that descriptor completes. |
| `hexkl_matmul_f16f16_f32` | G15 | `bin/runtime/src/HexKLAPI.cpp` | Full tiled HMX matmul built from HexKL micro calls. |
| `hexkl_micro_hmx_*`, `hexkl_micro_hmx_config_size` | G15 (from F4') | `libhexkl_micro.a` (HexKL SDK) | HMX tile primitives (§4.3b). |
| `hexkl_macro_*` | G15 (from F4) | `libhexkl_macro.a` (HexKL SDK) | Whole-matmul HMX API (§4.3c). |
| `qurt_mem_cache_clean` | G15 | QuRT | Cache maintenance. |
| `qhmath_hvx_<fn>_<a?hf/af>(in, out, n)` | A3 (libdevice) | `libqhmath_hvx.a` (Hexagon SDK QHL) | Array-wide HVX math. |
| `_hexagon_runtime_<fn>__vf/vhf(HVX_Vector...)` | G22 | `bin/runtime/IntrinsicsHVX.c` (always-inline wrappers) → `qhmath_hvx_*_vf/vhf` | Per-register HVX math; inlined after linking. |
| `mlirAsyncRuntime*` (CreateToken/Group, AddTokenToGroup, Execute, Await*, EmplaceToken, AddRef/DropRef, …) | G11 (from F9–F11) | `bin/runtime/multithreading/AsyncRuntime.cpp` (+ `HexagonThreadPool.cpp`) | Async tasks run on a QuRT-backed `HexagonThreadPool`; tokens and groups use reference counting. |
| `malloc` / `aligned_alloc` / `free` / `memrefCopy` | G18 (DDR `memref.alloc`, non-contiguous `memref.copy`) | QuRT libc; `aligned_alloc` override in `multithreading/HexagonCustomDefs.cpp`; `memrefCopy` in `test/utils/dsp/include/CRunnerUtils.cpp` | DDR heap and generic strided copy. |
| `llvm.hexagon.instrprof.custom` → `lwp_handler` | F13 | `bin/runtime/profiler/lwp_handler.S`, `test/utils/dsp/include/prof_utils.cc` | Records cycle counts per instrumented region. |
| `_mlir_ciface_<kernel>` | A1 + G12 | generated | Entry point that the generated C++ wrapper calls. |

### 9.2 How the runtime gets into the kernel: bitcode linking

The core runtime is **not** a normal library. It is compiled to LLVM bitcode, embedded in the compiler binary, and linked into each kernel's LLVM module:

1. **Build time** (`bin/runtime/CMakeLists.txt`): each of `IntrinsicsHVX.c`, `UserDMA.cc`, `RuntimeDMA.cc`, `VTCMPool.cpp`, `HexagonBuffer.cpp`, `HexagonBufferAlias.cpp`, `HexagonAPI.cpp`, `HexagonCAPI.cpp` and `HexKLAPI.cpp` is compiled with `hexagon-clang -emit-llvm -O2 -mvNN -mhvx -mhvx-qfloat -fvisibility=hidden` into a `.bc` file.
2. `bitcode2array.py` turns each `.bc` into a C++ byte array (`hexagon_runtime_module_<Name>[]`), built into the `hexagon_runtime` static library that is linked into the compiler.
3. **Compile time** (`bin/LinkRuntimeModules.cpp:linkRuntimeModules`):
   - Parses all 9 embedded modules.
   - Builds two maps: *function → defining module* and *module → declared functions*.
   - Starting from the kernel module, does a **BFS over undefined symbols** to find the runtime modules that are actually needed. For example, `hexagon_runtime_alloc_1d_dsp` pulls in `HexagonCAPI`, which pulls in `HexagonAPI`, then `HexagonBuffer` and `VTCMPool`.
   - Links only those modules with `llvm::Linker::linkInModule`.
4. Because this happens before codegen and the runtime has hidden visibility, LLVM can **inline runtime code into the kernel**. The `_hexagon_runtime_*` HVX wrappers are `always_inline` for this reason, and `enableHVXInlining` adds a full always-inliner pass.

### 9.3 Runtime initialization and memory managers

- `HexagonAPI::Global()` is a lazily created singleton (`static auto *inst = new HexagonAPI()`). Its constructor runs `AcquireResources()`:
  - **`VtcmPool`** acquires the device's VTCM with `HAP_compute_res_attr_init`, `compute_resource_query_VTCM`, `HAP_compute_res_attr_set_vtcm_param_v2`, `HAP_compute_res_acquire` and `..._get_vtcm_ptr_v2` (`bin/runtime/src/VTCMPool.cpp`), then serves sub-allocations from it.
  - **`BufferManager`** tracks every `HexagonBuffer`, whether DDR or VTCM.
  - **HMX** is acquired with `initialize_and_acquire_hmx()`.
- So the **first** runtime call a kernel makes acquires VTCM and HMX. `ReleaseResources()` releases them.
- In **external scratch mode** (`scratch>0`) the generated wrapper acquires VTCM itself (`HAP_compute_res_*` in `backend/triton_hexagon_launcher.py`), and the kernel only uses `memref.view`s into it (F2/F3).

### 9.4 Libraries linked when the `.so` is built

`HexagonExecutor.generate_shared_object()` (`backend/hexagon_executor.py`) links:

- the generated wrapper `.cpp` (built from the `*_launcher.py` templates; uses `test/utils/dsp/include/*`);
- the kernel `.o`, which already contains the linked-in core runtime bitcode;
- `libqhmath_hvx.a` (required) and `libqhmath.a` (if present) from the Hexagon SDK;
- `libhexkl_micro.a` and `libhexkl_macro.a` from `HEXKL_ROOT`;
- `libhexagon_mlir_async_runtime.a` (built from `bin/runtime/multithreading/`);
- with LWP enabled, also `prof_utils.cc` and `lwp_handler.S`.

### 9.5 Triton grid dispatch (not a compiler pass)

The Triton launch grid is **not** turned into IR loops. The kernel takes `num_programs_x/y/z, pid_x/y/z` as its last 6 arguments, and the generated wrapper builds one closure per program instance. It then dispatches them with `ThreadManager::exec` (real QuRT threads, `test/utils/dsp/include/multithreading.h`) when `enableThreadedDispatch` or `enableMultiThreading` is set, or with `exec_serial` otherwise.

So there are two independent levels of parallelism:

- **across program instances**, handled by the wrapper;
- **inside one instance**, handled by C14/F9 and the async runtime.

---

## 10. Option reference

The **Python** column is the default in `backend/hexagon_options.py`; "—" means Python doesn't send the option, so the TableGen default in `Passes.td` applies.

| Python field | Pass option | Python | TableGen | Controls |
|---|---|---|---|---|
| — | `puntBuffer` | — | true | A9 |
| — | `returnValueOptimization` (`rvo`) | — | true | B15 |
| `enableConversionToFp16` | same | False | true | A11 |
| `enableHexKL` / `hexKLMode` | same | False / "micro" | false / "micro" | B1–B5, F4/F4' |
| — | `enableMatmulToConv` | — | false | B6–B9 |
| `enableSeedLayoutConversions` | same | False | false | B6–B9, B12 |
| `enableConvTiling` / `convTileSizes` | same | False / "" | false / "" | B10 |
| `enableSCFThreading` | same | False | false | C1 |
| `fusion` / `fusionAllowRecompute` / `fusionDoMultiUse` | `fusion` / `far` / `fdmu` | True / False / True | true / false / true | C2 |
| — | `enableSlicing` / `slicingFactor` | — | false / 1 | C5 |
| — | `forceHVXCroutonization` | — | false | C7–C8 |
| `enableVTCMTiling` | same | **True** | true | C11 |
| `tileSizes` | `tile-sizes-override` | "" | "" | C11 (debug) |
| `scratch` | same | 0 | 0 | C11 budget, F2, F3 |
| `enableSplitReduceGeneric` | same | False | false | C13 |
| `enableMultiThreading` | same | False | false | C14 (+ threaded grid dispatch) |
| `enableVectorization` | same | **True** | true | D1, D10 |
| `enableSplitReduction` | same | False | true | D1 option |
| — | `useInterchangeVector` / `splitTilingRange` | — | false / true | D1 options |
| — | `addFastMath` | — | true | D14–D16 |
| `enableBufferization` | same | True | true | all of phase E |
| `enableDoubleBuffering` | same | False | false | E5, E11 |
| `enableSCFLoopUnroll` | same | False | false | E12 |
| `enableConvertToHexagonmem` | same | **True** | true | F1 |
| `enableHexagonmemCopyToDMA` | same | False | false | F6 |
| `enableLWP` / `disableLWPLoop` / `LWPloopDepth` | `enableLWP` / `loop` / `depth` | False / False / 1 | false / false / 1 | F13 |
| — | `expandBoolVec` | — | true | G7 |
| `enableCollapseAddressSpace` | same | True | true | G16 |
| — | `enableHexagonRoutines` | — | true | G22 |
| `device_type` | same | "hexagon" | "hexagon" | G14 symbol selection, runtime linking |
| `enableHVXInlining` | same | False | false | always-inliner after runtime linking |
| `lowerConstantsInSeparateSharedObjects` | same | False | false | `LowerConstantsSeparately` |
| `iterations`, `enableThreadedDispatch`, `deviceCleanup` | — | — | — | wrapper and executor only, not the compiler |

To set options for a single Triton kernel, pass them as launch kwargs, e.g. `kernel[grid](..., enableMultiThreading=True, enableHexKL=True)`. torch-mlir launchers take an options dict.

---

## 11. Debugging the compiler

| Tool / variable | Effect |
|---|---|
| `MLIR_ENABLE_DUMP=1` | Prints IR after **every** pass in the pipeline (`MLLVMIRTranslation.cpp`, `enableIRPrinting`, only when the IR changed). The IR after `HexagonLWP` is always printed. |
| `MLIR_ELIDE_LARGE_CONST_PRINT=1` | Elides large constants and resource blobs in dumps (otherwise elided above 16 elements). |
| `LLVM_IR_DUMP_TO_FILE=<path>` | Writes the final LLVM IR (after runtime linking and inlining). |
| `HEXAGON_ASM_DUMP=1` | Dumps the Hexagon assembly. |
| `-debug-only=<DEBUG_TYPE>` in `linalg-hexagon-opt` | Per-pass debug output, e.g. `linalg-to-llvm`, `hexagon-vectorize`, `hexagon-vector-lowering` (each pass defines `DEBUG_TYPE`). Requires an assertions-enabled build. |
| `linalg-hexagon-opt in.mlir --<pass-arg>` | Runs one pass. The argument is the name in `Pass<"...">`, e.g. `--vtcm-tiling`, `--hexagon-fusion`, `--hexagon-tiling`, `--hexagon-vectorization`, `--convert-to-hexagonmem`, `--hexagonmem-to-llvm`, `--linalg-to-llvm="enable-multi-threading=true"`. |
| `linalg-hexagon-translate` | MLIR → LLVM IR / object, including runtime linking (`bin/linalg-hexagon-translate.cpp`). |
| lit tests | `qcom_hexagon_backend/test/{Conversion,Dialect,Target,Transforms}`: FileCheck examples of the input and output IR for each pass. They're the quickest way to see what a pass does. |
| End-to-end tests | `test/python/triton/*`, `test/python/torch-mlir/*`, `test/python/mlir/*`. |

**Suggested way to learn the pipeline:** run `test/python/triton/test_vec_add.py` with `MLIR_ENAx BLE_DUMP=1` and follow one `linalg.generic` through these steps:

1. **C2:** fused.
2. **C11:** wrapped in a VTCM tile loop.
3. **D1:** inner loop of 32 or 64 elements.
4. **D10:** `vector<…>` ops.
5. **E3:** memrefs.
6. **F1:** `hexagonmem.alloc/copy`.
7. **G14:** `llvm.call @hexagon_runtime_alloc_1d_dsp`.

After that, read the matching lit test for any pass you want to understand in detail.
