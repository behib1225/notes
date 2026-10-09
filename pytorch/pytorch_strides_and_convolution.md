# Strides, Convolution, and Backpropagation in PyTorch

### A self-contained tutorial for engineering undergraduates

---

## How to use this tutorial

### Setup

Everything here runs on CPU. You need Python and PyTorch; nothing else.

```bash
pip install torch
```

Every example assumes these two imports, which are not repeated in each block:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

### Viewing this file

This is GitHub-flavoured Markdown with LaTeX math and collapsible `<details>` blocks for exercise solutions. It renders properly in **GitHub, VS Code's Markdown preview, Obsidian, Typora, or JupyterLab**. In a plain text editor the equations will appear as raw `$$...$$` and the solutions will be visible rather than hidden — still readable, but you lose the "try before you look" design of Part XI.

### Evidence policy

Every code block through Part VII was **executed against PyTorch 2.14** and the printed outputs are the real ones, pasted unedited. If you run something and get a different answer, that difference is a real finding worth chasing, not a typo in the text.

Parts VIII and IX discuss GPU performance and carry an explicit warning: those claims were **not** measured here. I flag them rather than blurring the line, because the difference between "I ran this" and "this is widely reported" is exactly the distinction a working engineer has to maintain.

### How to read it

Read with an interpreter open. The single most common way students fail to learn this material is by reading *about* strides instead of *printing* them. Every section has checkpoint questions with answers; do them before moving on.

**Suggested paths, depending on why you are here:**

| If you want… | Read | Roughly |
|---|---|---|
| To stop being confused by `view`/`reshape`/`contiguous` errors | Parts I–II, Part X | 1–2 hours |
| To understand how convolution is really implemented | Parts I–V | 3–4 hours |
| To write a custom layer with correct gradients | Parts I–VI, Exercises 11–12 | 5–6 hours |
| The full course | Everything, all 18 exercises | 10–15 hours |

**Roadmap.** Part I builds the memory model. Part II shows what "views" really are. Part III introduces `as_strided`, the sharp tool. Part IV defines convolution precisely. Part V connects the two: a convolution is a strided view followed by a matrix multiply. Part VI derives backpropagation by hand and verifies it four different ways. Parts VII–IX cover the variants and the performance engineering you will meet in industry. Part X is a debugging playbook. Part XI is 18 exercises with full solutions. Part XII is a one-page reference card to keep after you have read the rest.

**A warning about the word "stride."** It means two different things, and conflating them is the single biggest source of confusion in this topic.

| | **Memory stride** | **Convolution stride** |
|---|---|---|
| Where you see it | `t.stride()` | `nn.Conv2d(..., stride=2)` |
| What it is | How many *elements* to skip in the flat buffer to move one step along a dimension | How many *pixels* the kernel hops between output positions |
| Set by | The layout of the tensor | You, as a hyperparameter |
| Units | elements | pixels |

They are different concepts. The beautiful part — and the reason this tutorial puts them in one document — is that the second can be *implemented* by manipulating the first. That is Part V.

---

# Part I — How a tensor is actually stored

## 1.1 The flat buffer

A computer's memory is one-dimensional. It is a very long row of numbered slots. There is no such thing as a 2-D region of RAM, no such thing as a 4-D one. So when you write

```python
import torch
x = torch.arange(24).reshape(2, 3, 4)
```

PyTorch does **not** create a 2×3×4 box. It allocates **one flat array of 24 integers**, laid out `0, 1, 2, ..., 23`, and then attaches a small piece of bookkeeping that explains how to *pretend* that flat array is a 2×3×4 box.

That bookkeeping is the whole subject of Part I. It consists of exactly four things:

| Field | Accessor | Meaning |
|---|---|---|
| **Storage** | `x.untyped_storage()` | The flat buffer of raw numbers |
| **Shape** | `x.shape` | How many steps you may take along each axis |
| **Stride** | `x.stride()` | How many *elements* forward in the buffer one step along each axis costs |
| **Offset** | `x.storage_offset()` | Where in the buffer this tensor's element `[0,0,...,0]` lives |

A `Tensor` object in PyTorch is, to a first approximation, a pointer to a storage plus those three little tuples. That is it.

## 1.2 The one formula you must memorise

Everything in this tutorial follows from a single equation. Given a tensor `x` with stride $(s_0, s_1, \dots, s_{n-1})$ and offset $o$, the element `x[i_0, i_1, ..., i_{n-1}]` lives at flat position:

$$\text{flat\_index} = o + \sum_{k=0}^{n-1} i_k \cdot s_k$$

That is it. That is the entire memory model of PyTorch, NumPy, JAX, TensorFlow, and essentially every array library ever written.

Read it once more in words: **stride[k] is the price, in buffer positions, of taking one step along axis k.**

Let us verify it directly.

```python
import torch

x = torch.arange(24).reshape(2, 3, 4)

print("shape          :", tuple(x.shape))
print("stride         :", x.stride())
print("storage_offset :", x.storage_offset())
print("is_contiguous  :", x.is_contiguous())
```

```
shape          : (2, 3, 4)
stride         : (12, 4, 1)
storage_offset : 0
is_contiguous  : True
```

Now check the formula by hand. Where does `x[1, 2, 3]` live?

$$0 + 1 \times 12 + 2 \times 4 + 3 \times 1 = 23$$

Slot 23 of a buffer holding `0..23` contains the value 23. And indeed `x[1,2,3]` is `23`. The formula is not an analogy. It is what the C++ code does.

**Why (12, 4, 1)?** Walk it out physically:

- To move one step along the **last** axis (`x[0,0,0] → x[0,0,1]`), you move to the literal next number in the buffer. Cost: **1**.
- To move one step along the **middle** axis (`x[0,0,0] → x[0,1,0]`), you must skip an entire row of 4 numbers. Cost: **4**.
- To move one step along the **first** axis (`x[0,0,0] → x[1,0,0]`), you must skip an entire 3×4 slab of 12 numbers. Cost: **12**.

## 1.3 Computing contiguous strides yourself

A tensor is called **contiguous** (more precisely, *row-major contiguous*, also called *C-contiguous*) when its strides are exactly the "suffix products" of its shape:

$$s_k = \prod_{j=k+1}^{n-1} \text{shape}[j], \qquad s_{n-1} = 1$$

For shape `(2,3,4)`: $s_2 = 1$, $s_1 = 4$, $s_0 = 4 \times 3 = 12$. Which is what we printed.

You can write this in four lines:

```python
def contiguous_strides(shape):
    """Strides a fresh row-major tensor of this shape would have."""
    strides = [1] * len(shape)
    for k in range(len(shape) - 2, -1, -1):
        strides[k] = strides[k + 1] * shape[k + 1]
    return tuple(strides)

print(contiguous_strides((2, 3, 4)))    # (12, 4, 1)
print(contiguous_strides((8, 3, 224, 224)))  # (150528, 50176, 224, 1)
print(torch.empty(8, 3, 224, 224).stride())  # (150528, 50176, 224, 1)
```

```
(12, 4, 1)
(150528, 50176, 224, 1)
(150528, 50176, 224, 1)
```

The last two lines agree, which is the point.

> **Note on units.** PyTorch reports strides in **elements**. NumPy reports strides in **bytes**. If you move between the libraries, multiply or divide by `itemsize` (4 for `float32`, 2 for `float16`, 8 for `float64`). This trips up almost everyone once.

## 1.4 The NCHW convention

In vision, PyTorch's default layout for a batch of images is **NCHW**:

```
N = batch size
C = channels
H = height
W = width
```

So `x = torch.randn(8, 3, 224, 224)` has stride `(150528, 50176, 224, 1)`. Interpret those numbers as costs:

- Moving to the next **pixel to the right** costs 1. Cheap. Adjacent in memory.
- Moving to the next **row down** costs 224.
- Moving to the next **channel** costs 50176 = 224×224. Expensive — the red, green and blue values of the same pixel are 50,176 elements apart.
- Moving to the next **image** costs 150528 = 3×224×224.

That last observation has real consequences. In NCHW, the three colour values belonging to one physical pixel are *far apart* in memory. Convolution needs all channels of a pixel at once. This is precisely why NHWC ("channels-last") exists, and we return to it in Part IX.

## 1.5 Checkpoint questions

**Q1.1** A tensor has shape `(4, 5)` and stride `(5, 1)`, offset 0. What flat index holds `[2, 3]`?

**Q1.2** A tensor has shape `(4, 5)` and stride `(1, 4)`, offset 0. What flat index holds `[2, 3]`? What does this tensor *look like* relative to the one in Q1.1?

**Q1.3** Without running code: what is `torch.zeros(6, 1, 7).stride()`?

**Q1.4** Can two different index tuples map to the same flat index? If so, give an example of strides that would do it.

<details>
<summary><b>Answers</b></summary>

**A1.1** $0 + 2(5) + 3(1) = 13$.

**A1.2** $0 + 2(1) + 3(4) = 14$. Stride `(1,4)` means stepping along axis 0 is cheap and along axis 1 is expensive — the exact opposite of Q1.1. This is the **transpose**. Same buffer, swapped strides.

**A1.3** `(7, 7, 1)`. Suffix products: $s_2=1$, $s_1 = 1 \times 7 = 7$, $s_0 = 7 \times 1 = 7$. Note that a size-1 dimension makes two strides equal, because stepping along an axis of length 1 never actually happens. PyTorch is free to report anything for such an axis, and this occasionally surprises people writing stride assertions.

**A1.4** Yes — and this is not a bug, it is a feature. Stride `0` on an axis means "stepping along this axis costs nothing," so all indices along it alias the same memory. That is exactly how `expand` and broadcasting work (Section 2.4).

</details>

---

# Part II — Views: the free operations

## 2.1 The central idea

Because a tensor is just *(buffer, shape, stride, offset)*, a surprising number of operations can be performed by **editing the metadata and touching no data at all**. These are called **views**. They are effectively free: O(1) time, zero bytes copied, regardless of whether the tensor is 1 KB or 100 GB.

This is the most practically important idea in Part II. Reshaping a 10 GB activation tensor can cost nothing or can cost 10 GB of allocation plus a full memory copy, and the difference is whether the operation can be expressed as a stride change.

## 2.2 Transpose and permute

```python
x = torch.arange(24).reshape(2, 3, 4)
y = x.transpose(1, 2)          # swap axes 1 and 2

print("x :", tuple(x.shape), x.stride())
print("y :", tuple(y.shape), y.stride())
print("same buffer? ", y.data_ptr() == x.data_ptr())
print("y contiguous?", y.is_contiguous())
```

```
x : (2, 3, 4) (12, 4, 1)
y : (2, 4, 3) (12, 1, 4)
same buffer?  True
y contiguous? False
```

Look carefully: `transpose` swapped positions 1 and 2 **in both the shape tuple and the stride tuple**. No number was moved. `y` is now a 2×4×3 tensor that reads the same 24 integers in a different order, purely because the cost table changed.

And `y` is *not* contiguous, because `(12, 1, 4)` is not the suffix-product pattern for shape `(2,4,3)` (which would be `(12,3,1)`).

**`permute` is the general case:**

```python
z = x.permute(2, 0, 1)
print(tuple(z.shape), z.stride())   # (4, 2, 3) (1, 12, 4)
```

Shape and stride are reordered by the same permutation. Again, no data movement.

## 2.3 Slicing

Basic slicing is also a view. It adjusts the offset and possibly the strides.

```python
s = torch.arange(20).reshape(4, 5)
sub = s[1:3, ::2]

print(sub)
print("shape :", tuple(sub.shape))
print("stride:", sub.stride())
print("offset:", sub.storage_offset())
```

```
tensor([[ 5,  7,  9],
        [10, 12, 14]])
shape : (2, 3)
stride: (5, 2)
offset: 5
```

Decode this:

- `offset = 5`: we skipped row 0, which is 5 elements, so the view starts at buffer slot 5.
- `stride[0] = 5`: rows are still 5 apart — unchanged by slicing rows.
- `stride[1] = 2`: the step `::2` **doubled the cost of moving right**, so we visit every other column.

Verify with the formula: `sub[1, 2]` sits at $5 + 1(5) + 2(2) = 14$. The printed tensor shows `14` at position `[1,2]`. ✓

**This is the conceptual seed of strided convolution.** A convolution with `stride=2` will do exactly this: double the memory stride along the spatial axes so the window lands on every other pixel.

> **Caution:** *advanced* (fancy) indexing — `x[[0,2,3]]`, `x[mask]`, `x[tensor_of_indices]` — is **not** a view. It always copies. Only basic slicing with `start:stop:step` produces views.

## 2.4 `expand`: stride zero, and broadcasting demystified

```python
v = torch.tensor([1., 2., 3.])
e = v.expand(4, 3)

print(e)
print("stride       :", e.stride())
print("logical elems:", e.numel())
print("real elems   :", e.untyped_storage().nbytes() // 4)
```

```
tensor([[1., 2., 3.],
        [1., 2., 3.],
        [1., 2., 3.],
        [1., 2., 3.]])
stride       : (0, 1)
logical elems: 12
real elems   : 3
```

A 12-element tensor backed by 3 floats. The trick is `stride[0] = 0`: stepping down a row costs *nothing*, so every row reads the same three numbers.

**This is what broadcasting is.** When you write `bias.view(1, -1, 1, 1) + conv_output`, PyTorch is not materialising a full-size bias tensor. It sets stride 0 on the broadcast axes and reads the same values repeatedly. Broadcasting is free in memory because of stride 0.

Now the first classic bug:

```python
e = torch.zeros(3).expand(4, 3)
try:
    e.add_(1)                      # in-place write to an expanded tensor
except RuntimeError as err:
    print("RuntimeError:", str(err)[:110])
```

```
RuntimeError: unsupported operation: more than one element of the written-to tensor refers to a single memory location. Please clone()
```

PyTorch caught it. Writing into an expanded tensor is ambiguous — four logical cells share one physical cell, so "what should it hold?" has no answer. The fix is `.clone()`, which materialises a real 12-element buffer.

`repeat()` is the copying cousin of `expand()`: it allocates. Use `expand` when you only read (which is almost always), `repeat` when you must write.

## 2.5 `view` vs `reshape`: the interview question

`view` demands that the result be expressible as a pure metadata change. If it cannot be, `view` refuses rather than silently copying.

```python
x = torch.arange(24).reshape(2, 3, 4)
y = x.transpose(1, 2)              # stride (12, 1, 4) — not contiguous

try:
    y.view(-1)
except RuntimeError as err:
    print("RuntimeError:", str(err)[:150])

flat = y.reshape(-1)
print("reshape worked:", tuple(flat.shape))
print("shares memory with x?", flat.data_ptr() == x.data_ptr())
```

```
RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
flat worked: (24,)
shares memory with x? False
```

**Why `view` fails.** Flattening to 1-D asks for stride `(1,)` — "consecutive logical elements are consecutive in memory." But `y` reads the buffer in the order 0, 4, 8, 1, 5, 9, … That order is not achievable with any single stride. No metadata exists that describes it. So `view` correctly refuses.

`reshape` says "give me this shape by whatever means necessary": it returns a view if one exists, and otherwise silently copies. Here it copied — note the different `data_ptr`.

**Practical rule for production code:**

| Use | When |
|---|---|
| `view` | You *want* to fail loudly if a copy would be needed (perf-critical paths, and to document the assumption) |
| `reshape` | You want it to work and don't care about a possible copy |
| `.contiguous().view(...)` | You know a copy is needed and want it explicit |
| `flatten()` | Readability; behaves like `reshape` |

### The exact condition, stated once

You can stop guessing when a view is possible. Reshaping is always some combination of **splitting** one axis into several and **merging** several into one, and each has a precise rule:

> **Splitting** an axis of size $n$ and stride $s$ into axes of sizes $n_1 n_2 = n$ is **always** legal, with strides $(n_2 s,\ s)$.
>
> **Merging** adjacent axes $k$ and $k{+}1$ is legal **iff** $\ \text{stride}_k = \text{stride}_{k+1} \times \text{shape}_{k+1}$.

The merge condition says: *the two axes must already be nested without gaps and without overlap.* Axis $k$ must step over exactly one full span of axis $k{+}1$ — no more (there would be a gap), no less (they would overlap).

Check it against the failure above. Flattening `y` of shape `(2,4,3)` and stride `(12,1,4)` needs merging axes 1 and 2. Is $\text{stride}_1 = \text{stride}_2 \times \text{shape}_2$? That asks $1 = 4 \times 3 = 12$. No. Hence the error.

And against a success: `x` of shape `(2,3,4)`, stride `(12,4,1)`. Merging axes 1 and 2 asks $4 = 1 \times 4$. ✓ Then merging with axis 0 asks $12 = 4 \times 3$. ✓ So `x.view(-1)` works. **For a contiguous tensor this condition holds at every adjacent pair by construction** — which is exactly why `view` never fails on contiguous tensors, and why the suffix-product definition of contiguity is the definition it is.

`x.contiguous()` returns `x` unchanged if it already is contiguous (free), and otherwise produces a copy with suffix-product strides.

```python
y2 = y.contiguous()
print(tuple(y2.shape), y2.stride(), y2.is_contiguous())   # (2, 4, 3) (12, 3, 1) True
print("copied?", y2.data_ptr() != x.data_ptr())           # True
```

## 2.6 What PyTorch will *not* do: negative strides

NumPy implements `x[::-1]` as a view with a negative stride. **PyTorch does not support negative strides.**

```python
a = torch.arange(5)
f = a.flip(0)
print(f, f.stride(), "is a view?", f.data_ptr() == a.data_ptr())
```

```
tensor([4, 3, 2, 1, 0]) (1,) is a view? False
```

`flip` **copies**. This matters when you port NumPy code and when you flip kernels in convolution code (Section 4.2) — that flip is a real allocation, not free.

The practical consequence you will hit: `torch.from_numpy(numpy_array[::-1])` raises an error, because the negative stride cannot be represented. The fix is `np.ascontiguousarray(...)` or `.copy()` first.

## 2.7 Checkpoint questions

**Q2.1** `t = torch.randn(4, 6)`. What are the shape and stride of `t.T`? Is it contiguous?

**Q2.2** `u = torch.randn(2, 3, 4).permute(1, 0, 2)`. Will `u.view(6, 4)` succeed? Will `u.reshape(6, 4)`? Which one might copy?

**Q2.3** You have a 2 GB activation tensor and call `.permute(0,2,3,1).contiguous()`. How much memory does this transiently need, at peak?

**Q2.4** Why does `expand` allow stride 0 for reading but block in-place writes?

<details>
<summary><b>Answers</b></summary>

**A2.1** Shape `(6, 4)`, stride `(1, 6)`, not contiguous. `.T` swaps shape and stride entries.

**A2.2** `u` has shape `(3,2,4)` and stride `(4,12,1)`. Merging axes 0 and 1 into one of size 6 would require a single stride covering the sequence `0,1,2,3, 12,13,14,15, 4,5,...` — impossible. `view` fails; `reshape` succeeds by copying.

**A2.3** About **4 GB** at peak: the original 2 GB is still alive (the permuted view points into it) while the 2 GB copy is being written. This is a very common cause of "CUDA out of memory in a line that looks harmless." If the original is dead afterwards, freeing it doesn't help *during* the copy.

**A2.4** Reading through aliased memory is well-defined — you just get the same value repeatedly. Writing is not: four logical cells map to one physical cell, so the result depends on write order. PyTorch refuses rather than producing nondeterministic results.

</details>

---

# Part III — `as_strided`: the sharp tool

## 3.1 Constructing views by hand

Everything so far was a *convenience wrapper* around one primitive. `torch.as_strided(input, size, stride, storage_offset)` hands you the metadata dials directly:

> Give me a tensor of this shape, reading this buffer, with exactly these strides, starting at this offset.

No validation of overlap. No check that it makes sense. It is `reinterpret_cast` for tensors.

Watch it reproduce a transpose:

```python
x = torch.arange(12).reshape(3, 4)
manual_T = x.as_strided(size=(4, 3), stride=(1, 4))
print(torch.equal(manual_T, x.T))     # True
```

```
True
```

## 3.2 Sliding windows — the key trick

Here is the idea that makes convolution work. Take a 1-D signal and ask for every contiguous window of length 3:

```python
sig = torch.arange(8.)
windows = sig.as_strided(size=(6, 3), stride=(1, 1))
print(windows)
```

```
tensor([[0., 1., 2.],
        [1., 2., 3.],
        [2., 3., 4.],
        [3., 4., 5.],
        [4., 5., 6.],
        [5., 6., 7.]])
```

Stare at `stride=(1, 1)` until it clicks. It says:

- Moving one step **within** a window (axis 1) advances one element. Normal.
- Moving one step to the **next window** (axis 0) *also* advances one element.

So window `i` begins one element after window `i-1`, and consecutive windows overlap in 2 of their 3 elements. **The overlap costs nothing**, because all six windows are reading the same 8 floats. We created an 18-element logical tensor backed by 8 real numbers.

This is the whole secret. A sliding window — the fundamental operation of convolution — is a pure metadata change.

**Now add a stride hyperparameter.** Want windows starting at positions 0, 2, 4 instead of every position? Multiply the *window-index* stride by 2:

```python
strided_windows = sig.as_strided(size=(3, 3), stride=(2, 1))
print(strided_windows)
```

```
tensor([[0., 1., 2.],
        [2., 3., 4.],
        [4., 5., 6.]])
```

**Want dilation?** Dilation spreads out the taps *inside* a window, so multiply the *within-window* stride by the dilation factor:

```python
dilated = sig.as_strided(size=(4, 3), stride=(1, 2))
print(dilated)
```

```
tensor([[0., 2., 4.],
        [1., 3., 5.],
        [2., 4., 6.],
        [3., 5., 7.]])
```

Pause here, because Part V is now essentially already done:

> **Convolution stride multiplies the window-position stride. Dilation multiplies the within-window stride. Both are free.**

## 3.3 The safer wrapper: `Tensor.unfold`

`as_strided` has no guard rails. `Tensor.unfold(dim, size, step)` does exactly the sliding-window construction, with bounds checking:

```python
print(sig.unfold(0, 3, 1).shape)   # torch.Size([6, 3])
print(sig.unfold(0, 3, 2).shape)   # torch.Size([3, 3])
print(sig.unfold(0, 3, 2).stride())# (2, 1)
```

```
torch.Size([6, 3])
torch.Size([3, 3])
(2, 1)
```

`unfold` appends the window axis at the end and computes the output length for you. **Prefer `unfold` over `as_strided` whenever it can express what you need.** In this tutorial we use `as_strided` in places because seeing the raw strides is the pedagogical point; in production, reach for `unfold` first.

## 3.4 How `as_strided` will hurt you

`as_strided` does **not** check that your view stays inside the buffer.

```python
small = torch.arange(6.)
bad = small.as_strided(size=(3, 4), stride=(1, 1))
print(bad.shape, "but the buffer only holds", small.numel(), "elements")
print(bad)
```

```
torch.Size([3, 4]) but the buffer only holds 6 elements
tensor([[0., 1., 2., 3.],
        [1., 2., 3., 4.],
        [2., 3., 4., 5.]])
```

This particular one happens to stay in bounds (highest index touched is $2 + 3 = 5$). Push it a little further and you read whatever bytes happen to follow in memory — garbage, or another tensor's data, or a segmentation fault. There is no error message. On GPU it can corrupt unrelated tensors.

**The safety rule.** Before calling `as_strided`, compute the largest flat index you will touch:

$$\text{max\_index} = \text{offset} + \sum_k (\text{size}_k - 1)\cdot \text{stride}_k$$

and require `max_index < storage.numel()`.

```python
def check_as_strided(storage_numel, size, stride, offset=0):
    max_idx = offset + sum((sz - 1) * st for sz, st in zip(size, stride))
    assert max_idx < storage_numel, f"reads index {max_idx}, buffer has {storage_numel}"
    return max_idx

check_as_strided(small.numel(), (3, 4), (1, 1))     # ok, 5 < 6
try:
    check_as_strided(small.numel(), (4, 4), (1, 1)) # 6 >= 6
except AssertionError as err:
    print("caught:", err)
```

```
caught: reads index 6, buffer has 6
```

Keep that helper. Three lines of assertion have saved more debugging hours than any profiler.

## 3.5 Checkpoint questions

**Q3.1** From `torch.arange(10.)`, produce every window of length 4 with step 3, using `as_strided`. What shape and stride? How many windows?

**Q3.2** What does `x.as_strided((5, 5), (0, 1))` give, for `x = torch.arange(5.)`?

**Q3.3** You call `as_strided` on a tensor obtained from `y = big[3:]`. Does `storage_offset` matter? Why is forgetting it dangerous?

<details>
<summary><b>Answers</b></summary>

**A3.1** Window starts are 0, 3, 6 — position 9 would need elements 9,10,11,12, out of bounds. So 3 windows: `size=(3,4)`, `stride=(3,1)`. Max index $= 2(3) + 3(1) = 9 < 10$. ✓ The general count is $\lfloor (L - K)/\text{step} \rfloor + 1 = \lfloor 6/3 \rfloor + 1 = 3$.

**A3.2** Five identical rows `[0,1,2,3,4]` — stride 0 on axis 0 is a broadcast.

**A3.3** It matters enormously. `y.storage_offset()` is 3, not 0. `as_strided` **ignores the input tensor's offset unless you pass one explicitly** — it defaults to the input's offset in current PyTorch, but relying on defaults here is exactly the kind of thing that silently changes behaviour when someone refactors. Always pass `storage_offset=t.storage_offset()` explicitly when you build strided views from possibly-sliced tensors, and always run the bounds check against the *full storage* size.

</details>

---

# Part IV — What a convolution layer actually computes

## 4.1 The definition

For a 2-D convolution with input $x$ of shape $(N, C_{in}, H, W)$ and weight $w$ of shape $(C_{out}, C_{in}, K_H, K_W)$, PyTorch computes:

$$y[n, o, h, v] \;=\; b[o] \;+\; \sum_{c=0}^{C_{in}-1}\ \sum_{i=0}^{K_H-1}\ \sum_{j=0}^{K_W-1} x\big[n,\ c,\ s\cdot h + d\cdot i,\ s\cdot v + d\cdot j\big]\ \cdot\ w[o, c, i, j]$$

where $s$ is the convolution stride, $d$ is the dilation, and $x$ has already been zero-padded by $p$ on each side.

In words, and this is the sentence to remember:

> **Place a $C_{in} \times K_H \times K_W$ box on the input, multiply it elementwise by a filter of the same shape, sum every product into a single number. Slide the box, repeat.**

Every output channel $o$ has its own filter. Every output pixel is one dot product of length $C_{in} K_H K_W$.

## 4.2 PyTorch does cross-correlation, not convolution

Notice the index: $s h + d i$, with a **plus**. The mathematical definition of convolution has a minus — the kernel is flipped:

$$(x * w)[h] = \sum_i x[h - i]\,w[i] \qquad\text{vs.}\qquad (x \star w)[h] = \sum_i x[h + i]\,w[i]$$

The second is **cross-correlation**, and it is what `nn.Conv2d` implements. Let's confirm:

```python
import torch.nn.functional as F

a = torch.tensor([[[[1., 2, 3],
                    [4., 5, 6],
                    [7., 8, 9]]]])
k = torch.tensor([[[[1., 2],
                    [3., 4]]]])

print("F.conv2d (cross-correlation):", F.conv2d(a, k).flatten().tolist())
print("true convolution (flip k)   :", F.conv2d(a, k.flip(-1, -2)).flatten().tolist())
```

```
F.conv2d (cross-correlation): [37.0, 47.0, 67.0, 77.0]
true convolution (flip k)   : [23.0, 33.0, 53.0, 63.0]
```

Check the first number by hand: $1(1) + 2(2) + 4(3) + 5(4) = 1 + 4 + 12 + 20 = 37$. ✓ No flip.

**Does this matter?** For a learned layer, no — the filter is learned, so whether the library flips it or not just relabels which learned value sits in which slot, and gradient descent finds the appropriate one either way. It matters in three concrete situations:

1. **Porting hand-designed kernels** from signal processing / MATLAB / `scipy.signal.convolve2d`, which *do* flip. A Sobel edge detector loaded without flipping gives you the sign-flipped gradient.
2. **Comparing against textbook math** or a paper's equations.
3. **In the backward pass**, where a genuine flip appears (Part VI). The flip you keep hearing about in "backprop through conv is a convolution with the flipped kernel" is real — it just lives in the backward pass, not the forward one.

Since `flip` copies in PyTorch (Section 2.6), doing it every forward pass is wasteful. Flip once at load time.

## 4.3 The output size formula

$$H_{out} = \left\lfloor \frac{H + 2p - d\,(K_H - 1) - 1}{s} \right\rfloor + 1$$

and identically for $W$. Derive it rather than memorising it:

1. After padding, the usable height is $H + 2p$.
2. A dilated kernel of $K_H$ taps spaced $d$ apart physically spans $d(K_H - 1) + 1$ pixels. Call this the **effective kernel size** $K_{\text{eff}}$.
3. The last valid window *start* position is $H + 2p - K_{\text{eff}}$.
4. Window starts are $0, s, 2s, \dots$, so the number of them is $\lfloor (H + 2p - K_{\text{eff}})/s \rfloor + 1$.

Substituting $K_{\text{eff}} = d(K_H-1)+1$ gives the formula. The `+1` is the "fencepost": 4 windows need 3 gaps between them.

```python
def conv_out_size(L, k, stride=1, padding=0, dilation=1):
    k_eff = dilation * (k - 1) + 1
    return (L + 2 * padding - k_eff) // stride + 1

for (L, k, s, p, d) in [(224,3,1,1,1), (224,3,2,1,1), (224,7,2,3,1), (32,3,1,2,2), (28,5,1,0,1)]:
    mine = conv_out_size(L, k, s, p, d)
    real = F.conv2d(torch.zeros(1,1,L,L), torch.zeros(1,1,k,k),
                    stride=s, padding=p, dilation=d).shape[-1]
    print(f"L={L:4d} k={k} s={s} p={p} d={d} -> mine={mine:4d} torch={real:4d} {'OK' if mine==real else 'MISMATCH'}")
```

```
L= 224 k=3 s=1 p=1 d=1 -> mine= 224 torch= 224 OK
L= 224 k=3 s=2 p=1 d=1 -> mine= 112 torch= 112 OK
L= 224 k=7 s=2 p=3 d=1 -> mine= 112 torch= 112 OK
L=  32 k=3 s=1 p=2 d=2 -> mine=  32 torch=  32 OK
L=  28 k=5 s=1 p=0 d=1 -> mine=  24 torch=  24 OK
```

**Two facts worth internalising:**

- **"Same" padding** for odd $k$, $s=1$, $d=1$ needs $p = (k-1)/2$. Hence 3×3 with padding 1, 5×5 with padding 2, 7×7 with padding 3 — the pattern you see everywhere in ResNets.
- **The floor is lossy.** With $L=224$, $k=3$, $s=3$, $p=0$ you get 74 outputs and the last pixel of the input is never read by any window. Silent data loss. If your input sizes are not divisible you may be dropping the right edge and bottom of every image without warning.

### Two padding options you should know

**`padding='same'`** lets PyTorch compute the padding for you, including for dilated kernels — but only at stride 1:

```python
print(tuple(F.conv2d(torch.randn(1,1,7,7), torch.randn(1,1,5,5), padding='same').shape))
try:
    F.conv2d(torch.randn(1,1,7,7), torch.randn(1,1,3,3), stride=2, padding='same')
except RuntimeError as err:
    print("RuntimeError:", err)
```
```
(1, 1, 7, 7)
RuntimeError: padding='same' is not supported for strided convolutions
```

The restriction is not arbitrary: with a stride, "same" size is ambiguous — same as the input, or same as input/stride? PyTorch refuses to guess. `padding='valid'` is also accepted, and means zero padding.

**`padding_mode`** controls *what* is padded, not how much. `nn.Conv2d` accepts `'zeros'` (default), `'reflect'`, `'replicate'` and `'circular'`:

```python
m = nn.Conv2d(1, 1, 3, padding=1, padding_mode='reflect')
print(tuple(m(torch.randn(1, 1, 5, 5)).shape))      # (1, 1, 5, 5)
```

Zero padding injects a hard artificial edge into every image, which the network can and does learn to detect. Reflection padding avoids that and is common in super-resolution, style transfer and generative models; circular padding is the right choice for genuinely periodic data (panoramas, angular coordinates, some physics simulations). Note that these modes are implemented as an explicit `F.pad` before a zero-padding convolution, so they **do copy** — unlike the geometry tricks in the rest of this tutorial, this one costs memory.

## 4.4 The reference implementation

Before optimising anything, write the version you are certain is correct. Loops, no cleverness.

```python
def conv2d_naive(x, w, b=None, stride=1, padding=0, dilation=1):
    """Deliberately slow, deliberately obvious. Ground truth for tests."""
    if padding:
        x = F.pad(x, (padding, padding, padding, padding))
    N, C, H, W = x.shape
    Cout, _, KH, KW = w.shape
    OH = (H - dilation * (KH - 1) - 1) // stride + 1
    OW = (W - dilation * (KW - 1) - 1) // stride + 1

    out = torch.zeros(N, Cout, OH, OW, dtype=x.dtype)
    for n in range(N):
        for o in range(Cout):
            for h in range(OH):
                for v in range(OW):
                    acc = 0.0
                    for c in range(C):
                        for i in range(KH):
                            for j in range(KW):
                                acc += x[n, c, stride*h + dilation*i,
                                              stride*v + dilation*j] * w[o, c, i, j]
                    out[n, o, h, v] = acc
    if b is not None:
        out += b.view(1, -1, 1, 1)
    return out

torch.manual_seed(0)
x = torch.randn(1, 2, 6, 6)
w = torch.randn(3, 2, 3, 3)
b = torch.randn(3)

ref  = conv2d_naive(x, w, b, stride=2, padding=1)
real = F.conv2d(x, w, b, stride=2, padding=1)
print("max abs diff:", (ref - real).abs().max().item())
```

```
max abs diff: 4.76837158203125e-07
```

Seven nested loops. It agrees with the library kernel to float32 round-off. **Keep this function.** Every optimisation in the rest of this tutorial gets checked against it.

The difference of $5\times10^{-7}$ is not an error — it is floating-point non-associativity. `(a+b)+c ≠ a+(b+c)` in floating point, and the optimised kernel sums in a different order. Never test numerical code with `==`; use `torch.allclose` with a sensible tolerance.

## 4.5 Checkpoint questions

**Q4.1** Input `(8, 3, 224, 224)`, layer `nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)`. Output shape? Number of weight parameters (with bias)?

**Q4.2** How many multiply-accumulate operations (MACs) does that layer perform for the whole batch?

**Q4.3** You want a 3×3 kernel to see a 7×7 region of the input. What dilation, and what padding keeps the output size the same?

<details>
<summary><b>Answers</b></summary>

**A4.1** $H_{out} = \lfloor (224 + 6 - 7)/2 \rfloor + 1 = \lfloor 223/2 \rfloor + 1 = 111 + 1 = 112$. Output `(8, 64, 112, 112)`. Parameters: $64 \times 3 \times 7 \times 7 = 9408$ weights $+\ 64$ biases $= 9472$. (This is exactly ResNet's `conv1`.)

**A4.2** MACs $=$ output elements $\times$ dot-product length $= (8 \cdot 64 \cdot 112 \cdot 112) \times (3 \cdot 7 \cdot 7) = 6{,}422{,}528 \times 147 \approx 9.44 \times 10^{8}$ MACs, i.e. about 1.9 GFLOPs counting a multiply and an add separately. Per image that is ~118 MMACs — the standard quoted figure for ResNet conv1.

**A4.3** $K_{\text{eff}} = d(3-1)+1 = 7 \Rightarrow d = 3$. To keep size: $p = (K_{\text{eff}}-1)/2 = 3$. So `Conv2d(..., 3, dilation=3, padding=3)` — a 7×7 receptive field for the parameter cost and MAC cost of a 3×3. This is the trick behind dilated/atrous convolutions in segmentation networks.

</details>

---

# Part V — Convolution as a strided view plus a matrix multiply

## 5.1 The reframing

Look again at the definition:

$$y[n,o,h,v] = \sum_{c,i,j} x[n, c, sh+di, sv+dj]\; w[o,c,i,j]$$

Hold $(n,h,v)$ fixed. What remains is a **dot product** between two vectors of length $C_{in}K_HK_W$: the flattened patch of the input, and the flattened filter.

So a convolution is:

1. Extract every patch (there are $N \cdot H_{out} \cdot W_{out}$ of them), flatten each to a vector of length $C_{in}K_HK_W$. Stack into a matrix $P$ of shape $(N H_{out} W_{out}) \times (C_{in}K_HK_W)$.
2. Flatten the weights into $\tilde{W}$ of shape $(C_{in}K_HK_W) \times C_{out}$.
3. Compute $P\tilde{W}$.
4. Reshape.

This reframing is called **im2col** (image-to-column). Caffe popularised it around 2014, and it remains the standard *conceptual* model of how convolution is computed — though on GPUs the explicit-copy version was long ago superseded by the implicit variant described in Section 5.5. The reason the GEMM framing dominates is economic rather than mathematical: matrix multiplication is *the* most heavily optimised routine in computing. Decades of work in BLAS, and later in Tensor Cores, went into GEMM. Turning your problem into a GEMM means inheriting all of it for free.

**And here is where strides enter.** Step 1 — the expensive-looking one — is *free*, because it is a sliding window, and we already know sliding windows are pure metadata.

## 5.2 Deriving the strides

We want a 6-D view `patches` of shape

$$(N,\ C,\ H_{out},\ W_{out},\ K_H,\ K_W)$$

such that

$$\texttt{patches}[n,c,h,v,i,j] = x[n,\ c,\ sh + di,\ sv + dj]$$

Start from the padded input's strides, $(s_N, s_C, s_H, s_W)$. Now ask, for each of the six new axes, "one step along this axis changes the input index by how much, and therefore costs how many buffer positions?"

| New axis | Effect on input index | Memory stride |
|---|---|---|
| $n$ | $n \to n+1$ | $s_N$ |
| $c$ | $c \to c+1$ | $s_C$ |
| $h$ | row index $\to$ row $+\,s$ | $s_H \cdot s$ |
| $v$ | col index $\to$ col $+\,s$ | $s_W \cdot s$ |
| $i$ | row index $\to$ row $+\,d$ | $s_H \cdot d$ |
| $j$ | col index $\to$ col $+\,d$ | $s_W \cdot d$ |

That table **is** the implementation. Notice the two punchlines:

- **Convolution stride $s$ scales the output-position strides** ($h, v$): take bigger hops between windows.
- **Dilation $d$ scales the kernel-position strides** ($i, j$): spread the taps inside a window.

Both are multiplications on a metadata tuple. Neither moves a byte.

## 5.3 The implementation

```python
def conv2d_strided(x, w, b=None, stride=1, padding=0, dilation=1):
    """Convolution via a strided view + einsum. No patch data is copied."""
    if padding:
        x = F.pad(x, (padding, padding, padding, padding))

    N, C, H, W = x.shape
    Cout, _, KH, KW = w.shape
    s, d = stride, dilation

    OH = (H - d * (KH - 1) - 1) // s + 1
    OW = (W - d * (KW - 1) - 1) // s + 1

    sN, sC, sH, sW = x.stride()                 # strides of the (padded) input

    patches = x.as_strided(
        size  =(N,   C,   OH,     OW,     KH,     KW),
        stride=(sN,  sC,  sH * s, sW * s, sH * d, sW * d),
    )

    out = torch.einsum('nchwij,ocij->nohw', patches, w)
    if b is not None:
        out = out + b.view(1, -1, 1, 1)
    return out
```

That is the entire forward pass of a convolution layer. One `as_strided`, one `einsum`.

Test it across the whole configuration space:

```python
torch.manual_seed(0)
for (s, p, d) in [(1,0,1), (2,1,1), (1,2,2), (3,1,2)]:
    x = torch.randn(2, 3, 9, 11)
    w = torch.randn(4, 3, 3, 3)
    b = torch.randn(4)
    mine = conv2d_strided(x, w, b, s, p, d)
    real = F.conv2d(x, w, b, stride=s, padding=p, dilation=d)
    ok = torch.allclose(mine, real, atol=1e-4)
    print(f"stride={s} pad={p} dil={d}  shape={tuple(mine.shape)}  match={ok}  maxdiff={(mine-real).abs().max():.2e}")
```

```
stride=1 pad=0 dil=1  shape=(2, 4, 7, 9)  match=True  maxdiff=3.81e-06
stride=2 pad=1 dil=1  shape=(2, 4, 5, 6)  match=True  maxdiff=2.86e-06
stride=1 pad=2 dil=2  shape=(2, 4, 9, 11) match=True  maxdiff=2.86e-06
stride=3 pad=1 dil=2  shape=(2, 4, 3, 3)  match=True  maxdiff=1.67e-06
```

## 5.4 Reading the einsum

`'nchwij,ocij->nohw'` is worth decoding carefully, because einsum notation is standard in modern deep-learning code.

- Left operand `patches` is indexed `n c h w i j`.
- Right operand `w` is indexed `o c i j`.
- Output is indexed `n o h w`.
- **Any index appearing in the inputs but not the output is summed over.** Here that is `c`, `i`, `j` — exactly the three sums in the mathematical definition.
- `h` and `w` appear on the left and in the output but not in the weights: they are *batched over*, which is exactly right (each output position is an independent dot product).

So the einsum string is a literal transcription of the defining equation. This is why einsum is worth learning: the code and the math become the same object.

If you prefer to see the GEMM explicitly:

```python
def conv2d_gemm(x, w, b=None, stride=1, padding=0, dilation=1):
    if padding:
        x = F.pad(x, (padding,) * 4)
    N, C, H, W = x.shape
    Cout, _, KH, KW = w.shape
    s, d = stride, dilation
    OH = (H - d*(KH-1) - 1)//s + 1
    OW = (W - d*(KW-1) - 1)//s + 1
    sN, sC, sH, sW = x.stride()

    patches = x.as_strided((N, C, OH, OW, KH, KW),
                           (sN, sC, sH*s, sW*s, sH*d, sW*d))
    # (N, OH, OW, C, KH, KW) -> (N*OH*OW, C*KH*KW).  This reshape MUST copy.
    P = patches.permute(0, 2, 3, 1, 4, 5).reshape(N * OH * OW, C * KH * KW)
    Wm = w.reshape(Cout, C * KH * KW).t()          # (C*KH*KW, Cout)
    out = (P @ Wm).reshape(N, OH, OW, Cout).permute(0, 3, 1, 2)
    return out + b.view(1, -1, 1, 1) if b is not None else out

print(torch.allclose(conv2d_gemm(x, w, b, 2, 1, 1),
                     F.conv2d(x, w, b, stride=2, padding=1), atol=1e-4))
```

```
True
```

**Note the crucial comment.** The `reshape` to a 2-D matrix *cannot* be a view — overlapping windows cannot be flattened into a matrix without duplicating the overlapping elements. This is where im2col pays its price, and it is the subject of the next section.

## 5.5 The memory cost of im2col

Each input element appears in up to $K_H K_W$ windows — but with a convolution stride $s$, only about $1/s^2$ of the possible window positions are actually used. So the size ratio of the materialised patch matrix to the input is approximately

$$\text{blow-up} \;\approx\; \frac{K_H K_W}{s^2}$$

For the common case $s=1$ this is just $K_HK_W$. Do not forget the $s^2$: it is the difference between a 49× and a 12× estimate for a 7×7 stride-2 stem (Exercise 13).

```python
x = torch.randn(8, 64, 56, 56)
cols = F.unfold(x, kernel_size=3)           # the official im2col
print("input  :", x.numel() * 4 / 1e6, "MB")
print("patches:", cols.numel() * 4 / 1e6, "MB")
print("ratio  :", round(cols.numel() / x.numel(), 2))
```

```
input  : 6.42 MB
patches: 53.75 MB
ratio  : 8.37
```

An 8.4× blow-up (close to the $K_HK_W = 9$ ceiling; it falls short only because of edge effects). For a 5×5 kernel it approaches 25×.

This is the central engineering tension:

> **The strided view is free. Making the GEMM see it as a dense matrix is not.**

Real libraries resolve it with **implicit GEMM**: the matrix multiply kernel is written to compute patch addresses on the fly from strides as it loads tiles into shared memory, so the expanded matrix is never materialised in global memory. `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM` and `..._IMPLICIT_PRECOMP_GEMM` are exactly this. Your strided view is the toy version of the same idea; cuDNN's version fuses it into the GEMM's inner loop.

## 5.6 What it costs in practice

Here is the complete benchmark, so you can run it yourself rather than trust a table:

```python
import time
torch.set_num_threads(4)

def conv_taps(x, w):
    """Vectorised over pixels; loops only over the KH*KW kernel taps."""
    N, C, H, W = x.shape; Co, _, KH, KW = w.shape
    OH, OW = H - KH + 1, W - KW + 1
    out = torch.zeros(N, Co, OH, OW)
    for i in range(KH):
        for j in range(KW):
            out += torch.einsum('nchw,oc->nohw', x[:, :, i:i+OH, j:j+OW], w[:, :, i, j])
    return out

def conv_strided_bench(x, w):
    N, C, H, W = x.shape; Co, _, KH, KW = w.shape
    sN, sC, sH, sW = x.stride()
    pt = x.as_strided((N, C, H-KH+1, W-KW+1, KH, KW), (sN, sC, sH, sW, sH, sW))
    return torch.einsum('nchwij,ocij->nohw', pt, w)

def conv_unfold_bench(x, w):
    N, _, H, W = x.shape; Co, KH = w.shape[0], w.shape[2]
    return (w.view(Co, -1) @ F.unfold(x, KH)).view(N, Co, H-KH+1, W-KH+1)

def bench(f, x, w, n=5):
    f(x, w)                                  # warm-up
    t = time.perf_counter()
    for _ in range(n): f(x, w)
    return (time.perf_counter() - t) / n * 1000

x = torch.randn(8, 64, 56, 56); w = torch.randn(128, 64, 3, 3)
for name, f in [("tap loop (9 einsums)",  conv_taps),
                ("strided view + einsum", conv_strided_bench),
                ("F.unfold + matmul",     conv_unfold_bench),
                ("F.conv2d",              lambda a, b: F.conv2d(a, b))]:
    print(f"{name:24s} {bench(f, x, w):6.1f} ms")

print("all agree with F.conv2d:",
      all(torch.allclose(f(x, w), F.conv2d(x, w), atol=2e-3)
          for f in [conv_taps, conv_strided_bench, conv_unfold_bench]))
```

```
tap loop (9 einsums)       92.4 ms
strided view + einsum      67.1 ms
F.unfold + matmul          54.5 ms
F.conv2d                   24.5 ms
all agree with F.conv2d: True
```

(CPU, 4 threads. Expect ±20% run-to-run variance on a shared machine; the *ordering* is the stable finding, not the exact milliseconds.)

Two lessons.

**First:** the fully naive 7-loop version from Section 4.4 is not in this table, because on this input it takes **minutes**, not milliseconds. Merely vectorising it into 9 batched einsums — one per kernel tap, with all pixels handled at once — already brings it to 92 ms. Python loop overhead is catastrophic; the first rule of PyTorch performance is *get the loop inside the library*.

**Second, and more important:** every hand-written version here loses to `F.conv2d`, by 2.2× to 3.8×, on CPU. On GPU with cuDNN the gap is typically larger still. So:

> **Never ship a hand-written convolution.** Write it once, to understand it. Then call `F.conv2d`.

The point of Part V is not to give you a conv implementation. It is to let you reason about what `F.conv2d` does to memory, which is what you need when you are debugging an OOM, choosing a memory format, or writing a custom operator that has no library equivalent.

## 5.7 The official API: `unfold` and `fold`

PyTorch exposes im2col and its adjoint as `F.unfold` / `F.fold` (also `nn.Unfold` / `nn.Fold`).

```python
x = torch.randn(1, 2, 5, 5)
w = torch.randn(3, 2, 3, 3)

cols = F.unfold(x, kernel_size=3)       # (N, C*KH*KW, L)
print("unfold shape:", tuple(cols.shape))

out = (w.view(3, -1) @ cols).view(1, 3, 3, 3)
print("matches conv2d:", torch.allclose(out, F.conv2d(x, w), atol=1e-5))
```

```
unfold shape: (1, 18, 9)
matches conv2d: True
```

`F.unfold` gives $(N,\ C K_H K_W,\ L)$ where $L = H_{out}W_{out}$ — patches as *columns*. It supports `dilation`, `padding`, `stride`. Unlike `as_strided` it **copies**, and it is bounds-checked.

`F.fold` is the inverse-shaped operation, and it does something very important: **where windows overlap, it adds**.

```python
ones = torch.ones(1, 1 * 3 * 3, 9)
print(F.fold(ones, output_size=(5, 5), kernel_size=3).squeeze())
```

```
tensor([[1., 2., 3., 2., 1.],
        [2., 4., 6., 4., 2.],
        [3., 6., 9., 6., 3.],
        [2., 4., 6., 4., 2.],
        [1., 2., 3., 2., 1.]])
```

Each cell shows how many windows covered that pixel. `fold` is **not** the inverse of `unfold`; `fold(unfold(x)) ≠ x`. It is the **adjoint** (transpose) of `unfold`, in the precise linear-algebra sense: $\langle \text{unfold}(x), v\rangle = \langle x, \text{fold}(v)\rangle$ for all $x, v$.

```python
x = torch.randn(1, 2, 5, 5)
v = torch.randn(1, 18, 9)
lhs = (F.unfold(x, 3) * v).sum()
rhs = (x * F.fold(v, output_size=(5, 5), kernel_size=3)).sum()
print("adjoint identity holds:", torch.allclose(lhs, rhs, atol=1e-4))
```

```
adjoint identity holds: True
```

**Remember this.** In Part VI we will show that *the backward pass of any linear operation is its adjoint*. Since the forward pass extracts overlapping patches, the backward pass must **scatter-add** gradients back into the overlaps. `fold` is that scatter-add. You have just met the backward pass of convolution, before deriving it.

## 5.8 The same trick everywhere else

Once you see convolution as **patch extraction + reduction**, a large part of modern deep learning turns out to be the same two steps with a different reduction or a different patch geometry. This is the most transferable idea in the tutorial, so it is worth making explicit.

| Operation | Patch geometry | Reduction |
|---|---|---|
| Convolution | overlapping, stride $s$, dilation $d$ | weighted sum (`einsum`) |
| Max pooling | usually non-overlapping ($s = K$) | `amax` |
| Average pooling | same | `mean` |
| **ViT patch embedding** | **non-overlapping, $s = K$** | linear projection |
| Local / windowed attention | non-overlapping or sliding windows | attention within each window |

The Vision Transformer case deserves a moment, because students often learn ViTs as something categorically different from CNNs. Splitting a 224×224 image into 16×16 patches and linearly projecting each one **is a convolution** — a stride-16, kernel-16, non-overlapping one:

```python
img = torch.randn(1, 3, 224, 224)
patch_embed = nn.Conv2d(3, 768, kernel_size=16, stride=16)   # kernel == stride
tokens = patch_embed(img).flatten(2).transpose(1, 2)
print(tuple(tokens.shape))      # (1, 196, 768)  -> 14*14 = 196 tokens
```
```
(1, 196, 768)
```

That is how the reference ViT implementations actually do it. Setting `stride = kernel_size` makes the windows tile the image exactly instead of overlapping — so the patch view has *no* memory reuse, which is why the equivalent `unfold` here costs nothing extra (blow-up factor $K^2/s^2 = 1$, by the formula in Section 5.5).

The practical payoff: when you meet a new architecture described as "we partition the feature map into windows and do X," you already know how to implement it. Build the strided view, apply X, and let autograd handle the scatter-add.

## 5.9 Checkpoint questions

**Q5.1** For input `(N,C,H,W)` = `(4,32,28,28)`, kernel 5×5, stride 1, padding 2: what is the shape of the 6-D strided `patches` view? How many logical elements? How many *real* floats back it?

**Q5.2** In `conv2d_strided`, why must `x.stride()` be read **after** `F.pad`, not before?

**Q5.3** `patches.permute(0,2,3,1,4,5).reshape(N*OH*OW, C*KH*KW)` — argue from strides why this cannot be a view.

<details>
<summary><b>Answers</b></summary>

**A5.1** Padded input is 32×32, so $OH=OW=28$. Patches shape `(4, 32, 28, 28, 5, 5)` = 2,508,800 logical elements. Backed by the padded buffer: $4 \times 32 \times 32 \times 32 = 131{,}072$ floats. A **19×** logical-to-physical ratio, achieved for free.

**A5.2** `F.pad` allocates a **new, larger** tensor. Its row stride is $W + 2p$, not $W$. Using pre-pad strides would make every window read from the wrong row — an off-by-$2p$-per-row bug that produces plausible-looking garbage rather than a crash. This is the single most common bug when people write this code themselves.

**A5.3** The patch axes overlap: consecutive windows share memory. A view requires that the flat index be an affine function $o + \sum_k i_k s_k$ of the indices. Flattening $(h,v,i,j)$ into a single axis would require consecutive positions in the new axis to differ by a *constant* stride, but the actual address jumps are irregular (small within a window, then a jump at the window boundary). No single stride exists. Hence a copy — and hence the 8×–25× memory blow-up.

</details>

---

# Part VI — Backpropagation through convolution

This is the heart of the tutorial. We derive the three gradients by hand, interpret each one geometrically, express each as a standard operation, and verify all of it numerically.

## 6.1 What backpropagation actually asks for

Reverse-mode automatic differentiation works like this. Somewhere downstream there is a scalar loss $L$. Each operation is handed

$$g \;=\; \frac{\partial L}{\partial y}$$

— the gradient of the loss with respect to **its own output** — and must return the gradient with respect to **each of its inputs**. Nothing else. Each layer is a local, self-contained transformation of gradients.

For a convolution with inputs $x$, $w$, $b$ and output $y$, we must produce three things:

$$\frac{\partial L}{\partial x}, \qquad \frac{\partial L}{\partial w}, \qquad \frac{\partial L}{\partial b}$$

The tool is the multivariate chain rule. For any input element $\theta$:

$$\frac{\partial L}{\partial \theta} \;=\; \sum_{n,o,h,v} \frac{\partial L}{\partial y[n,o,h,v]} \cdot \frac{\partial y[n,o,h,v]}{\partial \theta} \;=\; \sum_{n,o,h,v} g[n,o,h,v]\cdot \frac{\partial y[n,o,h,v]}{\partial \theta}$$

**The sum is the important part.** It runs over *every output element that $\theta$ influenced*. Get the set of influenced outputs right and the derivation writes itself. Everything below is bookkeeping about which outputs each input touched.

Our forward equation again:

$$y[n,o,h,v] = b[o] + \sum_{c,i,j} x[n,c,\,sh+di,\,sv+dj]\; w[o,c,i,j] \tag{F}$$

## 6.2 Gradient with respect to the bias

$\partial y[n,o,h,v] / \partial b[o'] = 1$ if $o = o'$, else 0. So:

$$\boxed{\ \frac{\partial L}{\partial b[o]} = \sum_{n,h,v} g[n,o,h,v]\ }$$

**Intuition:** $b[o]$ was added to *every* spatial position of *every* image in output channel $o$. It influenced all of them equally, so it collects all their gradients.

```python
grad_b = g.sum(dim=(0, 2, 3))
```

This is the general pattern: **broadcasting forward becomes summation backward.** Recall from Section 2.4 that broadcasting is stride-0 reading — many logical positions aliasing one physical value. Its adjoint must accumulate all those contributions back into the one value. Broadcast forward ⟺ sum backward is the same duality as gather forward ⟺ scatter-add backward, which will reappear immediately.

## 6.3 Gradient with respect to the weights

Differentiate (F) with respect to $w[o,c,i,j]$. That specific weight appears once in the formula for each output element in channel $o$, multiplied by the input value at $(sh+di,\ sv+dj)$:

$$\frac{\partial y[n,o',h,v]}{\partial w[o,c,i,j]} = \begin{cases} x[n,c,\,sh+di,\,sv+dj] & o' = o\\ 0 & \text{otherwise}\end{cases}$$

Therefore:

$$\boxed{\ \frac{\partial L}{\partial w[o,c,i,j]} = \sum_{n,h,v} g[n,o,h,v]\cdot x[n,c,\,sh+di,\,sv+dj]\ }$$

**Intuition:** *"How much would the loss change if I nudged this one filter tap?"* That tap was applied at every spatial location. So you take the input values it multiplied against, weight each by how much the loss cared about that output, and add them up. It is a correlation between the input and the output gradient.

**And notice**: the term $x[n,c,sh+di,sv+dj]$ is *exactly* the `patches` tensor from Part V. We already built that view in the forward pass. So:

```python
grad_w = torch.einsum('nchwij,nohw->ocij', patches, g)
```

Compare with the forward einsum, `'nchwij,ocij->nohw'`. Same first operand; the other two index-sets have swapped roles. This is not a coincidence — it is the transpose structure of a bilinear form. Mechanically: to differentiate a product of two tensors with respect to one of them, keep the same einsum operands and swap the "other input" with the "output".

Critically, **the same strided view serves both passes**. Whatever stride and dilation trickery encoded the forward geometry automatically encodes the backward geometry too. You do not re-derive anything.

## 6.4 Gradient with respect to the input — the interesting one

$x[n,c,r,q]$ (row $r$, column $q$) contributes to output $y[n,o,h,v]$ exactly when

$$r = sh + di \quad\text{and}\quad q = sv + dj \qquad\text{for some valid } (i,j)$$

So:

$$\boxed{\ \frac{\partial L}{\partial x[n,c,r,q]} = \sum_{o}\ \sum_{\substack{h,v,i,j \\ sh+di=r,\ sv+dj=q}} g[n,o,h,v]\; w[o,c,i,j]\ }$$

**Intuition — and this is the sentence to remember:** *in the forward pass every input pixel was read by several overlapping windows; in the backward pass every input pixel collects gradient from every window that read it.* Forward **gathers** with overlap; backward **scatter-adds** into the overlap.

This is exactly the `unfold`/`fold` duality from Section 5.7, and exactly why `fold` adds rather than overwrites. You met the answer before you asked the question.

### Why people say "backprop through conv is a flipped convolution"

Take the simplest case, $s = 1$, $d = 1$, $p = 0$. The constraint becomes $h = r - i$, $v = q - j$:

$$\frac{\partial L}{\partial x[n,c,r,q]} = \sum_{o,i,j} g[n,o,\,r-i,\,q-j]\; w[o,c,i,j]$$

That **minus sign** is the signature of a true convolution (Section 4.2). So the input gradient is a genuine convolution of $g$ with $w$ — equivalently, a *cross-correlation of $g$ with the spatially flipped $w$*. It also needs zero-padding by $K-1$ on each side (a "full" convolution), because output positions $r-i$ fall outside the valid range near the borders. And the channel axes swap: $w$ maps $C_{in}\to C_{out}$ forward, so its adjoint maps $C_{out}\to C_{in}$.

Three transformations, each with a reason:

| Transformation | Why |
|---|---|
| Flip the kernel spatially | the index becomes $r-i$ instead of $h+i$ |
| Swap input/output channel axes | the adjoint of a linear map reverses its direction |
| Pad by $K-1$ ("full" convolution) | border outputs receive gradient from fewer windows |

### When stride > 1: dilate the gradient

With $s>1$, only every $s$-th input position is a window start, so most input pixels receive gradient from fewer windows. The clean way to express this: **insert $s-1$ zeros between adjacent elements of $g$**, then do the $s=1$ full flipped convolution.

The zeros are not a hack — they are the correct statement that those intermediate positions contributed nothing to any output. Let's verify this identity directly:

```python
torch.manual_seed(0)
x = torch.randn(1, 1, 7, 7, requires_grad=True)
w = torch.randn(1, 1, 3, 3)

y = F.conv2d(x, w, stride=2)
g = torch.randn_like(y)
grad_x_autograd = torch.autograd.grad(y, x, g)[0]

# Manual: dilate g by the stride, then full-convolve with the flipped kernel
OH = g.shape[-1]
g_dilated = torch.zeros(1, 1, (OH - 1) * 2 + 1, (OH - 1) * 2 + 1)
g_dilated[..., ::2, ::2] = g                       # <-- a strided write!

manual = F.conv2d(F.pad(g_dilated, (2, 2, 2, 2)),   # full padding = K-1 = 2
                  w.flip(-1, -2).transpose(0, 1))   # flip + swap channels
manual = F.pad(manual, (0, 7 - manual.shape[-1], 0, 7 - manual.shape[-2]))

print("dilate + flip + full-conv == autograd:",
      torch.allclose(grad_x_autograd, manual, atol=1e-5))
```

```
dilate + flip + full-conv == autograd: True
```

Note the line `g_dilated[..., ::2, ::2] = g`. Strided assignment — the exact mirror of the strided *read* that stride-2 convolution performs in the forward pass. Forward reads every second pixel; backward writes to every second pixel. The symmetry is not decorative; it is the adjoint.

The trailing `F.pad` handles the case where the floor in the output-size formula discarded some input pixels: those pixels influenced nothing, so their gradient is exactly zero.

### Closing the loop: the input gradient *is* `fold`

Section 5.7 promised that `fold` is the backward pass of patch extraction. Let us actually show it, because the claim is easy to state and easy to doubt.

If the forward pass is "gather patches, then contract with $w$", the backward pass must be "contract $g$ with $w$ to get a gradient *per patch element*, then scatter-add those patches back." The second step is exactly `F.fold`:

```python
torch.manual_seed(0)
x = torch.randn(2, 3, 9, 9, dtype=torch.double, requires_grad=True)
w = torch.randn(4, 3, 3, 3, dtype=torch.double)

y = F.conv2d(x, w, padding=1)
g = torch.randn_like(y)
grad_x_autograd = torch.autograd.grad(y, x, g)[0]

# 1. spread each output gradient back over the patch that produced it
grad_patches = torch.einsum('nohw,ocij->ncijhw', g, w).reshape(2, 3 * 3 * 3, -1)
# 2. scatter-add the overlapping patches back into image space
grad_x_fold = F.fold(grad_patches, output_size=(9, 9), kernel_size=3, padding=1)

print("grad_x == fold(g ⊗ w):", torch.allclose(grad_x_autograd, grad_x_fold))
```

```
grad_x == fold(g ⊗ w): True
```

Exactly equal. This is the single most compact statement of what backpropagation through a convolution *is*: **`unfold` forward, `fold` backward.** The einsum spreads each output's gradient over the patch it came from; `fold` performs the accumulation over overlaps that Section 6.4's constrained sum describes. If you remember one picture from Part VI, make it this one.

### The practical route: `conv_transpose2d`

You do not need to assemble flips and dilations yourself. `F.conv_transpose2d` **is** the adjoint of `F.conv2d`. That is its definition — "transposed convolution" means "the transpose of the convolution matrix," not "the inverse."

```python
grad_x = F.conv_transpose2d(g, w, stride=s, padding=p, dilation=d,
                            output_padding=(oph, opw))
```

`output_padding` exists precisely because of the floor. When $s>1$, several input sizes map to the same output size, so the adjoint cannot know which one to return. You tell it:

$$\text{output\_padding} = H_{in} + 2p - d(K_H-1) - 1 - (H_{out}-1)\,s$$

This is the output-size formula rearranged, measuring how much the floor threw away. But there is a much cleaner way to see it. Write $M = H_{in} + 2p - d(K_H-1) - 1$, so that $H_{out} = \lfloor M/s\rfloor + 1$. Substituting:

$$\text{output\_padding} = M - s\left\lfloor \frac{M}{s} \right\rfloor = M \bmod s$$

$$\boxed{\ \text{output\_padding} = \big(H_{in} + 2p - d(K_H-1) - 1\big) \bmod s\ }$$

**It is just the remainder** — the pixels at the bottom/right edge that the stride stepped over and no window ever reached. This form is easier to remember, and it immediately explains a constraint you will otherwise meet as a confusing error message: PyTorch requires `output_padding < stride`, which is automatic for a remainder. Verified exhaustively over several hundred $(H, p, K, s, d)$ combinations:

```python
for H in range(8, 25):
  for p in range(3):
    for k in (3, 5):
      for s in (1, 2, 3, 4):
        for d in (1, 2):
          M = H + 2*p - d*(k-1) - 1
          if M < 0: continue
          OH = M // s + 1
          assert H + 2*p - d*(k-1) - 1 - (OH-1)*s == M % s < s
print("output_padding == (H + 2p - d(K-1) - 1) mod s  — verified")
```
```
output_padding == (H + 2p - d(K-1) - 1) mod s  — verified
```

## 6.5 Putting it together: a custom autograd Function

Here is a complete, correct, self-contained convolution layer with hand-written forward and backward.

```python
class Conv2dFunction(torch.autograd.Function):
    """Convolution with hand-derived gradients. Educational; F.conv2d is faster."""

    @staticmethod
    def forward(ctx, x, w, b, stride, padding, dilation):
        ctx.save_for_backward(x, w)
        ctx.conf = (stride, padding, dilation)

        xp = F.pad(x, (padding,) * 4) if padding else x
        N, C, H, W = xp.shape
        Cout, _, KH, KW = w.shape
        s, d = stride, dilation
        OH = (H - d * (KH - 1) - 1) // s + 1
        OW = (W - d * (KW - 1) - 1) // s + 1

        sN, sC, sH, sW = xp.stride()
        patches = xp.as_strided((N, C, OH, OW, KH, KW),
                                (sN, sC, sH * s, sW * s, sH * d, sW * d))

        out = torch.einsum('nchwij,ocij->nohw', patches, w)
        if b is not None:
            out = out + b.view(1, -1, 1, 1)
        return out

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        s, p, d = ctx.conf
        g = g.contiguous()                       # incoming grad may be strided

        N, C, H, W = x.shape
        Cout, _, KH, KW = w.shape
        OH, OW = g.shape[2:]

        # --- rebuild the same strided view used in forward ---
        xp = F.pad(x, (p,) * 4) if p else x
        sN, sC, sH, sW = xp.stride()
        patches = xp.as_strided((N, C, OH, OW, KH, KW),
                                (sN, sC, sH * s, sW * s, sH * d, sW * d))

        # --- (1) bias: broadcast forward  =>  sum backward ---
        grad_b = g.sum(dim=(0, 2, 3))

        # --- (2) weights: correlate input patches with output gradient ---
        grad_w = torch.einsum('nchwij,nohw->ocij', patches, g)

        # --- (3) input: the adjoint, i.e. transposed convolution ---
        oph = (H + 2 * p - d * (KH - 1) - 1) % s        # the remainder (Section 6.4)
        opw = (W + 2 * p - d * (KW - 1) - 1) % s
        grad_x = F.conv_transpose2d(g, w, stride=s, padding=p, dilation=d,
                                    output_padding=(oph, opw))

        return grad_x, grad_w, grad_b, None, None, None   # None for non-tensor args
```

### Verification 1: `gradcheck`

`torch.autograd.gradcheck` compares your analytic gradients against finite differences computed in `float64`. If it passes, your derivation is right.

```python
for (s, p, d) in [(1,0,1), (2,1,1), (2,1,2)]:
    x = torch.randn(2, 3, 8, 9, dtype=torch.double, requires_grad=True)
    w = torch.randn(4, 3, 3, 3, dtype=torch.double, requires_grad=True)
    b = torch.randn(4,          dtype=torch.double, requires_grad=True)
    ok = torch.autograd.gradcheck(Conv2dFunction.apply, (x, w, b, s, p, d),
                                  eps=1e-6, atol=1e-6)
    print(f"stride={s} pad={p} dil={d}  gradcheck: {ok}")
```

```
stride=1 pad=0 dil=1  gradcheck: True
stride=2 pad=1 dil=1  gradcheck: True
stride=2 pad=1 dil=2  gradcheck: True
```

**Always use `float64` for `gradcheck`.** In `float32`, finite differences $\frac{f(\theta+\epsilon)-f(\theta-\epsilon)}{2\epsilon}$ suffer catastrophic cancellation and the test fails on correct code. Nearly every "gradcheck is broken" report is a dtype mistake.

### Verification 2: against PyTorch's own gradients

```python
for (s, p, d) in [(1,0,1), (2,1,1), (2,1,2), (3,2,2)]:
    x = torch.randn(2, 3, 10, 11, dtype=torch.double, requires_grad=True)
    w = torch.randn(4, 3, 3, 3,   dtype=torch.double, requires_grad=True)
    b = torch.randn(4,            dtype=torch.double, requires_grad=True)

    ref = F.conv2d(x, w, b, stride=s, padding=p, dilation=d)
    g   = torch.randn_like(ref)
    gx_ref, gw_ref, gb_ref = torch.autograd.grad(ref, [x, w, b], g)

    mine = Conv2dFunction.apply(x, w, b, s, p, d)
    gx, gw, gb = torch.autograd.grad(mine, [x, w, b], g)

    print(f"s={s} p={p} d={d}  grad_x:{torch.allclose(gx,gx_ref)}"
          f"  grad_w:{torch.allclose(gw,gw_ref)}  grad_b:{torch.allclose(gb,gb_ref)}")
```

```
s=1 p=0 d=1  grad_x:True  grad_w:True  grad_b:True
s=2 p=1 d=1  grad_x:True  grad_w:True  grad_b:True
s=2 p=1 d=2  grad_x:True  grad_w:True  grad_b:True
s=3 p=2 d=2  grad_x:True  grad_w:True  grad_b:True
```

All three gradients, all four configurations, exact to `float64` tolerance.

### Verification 3: it actually trains

`gradcheck` proves the derivatives are right at a point. It does **not** prove your layer works inside a real model — that it handles `nn.Module` plumbing, non-contiguous inputs, optimiser updates, or being stacked. That gap is where integration bugs live, so close it: drop the custom Function into an actual network and train it beside the built-in.

```python
import torch.nn as nn

class MyConv2d(nn.Conv2d):
    """Identical to nn.Conv2d, but routed through our hand-written autograd Function."""
    def forward(self, x):
        return Conv2dFunction.apply(x, self.weight, self.bias,
                                    self.stride[0], self.padding[0], self.dilation[0])

def make_net(conv):
    return nn.Sequential(
        conv(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        conv(8, 16, 3, stride=2, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10))

def train(conv, steps=30):
    torch.manual_seed(0)                    # identical initialisation
    net = make_net(conv)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    torch.manual_seed(1)                    # identical data
    X, Y = torch.randn(64, 1, 28, 28), torch.randint(0, 10, (64,))
    losses = []
    for _ in range(steps):
        loss = F.cross_entropy(net(X), Y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses

builtin, mine = train(nn.Conv2d), train(MyConv2d)
print("step   nn.Conv2d    MyConv2d")
for i in (0, 9, 19, 29):
    print(f"{i:4d}   {builtin[i]:9.5f}   {mine[i]:9.5f}")
print("max divergence over 30 steps:",
      max(abs(a - b) for a, b in zip(builtin, mine)))
```

```
step   nn.Conv2d    MyConv2d
   0     2.33798     2.33798
   9     2.27648     2.27648
  19     2.25483     2.25483
  29     2.24119     2.24119
max divergence over 30 steps: 2.384185791015625e-07
```

Two networks, same seed, same data, two entirely different convolution implementations — and the loss curves agree to float32 round-off over 30 optimiser steps. That is the test that lets you ship.

Note how little glue was needed: subclassing `nn.Conv2d` and overriding only `forward` inherits parameter registration, initialisation, `.to(device)`, and state-dict compatibility for free. This is the right pattern for a custom operator that is a drop-in replacement for a standard one.

### Verification 4: second derivatives

If anyone will ever call `backward(create_graph=True)` on your layer — gradient penalties (WGAN-GP), meta-learning, influence functions, some regularisers — your `backward` must itself be differentiable. Test it:

```python
x = torch.randn(1, 2, 6, 6, dtype=torch.double, requires_grad=True)
w = torch.randn(3, 2, 3, 3, dtype=torch.double, requires_grad=True)
b = torch.randn(3,          dtype=torch.double, requires_grad=True)
print("gradgradcheck:", torch.autograd.gradgradcheck(Conv2dFunction.apply, (x, w, b, 2, 1, 1)))
```
```
gradgradcheck: True
```

It passes *because* the `backward` above is written entirely in differentiable PyTorch operations (`einsum`, `sum`, `conv_transpose2d`). Had it used an in-place buffer fill or a `.data` access, first derivatives would still be correct and this check would fail — a bug that stays hidden until the day someone adds a gradient penalty.

### The shortcut you should know about first

PyTorch already ships the two hard gradients, and a tutorial that made you derive them for 3,000 words owes you this:

```python
import torch.nn.grad as G

x = torch.randn(2, 3, 9, 9, dtype=torch.double, requires_grad=True)
w = torch.randn(4, 3, 3, 3, dtype=torch.double, requires_grad=True)
out = F.conv2d(x, w, stride=2, padding=1)
g = torch.randn_like(out)

ref_x, ref_w = torch.autograd.grad(out, [x, w], g)
lib_x = G.conv2d_input(x.shape, w, g, stride=2, padding=1)
lib_w = G.conv2d_weight(x, w.shape, g, stride=2, padding=1)
print(torch.allclose(lib_x, ref_x), torch.allclose(lib_w, ref_w))
```
```
True True
```

`torch.nn.grad` provides `conv1d/2d/3d_input` and `conv1d/2d/3d_weight`. **Use these in real code** when you need convolution gradients explicitly — inside a custom operator, for a gradient penalty, for an adversarial attack. The derivation you just worked through is what lets you read their output and know when they are the right tool; it is not a reason to re-implement them.

## 6.6 The best option of all: let autograd differentiate `as_strided`

Here is a fact worth knowing before you spend a weekend hand-deriving something: **`as_strided` is itself a differentiable operation in PyTorch.** Its backward pass performs exactly the scatter-add into overlapping positions that Section 6.4 describes.

Which means the forward function from Part V is *already* fully differentiable, with no `backward` written at all:

```python
def conv_strided_autograd(x, w, b, s, p, d):
    xp = F.pad(x, (p,) * 4) if p else x
    N, C, H, W = xp.shape
    Cout, _, KH, KW = w.shape
    OH = (H - d*(KH-1) - 1)//s + 1
    OW = (W - d*(KW-1) - 1)//s + 1
    sN, sC, sH, sW = xp.stride()
    patches = xp.as_strided((N, C, OH, OW, KH, KW),
                            (sN, sC, sH*s, sW*s, sH*d, sW*d))
    return torch.einsum('nchwij,ocij->nohw', patches, w) + b.view(1, -1, 1, 1)

for (s, p, d) in [(1,0,1), (2,1,1), (2,1,2)]:
    x = torch.randn(2, 3, 9, 9, dtype=torch.double, requires_grad=True)
    w = torch.randn(4, 3, 3, 3, dtype=torch.double, requires_grad=True)
    b = torch.randn(4,          dtype=torch.double, requires_grad=True)
    ref = F.conv2d(x, w, b, stride=s, padding=p, dilation=d)
    g = torch.randn_like(ref)
    a = torch.autograd.grad(conv_strided_autograd(x,w,b,s,p,d), [x,w,b], g)
    e = torch.autograd.grad(ref, [x,w,b], g)
    print(f"s={s} p={p} d={d}  autograd-through-as_strided matches:",
          all(torch.allclose(u, v) for u, v in zip(a, e)))
```

```
s=1 p=0 d=1  autograd-through-as_strided matches: True
s=2 p=1 d=1  autograd-through-as_strided matches: True
s=2 p=1 d=2  autograd-through-as_strided matches: True
```

**The lesson for industry work.** When you build a custom operator, first ask whether you can express the forward pass entirely in differentiable PyTorch primitives. If you can, autograd gives you a correct backward pass for free, and you cannot introduce a gradient bug. Write a custom `backward` only when you have a concrete reason:

- you can beat autograd's memory usage (e.g. recompute instead of storing an activation);
- you can fuse several steps into one kernel;
- your forward pass calls out to non-differentiable code (a C++ solver, a simulator);
- the naive gradient is numerically unstable and you know a better-conditioned form.

If you do write one, `gradcheck` is not optional. A wrong gradient does not crash. It trains — slowly, to a worse optimum — and you will blame your learning rate for three weeks.

## 6.7 Memory: what backward forces you to keep

The backward pass needs `x` (for `grad_w`) and `w` (for `grad_x`). So **the input activation of every conv layer stays alive from forward until backward.** This, not the parameters, is what fills your GPU.

For ResNet-50 at batch 32, weights are ~100 MB while stored activations run to several GB. Three standard levers:

| Technique | Mechanism | Cost |
|---|---|---|
| Gradient checkpointing (`torch.utils.checkpoint`) | Discard activations, recompute in backward | ~30% more compute |
| Mixed precision (`torch.autocast`) | Store activations in bf16/fp16 | Halves activation memory |
| Smaller batch + gradient accumulation | Fewer activations alive at once | More optimiser steps |

And a stride-specific one: `bias=False` in any conv followed by BatchNorm. The bias is mathematically redundant there (BatchNorm subtracts the mean, removing any constant offset), so `nn.Conv2d(..., bias=False)` before `nn.BatchNorm2d` is standard practice. Look at any torchvision ResNet source and you will see it.

## 6.8 Checkpoint questions

**Q6.1** Why does `grad_b = g.sum(dim=(0,2,3))` sum over dims 0, 2 and 3 but not 1?

**Q6.2** In a stride-1, no-padding conv, which input pixel receives gradient from the *fewest* output positions? How many?

**Q6.3** You write a custom backward, `gradcheck` fails by a factor of about 2 on `grad_w` only. Where would you look first?

**Q6.4** True or false: `conv_transpose2d` undoes `conv2d`, so applying both recovers the original input.

<details>
<summary><b>Answers</b></summary>

**A6.1** Dim 1 is the output-channel axis, and there is one bias per output channel — so each channel keeps its own gradient. Dims 0, 2, 3 (batch, height, width) are the axes along which the bias was broadcast, and broadcasting forward means summing backward.

**A6.2** The corner pixels, e.g. `[0,0]`. Only one window (the one starting at the origin) ever covered them, so they receive gradient from exactly **1** output position per output channel. Interior pixels get $K_HK_W$. This non-uniformity is a real effect — it is one reason border pixels are learned less reliably, and part of the motivation for reflection or replication padding over zero padding in some architectures.

**A6.3** A factor of exactly 2 almost always means **double counting**: summing over an axis twice, or adding into an already-accumulated buffer, or forgetting that PyTorch *accumulates* gradients (`+=`) rather than assigning. Check whether you zeroed a buffer, and check for a duplicated reduction in the einsum string. Consistent-factor errors are structural; random-looking errors are usually index arithmetic.

**A6.4** **False.** "Transposed" means the transpose $A^\top$ of the convolution matrix $A$, not the inverse $A^{-1}$. $A^\top A \ne I$ in general. Transposed convolution restores the *shape*, not the *values*. In a decoder, its weights are learned independently of the encoder's — the name describes the shape-transforming and gradient-routing structure, nothing more. (The historical name "deconvolution" is worse and should be avoided; it means something entirely different in signal processing.)

</details>

---

# Part VII — Groups, depthwise, and transposed convolution

## 7.1 Grouped convolution

`groups=G` splits the channel axis into $G$ independent blocks. Group $k$ of the input channels is convolved only with group $k$ of the filters; the results are concatenated.

Requirements: $C_{in}$ and $C_{out}$ must both be divisible by $G$, and the weight has shape $(C_{out},\ C_{in}/G,\ K_H,\ K_W)$ — note the **divided** second dimension.

```python
x = torch.randn(1, 4, 6, 6)
w = torch.randn(6, 2, 3, 3)          # Cin/groups = 4/2 = 2

grouped = F.conv2d(x, w, groups=2)
manual  = torch.cat([F.conv2d(x[:, :2], w[:3]),
                     F.conv2d(x[:, 2:], w[3:])], dim=1)
print(tuple(grouped.shape), torch.allclose(grouped, manual, atol=1e-5))
```

```
(1, 6, 4, 4) True
```

Parameters and MACs both drop by a factor of $G$, because each filter sees $G$ times fewer input channels.

**Where you meet it:** AlexNet used `groups=2` to fit across two GPUs in 2012. ResNeXt uses grouped convolutions as its core design element. And `groups = C_in` gives depthwise convolution.

## 7.2 Depthwise separable convolution

Set `groups = C_in` with `C_out = C_in`: each input channel gets exactly one filter, and channels never mix. Follow it with a 1×1 convolution to mix channels. This pair — **depthwise + pointwise** — is the *depthwise separable convolution* behind MobileNet, EfficientNet, and most mobile vision models.

```python
import torch.nn as nn
standard  = nn.Conv2d(64, 64, 3, padding=1, bias=False)
depthwise = nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False)
pointwise = nn.Conv2d(64, 64, 1, bias=False)

n = lambda m: sum(p.numel() for p in m.parameters())
print("standard         :", n(standard))
print("depthwise+pointwise:", n(depthwise) + n(pointwise))
print("reduction        :", round(n(standard) / (n(depthwise) + n(pointwise)), 1), "x")
```

```
standard         : 36864
depthwise+pointwise: 4672
reduction        : 7.9 x
```

Roughly $\frac{1}{K^2} + \frac{1}{C_{out}} = \frac{1}{9} + \frac{1}{64} \approx 0.127$, i.e. ~7.9×. Same receptive field, same output shape, 8× fewer parameters and MACs.

**An industry caveat you should know.** Depthwise convolutions have poor arithmetic intensity: they move a lot of memory per multiply-add, because each filter is used on exactly one channel instead of being reused across $C_{in}$ of them. The consequence, widely reported and easy to reason about but *not* measured in this document, is that depthwise layers are often **memory-bandwidth bound** on GPUs, so an 8× reduction in FLOPs can yield far less than 8× in wall-clock speedup — sometimes almost none. This is the standard illustration of why FLOPs are a bad proxy for latency. Measure on the target hardware before claiming a speedup.

## 7.3 Transposed convolution

We met it in Part VI as the adjoint of convolution. Used as a *layer*, it upsamples.

```python
up = F.conv_transpose2d(torch.randn(1, 3, 4, 4), torch.randn(3, 5, 3, 3), stride=2)
print(tuple(up.shape))        # (1, 5, 9, 9)

print("Conv2d weight shape         :", tuple(nn.Conv2d(3, 5, 3).weight.shape))
print("ConvTranspose2d weight shape:", tuple(nn.ConvTranspose2d(3, 5, 3).weight.shape))
```

```
(1, 5, 9, 9)
Conv2d weight shape         : (5, 3, 3, 3)
ConvTranspose2d weight shape: (3, 5, 3, 3)
```

**Note the swapped weight layout:** `Conv2d` stores $(C_{out}, C_{in}/G, K_H, K_W)$, `ConvTranspose2d` stores $(C_{in}, C_{out}/G, K_H, K_W)$. This reflects the direction reversal of the adjoint, and it is a frequent source of shape errors when people port weights between the two.

Output size (the output-size formula solved for the input):

$$H_{out} = (H_{in}-1)\,s - 2p + d(K_H-1) + 1 + \text{output\_padding}$$

**The checkerboard-artifact problem.** When `kernel_size` is not divisible by `stride`, output pixels receive contributions from different numbers of input positions, producing periodic bright/dark patterns in generated images. Two standard fixes:

1. Use `kernel_size` divisible by `stride` (e.g. 4×4 kernel with stride 2).
2. Replace it entirely with `nn.Upsample(scale_factor=2, mode='nearest')` followed by a stride-1 `Conv2d`. This is the more common choice in modern generative architectures.

---

# Part VIII — Memory format: NCHW vs channels-last

> ### ⚠️ A note on evidence, for Parts VIII and IX
>
> Every code block up to this point was executed and its real output pasted in. **The performance claims in Parts VIII and IX were not measured on the machine that produced this tutorial** — it has no GPU, and channels-last and cuDNN algorithm selection are GPU phenomena. What follows is accurate as a description of *mechanism* (the stride layouts are printed from real runs, and those you can trust), but every statement about *speed* is reported, not verified here.
>
> Treat these two parts as a map of what to measure, not as measurements. Where I give a number, I say where it comes from. This distinction matters more than it may seem: a large fraction of confidently-repeated deep-learning performance folklore is someone's benchmark from 2019 on hardware you do not own.

## 8.1 The problem with NCHW

Return to Section 1.4. A `(8, 3, 224, 224)` NCHW tensor has stride `(150528, 50176, 224, 1)`. The R, G and B values of one pixel are **50,176 elements apart**.

Convolution consumes all channels of a spatial neighbourhood simultaneously. In NCHW, gathering them means touching $C_{in}$ widely separated memory regions. Each touch pulls a cache line; most of what that cache line holds is data you do not want yet. The hardware's memory system hates this.

## 8.2 Channels-last: same shape, different strides

```python
t  = torch.randn(2, 3, 4, 5)
tc = t.to(memory_format=torch.channels_last)

print("shape (both)        :", tuple(t.shape), tuple(tc.shape))
print("NCHW stride         :", t.stride())
print("channels_last stride:", tc.stride())
print("tc.is_contiguous()                              :", tc.is_contiguous())
print("tc.is_contiguous(memory_format=channels_last)   :",
      tc.is_contiguous(memory_format=torch.channels_last))
```

```
shape (both)        : (2, 3, 4, 5) (2, 3, 4, 5)
NCHW stride         : (60, 20, 5, 1)
channels_last stride: (60, 1, 15, 3)
tc.is_contiguous()                              : False
tc.is_contiguous(memory_format=channels_last)   : True
```

Read the channels-last stride `(60, 1, 15, 3)`:

- Channel stride is **1** — R, G, B of a pixel are now *adjacent in memory*.
- Width stride is 3 = $C$ — the next pixel right is one full channel-vector away.
- Height stride is 15 = $W \cdot C$.

Physically the buffer is ordered N, H, W, C. **The shape is unchanged** — still `(2,3,4,5)`, still indexed `t[n,c,h,w]`. Only the strides differ. Every piece of your code that indexes the tensor keeps working untouched. This is the payoff of the shape/stride separation: layout becomes a pure performance knob, invisible to semantics.

Note that `is_contiguous()` returns `False` — which is correct and expected, because the unqualified call means "row-major NCHW contiguous." You must pass `memory_format=` to ask the right question. **Code containing `assert t.is_contiguous()` will fire spuriously on channels-last tensors**, and this is a real and common source of breakage when teams enable channels-last on an existing codebase.

## 8.3 Using it

```python
model = model.to(memory_format=torch.channels_last)
x     = x.to(memory_format=torch.channels_last)
y     = model(x)
print(y.is_contiguous(memory_format=torch.channels_last))   # True — format propagates
```

Convolution layers propagate the memory format, so you convert the model and the input once at the top and it flows through the network. Mixing formats mid-network forces PyTorch to insert transposes — real copies — which can make things *slower*. Convert once, and profile to confirm no format thrash.

## 8.4 Why it wins

Two effects, both worth understanding:

1. **Cache locality.** The convolution's inner loop reads $C_{in}$ contiguous floats per spatial position instead of $C_{in}$ scattered ones. Every cache line is fully used.
2. **Tensor Cores.** NVIDIA's Tensor Cores (Volta and later) and cuDNN's fastest fp16/bf16 convolution kernels are written against an NHWC layout. If you hand cuDNN an NCHW tensor, it frequently transposes to NHWC internally, runs, and transposes back — you pay for two copies. Giving it NHWC directly skips both.

**The standard industry recipe** is channels-last together with mixed precision:

```python
model = model.to(memory_format=torch.channels_last).cuda()

for x, target in loader:
    x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = criterion(model(x), target.cuda(non_blocking=True))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

The two go together: channels-last mainly pays off when the underlying kernels are the NHWC Tensor Core ones, which requires fp16/bf16.

> **On `GradScaler`.** You will see `torch.amp.GradScaler` in most mixed-precision examples, and it is **not** used above on purpose. Its job is to rescale the loss so that small gradients do not flush to zero in **fp16**, which has only ~5 exponent bits. **bfloat16 has the same exponent range as fp32** and does not underflow that way, so a scaler is unnecessary with bf16 — harmless, but noise that suggests you are guarding against a problem you do not have. Use `GradScaler` with `dtype=torch.float16`; omit it with `dtype=torch.bfloat16`. On Ampere and newer, bf16 is usually the better default precisely because it removes this whole concern.

**But measure.** Channels-last is generally a win for convolution-heavy fp16/bf16 GPU training, and can be neutral or negative for small models, CPU inference, non-conv workloads, or networks with many layout-breaking operations. You will see specific speedup multipliers quoted in blog posts and tutorials; treat all of them, including any I might have given you, as anecdotes about someone else's hardware. The only number that means anything for your model is the one your own profiler prints. Benchmark like this:

```python
import time

def bench(model, x, n=50, warmup=10):
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000      # ms/iter
```

The `torch.cuda.synchronize()` calls are mandatory. CUDA is asynchronous; without them you are timing how fast Python can queue work, which is not a number anyone wants.

---

# Part IX — How convolution is really computed

You will never write the kernel. But you will choose flags, read profiler output, and explain performance anomalies, and for that you need to know what the algorithms are.

## 9.1 The four families

| Algorithm | Idea | Best for | Trade-off |
|---|---|---|---|
| **Direct** | Nested loops, tiled and vectorised by hand | Small channel counts, depthwise, edge devices | Hard to reach peak FLOPs |
| **im2col + GEMM** | Materialise patches, call BLAS | General purpose; the historical default | $K_HK_W\times$ memory blow-up |
| **Implicit GEMM** | GEMM with patch addresses computed from strides in the inner loop | The modern default on GPU | Complex kernel; many tuned variants |
| **FFT** | $x * w = \mathcal{F}^{-1}(\mathcal{F}x \cdot \mathcal{F}w)$ | Large kernels (7×7 and up) | Overhead and workspace dominate for small kernels |
| **Winograd** | Minimal-filtering algorithms trade multiplies for adds | Small kernels (3×3), stride 1 | Reduced numerical accuracy; narrow applicability |

**On FFT.** The convolution theorem turns convolution into pointwise multiplication in the frequency domain, taking an $O(K^2)$ per-output-pixel cost down to $O(\log n)$ amortised. Splendid asymptotics, but the transforms, the padding to compatible sizes, and the complex-valued workspace are large fixed costs. For the 3×3 kernels that dominate modern vision, FFT essentially never wins. It matters for large-kernel and scientific-computing convolutions.

**On Winograd.** The $F(2\times2, 3\times3)$ algorithm computes a 2×2 output tile from a 4×4 input tile using 16 multiplies instead of the direct method's 36 — a 2.25× reduction in multiplications, paid for with extra additions and three fixed transform matrices. It is a genuine workhorse for 3×3 stride-1 layers. The cost is numerical: the transforms have condition numbers greater than 1, so Winograd is measurably less accurate than direct convolution. This is a normal reason for tiny reproducibility differences between machines — it is not a bug.

## 9.2 What cuDNN does

cuDNN ships many algorithm implementations (`CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM`, `..._IMPLICIT_PRECOMP_GEMM`, `..._GEMM`, `..._DIRECT`, `..._FFT`, `..._FFT_TILING`, `..._WINOGRAD`, `..._WINOGRAD_NONFUSED`), each with its own workspace requirement and its own sweet spot in the space of shapes, strides, dtypes and layouts. Each of forward, backward-data and backward-filter is a separate choice.

PyTorch asks cuDNN to pick. You control that with one flag:

```python
torch.backends.cudnn.benchmark = True
```

With `benchmark=True`, the **first** time cuDNN sees a given input configuration it actually times the candidate algorithms and caches the winner.

- **Turn it on** when your input shapes are fixed (typical training with a constant batch size and image size). Expect a slow first iteration or two, then a steady speedup.
- **Turn it off** when shapes vary every batch (variable-length audio, varied image sizes, an odd final batch). Otherwise you re-benchmark constantly and it becomes a large net loss.

Related flags:

```python
torch.backends.cudnn.deterministic = True       # reproducible, usually slower
torch.use_deterministic_algorithms(True)         # stricter; errors on non-deterministic ops

# Numerical precision for fp32 conv/matmul on Ampere and later:
torch.backends.cuda.matmul.allow_tf32 = True     # default False for matmul
torch.backends.cudnn.allow_tf32     = True       # default True for convolutions
```

TF32 keeps fp32's exponent range but truncates the mantissa to 10 bits inside Tensor Cores. It is typically a large speedup and is fine for neural-network training; it is *not* fine for numerics that need full fp32 precision. Note the asymmetric defaults — convolutions use TF32 by default, matmuls do not. That surprises people comparing a conv net against a transformer.

> PyTorch is in the process of consolidating these under a newer `torch.backends.fp32_precision` / `torch.backends.cudnn.conv.fp32_precision` API. Both interfaces exist in 2.14. Check the docs for your exact version rather than trusting any tutorial, including this one, on flag names.

**`torch.compile`** adds another layer: it can fuse the convolution with its surrounding pointwise operations (bias, BatchNorm, activation), which removes intermediate tensors and memory traffic that no choice of convolution algorithm can help with. For new code, try `model = torch.compile(model)` before hand-optimising anything.

## 9.3 What this means for you

Practical priority order when a convolution is slow:

1. Are you accidentally forcing copies? (`.contiguous()` in a loop, mixed memory formats, `reshape` where a view should be possible)
2. Is the model in channels-last + autocast?
3. Is `cudnn.benchmark = True` appropriate for your shape pattern?
4. Have you tried `torch.compile`?
5. Only then think about the algorithm.

Steps 1–4 are configuration. Step 5 is research.

---

# Part X — A debugging playbook

Symptoms you will meet, and what they usually mean.

### "RuntimeError: view size is not compatible with input tensor's size and stride"

The tensor is non-contiguous — almost always after a `transpose` or `permute`. Either `.contiguous().view(...)` (explicit copy) or `.reshape(...)` (implicit copy). Ask yourself whether you can avoid the transpose instead; the copy may be the real cost.

### "CUDA out of memory" on an innocuous line

Look for a hidden copy. `.contiguous()`, `.reshape()` on a non-contiguous tensor, `F.unfold`, `permute().contiguous()` — each doubles peak memory for that tensor. Also remember Part VI: every conv's *input activation* is held until backward. Check with:

```python
print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
print(torch.cuda.max_memory_allocated() / 1e9, "GB peak")
torch.cuda.reset_peak_memory_stats()
```

### Silently wrong results after `as_strided`

Run the bounds check from Section 3.4. Then check that you read `.stride()` *after* any padding or reshaping, not before. Then check `storage_offset`.

### Training works but is 3× slower than a colleague's

In order: memory format mismatches causing transposes; `cudnn.benchmark` off with fixed shapes (or on with varying shapes); autocast not enabled; a Python loop where a batched op would do; `.item()` or `.cpu()` inside the training loop forcing a synchronisation every iteration.

### A custom layer trains, but worse than the built-in equivalent

Your backward pass is probably wrong. Run `gradcheck` in `float64`. A wrong gradient does not raise; it just descends in a slightly incorrect direction. This is the single most expensive class of bug in this field because the failure signal is indistinguishable from "the model is a bit worse."

### Results differ slightly between runs or machines

Expected. Floating-point addition is not associative, cuDNN may select different algorithms (`benchmark=True` makes this explicit), Winograd is less accurate than direct convolution, and atomic accumulations in backward passes have nondeterministic ordering. For bit-reproducibility:

```python
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
```

and accept the slowdown. Note that reproducibility is not guaranteed across PyTorch versions, CUDA versions, or different GPUs, and no flag changes that.

### Your model silently ignores the right/bottom edge of every image

Check the floor in the output-size formula (Section 4.3). If $(H + 2p - K_{\text{eff}})$ is not divisible by $s$, the trailing pixels are never covered by any window.

### `assert t.is_contiguous()` fires after enabling channels-last

Correct behaviour, wrong assertion. Use `t.is_contiguous(memory_format=torch.channels_last)`, or drop the assertion.

### A quick diagnostic to keep around

```python
def describe(t, name="t"):
    print(f"{name:12s} shape={tuple(t.shape)} stride={t.stride()} "
          f"offset={t.storage_offset()} dtype={t.dtype} "
          f"contig={t.is_contiguous()} "
          f"cl={t.is_contiguous(memory_format=torch.channels_last) if t.dim()==4 else '-'} "
          f"storage={t.untyped_storage().nbytes() // t.element_size()} elems")

x = torch.randn(2, 3, 4, 5)
describe(x, "x")
describe(x.permute(0, 2, 3, 1), "x.permute")
describe(x.to(memory_format=torch.channels_last), "x.channels_last")
```

```
x            shape=(2, 3, 4, 5) stride=(60, 20, 5, 1) offset=0 dtype=torch.float32 contig=True  cl=False storage=120 elems
x.permute    shape=(2, 4, 5, 3) stride=(60, 5, 1, 20) offset=0 dtype=torch.float32 contig=False cl=False storage=120 elems
x.channels_last shape=(2, 3, 4, 5) stride=(60, 1, 15, 3) offset=0 dtype=torch.float32 contig=False cl=True storage=120 elems
```

Note the third line carefully: `x.permute(0,2,3,1)` and `x.to(memory_format=channels_last)` have **the same physical layout** but **different logical shapes**. Channels-last keeps the NCHW *shape* and changes the strides; permute changes the shape. That distinction confuses nearly everyone once.

---

# Part XI — Exercises

Attempt each before reading the solution. Difficulty: ⭐ warm-up, ⭐⭐ standard, ⭐⭐⭐ requires real thought.

---

### Exercise 1 ⭐ — Predict the strides

Without running code, give `shape`, `stride` and `is_contiguous` for each:

```python
a = torch.arange(60).reshape(3, 4, 5)
b = a.transpose(0, 2)
c = a[:, 1:3, ::2]
d = a.permute(1, 2, 0)
e = a.unsqueeze(1)
```

<details><summary><b>Solution</b></summary>

- `a`: shape `(3,4,5)`, stride `(20,5,1)`, contiguous.
- `b`: shape `(5,4,3)`, stride `(1,5,20)` — axes 0 and 2 swapped in both tuples. Not contiguous.
- `c`: shape `(3,1,3)`... careful: `1:3` on an axis of length 4 gives 2 elements, and `::2` on length 5 gives 3. So shape `(3,2,3)`, stride `(20,5,2)`, offset $= 1 \times 5 = 5$. Not contiguous.
- `d`: shape `(4,5,3)`, stride `(5,1,20)`. Not contiguous.
- `e`: shape `(3,1,4,5)`, stride `(20,20,5,1)`. **Still contiguous** — inserting a size-1 axis cannot break contiguity, since you never step along it.
</details>

---

### Exercise 2 ⭐ — The indexing formula both ways

A tensor has `shape=(2,3,4)`, `stride=(1,8,2)`, `storage_offset=3`.

(a) Which flat index holds element `[1,2,3]`?
(b) What is the largest flat index this tensor touches? What minimum storage size does it need?
(c) Is it contiguous?

<details><summary><b>Solution</b></summary>

(a) $3 + 1(1) + 2(8) + 3(2) = 3 + 1 + 16 + 6 = 26$.

(b) $3 + (2-1)(1) + (3-1)(8) + (4-1)(2) = 3 + 1 + 16 + 6 = 26$. Storage must hold at least **27** elements.

(c) No. Contiguous strides for shape `(2,3,4)` are `(12,4,1)`.

But there is something interesting here, and it is worth checking rather than eyeballing:

```python
touched = sorted(3 + i0*1 + i1*8 + i2*2
                 for i0 in range(2) for i1 in range(3) for i2 in range(4))
print(len(touched), min(touched), max(touched))
print("gap-free?", touched == list(range(min(touched), max(touched) + 1)))
```
```
24 3 26
gap-free? True
```

The 24 elements fill slots 3 through 26 **with no gaps at all** — the tensor is perfectly dense, it just visits those slots in a non-row-major order. So "dense" and "contiguous" are *not* the same property. `is_contiguous()` asks specifically about row-major order, not about whether the memory is packed. A tensor can be gap-free and still fail `is_contiguous()`, which is exactly the situation a transposed tensor is in.
</details>

---

### Exercise 3 ⭐ — `view` or `reshape`?

For each, will `view` succeed? Will `reshape` copy?

```python
t = torch.randn(6, 4)
t.view(24)
t.t().view(24)
t.view(2, 3, 4)
t[::2].view(12)
t.unsqueeze(0).view(6, 4)
```

<details><summary><b>Solution</b></summary>

1. ✅ view. Contiguous, total size matches.
2. ❌ view fails (stride `(1,6)`; flattening needs an impossible stride). `reshape` copies.
3. ✅ view. Splitting an axis of a contiguous tensor is always view-able.
4. ❌ view fails. `t[::2]` has shape `(3,4)`, stride `(8,1)` — there is a gap of 4 between rows, so consecutive logical elements are not consecutive in memory. `reshape` copies.
5. ✅ view. Adding then removing a size-1 axis is free.

**The general rule:** a `view` is possible iff the requested shape can be obtained by splitting and/or merging *adjacent* axes that are already densely packed relative to each other.
</details>

---

### Exercise 4 ⭐⭐ — Build views with `as_strided`

Using only `torch.arange(12.)` and `as_strided`, construct:

(a) a 3×4 matrix in row-major order;
(b) its transpose, 4×3;
(c) a 3×3 matrix where every row is `[0,1,2]`;
(d) the "Hankel" matrix of 3-element sliding windows starting at 0, 2, 4, 6.

<details><summary><b>Solution</b></summary>

```python
v = torch.arange(12.)
a = v.as_strided((3, 4), (4, 1))    # row-major
b = v.as_strided((4, 3), (1, 4))    # transpose: swap shape and stride
c = v.as_strided((3, 3), (0, 1))    # stride 0 broadcasts the first three elements
d = v.as_strided((4, 3), (2, 1))    # window step 2, within-window step 1
```

Bounds check for (d): $3(2) + 2(1) = 8 < 12$. ✓
</details>

---

### Exercise 5 ⭐⭐ — Bounds checking

You are asked to build windows of length 5, step 3, from a signal of length 20.

(a) How many windows fit?
(b) What `size` and `stride`?
(c) What is the maximum flat index touched? Is it safe?
(d) What if the signal came from `sig = big_tensor[7:27]`?

<details><summary><b>Solution</b></summary>

(a) $\lfloor (20-5)/3 \rfloor + 1 = 5 + 1 = 6$.
(b) `size=(6,5)`, `stride=(3,1)`.
(c) $(6-1)(3) + (5-1)(1) = 15 + 4 = 19 < 20$. Safe, and it uses the final element exactly.
(d) Then `sig.storage_offset()` is 7, so the real flat indices run up to $7 + 19 = 26$, and the check must be against the *full* storage of `big_tensor`, not against `sig.numel()`. Pass `storage_offset=sig.storage_offset()` explicitly and validate `26 < big_tensor.untyped_storage().nbytes()//4`. This is the bug that the "it works on my test tensor but crashes in production" story is usually about.
</details>

---

### Exercise 6 ⭐⭐ — Implement `conv1d` with strides

Write `conv1d_strided(x, w, b, stride, padding, dilation)` for `x` of shape `(N, C, L)` and `w` of shape `(C_out, C, K)`. Verify against `F.conv1d`.

<details><summary><b>Solution</b></summary>

```python
def conv1d_strided(x, w, b=None, stride=1, padding=0, dilation=1):
    if padding:
        x = F.pad(x, (padding, padding))
    N, C, L = x.shape
    Cout, _, K = w.shape
    s, d = stride, dilation
    OL = (L - d * (K - 1) - 1) // s + 1
    sN, sC, sL = x.stride()                 # AFTER padding
    patches = x.as_strided((N, C, OL, K),
                           (sN, sC, sL * s, sL * d))
    out = torch.einsum('ncli,oci->nol', patches, w)
    return out + b.view(1, -1, 1) if b is not None else out

x = torch.randn(2, 3, 20); w = torch.randn(5, 3, 4); b = torch.randn(5)
for s, p, d in [(1,0,1), (2,1,1), (3,2,2)]:
    print(s, p, d, torch.allclose(conv1d_strided(x, w, b, s, p, d),
                                  F.conv1d(x, w, b, stride=s, padding=p, dilation=d), atol=1e-4))
```
```
1 0 1 True
2 1 1 True
3 2 2 True
```
The 2-D case collapses to this by deleting one spatial axis everywhere. If you can write this from scratch, you understand Part V.
</details>

---

### Exercise 7 ⭐⭐ — Shapes through a network

Input `(16, 3, 64, 64)`. Give the shape after each layer:

```
Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
MaxPool2d(2)
Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
AdaptiveAvgPool2d(1)
```

<details><summary><b>Solution</b></summary>

1. $(64 + 2 - 3)/1 + 1 = 64$ → `(16, 32, 64, 64)` (classic "same" 3×3)
2. `(16, 32, 32, 32)`
3. $\lfloor(32 + 2 - 3)/2\rfloor + 1 = 15 + 1 = 16$ → `(16, 64, 16, 16)`
4. $K_{\text{eff}} = 2(2)+1 = 5$; $(16 + 4 - 5)/1 + 1 = 16$ → `(16, 64, 16, 16)` (dilated but size-preserving)
5. `(16, 64, 1, 1)`
</details>

---

### Exercise 8 ⭐⭐ — Max pooling with strides

Implement 2-D max pooling using `as_strided`. Verify against `F.max_pool2d`.

<details><summary><b>Solution</b></summary>

```python
def maxpool2d_strided(x, k, stride=None):
    stride = stride or k                      # PyTorch default: stride == kernel_size
    N, C, H, W = x.shape
    OH = (H - k) // stride + 1
    OW = (W - k) // stride + 1
    sN, sC, sH, sW = x.stride()
    patches = x.as_strided((N, C, OH, OW, k, k),
                           (sN, sC, sH * stride, sW * stride, sH, sW))
    return patches.amax(dim=(-1, -2))

x = torch.randn(2, 3, 8, 8)
print(torch.allclose(maxpool2d_strided(x, 2),    F.max_pool2d(x, 2)))
print(torch.allclose(maxpool2d_strided(x, 3, 2), F.max_pool2d(x, 3, 2)))
```
```
True
True
```

**The point:** the *patch-extraction* geometry is identical for convolution and pooling. Only the reduction differs — `einsum` (weighted sum) versus `amax`. Average pooling would be `.mean(dim=(-1,-2))`. Once you see patch extraction as a separate concern, this whole family of layers collapses into one idea.
</details>

---

### Exercise 9 ⭐⭐ — Receptive field

Write a function computing the receptive field of a stack of layers, and apply it to (a) three 3×3 stride-1 convs, (b) ResNet's stem: 7×7 stride 2 then 3×3 max-pool stride 2, (c) three 3×3 convs with dilations 1, 2, 4.

<details><summary><b>Solution</b></summary>

Track the receptive field $r$ and the cumulative jump $j$ (how many input pixels one step in the current feature map corresponds to). Initialise $r=1$, $j=1$; for each layer with effective kernel $k_{\text{eff}} = d(k-1)+1$ and stride $s$:

$$r \leftarrow r + (k_{\text{eff}} - 1)\,j, \qquad j \leftarrow j \cdot s$$

```python
def receptive_field(layers):          # layers: list of (kernel, stride, dilation)
    r, j = 1, 1
    for k, s, d in layers:
        r += (d * (k - 1)) * j
        j *= s
    return r, j

print(receptive_field([(3,1,1)] * 3))              # (7, 1)
print(receptive_field([(7,2,1), (3,2,1)]))         # (11, 4)
print(receptive_field([(3,1,1), (3,1,2), (3,1,4)]))# (15, 1)
```
```
(7, 1)
(11, 4)
(15, 1)
```

Three 3×3 convs see 7×7 — the standard argument for replacing a 7×7 conv with three 3×3s (same receptive field, $3 \times 9 = 27$ weights per channel pair instead of 49, plus two extra nonlinearities). And case (c) shows the dilation trick: 15×15 receptive field, stride 1 throughout, so no resolution is lost — this is why dilated convolutions dominate semantic segmentation.
</details>

---

### Exercise 10 ⭐⭐⭐ — Grouped convolution with strides

Extend `conv2d_strided` to support `groups`. Verify against `F.conv2d(..., groups=G)`.

<details><summary><b>Solution</b></summary>

The patch extraction is unchanged — the geometry does not care about groups. What changes is the contraction: split the channel axis into (group, channel-within-group) and make `g` a *batched* index in the einsum rather than a summed one.

```python
def conv2d_grouped_strided(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
    if padding:
        x = F.pad(x, (padding,) * 4)
    N, C, H, W = x.shape
    Cout, Cg, KH, KW = w.shape            # Cg = C // groups
    s, d, G = stride, dilation, groups
    OH = (H - d*(KH-1) - 1)//s + 1
    OW = (W - d*(KW-1) - 1)//s + 1
    sN, sC, sH, sW = x.stride()

    patches = x.as_strided((N, C, OH, OW, KH, KW),
                           (sN, sC, sH*s, sW*s, sH*d, sW*d))
    patches = patches.view(N, G, Cg, OH, OW, KH, KW)   # split C -> (G, Cg): a free view
    wg      = w.view(G, Cout // G, Cg, KH, KW)

    out = torch.einsum('ngchwij,gocij->ngohw', patches, wg).reshape(N, Cout, OH, OW)
    return out + b.view(1, -1, 1, 1) if b is not None else out

x = torch.randn(2, 8, 10, 10); w = torch.randn(6, 4, 3, 3); b = torch.randn(6)
for s, p, d in [(1,0,1), (2,1,1), (1,2,2)]:
    print(s, p, d, torch.allclose(conv2d_grouped_strided(x, w, b, s, p, d, groups=2),
                                  F.conv2d(x, w, b, stride=s, padding=p, dilation=d, groups=2),
                                  atol=1e-4))
```
```
1 0 1 True
2 1 1 True
1 2 2 True
```

Two things worth noticing. First, `patches.view(N, G, Cg, ...)` **is a legal view even though `patches` is wildly non-contiguous** — splitting a single axis into two is always expressible as a stride change (the new strides are `sC*Cg` and `sC`). Second, `g` appears in both inputs *and* the output, which is einsum's notation for "batch over this index rather than summing it." That single letter is the entire semantics of grouping.
</details>

---

### Exercise 11 ⭐⭐⭐ — Derive the 1-D gradients

For $y[n,o,l] = \sum_{c,i} x[n,c,\,sl+di]\,w[o,c,i] + b[o]$, derive $\partial L/\partial w$, $\partial L/\partial b$, $\partial L/\partial x$ by hand, given $g = \partial L / \partial y$.

<details><summary><b>Solution</b></summary>

**Bias.** $b[o]$ was added to every $(n,l)$:
$$\frac{\partial L}{\partial b[o]} = \sum_{n,l} g[n,o,l]$$

**Weights.** $w[o,c,i]$ multiplies $x[n,c,sl+di]$ for every $(n,l)$:
$$\frac{\partial L}{\partial w[o,c,i]} = \sum_{n,l} g[n,o,l]\; x[n,c,\,sl+di]$$

**Input.** $x[n,c,q]$ influenced $y[n,o,l]$ whenever $sl+di=q$:
$$\frac{\partial L}{\partial x[n,c,q]} = \sum_{o}\ \sum_{\substack{l,i\\ sl+di=q}} g[n,o,l]\; w[o,c,i]$$

For $s=d=1$ this simplifies to $\sum_{o,i} g[n,o,q-i]\,w[o,c,i]$ — a genuine convolution (note the minus), i.e. a full cross-correlation with the flipped kernel and channels transposed. Exactly the 2-D result with one spatial axis removed, which is the useful sanity check: **the spatial axes are independent of the derivation's structure.**
</details>

---

### Exercise 12 ⭐⭐⭐ — Implement and verify `Conv1dFunction`

Turn Exercise 11 into a `torch.autograd.Function` with a hand-written `backward`, and pass `gradcheck`.

<details><summary><b>Solution</b></summary>

```python
class Conv1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, stride, padding, dilation):
        ctx.save_for_backward(x, w)
        ctx.conf = (stride, padding, dilation)
        xp = F.pad(x, (padding, padding)) if padding else x
        N, C, L = xp.shape
        Cout, _, K = w.shape
        s, d = stride, dilation
        OL = (L - d*(K-1) - 1)//s + 1
        sN, sC, sL = xp.stride()
        patches = xp.as_strided((N, C, OL, K), (sN, sC, sL*s, sL*d))
        return torch.einsum('ncli,oci->nol', patches, w) + b.view(1, -1, 1)

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        s, p, d = ctx.conf
        g = g.contiguous()
        N, C, L = x.shape
        Cout, _, K = w.shape
        OL = g.shape[-1]

        xp = F.pad(x, (p, p)) if p else x
        sN, sC, sL = xp.stride()
        patches = xp.as_strided((N, C, OL, K), (sN, sC, sL*s, sL*d))

        grad_b = g.sum(dim=(0, 2))
        grad_w = torch.einsum('ncli,nol->oci', patches, g)
        op = (L + 2*p - d*(K-1) - 1) % s
        grad_x = F.conv_transpose1d(g, w, stride=s, padding=p,
                                    dilation=d, output_padding=op)
        return grad_x, grad_w, grad_b, None, None, None

for s, p, d in [(1,0,1), (2,1,1), (2,2,2)]:
    x = torch.randn(2, 3, 12, dtype=torch.double, requires_grad=True)
    w = torch.randn(4, 3, 3,  dtype=torch.double, requires_grad=True)
    b = torch.randn(4,        dtype=torch.double, requires_grad=True)
    print(s, p, d, torch.autograd.gradcheck(Conv1dFunction.apply, (x, w, b, s, p, d)))
```
```
1 0 1 True
2 1 1 True
2 2 2 True
```

Compare `'ncli,oci->nol'` (forward) with `'ncli,nol->oci'` (weight gradient): the second operand and the output have swapped. That mechanical swap is the practical content of "differentiate a bilinear form with respect to one argument."
</details>

---

### Exercise 13 ⭐⭐ — Memory arithmetic

You train with input `(32, 3, 224, 224)` in float32. The first layer is `Conv2d(3, 64, 7, stride=2, padding=3)`.

(a) Input tensor size in MB?
(b) Output tensor size in MB?
(c) If this layer used explicit im2col, how large is the patch matrix?
(d) Why does `F.conv2d` not actually allocate (c)?

<details><summary><b>Solution</b></summary>

(a) $32 \times 3 \times 224 \times 224 \times 4 = 19{,}267{,}584$ B ≈ **19.3 MB**.
(b) Output is `(32, 64, 112, 112)`: $32 \times 64 \times 112 \times 112 \times 4 = 102{,}760{,}448$ B ≈ **102.8 MB**.
(c) Patch matrix: $(32 \cdot 112 \cdot 112) \times (3 \cdot 7 \cdot 7) = 401{,}408 \times 147 = 59{,}006{,}976$ elements ≈ **236 MB** — over 12× the input. (After padding, each input element appears in up to 49 windows, but the stride of 2 means only about a quarter of window positions are used, giving ~12× rather than ~49×.)
(d) Implicit GEMM (Section 5.5): the kernel computes patch addresses from strides while loading tiles into shared memory, so the expanded matrix never exists in global memory. The strided *view* is free; only materialising it costs.
</details>

---

### Exercise 14 ⭐ — Channels-last strides

`t = torch.randn(4, 16, 32, 32)`. Give `t.stride()` and `t.to(memory_format=torch.channels_last).stride()`. Does the shape change? Does `t.is_contiguous()` change?

<details><summary><b>Solution</b></summary>

- NCHW: `(16384, 1024, 32, 1)` — suffix products of `(4,16,32,32)`.
- channels-last: `(16384, 1, 512, 16)`. Channel stride 1; width stride $= C = 16$; height stride $= W \cdot C = 512$; batch stride unchanged at $16\cdot32\cdot32 = 16384$.
- **Shape is identical**, `(4, 16, 32, 32)`. Only the layout changed.
- `is_contiguous()` returns `False` for the channels-last tensor, because the unqualified form asks about NCHW row-major order. Use `is_contiguous(memory_format=torch.channels_last)`, which returns `True`.
</details>

---

### Exercise 15 ⭐⭐⭐ — Find the bug

This code produces wrong answers. Find all three bugs.

```python
def conv2d_buggy(x, w, stride=1, padding=0):
    N, C, H, W = x.shape
    Cout, _, KH, KW = w.shape
    sN, sC, sH, sW = x.stride()

    x = F.pad(x, (padding,) * 4)

    OH = (H - KH) // stride + 1
    OW = (W - KW) // stride + 1

    patches = x.as_strided((N, C, OH, OW, KH, KW),
                           (sN, sC, sH, sW, sH, sW))
    return torch.einsum('nchwij,ocij->nohw', patches, w)
```

<details><summary><b>Solution</b></summary>

**Bug 1 — strides read before padding.** `F.pad` allocates a new tensor whose row stride is $W + 2p$, not $W$. Every window then reads from the wrong row. Move the `x.stride()` call *after* the `F.pad`.

**Bug 2 — output size ignores padding.** Should be `(H + 2*padding - KH)//stride + 1`. Using the unpadded `H` makes the output too small, so the last windows are never computed. Worse, because bug 1 made the strides too small, the view happens to stay in bounds and you get silently wrong numbers instead of a crash.

**Bug 3 — convolution stride never applied.** The window-position strides must be `sH * stride` and `sW * stride`. As written, every configuration behaves as if `stride=1`, so the output has the right shape but the wrong contents whenever `stride > 1`.

Corrected:

```python
def conv2d_fixed(x, w, stride=1, padding=0):
    if padding:
        x = F.pad(x, (padding,) * 4)
    N, C, H, W = x.shape                 # shape AFTER padding
    Cout, _, KH, KW = w.shape
    sN, sC, sH, sW = x.stride()          # strides AFTER padding
    OH = (H - KH) // stride + 1
    OW = (W - KW) // stride + 1
    patches = x.as_strided((N, C, OH, OW, KH, KW),
                           (sN, sC, sH * stride, sW * stride, sH, sW))
    return torch.einsum('nchwij,ocij->nohw', patches, w)
```

**The meta-lesson:** all three bugs produce plausible-looking numbers rather than errors. `as_strided` has no type system protecting you. This is exactly why Section 4.4 insists on keeping a naive reference implementation and diffing against it.
</details>

---

### Exercise 16 ⭐⭐ — Why does `reshape` copy here?

```python
x = torch.randn(4, 8, 16, 16)
p = x.permute(0, 2, 3, 1)
q = p.reshape(4, -1)
print(q.data_ptr() == x.data_ptr())     # False
```

Explain from strides why no view exists. Then give a case where `permute` followed by `reshape` *is* free.

<details><summary><b>Solution</b></summary>

`p` has shape `(4,16,16,8)` and stride `(2048, 16, 1, 256)`. Flattening axes 1–3 into one axis of size 2048 would require every consecutive pair of logical elements to be a *constant* number of positions apart. But moving along the last axis (channels) costs 256, while moving along the axis before it costs 1. The address sequence is `0, 256, 512, ..., 1792, 1, 257, ...` — not an arithmetic progression. No single stride describes it, so a copy is required.

**A free case:** merging axes whose strides are already "nested" correctly. For example `x.permute(0,1,2,3)` (a no-op) then `.reshape(4, 8, 256)` merges H and W, which have strides 16 and 1 with $16 = 16 \times 1$ — the exact nesting a view needs. The general condition for merging axes $k$ and $k+1$ into one is $\text{stride}_k = \text{stride}_{k+1} \times \text{shape}_{k+1}$.
</details>

---

### Exercise 17 ⭐⭐ — Algorithm selection

For each layer, say which algorithm family you would expect cuDNN to choose, and why.

(a) `Conv2d(512, 512, 3, padding=1)` on a 14×14 feature map, fp16
(b) `Conv2d(3, 64, 11, stride=4)` on 224×224, fp32
(c) `Conv2d(256, 256, 3, groups=256)` (depthwise) on 56×56
(d) `Conv2d(64, 64, 1)` on 56×56

<details><summary><b>Solution</b></summary>

(a) **Winograd** or implicit GEMM with Tensor Cores. 3×3 stride-1 is Winograd's design point, the channel count is large so the GEMM is nicely shaped, and fp16 puts Tensor Cores in play. This is the hottest case in modern CNNs and the most heavily tuned.

(b) **Implicit GEMM or direct — not FFT**, despite the large kernel. The kernel size alone tempts you toward FFT, but the *stride of 4* rules it out. FFT-based convolution computes the result at **every** output position; with stride 4 in two dimensions you would then discard 15 out of every 16 values you paid to compute. The frequency-domain approach has no way to skip positions, so its advantage evaporates exactly when the stride grows. (This is a well-documented practical observation: for large strides the standard cuDNN path is competitive or faster, because FFT is only interesting when you want the convolution at all positions.) FFT convolution earns its place with **large kernels at stride 1** — long FIR filters in audio, some scientific computing — not with aggressively strided vision stems.

(c) **Direct**, specialised. Depthwise has $C_{in}/G = 1$, so the GEMM would be degenerate (a matrix multiply with inner dimension 9). There is no arithmetic intensity to exploit and the layer is memory-bandwidth bound. Libraries ship hand-written depthwise kernels for exactly this reason.

(d) A 1×1 convolution **is** a matrix multiply already — no patch extraction is needed, since each "patch" is a single pixel. Expect a straight GEMM over the $(N \cdot H \cdot W) \times C_{in}$ matrix. This is why 1×1 convolutions are so efficient and why bottleneck blocks use them freely.
</details>

---

### Exercise 18 ⭐⭐⭐ — A production scenario

You inherit a segmentation model. Training takes 4.5 hours/epoch on an A100. You observe:

- GPU utilisation oscillates between 30% and 95%
- `nvidia-smi` shows 38 GB of 80 GB used
- The model uses `nn.Upsample` + `Conv2d` in the decoder
- Input images are resized to varying sizes between 512 and 1024
- The code contains `x = x.permute(0, 2, 3, 1).contiguous()` inside the forward pass

List what you would investigate, in order, with reasoning.

<details><summary><b>Solution</b></summary>

**1. The oscillating utilisation — data loading, almost certainly.** 30%↔95% swings are the classic signature of a GPU starving while waiting for the input pipeline. Check `DataLoader(num_workers=...)`, `pin_memory=True`, `persistent_workers=True`, and whether the variable-size resize is being done on CPU per-image. **No amount of convolution tuning fixes a dataloader bottleneck**, so this comes first.

**2. The `permute(...).contiguous()` in the forward pass.** This is a full copy of an activation on every iteration, forward *and* (implicitly) backward. Ask why it is there. If it is feeding a channels-last-shaped op, switching the whole model to `memory_format=torch.channels_last` removes it and speeds up the convolutions at the same time. If it is feeding a linear layer, `einsum` or `flatten` on the right axes may avoid it.

**3. Varying input sizes vs `cudnn.benchmark`.** If `benchmark=True` with sizes ranging 512–1024, cuDNN re-benchmarks on every new shape — potentially a large net loss. Either bucket the sizes into a handful of fixed values (best: keeps `benchmark` useful) or set `benchmark=False`. Varying shapes also defeat `torch.compile`'s caching unless you use `dynamic=True`.

**4. Memory headroom.** 38/80 GB means you can roughly double the batch size, improving GPU efficiency and reducing per-epoch overhead — *provided* step 1 is fixed, otherwise you will just starve harder.

**5. Mixed precision.** If not already on, `torch.autocast('cuda', dtype=torch.bfloat16)` on an A100 is typically the single biggest win, and it is what makes channels-last pay off.

**6. The decoder.** `Upsample` + `Conv2d` is the *correct* choice for avoiding checkerboard artifacts (Section 7.3), so do not "fix" it. But check the interpolation mode and whether the convolution runs at full resolution when it could run before the upsample.

**7. Only now** look at the convolution algorithms themselves.

**The general principle:** work outward from the data pipeline, then memory layout, then precision, then configuration, and only then kernels. Most practitioners do this in exactly the reverse order and spend a week getting 5%.
</details>

---

# Part XII — Reference card

## The one formula

$$\text{flat\_index} = \text{storage\_offset} + \sum_k i_k \cdot \text{stride}_k$$

## Stride recipes

| Operation | What it does to metadata | Copies? |
|---|---|---|
| `reshape`/`view` (contiguous) | recompute suffix-product strides | no |
| `transpose(a,b)` / `permute` | permute shape and stride together | no |
| `t[a:b:c]` | offset `+= a*stride`; `stride *= c` | no |
| `expand` | set stride `0` on broadcast axes | no |
| `unsqueeze` | insert a size-1 axis | no |
| `unfold(dim,size,step)` | append window axis; window stride `= step*stride` | no |
| `as_strided` | set everything manually | no |
| `contiguous()` | rewrite buffer in row-major order | yes (if needed) |
| `repeat`, `flip`, `F.pad`, `F.unfold` | | **yes** |
| fancy indexing `x[idx]`, `x[mask]` | | **yes** |
| `reshape` (non-contiguous) | | **yes** |

## Convolution geometry

$$H_{out} = \left\lfloor \frac{H + 2p - d(K_H-1) - 1}{s}\right\rfloor + 1, \qquad K_{\text{eff}} = d(K_H-1)+1$$

Patch view for input strides $(s_N,s_C,s_H,s_W)$:

```
size   = (N,  C,  OH,     OW,     KH,     KW)
stride = (sN, sC, sH * s, sW * s, sH * d, sW * d)
                   ^^^^^  ^^^^^   ^^^^^   ^^^^^
                   conv stride    dilation
```

## Convolution gradients

| Gradient | Formula | PyTorch expression |
|---|---|---|
| $\partial L/\partial b$ | $\sum_{n,h,v} g$ | `g.sum(dim=(0,2,3))` |
| $\partial L/\partial w$ | $\sum_{n,h,v} g \cdot x_{\text{patch}}$ | `einsum('nchwij,nohw->ocij', patches, g)` |
| $\partial L/\partial x$ | adjoint / scatter-add | `F.conv_transpose2d(g, w, stride=s, padding=p, dilation=d, output_padding=...)` |

$$\text{output\_padding} = \big(H_{in} + 2p - d(K_H-1) - 1\big) \bmod s$$

Or skip all three: **`torch.nn.grad.conv2d_input(x.shape, w, g, ...)`** and **`torch.nn.grad.conv2d_weight(x, w.shape, g, ...)`** ship with PyTorch and compute the first two for you.

## View legality

- **Split** an axis: always legal.
- **Merge** adjacent axes $k, k{+}1$: legal iff $\ \text{stride}_k = \text{stride}_{k+1} \times \text{shape}_{k+1}$.

## Dualities worth memorising

| Forward | Backward |
|---|---|
| broadcast (stride 0) | sum |
| gather overlapping patches (`unfold`) | scatter-add (`fold`) — verified in §6.4 |
| stride $s$ (read every $s$-th) | dilate the gradient by $s$ |
| cross-correlation with $w$ | full convolution with flipped, channel-transposed $w$ |
| convolution | transposed convolution (the **adjoint**, not the inverse) |

## Performance checklist

```python
model = model.to(memory_format=torch.channels_last)   # NHWC layout
x     = x.to(memory_format=torch.channels_last)
torch.backends.cudnn.benchmark = True                  # ONLY if shapes are fixed
with torch.autocast('cuda', dtype=torch.bfloat16):     # Tensor Cores
    ...
model = torch.compile(model)                           # fuse conv + pointwise ops
nn.Conv2d(..., bias=False)                             # when followed by BatchNorm
```

## Correctness checklist for custom ops

1. Write a naive reference implementation first. Keep it forever.
2. Test against it with `torch.allclose`, never `==`.
3. Bounds-check every `as_strided`: `offset + Σ(size-1)·stride < storage.numel()`.
4. Read `.stride()` **after** any `pad`/`reshape`, never before.
5. `gradcheck` in `float64` for any hand-written `backward` — and `gradgradcheck` if anyone might use `create_graph=True`.
6. Train a real (tiny) model with your layer beside the built-in one and compare loss curves. `gradcheck` passing is not the same as the layer working.
7. Prefer expressing the forward pass in differentiable primitives and letting autograd do the rest.

---

# Where to go next

**Primary sources, in rough order of usefulness:**

- The PyTorch documentation for `torch.Tensor.stride`, `torch.as_strided`, `torch.nn.functional.unfold` / `fold`, `torch.nn.Conv2d`, and `torch.autograd.Function`. Read the notes sections, not just the signatures.
- The PyTorch *Channels Last Memory Format* tutorial, for the performance side of Part VIII.
- The PyTorch *Reproducibility* note, for the determinism flags in Part IX.
- The cuDNN Developer Guide, for the algorithm list and workspace behaviour.

**Papers worth reading once you are comfortable here:**

- Lavin & Gray, *Fast Algorithms for Convolutional Neural Networks* (2016) — the Winograd method.
- Chetlur et al., *cuDNN: Efficient Primitives for Deep Learning* (2014) — implicit GEMM, from the people who built it.
- Odena, Dumoulin & Olah, *Deconvolution and Checkerboard Artifacts* (Distill, 2016) — the clearest treatment of the transposed-convolution artifact problem.
- Dumoulin & Visin, *A Guide to Convolution Arithmetic for Deep Learning* (2016) — every shape formula, with animations.

**Things to build, in increasing order of difficulty:**

1. A strided implementation of average pooling and of `nn.Fold` from scratch.
2. A `Conv3d` using `as_strided` (six-to-seven-dimensional views; same table, one more row).
3. A custom autograd Function that saves memory by recomputing the patch view in `backward` instead of saving activations — then measure whether it actually helps.
4. A Triton or CUDA kernel for a fused conv + bias + ReLU, and a benchmark against `torch.compile`'s version.

---

## A closing thought

Almost every "advanced" idea in this tutorial reduces to one observation: **shape and layout are separate things.** Shape is what your code means. Stride is where the bytes are. Keeping them separate is what lets a transpose be free, a broadcast cost nothing, a sliding window be pure metadata, and a layout change be a pure performance decision that no other line of code needs to know about.

Once you see a convolution as *a strided view followed by a matrix multiply*, its backward pass as *the adjoint of that view*, and its performance as *a question about memory traffic rather than arithmetic*, the whole subject stops being a collection of formulas and becomes one idea applied repeatedly.

That is a good position to be in when you meet the operation that does not exist in any library yet — which, in industry, will happen sooner than you expect.
