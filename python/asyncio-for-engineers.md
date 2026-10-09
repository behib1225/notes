# Asyncio for Engineers

### A self-contained, hands-on tutorial — from first principles to production agentic AI

**Target audience:** undergraduate engineering students who already know Python (functions, classes, exceptions, generators at least by name, `with` blocks, list comprehensions).

**Python version:** 3.12. Every feature used here exists in 3.12. Where something is newer (3.13+) or older, it is flagged explicitly.

**Dependencies:** none. Everything runs on a stock CPython install. Real network and real LLM latency are simulated with `asyncio.sleep()`, which behaves *exactly* like real I/O from the event loop's point of view. This is deliberate: you learn the concurrency without fighting API keys, rate limits, or flaky WiFi.

**How to use this (self-study guide):**

1. **Read a chapter, then type the code.** Don't copy-paste — typing forces you to notice syntax you'd otherwise skim.
2. **Run it.** Every example in this tutorial was executed on Python 3.12.3 before publication, and several of the stated outputs corrected as a result. Yours should match. If they don't, that's a finding worth chasing.
3. **Attempt every exercise before reading on.** Solutions are printed inline, immediately after each problem, so cover the screen below the question. The harder exercises (from Chapter 11 onward) carry a `*Hint:*` line between the question and the solution — use that rung before you take the answer.
4. **Check yourself at the end of each Part** against the "you should now be able to" list.

**Pacing.** As a rough guide for self-study: Part I in one sitting, Parts II and III one chapter per sitting, Part IV in one, Part V spread over several. The capstone (Exercise 47) is a weekend project and is explicitly staged into four milestones.

**Accuracy note.** Where this tutorial states a measurement — timings, memory, GIL behaviour — it is a real number from a real run, not an estimate, and the machine it came from is named. Where a number would vary by hardware, that is said explicitly. Treat any tutorial that gives you benchmark figures without saying what produced them with suspicion, including this one if it slips.

---

## Table of Contents

**Part I — Foundations**
1. [Why async exists: the problem it solves](#1-why-async-exists-the-problem-it-solves)
2. [Coroutines: functions you can pause](#2-coroutines-functions-you-can-pause)
3. [The event loop: the thing that actually runs everything](#3-the-event-loop-the-thing-that-actually-runs-everything)

**Part II — Real concurrency**
4. [Tasks: how you actually get things running at the same time](#4-tasks-how-you-actually-get-things-running-at-the-same-time)
5. [Running many things: gather, TaskGroup, as_completed, wait](#5-running-many-things-gather-taskgroup-as_completed-wait)
6. [Timeouts and cancellation](#6-timeouts-and-cancellation)
7. [Errors: ExceptionGroup and `except*`](#7-errors-exceptiongroup-and-except)

**Part III — Coordination**
8. [Synchronization primitives: Lock, Semaphore, Event](#8-synchronization-primitives-lock-semaphore-event)
9. [Queues, worker pools, and pipelines](#9-queues-worker-pools-and-pipelines)
10. [Blocking code, threads, and the async boundary](#10-blocking-code-threads-and-the-async-boundary)

**Part IV — The rest of the language surface**
11. [`async with`, `async for`, and async generators](#11-async-with-async-for-and-async-generators)
12. [Networking with asyncio streams](#12-networking-with-asyncio-streams)
13. [Testing, debugging, and production hygiene](#13-testing-debugging-and-production-hygiene)
14. [The anti-pattern catalogue](#14-the-anti-pattern-catalogue)

**Part V — Agentic AI**
15. [Async patterns for agentic AI](#15-async-patterns-for-agentic-ai)
16. [Capstone project](#16-capstone-project)
17. [Quick reference card](#17-quick-reference-card)
18. [Glossary](#18-glossary)

47 exercises, all with worked solutions.

---

# Part I — Foundations

## 1. Why async exists: the problem it solves

### 1.1 Your program spends most of its life waiting

Open a terminal and time this:

```python
# file: blocking.py
import time

def fetch(name: str, seconds: float) -> str:
    print(f"  start  {name}")
    time.sleep(seconds)          # pretend this is a network request
    print(f"  finish {name}")
    return f"{name}-result"

def main() -> None:
    t0 = time.perf_counter()
    results = [fetch("A", 2), fetch("B", 2), fetch("C", 2)]
    print(results)
    print(f"elapsed: {time.perf_counter() - t0:.2f}s")

main()
```

Output:

```
  start  A
  finish A
  start  B
  finish B
  start  C
  finish C
['A-result', 'B-result', 'C-result']
elapsed: 6.01s
```

Six seconds. But here is the uncomfortable question: **what was your CPU doing for those six seconds?**

Essentially nothing. `time.sleep(2)` is standing in for a network request. During a real network request, your process sends a few hundred bytes to a socket and then sits there, completely idle, while electrons travel to a data centre in Virginia and back. Your 3 GHz processor, capable of billions of operations per second, executed roughly zero useful instructions for six seconds.

That's the waste asyncio is designed to eliminate.

### 1.2 I/O-bound vs CPU-bound — the single most important distinction

Before you write a line of async code, you must classify your workload.

| | **I/O-bound** | **CPU-bound** |
|---|---|---|
| What limits it | Waiting for something external | Raw computation |
| Examples | HTTP requests, LLM API calls, database queries, reading files, talking to a sensor over serial | Matrix multiplication, image filters, simulations, cryptography, parsing a 2 GB file |
| CPU usage while running | ~0% | 100% of one core |
| Right tool | **asyncio** (or threads) | **multiprocessing** (or NumPy/C extensions) |

**Asyncio makes I/O-bound programs faster. It makes CPU-bound programs slightly slower.** If you take nothing else from this tutorial, take that sentence.

Why does async help I/O but not CPU? Because async's entire trick is: *"while this task is waiting, let another task run."* If a task never waits — if it's just crunching numbers — there is no idle time to reclaim, and you've added overhead for nothing.

Agentic AI systems are almost perfectly I/O-bound. An agent making an LLM call waits 2–30 seconds for a response while doing essentially no work. This is why every serious agent framework is built on asyncio.

### 1.3 Concurrency is not parallelism

These two words get used interchangeably in casual speech, and it causes enormous confusion. They are different.

- **Parallelism** = multiple things happening *at the same instant*, on different CPU cores. Four cores, four computations, genuinely simultaneous.
- **Concurrency** = multiple things *in progress* over the same period, but not necessarily executing at the same instant. One worker, rapidly switching between several jobs, always working on whichever one is ready.

The classic analogy: **one chef, several dishes.**

A bad chef cooks serially: put the pasta on, stand and watch the pot for 10 minutes, then start chopping onions, then start the sauce. Total: 25 minutes.

A good chef is *concurrent*: put the pasta on, and while it boils, chop onions; while the onions sweat, grate the cheese; check the pasta when the timer goes. Total: 12 minutes. There is still only **one chef**. Nothing happened in parallel. The chef simply never stood idle.

Asyncio makes your program the good chef. It does **not** hire more chefs — that's `multiprocessing`.

```
SEQUENTIAL (blocking)
A: [====work====][~~~~~~~~ waiting ~~~~~~~~][==done==]
B:                                                    [====work====][~~~ waiting ~~~][==done==]
                                            time ───────────────────────────────────────────▶

CONCURRENT (asyncio, single thread)
A: [==work==][~~~~~~~~~~ waiting ~~~~~~~~~~][==done==]
B:          [==work==][~~~~~~ waiting ~~~~~~][==done==]
     ▲ CPU switches to B the moment A starts waiting
```

### 1.4 Cooperative multitasking: the deal you're making

Operating-system threads are **preemptively** scheduled. The OS can suspend a thread mid-instruction, at any moment, without asking. That's powerful but it means any shared variable can change under your feet at any time, which is why threaded code is riddled with subtle race conditions.

Asyncio is **cooperatively** scheduled. A task runs until it *voluntarily* yields control, which it does only at an `await` that actually needs to wait. Between two `await`s, your code is atomic — nothing else in the event loop can run. This makes async code dramatically easier to reason about than threaded code.

The price of that guarantee: **cooperation is mandatory**. If one task hogs the CPU without awaiting, *everything else in your program freezes.* There is no OS to rescue you. A single `time.sleep(5)` or a heavy `for` loop in the wrong place will stall your entire server. Chapter 10 is devoted to this.

> **Mental model to carry forward:** asyncio is a single-threaded scheduler running a to-do list. `await` means "I'm stuck, put me down and pick up something else; wake me when my thing is ready."

---

### Exercise 1 — Classification

Label each workload **I/O-bound** or **CPU-bound**, and say whether asyncio would help.

1. Downloading 500 product pages from an e-commerce site.
2. Computing the FFT of a 10-million-sample signal.
3. An agent that sends 12 prompts to an LLM API and summarizes the answers.
4. Resizing 2,000 JPEGs.
5. A web server that reads a row from Postgres and returns JSON.
6. Training a small neural network on CPU.
7. Polling 40 IoT temperature sensors over HTTP every 5 seconds.

**Solution**

1. **I/O-bound.** Huge win from asyncio — 500 sequential downloads at 200 ms each is 100 s; concurrently it's a few seconds.
2. **CPU-bound.** Asyncio does nothing. Use NumPy/SciPy (which release the GIL internally) or `multiprocessing`.
3. **I/O-bound.** The canonical asyncio use case. 12 × 4 s sequential = 48 s; concurrent ≈ 4 s.
4. **CPU-bound.** Use `ProcessPoolExecutor`. (Though the *disk reads* are I/O — in practice the decode/resize dominates.)
5. **I/O-bound.** Asyncio lets one process handle thousands of simultaneous connections instead of one-thread-per-request.
6. **CPU-bound.** Asyncio is the wrong tool entirely.
7. **I/O-bound.** Perfect fit, and a good example of *periodic* concurrency (Chapter 9).

The heuristic: **if the bottleneck is something outside your process, asyncio helps.**

---

### Exercise 2 — Predict the timings

A chef must: boil pasta (10 min, unattended), chop onions (4 min, hands-on), simmer sauce (8 min, unattended), grate cheese (2 min, hands-on).

(a) Sequential time? (b) Best concurrent time with one chef? (c) What's the hard lower bound, and why can't concurrency beat it?

**Solution**

(a) 10 + 4 + 8 + 2 = **24 minutes**.

(b) Start the pasta (t=0). While it boils, chop onions (t=0→4), start the sauce simmering (t=4→12), grate cheese during the simmer (t=4→6). Pasta finishes at t=10, sauce at t=12. **12 minutes.**

(c) The lower bound is the longest *unavoidable* chain of work. Here the sauce can't start until the onions are chopped, so 4 + 8 = 12 minutes. Concurrency removes *idle* time; it cannot remove *dependency* time or *hands-on* time. This is exactly the situation with `await` chains in an agent: if step 3 needs step 2's output, no amount of asyncio compresses that chain. Only genuinely independent work can overlap.

---

## 2. Coroutines: functions you can pause

### 2.1 `async def` makes a different kind of function

```python
def normal():
    return 42

async def coro():
    return 42

print(normal())   # 42
print(coro())     # <coroutine object coro at 0x7f...>
```

Calling `normal()` runs its body and returns 42. Calling `coro()` runs **none** of its body. It builds and returns a *coroutine object* — a suspended, not-yet-started computation. Think of it as a recipe card: possessing it is not the same as cooking it.

If you never run it, Python complains:

```
RuntimeWarning: coroutine 'coro' was never awaited
```

Memorize that warning. It means *you forgot an `await` or a `create_task` somewhere*, and it is one of the two or three most common async bugs.

### 2.2 `await` is how you run one

```python
import asyncio

async def get_answer() -> int:
    await asyncio.sleep(1)      # pause here, let others run, resume in ~1s
    return 42

async def main() -> None:
    value = await get_answer()  # run it, wait for the result, unwrap it
    print(value)

asyncio.run(main())
```

`await X` means three things at once:

1. **Start** running `X`.
2. **Suspend me** if `X` needs to wait, handing control back to the event loop so other tasks can run.
3. **Resume me** when `X` finishes, and evaluate to `X`'s return value.

You may only use `await` inside an `async def`. Using it at the top level of a script is a `SyntaxError`.

### 2.3 `asyncio.run()` is the bridge

Your program starts out in ordinary synchronous Python. `asyncio.run(coro)` is the door into async-land:

```python
asyncio.run(main())
```

It creates a fresh event loop, runs `main()` until it returns, cancels any leftover tasks, closes the loop, and returns the result.

Rules:
- Call it **once**, at the very top of your program. One `asyncio.run()` per process is the normal design.
- **Never** call it inside an `async def`. You'll get `RuntimeError: asyncio.run() cannot be called from a running event loop`.
- It doesn't work in Jupyter/IPython, because those already run an event loop. In a notebook, just write `await main()` directly in a cell.

### 2.4 The critical misunderstanding

Here is the mistake essentially every beginner makes:

```python
import asyncio, time

async def fetch(name: str, seconds: float) -> str:
    print(f"  start  {name}")
    await asyncio.sleep(seconds)
    print(f"  finish {name}")
    return f"{name}-result"

async def main() -> None:
    t0 = time.perf_counter()
    a = await fetch("A", 2)
    b = await fetch("B", 2)
    c = await fetch("C", 2)
    print([a, b, c])
    print(f"elapsed: {time.perf_counter() - t0:.2f}s")

asyncio.run(main())
```

```
  start  A
  finish A
  start  B
  finish B
  start  C
  finish C
['A-result', 'B-result', 'C-result']
elapsed: 6.00s
```

**Still six seconds.** We added `async` and `await` everywhere and gained absolutely nothing.

Why? Because `await` means *"wait right here for this to finish."* Three sequential `await`s are three sequential waits. The event loop dutifully offered to run something else while `fetch("A")` slept — but there was nothing else to run, because `fetch("B")` hadn't been started yet.

> **`async`/`await` does not create concurrency. It creates the *possibility* of concurrency.** To actually get it, you must launch work *before* you wait on it. That's Chapter 4.

### 2.5 `await` is not free, and not always a pause

```python
async def compute(x):
    return x * 2        # no await inside

async def main():
    print(await compute(21))
```

`compute` never awaits anything that blocks, so `await compute(21)` runs start-to-finish without ever yielding to the event loop. This matters: **an `await` only yields control if the thing being awaited actually has to wait.** Awaiting a coroutine full of pure computation gives other tasks no chance to run.

### 2.6 What can you `await`?

Exactly three kinds of things (collectively "awaitables"):

1. **Coroutines** — what `async def f()` returns.
2. **Tasks** — a coroutine that has been scheduled to run independently (Chapter 4).
3. **Futures** — a low-level "result will exist eventually" placeholder. You rarely create these yourself; libraries hand them to you.

Objects implementing `__await__` also work, but you will almost never write one.

---

### Exercise 3 — What does this print, and how long does it take?

```python
import asyncio, time

async def step(n):
    print(f"step {n} begin")
    await asyncio.sleep(1)
    print(f"step {n} end")
    return n * 10

async def main():
    t0 = time.perf_counter()
    x = step(1)                 # note: no await
    print("created coroutine")
    y = await step(2)
    print(f"y = {y}")
    print(f"{time.perf_counter() - t0:.1f}s")

asyncio.run(main())
```

**Solution**

```
created coroutine
step 2 begin
step 2 end
y = 20
1.0s
sys:1: RuntimeWarning: coroutine 'step' was never awaited
```

`step(1)` only *creates* a coroutine object — its body never runs, so "step 1 begin" is never printed. The line `print("created coroutine")` comes first. Only `step(2)` actually executes, so total time is 1 second, and Python warns about the abandoned coroutine when it is garbage-collected.

One caveat on that output: the `RuntimeWarning` fires when the abandoned coroutine is *garbage-collected*, so its exact position in the output is not guaranteed. On CPython it usually appears at the end, when `asyncio.run` tears down; don't treat its placement as deterministic.

**This is the #1 async bug in real code**, usually written as a forgotten `await` on a function that returns nothing useful — so you get no error, just silently missing behaviour. Enable warnings-as-errors in tests (`-W error::RuntimeWarning`) to catch it.

---

## 3. The event loop: the thing that actually runs everything

### 3.1 What it is

The event loop is a `while True:` loop in one thread. Stripped to its essence:

```
while there is work:
    1. Look at the "ready" queue of tasks that can make progress right now.
    2. Run each one until it hits an await that must wait (or finishes).
    3. Ask the OS: "which of my sockets/timers are now ready?"   ← select()/epoll/kqueue
       Block here if nothing is ready, so the CPU idles instead of spinning.
    4. Move every newly-ready task into the ready queue.
    5. Repeat.
```

Step 3 is the magic. The operating system has a syscall (`epoll` on Linux, `kqueue` on macOS, IOCP on Windows) that means *"here are 10,000 sockets; wake me when any of them has data."* One thread can therefore supervise tens of thousands of connections. That's why asyncio servers scale where thread-per-connection servers fall over.

### 3.2 The golden rule

> **Never block the event loop.**

There is exactly one thread. While your code runs between two `await`s, *nothing else in the entire program can happen* — no other task, no timer, no incoming connection. If you call `time.sleep(3)`, every user of your server waits 3 seconds.

| Blocking (forbidden) | Async equivalent |
|---|---|
| `time.sleep(n)` | `await asyncio.sleep(n)` |
| `requests.get(url)` | `await client.get(url)` with `httpx`/`aiohttp` |
| `open(f).read()` | `await asyncio.to_thread(path.read_text)` |
| `conn.execute(sql)` (sync driver) | `await conn.execute(sql)` (asyncpg, aiosqlite) |
| `subprocess.run(...)` | `await asyncio.create_subprocess_exec(...)` |
| `input()` | `await asyncio.to_thread(input)` |
| heavy computation | `await asyncio.to_thread(...)` / `ProcessPoolExecutor` |

### 3.3 Seeing it with your own eyes

```python
import asyncio, time

async def heartbeat():
    """Prints a tick every 0.25s — our 'is the loop alive?' monitor."""
    while True:
        print(f"  tick {time.strftime('%H:%M:%S')}")
        await asyncio.sleep(0.25)

async def polite_worker():
    print("polite worker: starting")
    await asyncio.sleep(1.5)          # yields control
    print("polite worker: done")

async def rude_worker():
    print("rude worker: starting")
    time.sleep(1.5)                   # BLOCKS THE WHOLE LOOP
    print("rude worker: done")

async def main():
    hb = asyncio.create_task(heartbeat())
    await polite_worker()
    print("--- now the rude one ---")
    await rude_worker()
    hb.cancel()

asyncio.run(main())
```

During `polite_worker` the ticks keep coming. During `rude_worker` they stop dead for 1.5 seconds, then several fire at once as the loop catches up. That gap is a frozen program. In a web service it's 1.5 seconds of every user's requests hanging.

### 3.4 Getting a handle on the loop

```python
loop = asyncio.get_running_loop()     # correct, inside async code
print(loop.time())                    # loop's monotonic clock (NOT wall clock)
```

Use `get_running_loop()`. The older `asyncio.get_event_loop()` has had shifting, deprecated semantics across recent Python versions — avoid it in new code. If you see `loop = asyncio.get_event_loop()` followed by `loop.run_until_complete(...)` in a tutorial, that tutorial predates Python 3.7 conventions; the modern form is `asyncio.run(...)`.

If you need more control than `asyncio.run` gives (e.g. running several top-level coroutines on one loop), use `asyncio.Runner` (3.11+):

```python
with asyncio.Runner() as runner:
    runner.run(setup())
    runner.run(main())
```

---

### Exercise 4 — Order of output

Predict the exact output.

```python
import asyncio

async def a():
    print("a1")
    await asyncio.sleep(0)
    print("a2")

async def b():
    print("b1")
    await asyncio.sleep(0)
    print("b2")

async def main():
    await a()
    await b()

asyncio.run(main())
```

Then predict it again with `main` changed to:

```python
async def main():
    await asyncio.gather(a(), b())
```

**Solution**

**Version 1:** `a1 a2 b1 b2`. Sequential awaits: `a()` runs to completion (the `sleep(0)` yields, but there's nothing else scheduled, so control comes right back) before `b()` starts.

**Version 2:** `a1 b1 a2 b2`. `gather` schedules both as tasks. `a` runs to its `sleep(0)` and yields; the loop then runs `b` up to *its* `sleep(0)`; then `a` resumes; then `b`.

`await asyncio.sleep(0)` is the idiomatic way to say **"yield control to the event loop right now, but don't actually wait."** It's how you politely break up a long computation so other tasks get a turn.

---

### Exercise 5 — Find and fix the blocking call

```python
import asyncio, time

async def handle_request(req_id: int):
    print(f"request {req_id} received")
    data = expensive_lookup(req_id)
    print(f"request {req_id} -> {data}")

def expensive_lookup(req_id: int) -> str:
    time.sleep(0.5)
    return f"row-{req_id}"

async def main():
    await asyncio.gather(*(handle_request(i) for i in range(5)))

t0 = time.perf_counter()
asyncio.run(main())
print(f"{time.perf_counter() - t0:.2f}s")
```

How long does this take, why, and what are two ways to fix it?

**Solution**

It takes **~2.5 seconds**, not 0.5. `gather` correctly schedules five concurrent tasks, but `expensive_lookup` uses `time.sleep`, which blocks the single thread. Each task monopolizes the loop for 0.5 s in turn. `gather` gave us concurrency; the blocking call destroyed it.

**Fix 1 — if the operation is genuinely I/O (a sync DB driver, a file read), push it to a thread:**

```python
data = await asyncio.to_thread(expensive_lookup, req_id)
```
Now ~0.5 s total. The thread blocks; the event loop doesn't.

**Fix 2 — if an async version of the library exists, use it:**

```python
async def expensive_lookup(req_id: int) -> str:
    await asyncio.sleep(0.5)          # e.g. await asyncpg_conn.fetchrow(...)
    return f"row-{req_id}"
```
Also ~0.5 s, and cheaper — no thread overhead, and it supports cancellation properly.

Prefer Fix 2 when a native async library exists; use Fix 1 when you're stuck with a synchronous dependency. Both are covered in depth in Chapter 10.

---

---

### ✓ Checkpoint — Part I

You should now be able to:

- Decide whether a given workload is **I/O-bound** or **CPU-bound**, and say whether asyncio will help.
- Explain the difference between **concurrency** and **parallelism** without using the word "simultaneous" loosely.
- Say what `async def f()` returns when you call it, and what happens if you never await it.
- Explain, in one sentence, why adding `async`/`await` to a sequential program does not make it faster.
- Name five blocking calls and their async replacements.

If any of those are shaky, re-read §1.2, §2.4, and §3.2 before continuing — the rest of the tutorial assumes them.

# Part II — Real concurrency

## 4. Tasks: how you actually get things running at the same time

### 4.1 `create_task` schedules; `await` collects

A **Task** wraps a coroutine and hands it to the event loop with the instruction *"run this whenever you have a spare moment."* Creating a task starts it in the background; you don't have to wait for it immediately.

```python
import asyncio, time

async def fetch(name: str, seconds: float) -> str:
    print(f"  start  {name}")
    await asyncio.sleep(seconds)
    print(f"  finish {name}")
    return f"{name}-result"

async def main() -> None:
    t0 = time.perf_counter()

    ta = asyncio.create_task(fetch("A", 2))   # starts running (soon)
    tb = asyncio.create_task(fetch("B", 2))
    tc = asyncio.create_task(fetch("C", 2))

    print("all three launched, none awaited yet")

    a = await ta            # now wait for results
    b = await tb
    c = await tc

    print([a, b, c])
    print(f"elapsed: {time.perf_counter() - t0:.2f}s")

asyncio.run(main())
```

```
all three launched, none awaited yet
  start  A
  start  B
  start  C
  finish A
  finish B
  finish C
['A-result', 'B-result', 'C-result']
elapsed: 2.00s
```

**Two seconds instead of six.** This is the whole point. Note the shape of the working pattern:

> **Launch everything first. Await afterwards.**

Note also that "all three launched" prints *before* "start A". `create_task` doesn't run the coroutine immediately — it puts it on the loop's ready queue. The coroutine body first executes when the current code next hits an `await` that yields (here, `await ta`).

### 4.2 Task lifecycle and inspection

```python
task = asyncio.create_task(fetch("X", 1), name="fetch-X")

task.done()        # False until it finishes/fails/is cancelled
task.cancelled()   # True only if it was cancelled
task.result()      # the return value — raises if not done, re-raises its exception
task.exception()   # the exception it raised, or None
task.get_name()    # "fetch-X" — worth setting, it shows up in debug output
task.cancel()      # request cancellation (Chapter 6)
```

Awaiting a task that has already finished returns instantly with its stored result. **You can await the same task many times** — unlike a coroutine, which can only be awaited once.

```python
t = asyncio.create_task(fetch("Y", 1))
print(await t)   # waits 1s, prints Y-result
print(await t)   # instant, prints Y-result again
```

### 4.3 The strong-reference trap (a real production bug)

The event loop holds only a **weak** reference to your tasks. If you create a fire-and-forget task and don't keep a reference to it, the garbage collector may destroy it mid-flight — and your work silently vanishes.

```python
# BROKEN: task may be garbage-collected before it completes
asyncio.create_task(log_analytics_event(payload))
```

The documented fix:

```python
_background: set[asyncio.Task] = set()

def spawn(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)   # auto-cleanup, no leak
    return task
```

This bug is intermittent and load-dependent — the worst kind. Better still: avoid fire-and-forget entirely and use a `TaskGroup` (§5.2), which keeps strong references for you.

---

### Exercise 6 — Fix the slow function

Make this finish in ~3 seconds instead of ~9, changing only `main`.

```python
import asyncio, time

async def download(url: str, secs: float) -> int:
    await asyncio.sleep(secs)
    return len(url) * 100

async def main():
    t0 = time.perf_counter()
    a = await download("http://a.com", 3)
    b = await download("http://bb.com", 3)
    c = await download("http://ccc.com", 3)
    print(a + b + c, f"{time.perf_counter() - t0:.1f}s")

asyncio.run(main())
```

**Solution**

```python
async def main():
    t0 = time.perf_counter()
    ta = asyncio.create_task(download("http://a.com", 3))
    tb = asyncio.create_task(download("http://bb.com", 3))
    tc = asyncio.create_task(download("http://ccc.com", 3))
    a, b, c = await ta, await tb, await tc
    print(a + b + c, f"{time.perf_counter() - t0:.1f}s")
```

~3.0 s. Equivalent and shorter:

```python
a, b, c = await asyncio.gather(
    download("http://a.com", 3),
    download("http://bb.com", 3),
    download("http://ccc.com", 3),
)
```

---

### Exercise 7 — Why is this still slow?

```python
async def main():
    results = []
    for url, secs in [("a", 2), ("b", 2), ("c", 2)]:
        task = asyncio.create_task(download(url, secs))
        results.append(await task)
    print(results)
```

**Solution**

**6 seconds.** The `await task` is *inside* the loop, so each iteration creates a task and then immediately blocks until it completes before creating the next one. Using `create_task` buys you nothing if you await it in the same breath.

Correct — two passes:

```python
async def main():
    tasks = [asyncio.create_task(download(u, s)) for u, s in [("a",2),("b",2),("c",2)]]
    results = [await t for t in tasks]      # all already running
    print(results)
```

**Rule of thumb: if `create_task` and its `await` are in the same loop body, you've written sequential code.**

---

### Exercise 8 — Trace the interleaving

```python
import asyncio

async def worker(name, steps):
    for i in range(steps):
        print(f"{name}:{i}")
        await asyncio.sleep(0.1)

async def main():
    t1 = asyncio.create_task(worker("A", 3))
    t2 = asyncio.create_task(worker("B", 2))
    print("main: launched")
    await t1
    print("main: t1 done")
    await t2
    print("main: t2 done")

asyncio.run(main())
```

**Solution**

```
main: launched
A:0
B:0
A:1
B:1
A:2
main: t1 done
main: t2 done
```

Walkthrough: both tasks are queued; `print("main: launched")` runs; `await t1` suspends `main`, letting the loop run A (prints `A:0`, sleeps) then B (`B:0`, sleeps). At t≈0.1 both wake in scheduling order: `A:1`, `B:1`. At t≈0.2: `A:2`, and B finishes its loop so `t2` completes. At t≈0.3 A finishes, `t1` completes, `main` resumes and prints "t1 done". `await t2` returns instantly because t2 is already done, so "t2 done" follows immediately.

Key insight: **the order in which you await does not change the order in which tasks run.** Tasks run as soon as the loop gets to them.

---

### Exercise 9 — Spot the bug

```python
async def process_all(items):
    for item in items:
        asyncio.create_task(handle(item))
    print("all done!")
```

Two serious bugs. Name them and fix.

**Solution**

**Bug 1:** `print("all done!")` is a lie. The tasks have been *scheduled*, not completed — in fact none of their bodies has even started, since `process_all` never awaits.

**Bug 2:** No references are kept to the tasks, so they may be garbage-collected mid-execution (§4.3). And if `process_all`'s caller finishes and `asyncio.run` returns, all pending tasks are cancelled outright.

Fix:

```python
async def process_all(items):
    async with asyncio.TaskGroup() as tg:
        for item in items:
            tg.create_task(handle(item))
    print("all done!")      # now true: TaskGroup waits on exit
```

---

## 5. Running many things: gather, TaskGroup, as_completed, wait

Four tools. Learning when to use which is most of the skill.

### 5.1 `asyncio.gather` — results in order

```python
results = await asyncio.gather(
    fetch("A", 1),
    fetch("B", 2),
    fetch("C", 3),
)
# results == ["A-result", "B-result", "C-result"]  — always in ARGUMENT order
```

`gather` wraps each coroutine in a task, runs them concurrently, and returns a list of results **in the order you passed them in**, regardless of which finished first. Total time = the slowest one (3 s here).

For a dynamic collection, unpack:

```python
urls = ["a", "b", "c", "d"]
results = await asyncio.gather(*(fetch(u, 1) for u in urls))
```

**Error behaviour (important):** by default, if any coroutine raises, `gather` immediately propagates that exception to you — but **the other tasks keep running**, unsupervised, with their results and errors discarded. That's a leak. To collect errors instead of raising:

```python
results = await asyncio.gather(*coros, return_exceptions=True)
for r in results:
    if isinstance(r, Exception):
        print("failed:", r)
    else:
        print("ok:", r)
```

With `return_exceptions=True`, exceptions are returned as values in the results list, position-matched to the inputs. Nothing is cancelled; everything runs to completion. This mode is genuinely useful — "try all 100 URLs, tell me which failed" — and is common in agent fan-outs.

### 5.2 `asyncio.TaskGroup` — the modern default (3.11+)

```python
async def main():
    async with asyncio.TaskGroup() as tg:
        ta = tg.create_task(fetch("A", 1))
        tb = tg.create_task(fetch("B", 2))
        tc = tg.create_task(fetch("C", 3))
    # on exiting the block, ALL tasks are guaranteed finished
    print(ta.result(), tb.result(), tc.result())
```

This is **structured concurrency**: tasks cannot outlive the block that created them. The `async with` doesn't exit until every child task is done. Guarantees you get for free:

- No task leaks. No forgotten strong references.
- **If one child raises, the group cancels all its siblings**, waits for them to unwind, and then raises an `ExceptionGroup` containing everything that went wrong.
- Errors are never silently swallowed.

Compare the failure modes:

| | `gather()` | `TaskGroup` |
|---|---|---|
| One child fails | Others keep running unsupervised | Siblings cancelled immediately |
| Error surfaces as | The first exception only | `ExceptionGroup` with *all* of them |
| Task references | You must manage them | Managed for you |
| Results | Returned as an ordered list | Read from each task's `.result()` |
| "Run all, collect failures" | `return_exceptions=True` | Catch per-task inside the coroutine |

**Default to `TaskGroup`** for anything that must all succeed together. Use `gather(..., return_exceptions=True)` when partial failure is acceptable and you want every result.

You can also add tasks dynamically:

```python
async with asyncio.TaskGroup() as tg:
    for item in queue_of_work:
        tg.create_task(handle(item))
```

One restriction: you can only call `tg.create_task()` while the group's block is active.

### 5.3 `asyncio.as_completed` — process results as they arrive

When the slowest result shouldn't hold up the fast ones:

```python
async def main():
    coros = [fetch("slow", 3), fetch("medium", 2), fetch("fast", 1)]
    for finished in asyncio.as_completed(coros):
        result = await finished
        print("got:", result)       # fast, medium, slow — completion order
```

You get results the instant each one is ready, instead of waiting 3 s for everything. Perfect for streaming partial results to a user or a progress bar.

Caveat: in 3.12, `as_completed` yields *futures*, not your original coroutines, so you can't tell which input a result came from. The standard workaround is to have each coroutine return its own identity:

```python
async def fetch_tagged(name, secs):
    return name, await fetch(name, secs)

for fut in asyncio.as_completed([fetch_tagged(n, s) for n, s in jobs]):
    name, result = await fut
    print(name, "→", result)
```

(Python 3.13 improved this by allowing `async for` and yielding the original awaitables — but on 3.12 use the tagging trick.)

### 5.4 `asyncio.wait` — low-level, for "first one wins"

```python
tasks = [asyncio.create_task(fetch(n, s)) for n, s in jobs]
done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

for t in done:
    print(t.result())
for t in pending:
    t.cancel()                      # YOU must clean up
await asyncio.gather(*pending, return_exceptions=True)
```

`return_when` accepts `FIRST_COMPLETED`, `FIRST_EXCEPTION`, or `ALL_COMPLETED` (default). Two traps:
- `wait` **never raises** your tasks' exceptions — they sit inside the returned task objects. You must check.
- It does **not** cancel the pending tasks. Forgetting to is a leak.

Note: since Python 3.11, `asyncio.wait` requires **Tasks**, not bare coroutines — wrap them with `create_task` first.

Use `wait` for genuine race patterns: "query three redundant mirrors, take whichever answers first." For everything else, `TaskGroup` or `gather` is clearer.

### 5.5 Choosing

```
Do all of them need to succeed?
├── YES → TaskGroup                          (structured, cancels siblings on error)
└── NO  → Do you need every result, including failures?
          ├── YES → gather(..., return_exceptions=True)
          └── NO  → Do you want results as they arrive?
                    ├── YES → as_completed
                    └── NO (just the first) → wait(..., FIRST_COMPLETED)
```

---

### Exercise 10 — gather vs TaskGroup under failure

```python
import asyncio

async def ok(name, secs):
    await asyncio.sleep(secs)
    print(f"{name} finished")
    return name

async def boom(secs):
    await asyncio.sleep(secs)
    raise ValueError("kaboom")

async def with_gather():
    try:
        await asyncio.gather(ok("A", 3), boom(1), ok("B", 3))
    except ValueError as e:
        print("gather caught:", e)
    await asyncio.sleep(4)          # keep the loop alive to observe stragglers

asyncio.run(with_gather())
```

What prints? Now rewrite with `TaskGroup` and compare.

**Solution**

**With `gather`:**
```
gather caught: kaboom
A finished
B finished
```
At t=1 `boom` raises and `gather` re-raises immediately — but A and B were never cancelled. They complete at t=3, printing merrily into the void. In a real system those are wasted API calls, wasted money, and possibly unwanted side effects (a half-finished multi-step write).

**With `TaskGroup`:**
```python
async def with_taskgroup():
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(ok("A", 3))
            tg.create_task(boom(1))
            tg.create_task(ok("B", 3))
    except* ValueError as eg:
        print("taskgroup caught:", eg.exceptions)
    await asyncio.sleep(4)
```
```
taskgroup caught: (ValueError('kaboom'),)
```
"A finished" and "B finished" never print — at t=1 the group cancelled its siblings. No wasted work, no zombie tasks. Note `except*` (Chapter 7) because `TaskGroup` raises an `ExceptionGroup`.

---

### Exercise 11 — Robust batch fetching

Write `fetch_all(jobs)` where `jobs` is a list of `(name, seconds)` and some names starting with `"bad"` raise `ConnectionError`. Return a dict `{name: result_or_None}` and print a summary of successes and failures. All jobs must run concurrently; one failure must not stop the others.

**Solution**

```python
import asyncio

async def fetch(name: str, secs: float) -> str:
    await asyncio.sleep(secs)
    if name.startswith("bad"):
        raise ConnectionError(f"could not reach {name}")
    return f"{name}-payload"

async def fetch_all(jobs: list[tuple[str, float]]) -> dict[str, str | None]:
    results = await asyncio.gather(
        *(fetch(n, s) for n, s in jobs),
        return_exceptions=True,
    )
    out: dict[str, str | None] = {}
    failures: list[tuple[str, BaseException]] = []
    for (name, _), res in zip(jobs, results):
        if isinstance(res, BaseException):
            out[name] = None
            failures.append((name, res))
        else:
            out[name] = res
    print(f"{len(jobs) - len(failures)} ok, {len(failures)} failed")
    for name, err in failures:
        print(f"  {name}: {type(err).__name__}: {err}")
    return out

asyncio.run(fetch_all([("a",1), ("bad-1",0.5), ("b",1.5), ("bad-2",0.2)]))
```

Two points of craft: `zip(jobs, results)` works because `gather` preserves input order — that guarantee is what makes the pairing safe. And check `isinstance(res, BaseException)`, not `Exception`, because `CancelledError` derives from `BaseException`.

---

### Exercise 12 — First responder wins

Three mirrors answer in 3 s, 1 s, and 2 s; the 1 s one raises. Write `fastest_success(mirrors)` that returns the first *successful* answer, cancels the rest, and raises only if all fail.

*Hint:* `asyncio.wait` returns `(done, pending)` and does **not** cancel the pending set for you. You will need a loop (the first task to finish may be the one that fails) and a `finally` block.

**Solution**

```python
import asyncio

async def mirror(name, secs, fail=False):
    await asyncio.sleep(secs)
    if fail:
        raise ConnectionError(f"{name} down")
    return f"data from {name}"

async def fastest_success(coros) -> str:
    tasks = {asyncio.create_task(c) for c in coros}
    errors = []
    try:
        while tasks:
            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                exc = t.exception()
                if exc is None:
                    return t.result()      # finally: cancels the rest
                errors.append(exc)
        raise ExceptionGroup("all mirrors failed", errors)
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

async def main():
    print(await fastest_success([
        mirror("eu", 3),
        mirror("us", 1, fail=True),
        mirror("ap", 2),
    ]))

asyncio.run(main())
```

Output: `data from ap` after ~2 s. The `us` mirror fails first at 1 s but we keep waiting; `ap` succeeds at 2 s; `eu` is cancelled.

The `finally` block is the part people forget. Without it, a `return` from inside the loop leaves the slow mirror running as a zombie. The `await asyncio.gather(*tasks, return_exceptions=True)` after cancelling is also important: `cancel()` only *requests* cancellation, and you should wait for the tasks to actually unwind before moving on.

---

## 6. Timeouts and cancellation

In production, **every network call needs a timeout**. An LLM API that normally answers in 3 seconds will one day hang for 600. Without a timeout, your agent hangs with it.

### 6.1 `asyncio.timeout` — the modern way (3.11+)

```python
async def main():
    try:
        async with asyncio.timeout(2.0):
            await fetch("slow-server", 10)
    except TimeoutError:
        print("gave up after 2s")
```

It wraps a *block*, so it can guard several operations at once:

```python
async with asyncio.timeout(30):
    plan   = await llm.complete("make a plan")
    facts  = await search(plan)
    answer = await llm.complete(f"answer using {facts}")
# 30s budget for the WHOLE sequence — exactly what you want for an agent turn
```

There's also `asyncio.timeout_at(deadline)` using the loop clock, for absolute deadlines:

```python
deadline = asyncio.get_running_loop().time() + 30
async with asyncio.timeout_at(deadline):
    ...
```

And you can reschedule a live timeout, which is how you build "reset the clock every time a token arrives" streaming logic:

```python
async with asyncio.timeout(5) as cm:
    async for token in stream:
        handle(token)
        cm.reschedule(asyncio.get_running_loop().time() + 5)
```

### 6.2 `asyncio.wait_for` — the older way

```python
try:
    result = await asyncio.wait_for(fetch("slow", 10), timeout=2.0)
except TimeoutError:
    print("timed out")
```

Wraps a *single* awaitable rather than a block. Still perfectly fine and very common in existing code. (Since 3.11, `asyncio.TimeoutError` is an alias for the built-in `TimeoutError`, so catching either works.)

### 6.3 How cancellation actually works

This is where async gets subtle, and where the difference between a junior and a senior engineer shows.

`task.cancel()` does **not** kill the task. It schedules a `CancelledError` to be raised *inside* the coroutine, at the point where it is currently suspended — i.e. at its current `await`. The task then unwinds normally, running `finally` blocks and `async with` cleanup on the way out.

```python
import asyncio

async def worker():
    print("worker: acquiring resource")
    try:
        await asyncio.sleep(10)
        print("worker: never reached")
    except asyncio.CancelledError:
        print("worker: cancelled — cleaning up")
        raise                       # ← ALWAYS re-raise
    finally:
        print("worker: resource released")

async def main():
    t = asyncio.create_task(worker())
    await asyncio.sleep(0.5)
    t.cancel()
    try:
        await t
    except asyncio.CancelledError:
        print("main: confirmed cancelled")

asyncio.run(main())
```

```
worker: acquiring resource
worker: cancelled — cleaning up
worker: resource released
main: confirmed cancelled
```

Three rules, in order of importance:

1. **If you catch `CancelledError`, re-raise it.** Swallowing it makes your task uncancellable, which breaks `TaskGroup`, breaks timeouts, and hangs shutdown. Catch it only to clean up.
2. **`CancelledError` inherits from `BaseException`, not `Exception`** (since 3.8). So `except Exception:` will *not* catch it — that's deliberate, and it's why `except Exception:` is safe to use for error handling in async code.
3. **A task can only be cancelled at an `await`.** A coroutine doing a 60-second CPU loop with no awaits is uncancellable. Another reason to keep computation off the loop.

Always cancel-then-await:

```python
task.cancel()
try:
    await task
except asyncio.CancelledError:
    pass
```

Without the `await`, you've only *requested* cancellation and may exit before the cleanup has run.

### 6.4 `asyncio.shield` — protecting critical sections

Occasionally an operation genuinely must not be interrupted — committing a transaction, releasing a lock on a remote resource.

```python
try:
    await asyncio.wait_for(asyncio.shield(commit_transaction()), timeout=5)
except TimeoutError:
    print("stopped waiting, but the commit is still running")
```

`shield` protects the inner operation from cancellation while allowing the *waiter* to give up.

**Notice the tension with §4.3.** Once the timeout fires, `commit_transaction()` is still running and *you have no handle on it*. That is precisely the untracked-work situation §4.3 warned about. `shield` does keep the inner task alive internally, so it won't be garbage-collected mid-flight — but you have lost the ability to await its result, observe its exception, or cancel it during shutdown. If it raises, the traceback goes nowhere.

So if you use `shield`, keep the handle yourself:

```python
commit = asyncio.create_task(commit_transaction())
try:
    await asyncio.wait_for(asyncio.shield(commit), timeout=5)
except TimeoutError:
    print("still committing; will confirm during shutdown")
    pending_commits.add(commit)          # someone must eventually await this
```

Use `shield` sparingly. The better answer is almost always to make the operation **idempotent** so it's safe to cancel and retry, rather than uncancellable.

### 6.5 The trap: what swallowed cancellation actually does

§6.3 said "always re-raise `CancelledError`." Here is exactly what goes wrong if you don't — and it is worse than most people expect, because **nothing raises an error.**

**Failure 1: the timeout is silently defeated.**

```python
import asyncio

async def swallow():
    try:
        await asyncio.sleep(10)
    except asyncio.CancelledError:
        return "finished normally"      # ← catches and RETURNS

async def main():
    async with asyncio.timeout(0.2):
        print(await swallow())

asyncio.run(main())
```

Output:

```
finished normally
```

No `TimeoutError`. No warning. No traceback. The deadline you carefully set was simply ignored, and your code proceeded as though everything was fine.

The mechanism: `asyncio.timeout` enforces its deadline by *cancelling the current task* and then converting that cancellation into `TimeoutError` on the way out of the block. When the inner coroutine absorbs the `CancelledError` and returns a value, there is no cancellation left to convert, so the block exits normally. Worse, the timeout is now spent — add `await asyncio.sleep(0.5)` after the `print` and it runs to completion, 0.7 s into a 0.2 s budget, still with no error. `asyncio.wait_for` behaves identically.

**Failure 2: the task becomes unkillable.**

Swallowing inside a loop is the version that takes down production systems:

```python
async def worker():
    while True:
        try:
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            continue                      # "recover and keep going"
```

This task cannot be stopped. `task.cancel()` delivers a `CancelledError` at the `sleep`; the handler discards it and loops back. A `TaskGroup` containing it never exits. Wrapping the group in an outer `asyncio.wait_for` does not help — that timeout also works by cancelling, and gets swallowed too. In testing this, the process had to be killed with a signal from outside; no amount of asyncio-level cancellation could reach it. In a real service, that is a pod that ignores SIGTERM and gets SIGKILLed mid-write.

**Failure 3: callers are told it succeeded.**

After a swallowed cancellation, `task.cancelled()` is `False` and `task.result()` returns the value the handler returned. Every caller concludes the work completed. Any bookkeeping that distinguishes "finished" from "aborted" is now wrong.

**The rule, stated operationally:**

> Catching `CancelledError` is fine. *Returning* from the handler is not. Either `raise`, or raise something else.

If all you want is cleanup, don't catch it at all — use `finally`, which runs on every exit path and cannot accidentally suppress anything:

```python
async def worker():
    try:
        while True:
            await do_chunk()
    finally:
        await release_resources()      # runs on normal exit, error, and cancellation
```

---

### Exercise 13 — Add a timeout

Modify so that `slow_api()` gets at most 1.5 s, and on timeout returns the string `"DEFAULT"` instead of crashing.

```python
async def slow_api():
    await asyncio.sleep(5)
    return "real data"
```

**Solution**

```python
import asyncio

async def slow_api():
    await asyncio.sleep(5)
    return "real data"

async def get_data() -> str:
    try:
        async with asyncio.timeout(1.5):
            return await slow_api()
    except TimeoutError:
        return "DEFAULT"

asyncio.run(get_data())
```

Equivalent with the older API:
```python
try:
    return await asyncio.wait_for(slow_api(), timeout=1.5)
except TimeoutError:
    return "DEFAULT"
```
Note that when the timeout fires, `slow_api` **is cancelled** — it doesn't keep running in the background. That's the behaviour you want, and it's why you shouldn't reach for `shield` by default.

---

### Exercise 14 — What's wrong here?

```python
async def worker():
    try:
        while True:
            await do_chunk()
    except asyncio.CancelledError:
        print("shutting down cleanly")
        # no re-raise
    except Exception as e:
        log(e)
```

**Solution**

`CancelledError` is caught and **not re-raised**. Consequences:

1. The task completes *normally* instead of being marked cancelled, so `task.cancelled()` returns `False` and callers think the work succeeded.
2. Inside a `TaskGroup` or `asyncio.timeout`, the machinery waits for a cancellation that never confirms, and the enclosing timeout can hang or raise confusing errors.
3. During shutdown, `asyncio.run()` can't drain the loop.

Fixed:

```python
async def worker():
    try:
        while True:
            await do_chunk()
    except asyncio.CancelledError:
        print("shutting down cleanly")
        raise                       # ← essential
    except Exception as e:
        log(e)
        raise
```

Better still, put cleanup in `finally` so it runs on every exit path:

```python
async def worker():
    try:
        while True:
            await do_chunk()
    finally:
        await release_resources()
```

---

### Exercise 15 — Cancellable countdown

Write `countdown(n)` that prints `n, n-1, ... 1` one per second and prints `"countdown aborted at k"` if cancelled. In `main`, start it with n=10 and cancel after 3.5 s.

**Solution**

```python
import asyncio

async def countdown(n: int) -> None:
    current = n
    try:
        while current > 0:
            print(current)
            await asyncio.sleep(1)
            current -= 1
        print("liftoff!")
    except asyncio.CancelledError:
        print(f"countdown aborted at {current}")
        raise

async def main():
    t = asyncio.create_task(countdown(10))
    await asyncio.sleep(3.5)
    t.cancel()
    try:
        await t
    except asyncio.CancelledError:
        print("main: task confirmed cancelled")

asyncio.run(main())
```

Prints 10, 9, 8, 7 (at t=0,1,2,3), then at t=3.5 the `sleep` is interrupted and it reports `countdown aborted at 7` — because `current` hadn't yet been decremented when the cancellation hit the `await`. Tracing exactly *which* `await` the cancellation lands on is the skill this exercise is teaching.

---

### Exercise 16 — Per-item timeouts inside a batch

Fetch 5 items concurrently, where each item gets its own 2-second timeout. Slow items should yield `None`, not kill the batch.

*Hint:* wrap the timeout around the single unit of work, not around the batch. Think about which function the `try` belongs in.

**Solution**

```python
import asyncio

async def fetch(i: int) -> str:
    await asyncio.sleep(i)            # item i takes i seconds
    return f"item-{i}"

async def fetch_with_timeout(i: int, limit: float = 2.0) -> str | None:
    try:
        async with asyncio.timeout(limit):
            return await fetch(i)
    except TimeoutError:
        print(f"item {i} timed out")
        return None

async def main():
    results = await asyncio.gather(*(fetch_with_timeout(i) for i in range(5)))
    print(results)

asyncio.run(main())
```

```
item 2 timed out
item 3 timed out
item 4 timed out
['item-0', 'item-1', None, None, None]
```

The pattern — **wrap the timeout around the individual unit of work, not the batch** — is the one you want for agent tool calls. Combine with an outer `asyncio.timeout` on the whole batch when you also need a global budget.

---

## 7. Errors: ExceptionGroup and `except*`

### 7.1 The problem

If ten tasks run concurrently and three fail, which exception should be raised? Classic Python can only raise one. Python 3.11 introduced `ExceptionGroup` so you can raise *all* of them.

### 7.2 `except*`

```python
import asyncio

async def flaky(name, exc=None):
    await asyncio.sleep(0.1)
    if exc:
        raise exc(f"{name} failed")
    return name

async def main():
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(flaky("a"))
            tg.create_task(flaky("b", ValueError))
            tg.create_task(flaky("c", ConnectionError))
    except* ValueError as eg:
        print("value errors:", [str(e) for e in eg.exceptions])
    except* ConnectionError as eg:
        print("connection errors:", [str(e) for e in eg.exceptions])

asyncio.run(main())
```

```
value errors: ['b failed']
connection errors: ['c failed']
```

`except*` differs from `except` in ways that matter:

- It **matches sub-exceptions inside a group** and gives you a sub-group containing only the matching ones.
- **Multiple `except*` clauses can all fire** for the same group — unlike ordinary `except`, where only the first match runs.
- `eg.exceptions` is the tuple of the actual exceptions.

To catch anything at all from a group:

```python
except* Exception as eg:
    for e in eg.exceptions:
        log.error("child failed: %r", e)
```

### 7.3 Interaction with cancellation

When a `TaskGroup` child fails, siblings receive `CancelledError`. Those cancellations are *not* reported in the resulting group — you only see genuine errors. But be aware that if the group itself is cancelled from outside, you may see `CancelledError` propagate rather than an `ExceptionGroup`.

A practical defensive pattern for agent code: **handle errors inside each task** so the group never has to tear itself down over one flaky tool call.

```python
async def safe_tool_call(tool, args) -> dict:
    try:
        return {"ok": True, "result": await tool(args)}
    except Exception as e:                  # NOT BaseException — lets cancellation through
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
```

This is how real agents behave: a failed tool call becomes an error *message* the model can read and react to, not a crash.

---

### Exercise 17 — Read the group

```python
async def main():
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(flaky("a", ValueError))
            tg.create_task(flaky("b", ValueError))
            tg.create_task(flaky("c", TypeError))
    except* ValueError as eg:
        print("V:", len(eg.exceptions))
    except* TypeError as eg:
        print("T:", len(eg.exceptions))
    print("after")
```

What prints?

**Solution**

```
V: 2
T: 1
after
```

Both handlers run — that's the defining feature of `except*`. The `ValueError` handler receives a sub-group with both value errors; the `TypeError` handler gets the one type error. Execution continues normally afterwards because every sub-exception was handled. Had there been a fourth task raising `KeyError`, the unhandled remainder would be re-raised as a group after the handlers ran.

---

### Exercise 18 — Convert crashes into messages

Write `run_tools(calls)` taking a list of `(tool_name, coroutine)`. Run them all concurrently. Return a list of dicts `{"tool": name, "ok": bool, "result"/"error": ...}`. No exception should ever escape.

**Solution**

```python
import asyncio

async def run_one(name: str, coro) -> dict:
    try:
        return {"tool": name, "ok": True, "result": await coro}
    except asyncio.CancelledError:
        raise                                      # never swallow cancellation
    except Exception as e:
        return {"tool": name, "ok": False, "error": f"{type(e).__name__}: {e}"}

async def run_tools(calls: list[tuple[str, object]]) -> list[dict]:
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(run_one(n, c)) for n, c in calls]
    return [t.result() for t in tasks]
```

Because `run_one` never lets a normal exception escape, the `TaskGroup` never tears down — every tool gets its full chance to run. The explicit `except asyncio.CancelledError: raise` is technically redundant (since `CancelledError` isn't an `Exception`) but it documents the intent, and protects you if someone later broadens the handler to `BaseException`.

This function is, essentially, the tool-dispatch core of an agent. You'll build on it in Chapter 15.

---

### Exercise 19 — Design question

Your agent runs 5 tools concurrently. Tool 3 raises. For each scenario, pick `TaskGroup` or `gather(return_exceptions=True)`:

(a) Tools are read-only lookups; the model can work with partial results.
(b) Tools write to a database and must be all-or-nothing.
(c) Tools are paid API calls and you want to stop burning money the moment anything breaks.
(d) You're benchmarking reliability and need every outcome recorded.

**Solution**

(a) **`gather(return_exceptions=True)`.** Partial results are useful; let everything finish and hand the model what you got, including a note about what failed.

(b) **`TaskGroup`** — but that's not sufficient. Cancelling in-flight writes does not undo committed ones. You need transactional semantics (a single DB transaction, a saga with compensating actions, or idempotent writes with a final commit step). Concurrency primitives give you *coordination*, not *atomicity*.

(c) **`TaskGroup`.** First failure cancels the siblings, cutting off further spend. This is the standard choice for expensive LLM fan-outs.

(d) **`gather(return_exceptions=True)`.** Cancelling siblings would destroy exactly the data you're trying to collect.

---

---

### ✓ Checkpoint — Part II

You should now be able to:

- Turn a sequential set of independent awaits into a concurrent one, and explain why `create_task` followed immediately by `await` in the same loop body is still sequential.
- Choose correctly between `TaskGroup`, `gather(return_exceptions=True)`, `as_completed`, and `wait`, and justify the choice in terms of failure behaviour.
- Put a timeout on both a single call and a multi-step block.
- Explain what `task.cancel()` actually does, and what breaks when a coroutine swallows `CancelledError`.
- Read an `ExceptionGroup` traceback and write `except*` handlers for it.

The single most valuable habit from this Part: **launch first, await later.**

# Part III — Coordination

## 8. Synchronization primitives: Lock, Semaphore, Event

### 8.1 "It's single-threaded, so why do I need locks?"

Excellent question, and the answer is instructive.

You do **not** need locks to protect a simple read-modify-write like `counter += 1`. In threaded code that's a race condition; in asyncio it's safe, because the event loop can only switch tasks at an `await`, and there is no `await` inside `counter += 1`.

You **do** need locks when an invariant must hold across an `await`:

```python
balance = 100

async def withdraw(amount):
    global balance
    if balance >= amount:             # check
        await asyncio.sleep(0.1)      # ← another task runs here!
        balance -= amount             # act
        return True
    return False

async def main():
    # two concurrent withdrawals of 100 from a balance of 100
    print(await asyncio.gather(withdraw(100), withdraw(100)))
    print("balance:", balance)
```

```
[True, True]
balance: -100
```

Both tasks passed the check before either performed the subtraction. This is a genuine **check-then-act race**, and it exists in asyncio exactly because there's an `await` in the middle.

> **The rule: if a sequence of operations must be atomic *and contains an `await`*, you need a lock.**

### 8.2 `asyncio.Lock`

```python
lock = asyncio.Lock()

async def withdraw(amount):
    global balance
    async with lock:                  # only one task inside at a time
        if balance >= amount:
            await asyncio.sleep(0.1)
            balance -= amount
            return True
        return False
```

Now the result is `[True, False]` and `balance: 0`.

Always use `async with lock:`, never manual `acquire`/`release` — the context manager releases the lock even if the body raises or is cancelled.

**Important:** create asyncio primitives inside the running loop (i.e. inside `async def` or as instance attributes created during async setup), not at module import time before a loop exists. And never use `threading.Lock` in async code — acquiring a contended one blocks the entire loop.

### 8.3 `asyncio.Semaphore` — the most useful one in practice

A semaphore is a lock that allows *N* holders instead of 1. It is how you enforce **concurrency limits**, and in agentic AI it is the single most important primitive after `TaskGroup`.

Why you need it: "run all 500 requests concurrently" sounds great until you hit the API's rate limit, exhaust your file descriptors, or get your key throttled. You want *controlled* concurrency — say 10 at a time.

```python
import asyncio, time

async def call_api(i: int, sem: asyncio.Semaphore) -> str:
    async with sem:                          # waits if 5 are already in flight
        print(f"  [{time.perf_counter()-T0:5.2f}] start {i}")
        await asyncio.sleep(1)
        return f"result-{i}"

async def main():
    sem = asyncio.Semaphore(5)               # at most 5 concurrent
    results = await asyncio.gather(*(call_api(i, sem) for i in range(20)))
    print(len(results), f"{time.perf_counter()-T0:.2f}s")

T0 = time.perf_counter()
asyncio.run(main())
```

All 20 tasks are created at once, but only 5 execute the API call at any moment. Total ≈ 4 s (20 ÷ 5 batches × 1 s). Change the semaphore to 20 and it's 1 s; change it to 1 and it's 20 s. **The semaphore is your throttle dial.**

A reusable helper worth memorizing:

```python
async def gather_limited(coros, limit: int):
    """Run coroutines concurrently, at most `limit` at a time."""
    sem = asyncio.Semaphore(limit)
    async def run(c):
        async with sem:
            return await c
    return await asyncio.gather(*(run(c) for c in coros))
```

> **Note on true rate limits.** A semaphore limits *concurrency* (how many at once), not *rate* (how many per second). If an API allows 60 requests/minute and each takes 0.1 s, a semaphore of 5 will happily send 3,000/minute. For real rate limiting you need a token bucket — you'll build one in Chapter 15.

### 8.4 `asyncio.Event` — one-to-many signalling

An `Event` is a flag tasks can wait on.

```python
import asyncio

async def worker(n: int, ready: asyncio.Event):
    print(f"worker {n} waiting for config")
    await ready.wait()                  # blocks until set()
    print(f"worker {n} running")

async def loader(ready: asyncio.Event):
    await asyncio.sleep(1)
    print("config loaded")
    ready.set()                         # releases ALL waiters at once

async def main():
    ready = asyncio.Event()
    async with asyncio.TaskGroup() as tg:
        for n in range(3):
            tg.create_task(worker(n, ready))
        tg.create_task(loader(ready))

asyncio.run(main())
```

Events are for startup gating, shutdown signalling, and "wake everyone when state X becomes true". `ready.is_set()` checks without waiting; `ready.clear()` resets it.

A standard graceful-shutdown skeleton:

```python
shutdown = asyncio.Event()

async def service_loop():
    while not shutdown.is_set():
        await do_work()

async def stop_after(seconds):
    await asyncio.sleep(seconds)
    shutdown.set()
```

### 8.5 The others, briefly

- **`asyncio.Condition`** — a lock plus the ability to `wait()` for a predicate and `notify()` waiters. Used when waiters need to re-check a condition under the lock. Rare in application code.
- **`asyncio.Barrier`** (3.11+) — blocks N tasks until all N have arrived, then releases them together. Useful for synchronized simulation steps.
- **`asyncio.BoundedSemaphore`** — like `Semaphore` but raises if you release more times than you acquired, catching bugs.

---

### Exercise 20 — Find the race

```python
import asyncio

inventory = {"widget": 1}

async def buy(customer: str):
    if inventory["widget"] > 0:
        await asyncio.sleep(0.05)          # charge the credit card
        inventory["widget"] -= 1
        print(f"{customer}: purchased")
    else:
        print(f"{customer}: sold out")

async def main():
    await asyncio.gather(buy("alice"), buy("bob"), buy("carol"))
    print(inventory)

asyncio.run(main())
```

What happens, and how do you fix it?

**Solution**

```
alice: purchased
bob: purchased
carol: purchased
{'widget': -2}
```

All three pass the stock check before any of them decrements, because the `await` between check and act lets the loop switch tasks. You've sold three widgets and have minus two.

Fix with a lock:

```python
lock = asyncio.Lock()

async def buy(customer: str):
    async with lock:
        if inventory["widget"] > 0:
            await asyncio.sleep(0.05)
            inventory["widget"] -= 1
            print(f"{customer}: purchased")
        else:
            print(f"{customer}: sold out")
```

Now: `alice: purchased`, `bob: sold out`, `carol: sold out`, `{'widget': 0}`.

**Design note:** holding a lock across a slow network call (the card charge) serializes your whole shop. The better production design is to *reserve* the stock under the lock, then charge outside it, then confirm or release the reservation:

```python
async def buy(customer):
    async with lock:                         # fast critical section only
        if inventory["widget"] <= 0:
            print(f"{customer}: sold out"); return
        inventory["widget"] -= 1             # reserve immediately
    try:
        await charge_card(customer)          # slow part, no lock held
    except PaymentError:
        async with lock:
            inventory["widget"] += 1         # give it back
        raise
```
**Keep critical sections short. Never hold a lock across a network call if you can avoid it.**

---

### Exercise 21 — Throttle the fan-out

You must call an API 50 times. Each call takes 0.5 s. The API tolerates at most 8 concurrent connections. Write it, print the elapsed time, and predict that time first.

**Solution**

Prediction: ⌈50/8⌉ = 7 waves × 0.5 s ≈ **3.5 s** (vs 25 s sequential, vs 0.5 s unlimited-but-banned).

```python
import asyncio, time

async def call_api(i: int) -> int:
    await asyncio.sleep(0.5)
    return i * i

async def main():
    sem = asyncio.Semaphore(8)

    async def limited(i: int) -> int:
        async with sem:
            return await call_api(i)

    t0 = time.perf_counter()
    results = await asyncio.gather(*(limited(i) for i in range(50)))
    print(sum(results), f"{time.perf_counter() - t0:.2f}s")

asyncio.run(main())
```

Output: `40425 3.50s`.

Note that we create all 50 coroutines at once, and that's fine — the semaphore controls how many are *inside the API call*, not how many exist. But "fine" has a price worth knowing. Measured with `tracemalloc`, **10,000 pending tasks occupy about 13.9 MB — roughly 1.4 KB each** (the coroutine object itself is small; the frame, the Task wrapper, and the loop's bookkeeping are the rest).

So: 1,000 pending tasks ≈ 1.4 MB, negligible. One million ≈ 1.4 GB, which is an outage. If you're fanning out over an unbounded input — every row of a database, every file in a bucket — don't materialize all the tasks. Stream the work through a bounded queue (Chapter 9) instead, so the number of live tasks stays fixed regardless of input size.

---

### Exercise 22 — Startup gate

Three workers must not process anything until a (slow) config load completes, and must all stop when a shutdown signal fires. Implement with two `Event`s.

**Solution**

```python
import asyncio

async def load_config(ready: asyncio.Event) -> None:
    print("loading config...")
    await asyncio.sleep(1)
    print("config ready")
    ready.set()

async def worker(n: int, ready: asyncio.Event, shutdown: asyncio.Event) -> None:
    await ready.wait()
    count = 0
    while not shutdown.is_set():
        count += 1
        print(f"worker {n} job {count}")
        await asyncio.sleep(0.4)
    print(f"worker {n} stopped after {count} jobs")

async def main():
    ready, shutdown = asyncio.Event(), asyncio.Event()
    async with asyncio.TaskGroup() as tg:
        tg.create_task(load_config(ready))
        for n in range(3):
            tg.create_task(worker(n, ready, shutdown))
        async def stopper():
            await asyncio.sleep(3)
            print("shutdown requested")
            shutdown.set()
        tg.create_task(stopper())

asyncio.run(main())
```

One subtlety worth noticing: a worker sleeping when `shutdown.set()` fires still finishes its sleep before re-checking the flag, so shutdown can lag by up to 0.4 s. For responsive shutdown, race the sleep against the event:

```python
try:
    await asyncio.wait_for(shutdown.wait(), timeout=0.4)
    break                      # shutdown fired
except TimeoutError:
    pass                       # normal 0.4s tick, keep going
```

---

### Exercise 23 — Concurrency vs rate

An API allows **10 requests per second**. Your calls each take 2 s. Someone suggests `Semaphore(10)`. Is that correct?

**Solution**

**No.** A semaphore of 10 with 2-second calls means a new request starts roughly every 0.2 s once the pipeline is full — that's **5 requests/second**, which happens to be under the limit, but only by accident. If latency dropped to 0.05 s, the same semaphore would fire 200 requests/second and get you banned.

Concurrency limit and rate limit are different constraints:
- **Concurrency** = how many are in flight simultaneously → `Semaphore`.
- **Rate** = how many start per unit time → token bucket / leaky bucket.

The relationship is Little's Law: `concurrency ≈ rate × latency`. To hold 10 req/s with 2 s latency you need ~20 in flight. But you should enforce the rate directly rather than inferring it, because latency varies. In production you typically want **both**: a semaphore to cap resource usage and a token bucket to respect the documented rate limit. You'll build the token bucket in §15.4.

---

## 9. Queues, worker pools, and pipelines

### 9.1 Why queues

`gather` and `TaskGroup` assume you know all the work up front. Queues handle the case where work **arrives over time**, or where you want a fixed pool of workers chewing through a backlog with backpressure.

`asyncio.Queue` is the async analogue of `queue.Queue`. Its methods are coroutines:

```python
q = asyncio.Queue(maxsize=100)     # maxsize=0 means unbounded

await q.put(item)       # waits if the queue is full  ← backpressure
item = await q.get()    # waits if the queue is empty
q.put_nowait(item)      # raises QueueFull instead of waiting
q.get_nowait()          # raises QueueEmpty instead of waiting
q.task_done()           # "I finished the item I got"
await q.join()          # wait until every put item has been task_done()'d
q.qsize()               # current size
```

**Backpressure** is why `maxsize` matters. If a producer generates 100,000 items faster than consumers can process them, an unbounded queue eats all your RAM. With `maxsize`, `await q.put()` blocks the producer until there's room — the system self-regulates. Always set a `maxsize` in production.

### 9.2 The worker-pool pattern

This is the single most useful concurrency shape in backend engineering. Learn it cold.

```python
import asyncio, random

async def worker(name: str, queue: asyncio.Queue) -> None:
    while True:
        job = await queue.get()
        try:
            print(f"{name} processing {job}")
            await asyncio.sleep(random.uniform(0.2, 0.6))
            print(f"{name} finished {job}")
        except Exception as e:
            print(f"{name} failed {job}: {e}")      # never let a worker die
        finally:
            queue.task_done()                        # ALWAYS, even on failure

async def main():
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=10)

    workers = [asyncio.create_task(worker(f"w{i}", queue)) for i in range(3)]

    for n in range(10):
        await queue.put(f"job-{n}")

    await queue.join()                 # wait for all work to be done
    print("all jobs complete")

    for w in workers:                  # workers loop forever; shut them down
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

asyncio.run(main())
```

The four parts that make it correct:

1. Workers loop forever on `await queue.get()` — they idle for free when there's nothing to do.
2. `task_done()` is in a `finally`, so a failing job still counts. **Forget this and `queue.join()` hangs forever** — a classic production incident.
3. `await queue.join()` is how you know the work is finished.
4. Explicit cancel-and-await shutdown. `return_exceptions=True` absorbs the `CancelledError`s.

An alternative to cancellation is the **sentinel** pattern — push one `None` per worker and have each worker `break` on it. That's a graceful shutdown that lets in-flight jobs finish naturally:

```python
for _ in workers:
    await queue.put(None)

# in the worker:
job = await queue.get()
if job is None:
    queue.task_done()
    break
```

(Python 3.13 added `Queue.shutdown()`, which makes this cleaner. On 3.12, use a sentinel or cancellation.)

### 9.3 Pipelines

Chain queues to build multi-stage processing — the async version of a Unix pipe. This maps directly onto agent architectures (fetch → parse → summarize → store).

```python
import asyncio

async def producer(out_q: asyncio.Queue, n: int) -> None:
    for i in range(n):
        await asyncio.sleep(0.1)                 # e.g. reading from a stream
        await out_q.put(f"raw-{i}")
    await out_q.put(None)                        # end-of-stream sentinel

async def transformer(in_q: asyncio.Queue, out_q: asyncio.Queue) -> None:
    while True:
        item = await in_q.get()
        if item is None:
            await out_q.put(None)
            in_q.task_done()
            break
        await asyncio.sleep(0.2)                 # e.g. an LLM call
        await out_q.put(item.replace("raw", "clean").upper())
        in_q.task_done()

async def consumer(in_q: asyncio.Queue) -> None:
    while True:
        item = await in_q.get()
        if item is None:
            in_q.task_done()
            break
        print("saved:", item)
        in_q.task_done()

async def main():
    q1: asyncio.Queue = asyncio.Queue(maxsize=5)
    q2: asyncio.Queue = asyncio.Queue(maxsize=5)
    async with asyncio.TaskGroup() as tg:
        tg.create_task(producer(q1, 8))
        tg.create_task(transformer(q1, q2))
        tg.create_task(consumer(q2))

asyncio.run(main())
```

All three stages run concurrently: while the consumer saves item 3, the transformer is working on item 4 and the producer is fetching item 5. Throughput becomes the slowest stage rather than the sum of all stages — and with bounded queues, the fast stages automatically slow down to match.

### 9.4 Priority queues

`asyncio.PriorityQueue` pops the smallest item first, and `asyncio.LifoQueue` pops the newest. Use tuples of `(priority, counter, payload)`; the counter breaks ties so Python never tries to compare the payloads.

```python
import itertools
counter = itertools.count()

pq = asyncio.PriorityQueue()
await pq.put((1, next(counter), "urgent user request"))
await pq.put((5, next(counter), "background reindex"))
```

---

### Exercise 24 — Build a URL crawler pool

20 URLs, 4 workers, each fetch takes between 0.1 and 0.5 s and URLs containing `"bad"` raise. Collect successful results into a shared list and print how many succeeded and failed.

*Hint:* four parts — a bounded queue, N worker tasks looping on `get()`, `await queue.join()`, then cancel-and-await the workers. Where does `task_done()` have to go?

**Solution**

```python
import asyncio, random

async def fetch(url: str) -> str:
    await asyncio.sleep(random.uniform(0.1, 0.5))
    if "bad" in url:
        raise ConnectionError(f"failed: {url}")
    return f"<html>{url}</html>"

async def worker(name, queue, results, errors):
    while True:
        url = await queue.get()
        try:
            results.append(await fetch(url))
        except Exception as e:
            errors.append(str(e))
        finally:
            queue.task_done()

async def main():
    urls = [f"http://site{i}.com" if i % 5 else f"http://bad{i}.com" for i in range(20)]
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=8)
    results: list[str] = []
    errors: list[str] = []

    workers = [asyncio.create_task(worker(f"w{i}", queue, results, errors))
               for i in range(4)]

    for u in urls:
        await queue.put(u)
    await queue.join()

    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

    print(f"{len(results)} ok, {len(errors)} failed")

asyncio.run(main())
```

Appending to a shared list needs no lock: `list.append` contains no `await`, so it's atomic under cooperative scheduling. If you were doing read-modify-write across an `await`, you'd need one.

---

### Exercise 25 — Diagnose the hang

```python
async def worker(q):
    while True:
        item = await q.get()
        result = await process(item)     # sometimes raises
        await save(result)
        q.task_done()

async def main():
    q = asyncio.Queue()
    ws = [asyncio.create_task(worker(q)) for _ in range(3)]
    for i in range(10):
        await q.put(i)
    await q.join()
    print("done")
```

Sometimes this prints "done"; sometimes it hangs forever. Why, and what's the fix?

**Solution**

Two linked bugs.

1. **`task_done()` is not in a `finally`.** If `process(item)` raises, the exception propagates out of the `while` loop, the worker task dies, and `task_done()` is never called for that item. `q.join()` waits for a count that will never reach zero → permanent hang.
2. **Dead workers are never noticed.** After three failures you have zero workers and a queue full of items, with no error printed anywhere — the exceptions are sitting unretrieved inside the dead task objects.

Fixed:

```python
import logging

async def worker(q):
    while True:
        item = await q.get()
        try:
            result = await process(item)
            await save(result)
        except Exception:
            logging.exception("job %r failed", item)    # log, don't die
        finally:
            q.task_done()                               # exactly once per get()
```

Three things this gets right:

- `task_done()` lives in `finally`, so it is called exactly once per `get()` on **every** path — success, failure, and cancellation. Calling it twice is as bad as calling it zero times: `join()` would then return while work is still outstanding.
- `except Exception` does not catch `CancelledError` (which is a `BaseException`), so cancellation still propagates and shuts the worker down cleanly. You do not need a separate `except asyncio.CancelledError` clause here, and adding one that also calls `task_done()` would produce the double-call bug.
- The worker survives a bad job. A worker pool where one poisoned message kills a worker degrades silently until you have zero workers.

**Diagnosis tip:** when an async program hangs, dump the tasks:

```python
for t in asyncio.all_tasks():
    print(t.get_name(), t.get_coro())
    t.print_stack()
```

---

### Exercise 26 — Backpressure demonstration

Write a fast producer (10 items/second) and one slow consumer (1 item/second), with `maxsize=3`. Print the queue size on each put. Then run with `maxsize=0` and compare.

**Solution**

```python
import asyncio, time

async def producer(q: asyncio.Queue, n: int):
    for i in range(n):
        await q.put(i)
        print(f"[{time.perf_counter()-T0:5.2f}] put {i}, qsize={q.qsize()}")
        await asyncio.sleep(0.1)
    await q.put(None)

async def consumer(q: asyncio.Queue):
    while (item := await q.get()) is not None:
        await asyncio.sleep(1.0)
        print(f"[{time.perf_counter()-T0:5.2f}] consumed {item}")

async def main(maxsize: int):
    q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
    async with asyncio.TaskGroup() as tg:
        tg.create_task(producer(q, 10))
        tg.create_task(consumer(q))

T0 = time.perf_counter()
asyncio.run(main(maxsize=3))
```

With `maxsize=3`: the first four puts are instant, then the producer is throttled — each subsequent `put` waits for the consumer to free a slot, so puts settle into a 1-per-second rhythm. Memory stays flat.

With `maxsize=0`: all 10 puts complete within 1 second; the queue balloons while the consumer plods along. With 10 items that's harmless. With a producer reading a 50 GB log file it's an out-of-memory crash.

**`maxsize` is not a performance knob; it's a safety mechanism.** Set it.

---

## 10. Blocking code, threads, and the async boundary

### 10.1 The problem restated

One thread runs everything. A blocking call freezes the world. But real projects are full of synchronous libraries — `requests`, `psycopg2`, `pandas`, `PIL`, `boto3`, and the entire `open()`/filesystem API. You need a bridge.

### 10.2 `asyncio.to_thread` — the everyday tool (3.9+)

```python
import asyncio, time

def blocking_io(name: str) -> str:
    time.sleep(2)                       # a sync library call
    return f"{name} done"

async def main():
    t0 = time.perf_counter()
    results = await asyncio.gather(
        asyncio.to_thread(blocking_io, "a"),
        asyncio.to_thread(blocking_io, "b"),
        asyncio.to_thread(blocking_io, "c"),
    )
    print(results, f"{time.perf_counter()-t0:.2f}s")     # ~2.00s

asyncio.run(main())
```

`await asyncio.to_thread(fn, *args, **kwargs)` runs `fn` in a worker thread from the default `ThreadPoolExecutor` and gives you back an awaitable. The event loop stays free.

**Why threads help here despite the GIL:** the GIL is released during blocking I/O syscalls. So a thread stuck in `time.sleep` or `socket.recv` or a database driver's network read is not holding the GIL, and your event loop thread runs normally. This is why `to_thread` works for I/O and *not* for pure Python computation.

Caveats:
- The default executor has a limited number of threads (`min(32, cpu_count + 4)`). Don't push 10,000 calls through it.
- The function runs in another thread, so any data it touches is subject to real, preemptive thread races. Keep it self-contained.
- **`to_thread` calls cannot be cancelled.** `task.cancel()` will stop you waiting, but the thread keeps running to completion. Design accordingly.

### 10.3 CPU-bound work: `ProcessPoolExecutor`

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

def crunch(n: int) -> int:
    return sum(i * i for i in range(n))      # pure CPU, holds the GIL

async def main():
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as pool:
        results = await asyncio.gather(*(
            loop.run_in_executor(pool, crunch, 10_000_000) for _ in range(4)
        ))
    print(results)

if __name__ == "__main__":       # required on Windows/macOS spawn
    asyncio.run(main())
```

Separate processes mean separate GILs mean genuine parallelism across cores. The cost: arguments and return values must be picklable and get copied between processes, so it's only worth it when the computation is much more expensive than the transfer.

### 10.4 Decision table

| Situation | Tool |
|---|---|
| A native async library exists (`httpx`, `aiohttp`, `asyncpg`, `aiofiles`, `aiosqlite`) | Use it — best option by far |
| Synchronous I/O library you can't replace | `asyncio.to_thread` |
| Short CPU work (< ~10 ms) | Just run it inline; not worth the overhead |
| Long CPU work | `ProcessPoolExecutor` via `run_in_executor` |
| Long CPU work you control the code of | Break it up with periodic `await asyncio.sleep(0)` |
| Running an external program | `asyncio.create_subprocess_exec` |

### 10.5 Yielding inside a long loop

If you must do a long computation on the loop and can't move it, at least let others breathe:

```python
async def process_many(items):
    for i, item in enumerate(items):
        crunch(item)
        if i % 100 == 0:
            await asyncio.sleep(0)     # give the loop a turn
```

This is a stopgap, not a solution — it still consumes 100% of the loop's CPU. Use it when a thread or process is genuinely impractical.

### 10.6 Function colouring: async is contagious

This is the most important *architectural* fact about asyncio, and it is usually discovered the hard way, three days into a migration.

A normal function can call anything. An `async def` can call anything. But a **normal function cannot `await`**. So the moment one deep function needs to become async, every function that calls it must become async, and every function that calls *those*, all the way up to `asyncio.run`.

```python
# before
def load_config(): return parse(read_file("config.yml"))
def build_client(): return Client(load_config())
def handle(req):   return build_client().process(req)

# you switch read_file to an async library... and now:
async def load_config(): return parse(await read_file("config.yml"))
async def build_client(): return Client(await load_config())
async def handle(req):   return (await build_client()).process(req)
```

One leaf change rewrote three signatures, and every call site of all three. People call this the **"coloured functions" problem**: sync and async are two colours, and a red function can't call a blue one without changing colour itself.

Practical consequences:

- **You cannot adopt asyncio in one corner of a codebase.** It spreads along the call graph to the entry point. Budget for that before you start.
- **Libraries must pick a colour, or ship both.** This is why you see `requests` *and* `httpx`, `psycopg2` *and* `asyncpg`.
- **The escape hatches are asymmetric and both cost something.** Going async→sync means `asyncio.to_thread` (a real thread). Going sync→async means `asyncio.run_coroutine_threadsafe` (§10.7) or, badly, `asyncio.run()` in the middle of your program, which cannot be nested and will raise if a loop is already running.

If you see someone reach for `nest_asyncio` to patch around this, it is a symptom: something in the architecture is trying to be both colours at once.

### 10.7 Crossing the boundary: threads and the loop

Sooner or later you will bridge a synchronous SDK into an async program — a vendor client that delivers results via callbacks on its own thread, say. Three rules govern that boundary, and violating them produces bugs that appear only under load.

**Rule 1: `asyncio` primitives are not thread-safe.** `asyncio.Queue`, `asyncio.Event`, and `asyncio.Lock` are designed for a single thread and do no locking. Calling `queue.put_nowait()` from another thread can corrupt the queue's internal state or, more commonly, fail to wake the waiting coroutine at all — the item goes in and nothing notices. Use `loop.call_soon_threadsafe` to schedule the operation onto the loop's thread:

```python
import asyncio, threading, time

async def main():
    loop = asyncio.get_running_loop()
    aq: asyncio.Queue = asyncio.Queue()

    def sdk_callback_thread():
        """A sync SDK delivering events on its own thread."""
        for i in range(5):
            time.sleep(0.05)
            loop.call_soon_threadsafe(aq.put_nowait, f"event-{i}")   # ← correct
        loop.call_soon_threadsafe(aq.put_nowait, None)

    threading.Thread(target=sdk_callback_thread, daemon=True).start()

    while (item := await aq.get()) is not None:
        print("got", item)

asyncio.run(main())
```

If you need a queue that both sides touch directly, use a `queue.Queue` (which *is* thread-safe) and drain it with `await asyncio.to_thread(q.get)`. Likewise, wait on a `threading.Event` with `await asyncio.to_thread(event.wait)` — never call its blocking `.wait()` on the loop.

**Rule 2: to call *into* async from a thread, use `run_coroutine_threadsafe`.** It returns a `concurrent.futures.Future`, not an asyncio one, so `.result()` blocks the calling thread — which is exactly what you want, since that thread isn't the loop.

```python
def sync_worker(loop):
    fut = asyncio.run_coroutine_threadsafe(async_tool(21), loop)
    return fut.result(timeout=2)        # blocks this thread, not the loop
```

**Rule 3: one loop, one thread, and don't mix loops.** An `asyncio.Lock` created on loop A is meaningless to loop B. This is why you create asyncio primitives inside your async setup rather than at module import time.

### 10.8 When *not* to use asyncio

A tutorial that only argues for its subject is advertising. Asyncio is the wrong choice more often than its popularity suggests.

**Use threads (`ThreadPoolExecutor`) instead when:**
- You have a synchronous codebase and a bounded amount of I/O — a script making 20 HTTP calls. `ThreadPoolExecutor(max_workers=20).map(requests.get, urls)` is three lines, needs no rewrite, and is fast enough. Async would force you to recolour the whole program for no measured benefit.
- Your libraries have no async equivalent. Wrapping everything in `to_thread` gives you async syntax over a thread pool — all of the complexity, none of the advantage.
- The team doesn't know asyncio and the deadline is Friday. Threads fail in more familiar ways.

**Use processes instead when:** the work is CPU-bound. Always.

**Use asyncio when:**
- Concurrency is **high** — hundreds or thousands of simultaneous operations, where one OS thread each (roughly 8 MB of stack apiece by default) stops being viable.
- The work is **I/O-dominated**, so there's real idle time to reclaim.
- You need **structured cancellation** — timeouts and budgets that reliably tear down in-flight work. This is genuinely hard with threads, which cannot be killed, and it is the reason agent frameworks choose asyncio even at modest concurrency.
- The ecosystem is already async: FastAPI, WebSockets, LLM SDKs.

**The honest summary:** for an agent that makes 10 concurrent LLM calls with a 30-second budget and clean cancellation, asyncio is clearly right. For a nightly script that scrapes 50 pages, a thread pool is right and asyncio is résumé-driven engineering. Know which one you're building.

---

### Exercise 27 — Fix the frozen server

```python
import asyncio, time, hashlib

async def handle(user_id: int):
    print(f"request {user_id} start")
    pw_hash = hashlib.pbkdf2_hmac("sha256", b"password", b"salt", 600_000)
    db_row = fetch_from_db(user_id)
    print(f"request {user_id} done")
    return pw_hash.hex()[:8], db_row

def fetch_from_db(user_id: int) -> str:
    time.sleep(0.3)
    return f"user-{user_id}"

async def main():
    t0 = time.perf_counter()
    await asyncio.gather(*(handle(i) for i in range(4)))
    print(f"{time.perf_counter()-t0:.2f}s")

asyncio.run(main())
```

Two different blocking problems. Identify and fix each with the right tool.

**Solution**

**Problem 1 — `pbkdf2_hmac` is CPU-bound** (deliberately: 600,000 rounds). A process pool is the safe, general answer for CPU work, so that's what the solution uses.

A caveat on the "threads never help CPU work" rule, because this particular function is a genuine exception: `hashlib`'s key-derivation functions release the GIL while they run, so `to_thread` *would* parallelize this one on a multi-core machine. The same is true of NumPy, `zlib`, image codecs, and most C extensions that do heavy work. The rule you should carry is narrower and more accurate than "threads never help CPU work":

> Threads parallelize **anything that releases the GIL** — blocking I/O and GIL-releasing C extensions. They do not parallelize **pure Python bytecode**.

Since you usually can't be sure which category a dependency falls into, default to a process pool for CPU work and switch to threads only after measuring a real speedup.

**Problem 2 — `fetch_from_db` is blocking I/O.** `to_thread` is exactly right.

```python
import asyncio, time, hashlib
from concurrent.futures import ProcessPoolExecutor

def hash_password() -> bytes:
    return hashlib.pbkdf2_hmac("sha256", b"password", b"salt", 600_000)

def fetch_from_db(user_id: int) -> str:
    time.sleep(0.3)
    return f"user-{user_id}"

async def handle(user_id: int, pool) -> tuple[str, str]:
    loop = asyncio.get_running_loop()
    print(f"request {user_id} start")
    pw_hash, db_row = await asyncio.gather(
        loop.run_in_executor(pool, hash_password),      # CPU → process
        asyncio.to_thread(fetch_from_db, user_id),      # I/O → thread
    )
    print(f"request {user_id} done")
    return pw_hash.hex()[:8], db_row

async def main():
    t0 = time.perf_counter()
    with ProcessPoolExecutor() as pool:
        await asyncio.gather(*(handle(i, pool) for i in range(4)))
    print(f"{time.perf_counter()-t0:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
```

Two lessons: (1) diagnose each blocking call by *type* — CPU or I/O — and route it accordingly; (2) notice we also made the hash and the DB lookup run concurrently *with each other* inside a single request, since they're independent.

---

### Exercise 28 — True or false

(a) `asyncio.to_thread` gives parallel execution of CPU-bound Python code.
(b) Cancelling a task that's awaiting `to_thread` stops the underlying function.
(c) Using `requests.get` inside an `async def` is fine if you `await` something else afterwards.
(d) `ProcessPoolExecutor` can run any function you like.
(e) `await asyncio.sleep(0)` lets other tasks run without a real delay.

**Solution**

(a) **False for pure Python; true for GIL-releasing C code.** The GIL prevents parallel execution of Python *bytecode*, so `to_thread` will not speed up a Python `for` loop. It does parallelize blocking I/O and C extensions that release the GIL (NumPy, hashlib's KDFs, compression libraries).

A related nuance people get wrong in the other direction: putting CPU-bound Python in a thread does **not** freeze the event loop. CPython switches between threads every few milliseconds (the `sys.setswitchinterval` default is 5 ms), so the loop thread still gets scheduled and your server keeps responding. It just doesn't go any faster. `to_thread` on pure-Python CPU work is therefore useless-but-harmless, whereas running that same code *directly on the loop* is actively harmful.

(b) **False.** Cancellation stops you *waiting* for the result; the thread runs to completion regardless. There is no safe way to kill a Python thread. Threads leak if you cancel aggressively — a real consideration in agent systems with tight timeouts.

(c) **False, and this is the most common real-world async bug.** `requests.get` blocks the event loop for the entire duration of the HTTP call, no matter what comes after. Use `httpx.AsyncClient` or `aiohttp`, or wrap it in `to_thread`.

(d) **False.** Arguments and return values must be picklable, and on `spawn` platforms (Windows, macOS) the target must be importable at module level — no lambdas, no closures, no locally-defined functions. You also need the `if __name__ == "__main__":` guard.

(e) **True.** It's the canonical "yield to the scheduler" idiom.

---

### Exercise 29 — Benchmark the three approaches

For `sum(i*i for i in range(5_000_000))` repeated 4 times, measure: (a) sequential, (b) `to_thread`, (c) `ProcessPoolExecutor`. Predict the ranking first.

**Solution**

```python
import asyncio, time
from concurrent.futures import ProcessPoolExecutor

def crunch(n: int = 5_000_000) -> int:
    return sum(i * i for i in range(n))

async def sequential():
    return [crunch() for _ in range(4)]

async def with_threads():
    return await asyncio.gather(*(asyncio.to_thread(crunch) for _ in range(4)))

async def with_processes():
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=4) as pool:
        return await asyncio.gather(*(loop.run_in_executor(pool, crunch) for _ in range(4)))

async def main():
    for name, fn in [("sequential", sequential), ("threads", with_threads),
                     ("processes", with_processes)]:
        t0 = time.perf_counter()
        await fn()
        print(f"{name:12s} {time.perf_counter()-t0:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
```

**Do not trust predicted numbers, including mine — run it.** That is the entire point of the exercise, so here are real measurements rather than plausible-sounding ones.

On a **single-core container** (Python 3.12.3):

```
pure-python seq      1.12s
pure-python threads  1.12s
pure-python procs    1.15s
```

Threads: no gain, as expected. Processes: **also no gain, plus overhead.** With one core there is nothing to parallelize onto, so spawning processes only costs you fork and pickling time.

On a 4-core machine you should instead see roughly `1.1 / 1.1 / 0.35` — processes winning by close to the core count, minus overhead.

Two takeaways, and the second is the one that gets missed:

1. **Threads do not speed up pure-Python CPU work.** Only processes, or GIL-releasing C code, do.
2. **`ProcessPoolExecutor` only helps if you actually have cores.** CI runners, serverless functions, and Kubernetes pods with `cpu: 1` are all single-core in practice even on a 64-core host, because the CPU quota is enforced by the scheduler. Code that got faster on your laptop can get *slower* in production. Check `os.cpu_count()` and, on Linux, `len(os.sched_getaffinity(0))`, which respects CPU affinity masks that `cpu_count()` ignores.

(Footnote: CPython 3.13 shipped an experimental free-threaded build without the GIL, and 3.12 added per-interpreter GILs at the C level. Neither changes the advice for normal application code in 2026 — the interpreter you deploy still has a GIL.)

---

### Exercise 30 — Choose the tool

For each, pick **asyncio**, **threads**, or **processes**, and justify in one sentence.

1. A nightly script that fetches 40 REST endpoints with `requests` and writes a CSV.
2. A chat backend holding 5,000 mostly-idle WebSocket connections.
3. An agent making 8 concurrent LLM calls that must abort cleanly after 20 s.
4. Re-encoding 500 videos with a pure-Python codec.
5. A Django view that calls two internal microservices.
6. A sensor logger polling 200 devices over HTTP every second, on a Raspberry Pi.

**Solution**

1. **Threads.** Bounded work, an existing sync codebase, no async library in play. `ThreadPoolExecutor` is a three-line change; asyncio would mean recolouring the whole script (§10.6) for a job that already finishes in seconds.
2. **Asyncio.** 5,000 OS threads means gigabytes of reserved stack and heavy scheduler churn; 5,000 idle coroutines is a few megabytes (see Exercise 21's measurement). This is the case asyncio was built for.
3. **Asyncio** — and the deciding factor is **cancellation, not speed**. Threads cannot be killed, so a 20-second budget enforced with threads means abandoning work that keeps running and keeps costing money.
4. **Processes.** Purely CPU-bound. (If you shell out to `ffmpeg` instead, `asyncio.create_subprocess_exec` is also fine, because the CPU work happens in another process and you're back to waiting on I/O.)
5. **Follow the framework's colour.** Sync Django → `ThreadPoolExecutor`. Async Django or FastAPI → asyncio with `httpx`. Mixing colours inside one request handler is how you end up blocking an event loop.
6. **Asyncio.** 200 concurrent connections on constrained hardware is exactly where per-thread memory hurts, and the work is pure I/O.

---

### Exercise 31 — Bridge a synchronous SDK

A vendor SDK delivers results by invoking a callback **on its own thread**. Collect 5 results into an async program and process them as they arrive.

```python
import threading, time

def vendor_sdk_subscribe(callback):       # given; runs callback on its own thread
    def run():
        for i in range(5):
            time.sleep(0.05)
            callback(f"result-{i}")
        callback(None)
    threading.Thread(target=run, daemon=True).start()
```

*Hint:* the obvious bridge is to pass `aq.put_nowait` straight in as the callback. Re-read §10.7 Rule 1 before deciding whether that is safe.

**Solution**

The naive bridge — `vendor_sdk_subscribe(aq.put_nowait)` with an `asyncio.Queue` — is **unsafe**. `asyncio.Queue` does no locking, and its wake-up machinery assumes it is only ever touched from the loop's own thread. It will often appear to work in testing and then drop wake-ups under load, leaving a consumer blocked on `get()` while items sit in the queue. That is a miserable bug to diagnose because the data is right there.

```python
import asyncio

async def main():
    loop = asyncio.get_running_loop()
    aq: asyncio.Queue = asyncio.Queue()

    def on_result(item):                          # runs on the SDK's thread
        loop.call_soon_threadsafe(aq.put_nowait, item)

    vendor_sdk_subscribe(on_result)

    while (item := await aq.get()) is not None:
        print("processing", item)
    print("stream complete")

asyncio.run(main())
```

`call_soon_threadsafe` is the **only** loop method safe to call from another thread. It hands the callable to the loop, which runs it on its own thread at the next opportunity, waking the loop if it is idle.

Two alternatives worth knowing:
- Use a thread-safe `queue.Queue` and drain it with `await asyncio.to_thread(q.get)`.
- If the callback itself needs to `await` something, use `asyncio.run_coroutine_threadsafe(coro, loop)`, which returns a `concurrent.futures.Future` whose `.result()` blocks the *calling* thread.

---

---

### ✓ Checkpoint — Part III

You should now be able to:

- Identify a check-then-act race in single-threaded async code, and explain why the `await` in the middle is what creates it.
- Bound concurrency with a `Semaphore` and explain why that is *not* a rate limit.
- Build a worker pool with a bounded queue, and say exactly where `task_done()` must go and why.
- Route a blocking dependency to a thread or a process, and justify which.
- Explain **function colouring** and argue, with a concrete case, for *not* using asyncio.
- Safely deliver data from a non-loop thread into an async program.

# Part IV — The rest of the language surface

## 11. `async with`, `async for`, and async generators

### 11.1 `async with` — context managers that await

A normal context manager has `__enter__` and `__exit__`. An async one has `__aenter__` and `__aexit__`, both coroutines, so setup and teardown can perform I/O.

```python
class DatabaseConnection:
    def __init__(self, dsn: str):
        self.dsn = dsn

    async def __aenter__(self):
        print(f"connecting to {self.dsn}")
        await asyncio.sleep(0.5)             # real handshake
        return self

    async def __aexit__(self, exc_type, exc, tb):
        print("closing connection")
        await asyncio.sleep(0.2)             # graceful close
        return False                          # don't suppress exceptions

    async def query(self, sql: str) -> list[str]:
        await asyncio.sleep(0.1)
        return [f"row for {sql}"]

async def main():
    async with DatabaseConnection("postgres://localhost") as db:
        print(await db.query("SELECT 1"))

asyncio.run(main())
```

The cleaner way, using `contextlib`:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def database(dsn: str):
    conn = await connect(dsn)
    try:
        yield conn
    finally:
        await conn.close()          # runs even on exception or cancellation

async def main():
    async with database("postgres://localhost") as db:
        ...
```

You'll use `async with` constantly with HTTP clients, database pools, locks, semaphores, timeouts, and `TaskGroup`. The `try/finally` around the `yield` is what makes it cancellation-safe — that's not optional.

`contextlib.AsyncExitStack` handles a dynamic number of them:

```python
from contextlib import AsyncExitStack

async with AsyncExitStack() as stack:
    clients = [await stack.enter_async_context(make_client(u)) for u in urls]
    # all clients cleanly closed on exit, in reverse order
```

### 11.2 `async for` and async iterators

An async iterator defines `__aiter__` and `__anext__` (a coroutine), and signals the end with `StopAsyncIteration`. You consume it with `async for`.

This is how you handle **data that arrives over time**: streaming HTTP responses, paginated APIs, WebSocket messages, and — crucially — LLM token streams.

```python
class Paginator:
    """Fetches pages lazily; the caller never sees the pagination."""
    def __init__(self, total_pages: int):
        self.total_pages = total_pages
        self.page = 0

    def __aiter__(self):
        return self

    async def __anext__(self) -> list[str]:
        if self.page >= self.total_pages:
            raise StopAsyncIteration
        await asyncio.sleep(0.3)              # fetch the next page
        self.page += 1
        return [f"item-{self.page}-{i}" for i in range(3)]

async def main():
    async for page in Paginator(4):
        print(page)
```

### 11.3 Async generators — much nicer

Writing `__anext__` by hand is tedious. An `async def` containing `yield` is an **async generator**, and it does all of that for you:

```python
async def paginate(total_pages: int):
    for page in range(1, total_pages + 1):
        await asyncio.sleep(0.3)
        yield [f"item-{page}-{i}" for i in range(3)]

async def main():
    async for items in paginate(4):
        print(items)
```

Async generators are the natural way to express streaming, and they compose beautifully:

```python
async def token_stream(text: str):
    """Simulates an LLM streaming tokens."""
    for word in text.split():
        await asyncio.sleep(0.05)
        yield word + " "

async def uppercase(stream):
    async for token in stream:
        yield token.upper()

async def take(stream, n: int):
    count = 0
    async for item in stream:
        if count >= n:
            break
        yield item
        count += 1

async def main():
    async for tok in take(uppercase(token_stream("the quick brown fox jumps")), 3):
        print(tok, end="", flush=True)
    print()
```

**One important caution:** async generators need their `aclose()` called to run their cleanup code, and that happens via garbage collection at unpredictable times. If your generator holds a resource, wrap it:

```python
from contextlib import aclosing

async with aclosing(paginate(4)) as pages:
    async for page in pages:
        ...
```

### 11.4 Async comprehensions

```python
results = [x async for x in paginate(3)]
doubled = [x * 2 async for x in numbers() if x % 2 == 0]
total   = sum([x async for x in numbers()])
```

Note `async for` inside the comprehension, and that these still run **sequentially** — a comprehension over an async iterator is not concurrent. For concurrency you still need tasks.

---

### Exercise 32 — Write a rate-limited async generator

Write `throttled(items, per_second)` that yields items no faster than `per_second` per second, regardless of how fast the consumer asks.

*Hint:* don't sleep a fixed interval each iteration — that drifts, because the consumer's own work adds up. Track the *next allowed emission time* and sleep until it. Use `loop.time()`, not `time.time()`.

**Solution**

```python
import asyncio, time

async def throttled(items, per_second: float):
    interval = 1.0 / per_second
    loop = asyncio.get_running_loop()
    next_time = loop.time()
    for item in items:
        now = loop.time()
        if now < next_time:
            await asyncio.sleep(next_time - now)
        next_time = max(now, next_time) + interval
        yield item

async def main():
    t0 = time.perf_counter()
    async for x in throttled(range(6), per_second=4):
        print(f"[{time.perf_counter()-t0:4.2f}] {x}")

asyncio.run(main())
```

Items emerge roughly every 0.25 s. Two details of craft: using `loop.time()` (a monotonic clock, immune to system clock changes) rather than `time.time()`, and scheduling against `next_time` rather than sleeping a fixed interval — the latter would drift, since the consumer's own processing time adds up.

---

### Exercise 33 — Async context manager with timing

Write `@asynccontextmanager`-based `timed(label)` that prints how long the block took, including when the block raises.

**Solution**

```python
import asyncio, time
from contextlib import asynccontextmanager

@asynccontextmanager
async def timed(label: str):
    t0 = time.perf_counter()
    status = "ok"
    try:
        yield
    except asyncio.CancelledError:
        status = "cancelled"
        raise
    except Exception as e:
        status = f"failed ({type(e).__name__})"
        raise
    finally:
        print(f"{label}: {time.perf_counter() - t0:.3f}s [{status}]")

async def main():
    async with timed("fast op"):
        await asyncio.sleep(0.2)
    try:
        async with timed("broken op"):
            await asyncio.sleep(0.1)
            raise ValueError("nope")
    except ValueError:
        pass

asyncio.run(main())
```

The `finally` is what makes the timing correct on every path. This tiny utility is genuinely useful — drop it around LLM calls to see where an agent's latency actually goes.

---

### Exercise 34 — Merge two async streams

Write `merge(*streams)` that yields items from several async generators as they arrive, interleaved by arrival time, finishing when all are exhausted.

*Hint:* you can't `async for` over two generators at once. Drain each into a shared `asyncio.Queue` with one task per stream, and push a unique sentinel when each finishes so you know when all are done.

**Solution**

```python
import asyncio

async def ticker(name: str, interval: float, count: int):
    for i in range(count):
        await asyncio.sleep(interval)
        yield f"{name}-{i}"

async def merge(*streams):
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()

    async def drain(stream):
        try:
            async for item in stream:
                await queue.put(item)
        finally:
            await queue.put(SENTINEL)

    tasks = [asyncio.create_task(drain(s)) for s in streams]
    try:
        remaining = len(streams)
        while remaining:
            item = await queue.get()
            if item is SENTINEL:
                remaining -= 1
            else:
                yield item
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

async def main():
    async for item in merge(ticker("fast", 0.1, 5), ticker("slow", 0.35, 3)):
        print(item)

asyncio.run(main())
```

Output interleaves by timing: `fast-0, fast-1, fast-2, slow-0, fast-3, ...`

The queue-plus-sentinel-per-stream structure is the standard way to fan several async streams into one. You'd use it to merge token streams from several models running concurrently, or to combine an LLM stream with a cancellation channel.

---

## 12. Networking with asyncio streams

You'll usually use a library (FastAPI, `httpx`, `aiohttp`, `websockets`), but understanding the primitives demystifies all of them — and asyncio's stream API is genuinely pleasant.

### 12.1 A TCP echo server

```python
import asyncio

async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    addr = writer.get_extra_info("peername")
    print(f"connected: {addr}")
    try:
        while data := await reader.readline():
            message = data.decode().rstrip()
            if message == "quit":
                break
            writer.write(f"echo: {message}\n".encode())
            await writer.drain()              # respect backpressure
    except asyncio.CancelledError:
        raise
    finally:
        print(f"closing: {addr}")
        writer.close()
        await writer.wait_closed()

async def main():
    server = await asyncio.start_server(handle_client, "127.0.0.1", 8888)
    async with server:
        print("serving on 127.0.0.1:8888")
        await server.serve_forever()

asyncio.run(main())
```

`start_server` spawns a new task per connection automatically. One process, one thread, thousands of simultaneous clients.

`await writer.drain()` is the part people skip and regret. `writer.write()` buffers without blocking; if you write faster than the client reads, the buffer grows without bound. `drain()` waits until the buffer drains below a watermark — it's **backpressure for sockets**, the same idea as `Queue(maxsize=...)`.

### 12.2 The matching client

```python
import asyncio

async def client(name: str):
    reader, writer = await asyncio.open_connection("127.0.0.1", 8888)
    try:
        for i in range(3):
            writer.write(f"{name} message {i}\n".encode())
            await writer.drain()
            response = await reader.readline()
            print(response.decode().rstrip())
            await asyncio.sleep(0.2)
        writer.write(b"quit\n")
        await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()

async def main():
    await asyncio.gather(*(client(f"client-{i}") for i in range(5)))

asyncio.run(main())
```

Run the server in one terminal and the client in another: five clients are served concurrently by a single-threaded server.

### 12.3 Subprocesses

```python
async def run_command(cmd: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    return proc.returncode, stdout.decode()

async def main():
    results = await asyncio.gather(
        run_command("sleep 1 && echo one"),
        run_command("sleep 1 && echo two"),
    )
    print(results)       # ~1 second total
```

This matters for agents: running shell commands, linters, tests, or sandboxed code as *tools* should use `create_subprocess_exec`, not `subprocess.run`, or you block the loop for the duration of the command.

---

### Exercise 35 — A tiny chat broadcast server

Extend the echo server so every message is broadcast to all connected clients. Handle disconnects without crashing.

**Solution**

```python
import asyncio

clients: set[asyncio.StreamWriter] = set()

async def broadcast(message: str, sender: asyncio.StreamWriter) -> None:
    dead = []
    for w in clients:
        if w is sender:
            continue
        try:
            w.write(message.encode())
            await w.drain()
        except (ConnectionResetError, BrokenPipeError):
            dead.append(w)
    for w in dead:
        clients.discard(w)

async def handle_client(reader, writer):
    addr = writer.get_extra_info("peername")
    clients.add(writer)
    await broadcast(f"*** {addr} joined ({len(clients)} online)\n", writer)
    try:
        while data := await reader.readline():
            text = data.decode().rstrip()
            if text == "quit":
                break
            await broadcast(f"[{addr[1]}] {text}\n", writer)
    except (ConnectionResetError, asyncio.IncompleteReadError):
        pass
    finally:
        clients.discard(writer)
        await broadcast(f"*** {addr} left\n", writer)
        writer.close()
        try:
            await writer.wait_closed()
        except ConnectionResetError:
            pass

async def main():
    server = await asyncio.start_server(handle_client, "127.0.0.1", 8888)
    async with server:
        print("chat server on 127.0.0.1:8888 — connect with: nc 127.0.0.1 8888")
        await server.serve_forever()

asyncio.run(main())
```

Test with several `nc 127.0.0.1 8888` terminals. The instructive parts: collecting dead writers during iteration rather than mutating the set while looping over it, and tolerating exceptions during close — networks break at inconvenient moments and a server that crashes on a client's rude disconnect is not a server.

---

## 13. Testing, debugging, and production hygiene

### 13.1 Testing with the standard library

`unittest.IsolatedAsyncioTestCase` needs no dependencies:

```python
import unittest, asyncio

async def fetch_user(uid: int) -> dict:
    await asyncio.sleep(0.01)
    if uid < 0:
        raise ValueError("bad id")
    return {"id": uid, "name": f"user{uid}"}

class TestFetchUser(unittest.IsolatedAsyncioTestCase):
    async def test_returns_user(self):
        self.assertEqual((await fetch_user(1))["name"], "user1")

    async def test_raises_on_bad_id(self):
        with self.assertRaises(ValueError):
            await fetch_user(-1)

    async def test_concurrent(self):
        users = await asyncio.gather(*(fetch_user(i) for i in range(5)))
        self.assertEqual(len(users), 5)

    async def test_timeout(self):
        async def slow():
            await asyncio.sleep(10)
        with self.assertRaises(TimeoutError):
            async with asyncio.timeout(0.05):
                await slow()

if __name__ == "__main__":
    unittest.main()
```

### 13.2 Testing with pytest

In industry you'll almost certainly use `pytest-asyncio` (`pip install pytest-asyncio`):

```python
# pytest.ini  ->  [pytest]\nasyncio_mode = auto
import pytest, asyncio

@pytest.mark.asyncio
async def test_fetch_user():
    assert (await fetch_user(1))["id"] == 1

@pytest.fixture
async def client():
    c = await make_client()
    yield c
    await c.close()
```

Two practical habits:
- **Mock async functions with `unittest.mock.AsyncMock`**, not `MagicMock` — a `MagicMock` returns a non-awaitable and you'll get `TypeError: object MagicMock can't be used in 'await' expression`.
- **Don't sleep in tests.** A suite that honestly waits out every timeout takes minutes and gets deleted by your teammates.

That second habit needs a technique, not just an instruction. Three, in order of preference:

**(a) Shrink the constants.** Make timeouts and delays parameters with production defaults, then pass tiny values in tests.

```python
async def fetch_with_retry(fetcher, *, attempts=3, base_delay=1.0): ...

await fetch_with_retry(flaky, base_delay=0.001)      # 1000x faster test
```

**(b) Synchronize on events, not on sleeps.** `await asyncio.sleep(0.5)` in a test is a guess; under CI load it becomes a flaky failure. Wait for the thing you actually care about:

```python
started = asyncio.Event()

async def worker():
    started.set()
    await do_work()

task = asyncio.create_task(worker())
await started.wait()          # deterministic, and as fast as the machine allows
task.cancel()
```

**(c) Inject the clock.** For logic that reasons about time — token buckets, circuit breakers, deadlines — take `now()` as a dependency instead of calling `loop.time()` directly. Then a test can advance time instantly:

```python
class TokenBucket:
    def __init__(self, rate, *, now=None):
        self._now = now or (lambda: asyncio.get_running_loop().time())

# in the test:
fake = 0.0
bucket = TokenBucket(rate=5, now=lambda: fake)
fake += 10.0                  # ten seconds pass, instantly
```

Rule of thumb: if a test contains `asyncio.sleep` with a number larger than about 0.01, it is either slow, flaky, or both.

### 13.3 Debug mode

```python
asyncio.run(main(), debug=True)
```
or `PYTHONASYNCIODEBUG=1 python app.py`. Debug mode:
- warns about coroutines that were never awaited,
- logs callbacks that took longer than 100 ms (`loop.slow_callback_duration`) — **this is your blocking-call detector**,
- logs tasks destroyed while still pending,
- gives better tracebacks.

Turn it on whenever an async program behaves strangely.

### 13.4 Inspecting a running program

```python
async def dump_tasks():
    for t in asyncio.all_tasks():
        print(f"{t.get_name():20s} done={t.done()} coro={t.get_coro().__qualname__}")

# and to see where a hung task is stuck:
task.print_stack()
```

`asyncio.current_task()` returns the task you're inside. Naming tasks (`create_task(coro, name="llm-call-3")`) turns incomprehensible dumps into readable ones — do it for anything long-lived.

### 13.5 Logging and `contextvars`

Each Task gets a copy of the current context, so `contextvars` follow your logical flow of control the way `threading.local` follows a thread. This is how you attach a request or trace ID to every log line inside an agent run:

```python
import contextvars, logging

request_id = contextvars.ContextVar("request_id", default="-")

class RequestFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id.get()
        return True

async def handle(rid: str):
    request_id.set(rid)            # visible to every coroutine this one awaits
    await do_work()                # logs here carry rid automatically
```

Note the copy semantics: a child task inherits the value at creation time, and changes it makes afterwards do **not** propagate back to the parent.

### 13.6 Graceful shutdown

Real services must finish in-flight work when Kubernetes sends SIGTERM.

```python
import asyncio, signal

async def main():
    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown.set)      # Unix only

    async with asyncio.TaskGroup() as tg:
        tg.create_task(service_loop(shutdown))
        tg.create_task(metrics_loop(shutdown))

    print("drained and exiting")

asyncio.run(main())
```

Note `loop.add_signal_handler` is not available on Windows; there, use `signal.signal` with a thread-safe `loop.call_soon_threadsafe(shutdown.set)`.

### 13.7 Performance notes

- **`uvloop`** (`pip install uvloop`) is a drop-in Cython event loop, typically 2–4× faster than the default on Linux/macOS. One line: `uvloop.install()` or `asyncio.run(main(), loop_factory=uvloop.new_event_loop)`. Not available on Windows.
- **Eager task factory** (3.12+): `loop.set_task_factory(asyncio.eager_task_factory)` starts tasks synchronously up to their first real suspension, which removes a scheduling round-trip for tasks that finish quickly (cache hits, for instance). Measure before adopting.
- **Don't create tasks you don't need.** `await coro()` is much cheaper than `await asyncio.create_task(coro())` when there's no concurrency to gain.
- Profile with `py-spy` or `yappi`; standard `cProfile` output is hard to read for async code.

---

### Exercise 36 — Write the tests

Write `IsolatedAsyncioTestCase` tests for this function covering: success, retry-then-success, and exhausted retries.

```python
async def fetch_with_retry(fetcher, attempts: int = 3) -> str:
    last = None
    for i in range(attempts):
        try:
            return await fetcher()
        except ConnectionError as e:
            last = e
            await asyncio.sleep(0.01 * 2 ** i)
    raise last
```

*Hint:* you don't need a mocking library. A closure over a `nonlocal` counter makes a perfectly good async fake. Remember to test the case that should **not** be retried.

**Solution**

```python
import unittest, asyncio

class TestFetchWithRetry(unittest.IsolatedAsyncioTestCase):

    async def test_success_first_try(self):
        calls = 0
        async def fetcher():
            nonlocal calls
            calls += 1
            return "ok"
        self.assertEqual(await fetch_with_retry(fetcher), "ok")
        self.assertEqual(calls, 1)

    async def test_succeeds_after_failures(self):
        calls = 0
        async def fetcher():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise ConnectionError("flaky")
            return "ok"
        self.assertEqual(await fetch_with_retry(fetcher), "ok")
        self.assertEqual(calls, 3)

    async def test_gives_up(self):
        calls = 0
        async def fetcher():
            nonlocal calls
            calls += 1
            raise ConnectionError("always down")
        with self.assertRaises(ConnectionError):
            await fetch_with_retry(fetcher, attempts=3)
        self.assertEqual(calls, 3)

    async def test_other_errors_not_retried(self):
        calls = 0
        async def fetcher():
            nonlocal calls
            calls += 1
            raise ValueError("bad input")
        with self.assertRaises(ValueError):
            await fetch_with_retry(fetcher)
        self.assertEqual(calls, 1)      # ValueError is not retryable

if __name__ == "__main__":
    unittest.main()
```

Closures over `nonlocal calls` make excellent lightweight async fakes — no mocking library required. The fourth test is the one students forget to write, and it's the one that catches the bug where someone broadens `except ConnectionError` to `except Exception` and starts retrying genuine programming errors 3 times.

---

### Exercise 37 — Find the blocking call with debug mode

```python
import asyncio, time

async def handler(n):
    await asyncio.sleep(0.01)
    result = sum(i**2 for i in range(3_000_000))    # sneaky
    await asyncio.sleep(0.01)
    return result

async def main():
    await asyncio.gather(*(handler(i) for i in range(5)))

asyncio.run(main())
```

Use debug mode to prove where the stall is, then fix it.

**Solution**

Run with `asyncio.run(main(), debug=True)` and you'll see repeated warnings like:

```
Executing <Task ... handler() ...> took 0.412 seconds
```

The loop's slow-callback detector (default threshold 100 ms) flags the CPU-bound `sum()` — this is the fastest way to find blocking code in an unfamiliar codebase. Lower the threshold to catch subtler offenders:

```python
loop = asyncio.get_running_loop()
loop.slow_callback_duration = 0.05
```

Fix, since it's genuine CPU work:

```python
from concurrent.futures import ProcessPoolExecutor

def crunch() -> int:
    return sum(i**2 for i in range(3_000_000))

async def handler(n, pool):
    await asyncio.sleep(0.01)
    result = await asyncio.get_running_loop().run_in_executor(pool, crunch)
    await asyncio.sleep(0.01)
    return result

async def main():
    with ProcessPoolExecutor() as pool:
        await asyncio.gather(*(handler(i, pool) for i in range(5)))

if __name__ == "__main__":
    asyncio.run(main(), debug=True)
```

---

### 13.8 Resource lifecycle: one client per process

Here is the single most common reason a team migrates to asyncio and sees *no speedup at all*:

```python
# WRONG — a new client, and a new connection pool, on every call
async def fetch(url):
    async with httpx.AsyncClient() as client:
        return await client.get(url)
```

Every call builds a fresh connection pool, opens a new TCP connection, performs a new TLS handshake (one or two extra round trips), makes one request, and throws the pool away. For a 30 ms API you may be paying 150 ms of setup. Run a hundred of these concurrently and you also open a hundred sockets, which is how you meet your file-descriptor limit.

Create the client **once**, at application startup, and share it:

```python
import contextlib, httpx

class Api:
    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
        return self

    async def __aexit__(self, *exc):
        await self._client.aclose()

    async def get(self, url: str):
        return await self._client.get(url)


async def main():
    async with Api() as api:                  # one pool for the whole program
        await asyncio.gather(*(api.get(u) for u in urls))
```

In FastAPI this belongs in the `lifespan` handler; in a worker, in the startup path. The general principle covers database pools, Redis clients, and LLM SDK clients equally: **expensive, poolable resources are created once per process and closed on shutdown.**

Note the two limits set explicitly. `max_connections` is another concurrency cap — and if you also have a `Semaphore(200)` in front of a client limited to 100 connections, the semaphore is not doing what you think. Keep your limits consistent and in one place.

### 13.9 When it hangs: a diagnostic procedure

Async programs fail by hanging more often than by crashing. Work through this in order.

**1. Is anything still running?** Dump the task list.

```python
async def dump():
    for t in asyncio.all_tasks():
        print(f"{t.get_name():25s} done={t.done()}")
        t.print_stack(limit=5)
```

Every hang is either "a task is stuck at an `await` that never completes" or "a task that should exist doesn't." The stack tells you which `await` — that is usually the whole diagnosis.

**2. Is the loop blocked?** Run with `debug=True` and watch for `Executing <Task…> took N seconds`. If the loop is blocked, no task will progress and the stacks will look frozen at arbitrary points.

**3. Check the classic deadlocks**, in rough order of frequency:

| Symptom | Cause |
|---|---|
| `await queue.join()` never returns | `task_done()` missed on some path — usually an exception before it. Put it in `finally`. |
| `join()` hangs and no work is happening | Every worker has died. Their exceptions are sitting unretrieved inside dead task objects. |
| `TaskGroup` never exits | A child swallows `CancelledError` (§6.5), or a child awaits something that never completes. |
| Whole program hangs at shutdown | A non-daemon thread from `to_thread` is still running; threads cannot be cancelled. |
| Two coroutines each hang on a lock | Lock-ordering deadlock — A holds `x` wants `y`, B holds `y` wants `x`. Always acquire locks in a fixed global order. |
| `await` on a task that never gets scheduled | You awaited a coroutine from inside itself, or created a cycle of awaits. |

**4. Self-deadlock.** This one is specific to asyncio and catches people:

```python
async def main():
    task = asyncio.create_task(worker())
    await task            # fine

async def worker():
    await asyncio.current_task()      # DEADLOCK: waiting for itself
```

The same happens more subtly when a coroutine awaits a `Future` that only its own later code would resolve.

**5. Reproduce it small.** Async hangs are usually timing-dependent. Shrink concurrency to 2, add `print` statements around every `await` in the suspect path, and run with `debug=True`. A hang with two tasks is a puzzle; a hang with two hundred is a lottery.

---

## 14. The anti-pattern catalogue

A checklist to run against your own code, and an excellent source of code-review comments.

**1. Sequential awaits on independent work**
```python
a = await fetch_user(); b = await fetch_orders()          # ✗ 2× slower
a, b = await asyncio.gather(fetch_user(), fetch_orders()) # ✓
```

**2. `create_task` immediately awaited in a loop** — sequential in disguise; see the "why is this still slow?" exercise in §4.

**3. Blocking calls in async functions** — `time.sleep`, `requests`, `open().read()`, heavy loops. Debug mode finds them.

**4. Fire-and-forget without a reference** — the task may be garbage-collected. Use `TaskGroup`, or keep a set.

**5. Swallowing `CancelledError`** — always re-raise.

**6. No timeouts on network calls** — every external call needs one.

**7. `gather` when you meant `TaskGroup`** — leaves siblings running after a failure.

**8. Unbounded concurrency** — `gather` over 10,000 URLs will exhaust file descriptors and get you rate-limited. Add a `Semaphore`.

**9. Unbounded queues** — set `maxsize` for backpressure.

**10. `task_done()` outside `finally`** — hangs `queue.join()` forever.

**11. Calling `asyncio.run()` more than once, or nested** — one per process, at the top.

**12. Reaching for `nest_asyncio`** — a monkey-patch that lets you nest event loops. It's a sign your architecture is wrong. In notebooks, just `await` directly.

**13. `async def` on functions that never await** — usually pure overhead and API noise; if there's no `await` in the body, make it a normal function. *Legitimate exceptions:* implementing an async protocol or abstract method whose other implementations do await; a test double or in-memory stub standing in for something async; and a public API you intend to make async later, where changing the signature would be a breaking change (§10.6 — colour is part of your interface).

**14. Sharing a client/session per request** — create one `httpx.AsyncClient` or connection pool for the process and reuse it. Creating one per call destroys connection pooling and is often the real cause of "async didn't make it faster."

**15. Mixing loops across threads** — call `loop.call_soon_threadsafe()` or `asyncio.run_coroutine_threadsafe()` to touch a loop from another thread. Anything else is a race.

**16. Holding a lock across a network call** — serializes the system. Keep critical sections short.

---

---

### Exercise 38 — Code review

This is the exercise closest to your actual first job. A teammate opens a pull request with the module below. Review it against Chapter 14 and write the comments you would leave. There are **nine** distinct defects.

```python
import asyncio, time, requests

results = []
lock = asyncio.Lock()

async def fetch_page(url):
    resp = requests.get(url)
    return resp.text

async def process(url):
    async with lock:
        html = await fetch_page(url)
        time.sleep(0.1)
        results.append(html)

async def run_all(urls):
    for url in urls:
        task = asyncio.create_task(process(url))
        await task
    asyncio.create_task(send_metrics(len(results)))
    print("done")

async def worker(queue):
    while True:
        item = await queue.get()
        try:
            await handle(item)
        except asyncio.CancelledError:
            print("stopping")
        queue.task_done()

def main(urls):
    q = asyncio.Queue()
    asyncio.run(run_all(urls))
    asyncio.run(drain(q))
```

*Hint:* work top to bottom and check each function against the checklist — blocking calls, concurrency shape, cancellation, task references, resource limits, and entry points.

**Solution**

1. **`requests.get` inside `async def`** (anti-pattern 3). Blocks the event loop for the whole HTTP call, which defeats the entire design. Use `httpx.AsyncClient`, or `await asyncio.to_thread(requests.get, url)` if you're stuck with it. Note that `fetch_page` also has no `await` in it at all (anti-pattern 13) — the `async` keyword is decoration over synchronous code.

2. **No shared client** (anti-pattern 14). Even after switching to `httpx`, a per-call client would rebuild the connection pool every time (§13.8). Create one client at startup and pass it in.

3. **No timeout on the network call** (anti-pattern 6). Every external call needs one.

4. **`time.sleep(0.1)` on the loop** (anti-pattern 3). Should be `await asyncio.sleep(0.1)`.

5. **A lock held across network I/O** (anti-pattern 16, §8.2). This serializes every URL — the program is now strictly sequential, and slower than the sync version it replaced. The lock isn't needed at all here: `results.append` contains no `await`, so it is already atomic under cooperative scheduling.

6. **`create_task` immediately awaited in a loop** (anti-pattern 2, §4 Exercise 7). Sequential in disguise. Use a `TaskGroup`, with a `Semaphore` to bound concurrency (anti-pattern 8), since `urls` may be long.

7. **Fire-and-forget `send_metrics` with no reference** (anti-pattern 4, §4.3). May be garbage-collected mid-flight; will certainly be cancelled when `asyncio.run` returns a moment later. Either await it or put it in the `TaskGroup`. Relatedly, `print("done")` is only truthful once the tasks are actually awaited (§4, Exercise 9).

8. **`worker` swallows `CancelledError`** (anti-pattern 5, §6.5). It prints and falls through to `queue.task_done()`, then loops forever — this worker cannot be shut down, and it will hang `TaskGroup` exit and process shutdown. Add `raise`. Separately, `task_done()` is outside a `try/finally` (anti-pattern 10), so any exception from `handle` skips it and `queue.join()` hangs forever.

9. **Two `asyncio.run` calls, and an `asyncio.Queue` created outside a loop** (anti-pattern 11, §8.2). Each `run` creates and destroys a separate event loop, so `q` — built before any loop exists and used in the second one — belongs to neither. Use one `asyncio.run(main())` as the single entry point and create the queue inside it.

A rewritten version:

```python
import asyncio, httpx

async def fetch_page(client: httpx.AsyncClient, url: str) -> str:
    resp = await client.get(url, timeout=10.0)
    resp.raise_for_status()
    return resp.text

async def run_all(urls: list[str], *, concurrency: int = 10) -> list[str]:
    results: list[str] = []
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        async def process(url: str) -> None:
            async with sem:
                results.append(await fetch_page(client, url))

        async with asyncio.TaskGroup() as tg:
            for url in urls:
                tg.create_task(process(url))
            tg.create_task(send_metrics(len(urls)))

    print(f"done: {len(results)} pages")
    return results

async def worker(queue: asyncio.Queue) -> None:
    while True:
        item = await queue.get()
        try:
            await handle(item)
        except Exception:
            logging.exception("failed on %r", item)
        finally:
            queue.task_done()

if __name__ == "__main__":
    asyncio.run(run_all(URLS))
```

If you found six or more, you can review async code. If you found all nine, including the lock-serialization and the two-loops bug, you are ahead of most working engineers.

---

### ✓ Checkpoint — Part IV

You should now be able to:

- Write an async context manager and an async generator, and say why `aclose()` matters.
- Write tests for async code without sleeping through real timeouts.
- Find a blocking call in an unfamiliar codebase using debug mode.
- Work through a hang systematically instead of guessing.
- Review someone else's async code against a checklist and name the defects.

# Part V — Agentic AI

## 15. Async patterns for agentic AI

### 15.1 Why every agent framework is async

An "agentic" system is a program that calls a language model in a loop: the model proposes an action, your code executes it, the result goes back to the model, repeat. The profile of that workload:

| Operation | Typical latency | CPU used |
|---|---|---|
| LLM completion | 1–30 s | ~0 |
| LLM streaming, first token | 0.2–2 s | ~0 |
| Web search tool | 0.3–3 s | ~0 |
| Vector DB query | 20–200 ms | ~0 |
| SQL query | 5–500 ms | ~0 |
| Your orchestration logic | < 1 ms | some |

Your code accounts for well under 1% of wall-clock time. Everything else is waiting. A sequential agent doing 8 LLM calls at 4 seconds each takes 32 seconds; if 6 of those are independent, an async agent takes about 12. When you're serving 500 users, sequential is not merely slower — it's a different cost structure, because you need 500× the machines.

This chapter builds the patterns that appear in every production agent system. Everything runs with no API key.

### 15.1b How to run this chapter's code

The rest of the tutorial is copy-one-block-and-run. This chapter is different: it builds a small library incrementally, because that's how real agent code is organized and because repeating 60 lines of retry logic in every example would obscure the point.

You will create **two files** in one directory and add to them as you read:

| File | Contents | Introduced in |
|---|---|---|
| `mockllm.py` | `MockLLM` and its exception types | §15.2 |
| `agentlib.py` | `ToolCall`, `TOOLS`, `run_tool`, `dispatch`, `with_retry`, `TokenBucket`, `ThrottledLLM` | §15.3–15.5 |

Every later block in this chapter begins with the imports it needs, so you can see exactly what it depends on. Where a block says `from agentlib import ...`, that name was defined in an earlier section and should already be in your file. If you'd rather not build it up, the capstone solution in Chapter 16 is a **single self-contained file** with everything inlined — read that instead.

### 15.2 The mock LLM client

Save this as `mockllm.py`; later sections import it.

```python
# mockllm.py
import asyncio
import random
from dataclasses import dataclass, field


class RateLimitError(Exception):
    """429 — back off and retry."""

class ModelOverloadedError(Exception):
    """529/503 — transient, retry."""

class InvalidRequestError(Exception):
    """400 — your fault, do NOT retry."""


@dataclass
class MockLLM:
    """A stand-in for a real LLM client, with realistic latency and failures."""
    name: str = "mock-sonnet"
    latency: tuple[float, float] = (0.4, 1.4)
    failure_rate: float = 0.0
    seed: int | None = None
    calls: int = field(default=0, init=False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    async def complete(self, prompt: str, *, max_tokens: int = 256) -> str:
        self.calls += 1
        if not prompt.strip():
            raise InvalidRequestError("empty prompt")
        await asyncio.sleep(self._rng.uniform(*self.latency))
        if self._rng.random() < self.failure_rate:
            raise self._rng.choice([RateLimitError, ModelOverloadedError])(
                f"{self.name} transient failure"
            )
        head = prompt.strip().splitlines()[-1][:50]
        return f"[{self.name}] response to: {head}"

    async def stream(self, prompt: str):
        """Async generator of token strings, like a real streaming endpoint."""
        self.calls += 1
        await asyncio.sleep(self._rng.uniform(*self.latency) * 0.25)   # time-to-first-token
        words = f"Considering the request about {prompt.strip()[:40]} , " \
                f"the answer is that this depends on several factors .".split()
        for w in words:
            await asyncio.sleep(0.05)
            yield w + " "
```

`failure_rate`, `latency`, and `seed` let you reproduce the bad days on demand. Building a fake client like this *before* you build the agent is a professional habit: it makes your tests fast, free, and deterministic.

### 15.3 Concurrent tool calls — the highest-value optimization

When a model returns several tool calls in one turn, they are usually independent. Running them sequentially is the most common performance mistake in agent code.

```python
# agentlib.py  (part 1 of 3 — tool dispatch)
import asyncio
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict


async def web_search(query: str) -> str:
    await asyncio.sleep(1.2)
    return f"3 results for {query!r}"

async def vector_search(query: str, k: int = 5) -> str:
    await asyncio.sleep(0.4)
    return f"top-{k} chunks for {query!r}"

async def sql_query(table: str) -> str:
    await asyncio.sleep(0.8)
    return f"142 rows from {table}"

async def flaky_api(endpoint: str) -> str:
    await asyncio.sleep(0.3)
    raise ConnectionError(f"{endpoint} unreachable")

async def very_slow(x: str) -> str:
    await asyncio.sleep(30)
    return "eventually"


TOOLS = {
    "web_search": web_search,
    "vector_search": vector_search,
    "sql_query": sql_query,
    "flaky_api": flaky_api,
    "very_slow": very_slow,
}


async def run_tool(call: ToolCall, *, timeout: float = 5.0) -> dict:
    """Execute one tool. NEVER raises — failures become messages for the model."""
    fn = TOOLS.get(call.name)
    if fn is None:
        return {"tool": call.name, "ok": False,
                "error": f"unknown tool {call.name!r}; available: {sorted(TOOLS)}"}
    try:
        async with asyncio.timeout(timeout):
            return {"tool": call.name, "ok": True, "result": await fn(**call.args)}
    except TimeoutError:
        return {"tool": call.name, "ok": False,
                "error": f"tool timed out after {timeout}s"}
    except Exception as e:
        return {"tool": call.name, "ok": False,
                "error": f"{type(e).__name__}: {e}"}


async def dispatch(calls: list[ToolCall], *, timeout: float = 5.0) -> list[dict]:
    """Run all tool calls concurrently, preserving input order."""
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(run_tool(c, timeout=timeout)) for c in calls]
    return [t.result() for t in tasks]


# ── demo; guarded so `import agentlib` does not run it ──────────────
async def _demo():
    import time
    calls = [
        ToolCall("web_search", {"query": "asyncio best practices"}),
        ToolCall("vector_search", {"query": "internal docs", "k": 3}),
        ToolCall("sql_query", {"table": "orders"}),
        ToolCall("flaky_api", {"endpoint": "/v1/legacy"}),
        ToolCall("nonexistent", {}),
        ToolCall("very_slow", {"x": "hi"}),
    ]
    t0 = time.perf_counter()
    for r in await dispatch(calls):
        print(r)
    print(f"{time.perf_counter()-t0:.2f}s")


if __name__ == "__main__":
    asyncio.run(_demo())
```

Run it with `python agentlib.py`. The `if __name__ == "__main__"` guard matters here: without it, every later `from agentlib import dispatch` would run this five-second demo as a side effect of the import.

Roughly 5 seconds (bounded by the tool timeout), versus 32+ sequential. Three design decisions worth internalizing:

1. **`run_tool` never raises.** A failed tool becomes `{"ok": False, "error": ...}` that goes back into the conversation, so the model can try a different approach. An exception would kill the whole turn. This is the difference between an agent that recovers and one that crashes.
2. **Per-tool timeouts.** One hung tool must not hold the turn hostage.
3. **`TaskGroup` with ordered results.** Since `run_tool` swallows normal exceptions, the group never tears down; reading `t.result()` in order re-associates results with the calls that produced them.

### 15.4 Rate limiting: semaphore + token bucket

Providers limit both concurrent requests and requests (or tokens) per minute. You need both controls.

```python
# agentlib.py  (part 2 of 3 — rate limiting)
import asyncio


class TokenBucket:
    """Classic token bucket: `rate` tokens accrue per second, up to `capacity`."""

    def __init__(self, rate: float, capacity: float | None = None):
        self.rate = rate
        self.capacity = capacity if capacity is not None else rate
        self._tokens = self.capacity
        self._updated: float | None = None
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        if tokens > self.capacity:
            raise ValueError("request larger than bucket capacity")
        async with self._lock:                 # serialize waiters → FIFO fairness
            loop = asyncio.get_running_loop()
            if self._updated is None:
                self._updated = loop.time()
            while True:
                now = loop.time()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._updated) * self.rate
                )
                self._updated = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                await asyncio.sleep((tokens - self._tokens) / self.rate)


class ThrottledLLM:
    """Wraps a client with a concurrency cap AND a request-rate cap."""

    def __init__(self, llm, *, max_concurrent: int = 8, requests_per_second: float = 5):
        self._llm = llm
        self._sem = asyncio.Semaphore(max_concurrent)
        self._bucket = TokenBucket(requests_per_second, capacity=requests_per_second)

    async def complete(self, prompt: str, **kw) -> str:
        await self._bucket.acquire()           # rate limit
        async with self._sem:                  # concurrency limit
            return await self._llm.complete(prompt, **kw)
```

Why both, restated from §8.3: the semaphore caps how many requests are *in flight* (protecting sockets, memory, and the provider's concurrency limit); the bucket caps how many *start per second* (protecting the documented rate limit). Neither substitutes for the other.

For token-based limits (e.g. 200,000 input tokens/minute), use the same bucket with `await bucket.acquire(estimated_tokens)`.

### 15.5 Retries with exponential backoff and jitter

```python
# agentlib.py  (part 3 of 3 — retries)
import asyncio
import random
from mockllm import RateLimitError, ModelOverloadedError, InvalidRequestError

RETRYABLE = (RateLimitError, ModelOverloadedError, TimeoutError, ConnectionError)


async def with_retry(
    make_coro,                    # zero-arg callable returning a NEW coroutine
    *,
    attempts: int = 5,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    retry_on: tuple = RETRYABLE,
):
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await make_coro()
        except asyncio.CancelledError:
            raise                                     # never retry a cancellation
        except retry_on as e:
            last_exc = e
            if attempt == attempts - 1:
                break
            delay = min(max_delay, base_delay * (2 ** attempt))
            delay *= 0.5 + random.random() / 2        # full-ish jitter: 50–100%
            print(f"  attempt {attempt+1} failed ({type(e).__name__}); "
                  f"retrying in {delay:.2f}s")
            await asyncio.sleep(delay)
    raise last_exc
```

Use it as:

```python
answer = await with_retry(lambda: llm.complete("summarize this"))
```

Four things this gets right, each of which is a bug if you get it wrong:

- **`make_coro` is a factory, not a coroutine.** A coroutine object can only be awaited once; retrying a used one raises `RuntimeError: cannot reuse already awaited coroutine`. This catches almost everyone the first time.
- **Exponential backoff.** 0.5, 1, 2, 4, 8 s. Retrying immediately against a rate limiter makes things worse.
- **Jitter.** Without randomness, 200 clients that all failed at the same instant retry at the same instant — the *thundering herd*. Jitter spreads them out. This is not a nicety; it's the difference between recovering and staying down.
- **Selective retries.** `InvalidRequestError` isn't in `RETRYABLE`, so a malformed prompt fails fast instead of burning five attempts. Retrying non-transient errors wastes money and hides bugs.

Related: **circuit breakers.** If a dependency has failed 10 times in a row, stop calling it for 30 seconds and fail fast instead. You'll build one in the last exercise of this chapter.

### 15.6 Streaming

Streaming changes perceived latency enormously: first token at 0.3 s feels instant even if the full answer takes 8 s.

```python
import asyncio
from mockllm import MockLLM


async def stream_with_idle_timeout(llm, prompt: str, *, idle_timeout: float = 2.0,
                                   total_timeout: float = 30.0) -> str:
    """Stream tokens, failing if the stream stalls OR if the whole thing runs long."""
    chunks: list[str] = []
    loop = asyncio.get_running_loop()
    hard_deadline = loop.time() + total_timeout

    stream = llm.stream(prompt)
    try:
        async with asyncio.timeout(min(idle_timeout, total_timeout)) as cm:
            async for token in stream:
                chunks.append(token)
                print(token, end="", flush=True)
                # reset the idle clock on every token, but never past the hard deadline
                cm.reschedule(min(loop.time() + idle_timeout, hard_deadline))
    except TimeoutError:
        print("\n[stream stalled or exceeded budget]")
    finally:
        await stream.aclose()
    print()
    return "".join(chunks)


async def main():
    await stream_with_idle_timeout(MockLLM(seed=1), "explain backpressure")

asyncio.run(main())
```

The **idle timeout** is the right primitive for streams. A total timeout alone either kills legitimately long answers or lets a dead connection hang for the full budget. `cm.reschedule()` gives you "fail if no token for 2 seconds," which is what you actually mean, and the `min(..., hard_deadline)` keeps an overall cap.

Fanning a stream out to several consumers (the UI, a logger, a moderation filter) is the stream-merge exercise from §11 run in reverse:

```python
async def tee(source, *queues: asyncio.Queue):
    try:
        async for item in source:
            for q in queues:
                await q.put(item)
    finally:
        for q in queues:
            await q.put(None)
```

### 15.7 The agent loop, with a budget

Here is a complete, runnable single-agent loop containing everything so far: a time budget, retries, concurrent tool calls, per-tool timeouts, and graceful degradation.

```python
import asyncio
from dataclasses import dataclass, field

from mockllm import MockLLM
from agentlib import ToolCall, dispatch, with_retry


@dataclass
class AgentResult:
    answer: str
    steps: int
    llm_calls: int
    elapsed: float
    truncated: bool = False
    transcript: list[str] = field(default_factory=list)


def decide_tools(step: int) -> list[ToolCall]:
    """Stands in for parsing the model's tool_use blocks."""
    plans = {
        0: [ToolCall("web_search", {"query": "market size 2026"}),
            ToolCall("vector_search", {"query": "internal forecasts", "k": 3})],
        1: [ToolCall("sql_query", {"table": "revenue"}),
            ToolCall("flaky_api", {"endpoint": "/v1/pricing"})],
    }
    return plans.get(step, [])


async def run_agent(goal: str, llm, *, max_steps: int = 4,
                    budget_seconds: float = 20.0) -> AgentResult:
    loop = asyncio.get_running_loop()
    start = loop.time()
    deadline = start + budget_seconds
    transcript = [f"GOAL: {goal}"]
    truncated = False
    step = 0

    try:
        async with asyncio.timeout_at(deadline):          # hard budget for the turn
            for step in range(max_steps):
                thought = await with_retry(
                    lambda: llm.complete("\n".join(transcript)))
                transcript.append(f"THOUGHT[{step}]: {thought}")

                calls = decide_tools(step)
                if not calls:
                    break

                results = await dispatch(calls, timeout=3.0)
                for r in results:
                    line = (f"OBSERVATION[{step}] {r['tool']}: "
                            f"{r.get('result') or 'ERROR ' + r['error']}")
                    transcript.append(line)

            final = await with_retry(
                lambda: llm.complete("\n".join(transcript) + "\nFINAL ANSWER:"))
    except TimeoutError:
        truncated = True
        final = ("Budget exhausted. Partial findings:\n  "
                 + "\n  ".join(transcript[-4:]))

    return AgentResult(
        answer=final, steps=step + 1, llm_calls=llm.calls,
        elapsed=loop.time() - start, truncated=truncated, transcript=transcript,
    )


async def main():
    llm = MockLLM(seed=7, failure_rate=0.25)
    res = await run_agent("Estimate the 2026 market size", llm, budget_seconds=20)
    print("\n".join(res.transcript))
    print("-" * 60)
    print(res.answer)
    print(f"steps={res.steps} llm_calls={res.llm_calls} "
          f"elapsed={res.elapsed:.2f}s truncated={res.truncated}")

asyncio.run(main())
```

Note the shape: **one `asyncio.timeout_at` wrapping the whole turn**, with per-tool timeouts nested inside. When the outer budget blows, everything in flight is cancelled and you return a partial answer rather than an exception. Users strongly prefer "here's what I found in 20 seconds" to a spinner followed by a stack trace.

### 15.8 Fan-out / fan-in: map-reduce over a corpus

Summarize 40 documents and combine the summaries — the classic RAG-adjacent workload.

```python
import asyncio

from mockllm import MockLLM
from agentlib import with_retry


async def summarize_corpus(docs: list[str], llm, *, concurrency: int = 6) -> str:
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def summarize_one(i: int, doc: str) -> tuple[int, str | None]:
        nonlocal done
        async with sem:
            try:
                async with asyncio.timeout(10):
                    out = await with_retry(lambda: llm.complete(f"Summarize: {doc}"))
                return i, out
            except Exception as e:                 # TimeoutError is an Exception too
                print(f"\n  doc {i} failed: {type(e).__name__}: {e}")
                return i, None
            finally:
                done += 1
                print(f"\r  {done}/{len(docs)} summarized", end="", flush=True)

    pairs = await asyncio.gather(*(summarize_one(i, d) for i, d in enumerate(docs)))
    print()

    summaries = [s for _, s in sorted(pairs) if s is not None]
    failed = len(docs) - len(summaries)
    if failed:
        print(f"  ({failed} documents failed and were skipped)")

    # reduce step — for very large corpora, do this hierarchically
    joined = "\n".join(f"- {s}" for s in summaries)
    return await with_retry(lambda: llm.complete(f"Combine these summaries:\n{joined}"))


async def main():
    docs = [f"document number {i} about topic {i % 7}" for i in range(40)]
    print(await summarize_corpus(docs, MockLLM(seed=3, failure_rate=0.1)))

asyncio.run(main())
```

Four production details embedded here:

- **Bounded concurrency** — `Semaphore(6)`, not 40 simultaneous API calls.
- **Partial failure tolerated** — one bad document doesn't lose the other 39.
- **Order preserved** — return the index alongside the summary and sort. With `gather` the order is already preserved, but being explicit survives a later refactor to `as_completed`.
- **Progress reporting** — the `nonlocal done` counter. On a 40-document job users need feedback.

For a corpus of thousands, make the reduce step hierarchical: summarize in groups of 20, then summarize the summaries, recursively, until one fits the context window.

### 15.9 Multi-agent orchestration

Two shapes cover most cases.

**Parallel specialists (fan-out), then synthesis:**

```python
async def panel(question: str, llm) -> str:
    roles = ["economist", "engineer", "lawyer", "ethicist"]
    async with asyncio.TaskGroup() as tg:
        tasks = {
            role: tg.create_task(
                with_retry(lambda r=role: llm.complete(f"As a {r}, answer: {question}"))
            )
            for role in roles
        }
    views = "\n".join(f"{role}: {t.result()}" for role, t in tasks.items())
    return await llm.complete(f"Synthesize these expert views:\n{views}")
```

Note `lambda r=role:` — binding the loop variable as a default argument. Without it, every lambda would close over the *final* value of `role`. This is a classic Python gotcha, made more painful by async because the lambdas execute later.

**Pipeline of agents (queues):** researcher → writer → critic, all running concurrently on a stream of topics.

```python
import asyncio

from mockllm import MockLLM

SENTINEL = None

async def researcher(topics, out_q, llm):
    for topic in topics:
        notes = await llm.complete(f"Research notes on {topic}")
        await out_q.put((topic, notes))
    await out_q.put(SENTINEL)

async def writer(in_q, out_q, llm):
    while (item := await in_q.get()) is not SENTINEL:
        topic, notes = item
        draft = await llm.complete(f"Write a paragraph on {topic} using {notes}")
        await out_q.put((topic, draft))
    await out_q.put(SENTINEL)

async def critic(in_q, results, llm):
    while (item := await in_q.get()) is not SENTINEL:
        topic, draft = item
        review = await llm.complete(f"Critique this draft: {draft}")
        results.append((topic, draft, review))
        print(f"  finished: {topic}")

async def main():
    llm = MockLLM(seed=11, latency=(0.2, 0.5))
    q1: asyncio.Queue = asyncio.Queue(maxsize=3)
    q2: asyncio.Queue = asyncio.Queue(maxsize=3)
    results: list = []
    topics = ["solar storage", "grid inertia", "demand response", "hydrogen"]

    async with asyncio.TaskGroup() as tg:
        tg.create_task(researcher(topics, q1, llm))
        tg.create_task(writer(q1, q2, llm))
        tg.create_task(critic(q2, results, llm))

    print(f"{len(results)} topics completed, {llm.calls} LLM calls")

asyncio.run(main())
```

While the critic reviews topic 1, the writer drafts topic 2 and the researcher investigates topic 3. Throughput is set by the slowest stage rather than the sum of all three — and `maxsize=3` means a slow critic automatically slows the researcher instead of piling up unbounded work.

### 15.10 Hedged requests

When tail latency matters more than cost, send a duplicate request after a delay and take whichever answers first.

```python
async def hedged(make_coro, *, hedge_after: float = 2.0, max_hedges: int = 1):
    tasks = {asyncio.create_task(make_coro())}
    try:
        for _ in range(max_hedges + 1):
            done, pending = await asyncio.wait(
                tasks, timeout=hedge_after, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                if t.exception() is None:
                    return t.result()
            tasks = pending
            if not tasks:
                raise RuntimeError("all hedged attempts failed")
            tasks.add(asyncio.create_task(make_coro()))     # fire a backup
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        return next(iter(done)).result()
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
```

The trade-off is explicit: you pay for extra calls to cut p99 latency. Only hedge **idempotent** operations — never hedge a request that charges a card or sends an email.

### 15.11 Cancellation and human-in-the-loop

Users close tabs. Approvals get denied. Your agent must stop *promptly* and clean up.

```python
async def agent_with_cancel(goal: str, llm, cancel: asyncio.Event):
    agent_task = asyncio.create_task(run_agent(goal, llm), name="agent")
    cancel_task = asyncio.create_task(cancel.wait(), name="cancel-watch")

    done, pending = await asyncio.wait(
        {agent_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED)

    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    if agent_task in done:
        return agent_task.result()
    return "cancelled by user"
```

Because `run_agent` is built from `async with`, `TaskGroup`, and `try/finally`, cancellation propagates all the way down: in-flight LLM calls are cancelled, tool timeouts are cleaned up, and connections are closed. **That's the payoff for writing cancellation-correct code throughout** — you get responsive cancellation for free at the top.

For human approval gates, an `Event` per pending decision plus an `asyncio.timeout` gives you "wait up to 5 minutes for approval, then abort":

```python
async def request_approval(action: str, approvals: dict[str, asyncio.Event]) -> bool:
    ev = approvals.setdefault(action, asyncio.Event())
    try:
        async with asyncio.timeout(300):
            await ev.wait()
        return True
    except TimeoutError:
        return False
```

### 15.12 Production checklist

- [ ] Every external call has a timeout; the whole agent turn has a budget.
- [ ] Retries use exponential backoff **with jitter**, and only on transient errors.
- [ ] Concurrency is bounded by a semaphore; rate is bounded by a token bucket.
- [ ] Tool failures become messages to the model, not exceptions.
- [ ] One shared HTTP client / connection pool per process, not one per request.
- [ ] `TaskGroup` for anything that must succeed together; no fire-and-forget without references.
- [ ] `CancelledError` is always re-raised; cleanup lives in `finally`.
- [ ] Bounded queues everywhere (backpressure).
- [ ] Structured logs carry a trace ID via `contextvars`.
- [ ] Graceful shutdown on SIGTERM drains in-flight work.
- [ ] Debug mode and `slow_callback_duration` used in staging to catch blocking calls.
- [ ] No synchronous SDK calls (`requests`, `openai` sync client, sync DB drivers) inside `async def` — or they're wrapped in `to_thread`.
- [ ] Cost and token usage tracked per turn; a runaway loop has a hard step cap.

---

### Exercise 39 — Make the tool dispatch concurrent

```python
async def handle_turn(model_output):
    results = []
    for call in model_output["tool_calls"]:
        results.append(await TOOLS[call["name"]](**call["args"]))
    return results
```

Three tools take 1.2 s, 0.4 s, 0.8 s. Rewrite it so it takes ~1.2 s, tolerates individual failures, and gives each tool a 3-second timeout.

*Hint:* the per-call wrapper is where timeouts and error-trapping live; the `TaskGroup` just runs the wrappers. Ask yourself what must be true for the group never to tear itself down.

**Solution**

```python
async def handle_turn(model_output, *, timeout: float = 3.0) -> list[dict]:
    async def run(call: dict) -> dict:
        fn = TOOLS.get(call["name"])
        if fn is None:
            return {"tool": call["name"], "ok": False, "error": "unknown tool"}
        try:
            async with asyncio.timeout(timeout):
                return {"tool": call["name"], "ok": True,
                        "result": await fn(**call["args"])}
        except TimeoutError:
            return {"tool": call["name"], "ok": False, "error": "timeout"}
        except Exception as e:
            return {"tool": call["name"], "ok": False,
                    "error": f"{type(e).__name__}: {e}"}

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(run(c)) for c in model_output["tool_calls"]]
    return [t.result() for t in tasks]
```

2.4 s → 1.2 s. The gain scales: with 6 tool calls averaging 1 s you go from 6 s to ~1 s, and that difference compounds across every step of a multi-step agent.

---

### Exercise 40 — Retries + rate limits together

Call `MockLLM(failure_rate=0.4)` 20 times with: at most 4 concurrent, up to 4 retry attempts with jittered exponential backoff, and a 10-second overall budget. Report successes, failures, and total attempts.

*Hint:* think carefully about whether the backoff sleep happens *inside* or *outside* the semaphore. One of the two collapses your throughput exactly when the system is already struggling.

**Solution**

```python
import asyncio, random, time
from mockllm import MockLLM, RateLimitError, ModelOverloadedError

async def main():
    llm = MockLLM(seed=5, failure_rate=0.4, latency=(0.1, 0.3))
    sem = asyncio.Semaphore(4)
    attempts_total = 0

    async def one(i: int) -> str | None:
        nonlocal attempts_total
        for attempt in range(4):
            async with sem:
                attempts_total += 1
                try:
                    return await llm.complete(f"prompt {i}")
                except (RateLimitError, ModelOverloadedError):
                    pass
            # back off OUTSIDE the semaphore so we don't hold a slot while sleeping
            delay = min(4.0, 0.2 * 2 ** attempt) * (0.5 + random.random() / 2)
            await asyncio.sleep(delay)
        return None

    t0 = time.perf_counter()
    try:
        async with asyncio.timeout(10):
            results = await asyncio.gather(*(one(i) for i in range(20)))
    except TimeoutError:
        print("budget exhausted")
        return

    ok = sum(r is not None for r in results)
    print(f"{ok} ok, {20-ok} failed, {attempts_total} total attempts, "
          f"{time.perf_counter()-t0:.2f}s")

asyncio.run(main())
```

The detail that separates this from a naive implementation: **the backoff sleep happens outside the `async with sem` block.** If you sleep while holding a semaphore slot, a failing request blocks a healthy one from starting — your throughput collapses exactly when the system is already under stress. Hold the limited resource only while actually using it.

---

### Exercise 41 — Build and verify a token bucket

Use the `TokenBucket` from §15.4 to enforce 5 requests/second across 20 concurrent tasks. Print each acquisition timestamp and verify no 1-second window contains more than 5.

**Solution**

```python
import asyncio, time

async def main():
    bucket = TokenBucket(rate=5, capacity=5)
    stamps: list[float] = []
    t0 = time.perf_counter()

    async def caller(i: int):
        await bucket.acquire()
        stamps.append(time.perf_counter() - t0)
        print(f"[{stamps[-1]:5.2f}] request {i:2d}")

    await asyncio.gather(*(caller(i) for i in range(20)))

    # verify: no 1-second sliding window holds more than `capacity` acquisitions
    worst = max(sum(1 for s in stamps if t <= s < t + 1.0) for t in stamps)
    print(f"max requests in any 1s window: {worst}")

asyncio.run(main())
```

You'll see the first 5 fire instantly (the bucket starts full), then a steady one-every-0.2-seconds cadence, finishing at ~3.0 s.

**Now look at the verification output — it prints 9, not 5.** This is the most instructive part of the exercise, and it surprises almost everyone. The window `[0, 1)` contains the initial burst of 5 *plus* the four that accrue at 0.2, 0.4, 0.6 and 0.8 s. A full bucket permits a burst of `capacity` **on top of** the sustained rate, so the worst-case count in any one-second sliding window is about `capacity + rate`, not `rate`.

Consequences you must design around:

- If the provider enforces a strict **sliding** window of N requests/second, set `capacity + rate ≤ N`. For N=5, that means something like `TokenBucket(rate=2.5, capacity=2.5)`.
- If the provider uses a **fixed** window (resets on the second boundary), bursts are more forgiving — but you don't usually get to know which one they use.
- Either way, configure for roughly 80% of the documented limit and watch for 429s.

"5 per second" is ambiguous until you say whether the window is fixed or sliding and whether bursts are allowed. Measuring your limiter rather than trusting it — exactly what the verification line in this exercise does — is a habit worth keeping.

---

### Exercise 42 — Streaming with a stall detector

Write `robust_stream(llm, prompt)` that: prints tokens as they arrive; aborts if no token arrives for 1.5 s; aborts if the total exceeds 10 s; and always returns whatever text it managed to collect.

*Hint:* `asyncio.timeout` returns a context object with a `.reschedule()` method. Compute the new deadline on each token, but never let it exceed the hard cap. Don't forget `aclose()`.

**Solution**

```python
import asyncio
from mockllm import MockLLM

async def robust_stream(llm, prompt: str, *, idle: float = 1.5,
                        total: float = 10.0) -> tuple[str, str]:
    """Returns (text, status) where status is 'complete' | 'stalled' | 'overran'."""
    loop = asyncio.get_running_loop()
    hard = loop.time() + total
    chunks: list[str] = []
    status = "complete"
    stream = llm.stream(prompt)
    try:
        async with asyncio.timeout(min(idle, total)) as cm:
            async for tok in stream:
                chunks.append(tok)
                print(tok, end="", flush=True)
                now = loop.time()
                if now >= hard:
                    status = "overran"
                    break
                cm.reschedule(min(now + idle, hard))
    except TimeoutError:
        status = "overran" if loop.time() >= hard - 0.01 else "stalled"
    finally:
        await stream.aclose()
    print(f"\n[{status}, {len(''.join(chunks))} chars]")
    return "".join(chunks), status

async def main():
    await robust_stream(MockLLM(seed=2), "explain token buckets")
    # a stalling stream:
    class Stalling(MockLLM):
        async def stream(self, prompt):
            yield "start "
            await asyncio.sleep(99)
            yield "never "
    await robust_stream(Stalling(), "this will stall")

asyncio.run(main())
```

Two things to notice. `await stream.aclose()` in the `finally` matters: abandoning an async generator leaves its cleanup to the garbage collector, which with a real HTTP client means a socket held open for an unpredictable amount of time. And the partial text is still returned — in a chat UI you show what arrived and let the user retry, rather than discarding it.

---

### Exercise 43 — Hierarchical summarization

Summarize 100 documents where the reduce step can only handle 10 summaries at a time. Build a summarization *tree*: batches of 10 → 10 intermediate summaries → 1 final. Bound concurrency at 8.

*Hint:* a `while len(level) > 1` loop where each pass replaces `level` with the summaries of its batches. Each pass is concurrent; only the passes are sequential.

**Solution**

```python
import asyncio

from mockllm import MockLLM
from agentlib import with_retry

async def summarize_hierarchically(docs: list[str], llm, *,
                                   batch: int = 10, concurrency: int = 8) -> str:
    sem = asyncio.Semaphore(concurrency)

    async def call(prompt: str) -> str:
        async with sem:
            return await with_retry(lambda: llm.complete(prompt))

    level = docs
    depth = 0
    while len(level) > 1:
        groups = [level[i:i + batch] for i in range(0, len(level), batch)]
        print(f"  level {depth}: {len(level)} items -> {len(groups)} groups")
        level = await asyncio.gather(*(
            call("Summarize these:\n" + "\n".join(f"- {t}" for t in g))
            for g in groups
        ))
        depth += 1
    return level[0]

async def main():
    docs = [f"document {i}" for i in range(100)]
    llm = MockLLM(seed=4, latency=(0.1, 0.3))
    print(await summarize_hierarchically(docs, llm))
    print(f"{llm.calls} LLM calls")

asyncio.run(main())
```

```
  level 0: 100 items -> 10 groups
  level 1: 10 items -> 1 groups
11 LLM calls
```

Count the calls: 10 at the first level plus 1 at the second. Each level is fully concurrent; only the *levels* are sequential. Depth is `log_batch(n)`, so 10,000 documents is four levels — roughly 500 + 50 + 5 + 1 calls, finishing in four waves rather than 10,000 sequential calls. This is how production RAG and "summarize my entire Drive" features are built, and the pattern generalizes to any associative reduce (voting, ranking, deduplication).

---

### Exercise 44 — Model racing with fallback

You have a fast-but-weak model and a slow-but-strong one. Return the strong model's answer if it arrives within 3 s; otherwise return the fast model's answer. Never wait more than 3 s, and don't leave tasks running.

*Hint:* launch both *before* you start waiting — otherwise the fallback only begins after you've already spent the patience budget. The `finally` block is not optional.

**Solution**

```python
import asyncio
from mockllm import MockLLM, RateLimitError, ModelOverloadedError

async def race_with_fallback(prompt: str, strong, fast, *, patience: float = 3.0) -> tuple[str, str]:
    strong_task = asyncio.create_task(strong.complete(prompt), name="strong")
    fast_task = asyncio.create_task(fast.complete(prompt), name="fast")
    try:
        try:
            async with asyncio.timeout(patience):
                return await strong_task, "strong"
        except (TimeoutError, ConnectionError, RateLimitError, ModelOverloadedError):
            return await fast_task, "fast (fallback)"
    finally:
        for t in (strong_task, fast_task):
            t.cancel()
        await asyncio.gather(strong_task, fast_task, return_exceptions=True)

async def main():
    strong = MockLLM("opus-like", latency=(4.0, 5.0))
    fast   = MockLLM("haiku-like", latency=(0.3, 0.6))
    answer, source = await race_with_fallback("hard question", strong, fast)
    print(f"{source}: {answer}")

asyncio.run(main())
```

Both models are launched immediately — that's the point. Waiting 3 s for the strong model *and then* starting the fast one would take 3.5 s; launching both means the fallback is already sitting there, finished, when the patience expires.

Two caveats worth stating in a design review: you pay for both calls on every request, and the `finally` cancellation is what stops the strong model from continuing to burn tokens after you've given up on it. If you'd rather not pay double, use the hedging pattern from §15.10 and only launch the fast model *after* the patience elapses.

---

### Exercise 45 — Budget-aware agent

Extend `run_agent` so it tracks an estimated **cost** budget in addition to time: each LLM call costs 1.0 units, each tool call 0.1. Stop when either budget is exhausted, and report which one ran out.

*Hint:* two independent kill-switches. Time is enforced by `asyncio.timeout_at`; cost has to be checked explicitly at the top of each step.

**Solution**

```python
import asyncio
from dataclasses import dataclass

@dataclass
class Budget:
    seconds: float
    units: float
    spent: float = 0.0

    def charge(self, amount: float) -> None:
        self.spent += amount

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.units

async def run_agent_budgeted(goal: str, llm, budget: Budget, *, max_steps: int = 6):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget.seconds
    transcript = [f"GOAL: {goal}"]
    reason = "completed"

    try:
        async with asyncio.timeout_at(deadline):
            for step in range(max_steps):
                if budget.exhausted:
                    reason = "cost budget exhausted"
                    break
                budget.charge(1.0)
                transcript.append(await with_retry(
                    lambda: llm.complete("\n".join(transcript))))

                calls = decide_tools(step)
                if not calls:
                    break
                budget.charge(0.1 * len(calls))
                for r in await dispatch(calls, timeout=3.0):
                    transcript.append(f"OBS {r['tool']}: {r.get('result') or r['error']}")
            else:
                reason = "step limit reached"
    except TimeoutError:
        reason = "time budget exhausted"

    return {"transcript": transcript, "reason": reason,
            "spent": round(budget.spent, 2),
            "elapsed": round(loop.time() - deadline + budget.seconds, 2)}

async def main():
    from mockllm import MockLLM
    out = await run_agent_budgeted("analyse the market",
                                   MockLLM(seed=9, latency=(0.3, 0.8)),
                                   Budget(seconds=15, units=4.0))
    print(out["reason"], "| spent:", out["spent"], "| elapsed:", out["elapsed"])

asyncio.run(main())
```

Two independent kill-switches — wall-clock and spend — is standard practice. Time budgets protect the *user experience*; cost budgets protect the *invoice*. An agent that loops on a tool that keeps failing will happily spend $400 overnight without the second one. Add a third in real systems: a hard step cap, which you have here as `max_steps`.

---

### Exercise 46 — Circuit breaker

Implement `CircuitBreaker` with three states: **closed** (calls pass through), **open** (calls fail immediately), **half-open** (one trial call allowed). Open after 3 consecutive failures; after 5 s go half-open; a successful trial closes it, a failed trial re-opens it. Make it safe under concurrency.

*Hint:* the state machine is small; the concurrency is the hard part. Which lines need the lock, and — more importantly — which line must **not** be inside it?

**Solution**

```python
import asyncio

class CircuitOpenError(Exception):
    pass

class CircuitBreaker:
    def __init__(self, *, failure_threshold: int = 3, recovery_time: float = 5.0):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = asyncio.Lock()
        self._trial = False          # a half-open trial is in flight

    def state(self, now: float) -> str:
        if self._opened_at is None:
            return "closed"
        if now - self._opened_at < self.recovery_time:
            return "open"
        return "half-open"

    async def call(self, make_coro):
        async with self._lock:
            loop = asyncio.get_running_loop()
            if self._opened_at is not None:
                if loop.time() - self._opened_at < self.recovery_time:
                    raise CircuitOpenError("circuit open; failing fast")
                if self._trial:
                    raise CircuitOpenError("circuit half-open; trial in progress")
                self._trial = True                     # this call is the trial

        try:
            result = await make_coro()
        except asyncio.CancelledError:
            async with self._lock:
                self._trial = False
            raise
        except Exception:
            async with self._lock:
                self._failures += 1
                self._trial = False
                if self._failures >= self.failure_threshold:
                    self._opened_at = asyncio.get_running_loop().time()
            raise
        else:
            async with self._lock:
                self._failures = 0
                self._opened_at = None
                self._trial = False
            return result


async def demo():
    from mockllm import MockLLM, ModelOverloadedError
    breaker = CircuitBreaker(failure_threshold=3, recovery_time=2.0)
    llm = MockLLM(seed=1, failure_rate=1.0, latency=(0.05, 0.1))

    for i in range(6):
        try:
            print(i, await breaker.call(lambda: llm.complete("hi")))
        except CircuitOpenError as e:
            print(i, "SHORT-CIRCUITED:", e)
        except Exception as e:
            print(i, "failed:", type(e).__name__)
        await asyncio.sleep(0.2)

    print("waiting for recovery window...")
    await asyncio.sleep(2.1)
    llm.failure_rate = 0.0
    print("after recovery:", await breaker.call(lambda: llm.complete("hi")))
    print("actual upstream calls made:", llm.calls)

asyncio.run(demo())
```

Calls 0–2 fail and trip the breaker; calls 3–5 are rejected in microseconds without touching the upstream service. That's the entire point: when a dependency is down, hammering it with retries from every one of your instances is how a partial outage becomes a total one. Failing fast sheds load and gives the dependency room to recover.

Note the concurrency care: state transitions happen under `self._lock`, and the lock is *not* held during `await make_coro()` — holding it there would serialize every call through the breaker and destroy your throughput. This is the §8.2 "keep critical sections short" rule applied in anger.

Follow-up exercises: add a success threshold (require two consecutive successful trials before closing), and expose metrics so a dashboard can show how long the breaker has been open.

---

---

### ✓ Checkpoint — Part V

You should now be able to:

- Dispatch a model's tool calls concurrently, with per-tool timeouts, such that a failed tool becomes a *message* rather than a crash.
- Combine a concurrency limit, a rate limit, retries with jittered backoff, and a circuit breaker without any of them undermining the others.
- Give an agent turn a time budget *and* a cost budget, and return a partial answer when either expires.
- Stream tokens with an idle-timeout stall detector.
- Structure a multi-agent system as either a concurrent fan-out or a bounded pipeline, and say which fits a given problem.

## 16. Capstone project

### Exercise 47 — A production-shaped research agent

Build a single-file agent that ties together everything in this tutorial. Requirements:

1. Accept a list of research questions and process them **concurrently**, at most 3 questions at a time.
2. For each question, run up to 3 reasoning steps. Each step may issue several tool calls, which must run concurrently with a 4-second per-tool timeout.
3. Every LLM call goes through retry-with-jittered-backoff and a shared rate limiter (5 req/s, 6 concurrent).
4. Each question has a 25-second budget. On expiry, return a partial answer rather than an error.
5. A circuit breaker protects the flaky tool; when open, calls fail instantly.
6. Report per-question: answer, steps taken, LLM calls, tool calls, elapsed time, and termination reason.
7. Support graceful cancellation: `Ctrl-C` (or a cancel `Event`) stops everything and still prints the partial report.
8. Nothing crashes. No unhandled exception may escape `main()`.

**Build it in four milestones.** Do not attempt this in one sitting — each milestone runs on its own, so you can test as you go.

| Milestone | Goal | You should see |
|---|---|---|
| **M1 — sequential skeleton** | One question, one reasoning step, tools called *sequentially*, no limits, no error handling. | It works and it's slow. Record the time; this is your baseline. |
| **M2 — concurrency** | Tool calls run under a `TaskGroup`; questions run under `gather` with a `Semaphore(3)`. Still no timeouts. | Roughly a 4–5× speedup over M1. One hung tool still freezes everything — that's the next milestone. |
| **M3 — resilience** | Add per-tool timeouts, a per-question budget, `run_tool` that never raises, and retry-with-backoff. | `stuck_tool` costs 4 s, not 60. Failures appear as messages, not tracebacks. |
| **M4 — production limits** | Add the shared `ThrottledLLM` (semaphore + token bucket), the circuit breaker, the report dataclass, and graceful cancellation. | Ctrl-C exits in under a second; the flaky tool short-circuits after 3 failures. |

Before you write M2, sketch the task graph on paper: which parts are concurrent, which are sequential, and where each limit is enforced. The table at the end of the solution is the answer — don't look first.

**Common mistakes to watch for as you go**

- M2: creating the `Semaphore` inside `research` instead of sharing one. Each question then gets its own limiter (see the note at the end).
- M3: putting the retry *inside* the timeout so retries eat the budget silently, or *outside* so a hung call is retried forever.
- M4: holding the circuit breaker's lock across the upstream call, which serializes every request through it.

**Solution**

```python
"""capstone.py — a production-shaped concurrent research agent.
Run:  python capstone.py     (Ctrl-C at any point to test cancellation)
"""
from __future__ import annotations

import asyncio
import random
import signal
import time
from dataclasses import dataclass, field

from mockllm import (MockLLM, RateLimitError, ModelOverloadedError,
                     InvalidRequestError)

RETRYABLE = (RateLimitError, ModelOverloadedError, TimeoutError, ConnectionError)


# ─────────────────────────── infrastructure ───────────────────────────

class TokenBucket:
    def __init__(self, rate: float, capacity: float | None = None):
        self.rate, self.capacity = rate, capacity if capacity is not None else rate
        self._tokens, self._updated = self.capacity, None
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            if self._updated is None:
                self._updated = loop.time()
            while True:
                now = loop.time()
                self._tokens = min(self.capacity,
                                   self._tokens + (now - self._updated) * self.rate)
                self._updated = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                await asyncio.sleep((tokens - self._tokens) / self.rate)


class CircuitOpenError(Exception):
    pass


class CircuitBreaker:
    def __init__(self, *, threshold: int = 3, recovery: float = 5.0):
        self.threshold, self.recovery = threshold, recovery
        self._failures, self._opened_at, self._trial = 0, None, False
        self._lock = asyncio.Lock()

    async def call(self, make_coro):
        async with self._lock:
            loop = asyncio.get_running_loop()
            if self._opened_at is not None:
                if loop.time() - self._opened_at < self.recovery:
                    raise CircuitOpenError("circuit open")
                if self._trial:
                    raise CircuitOpenError("trial in progress")
                self._trial = True
        try:
            result = await make_coro()
        except asyncio.CancelledError:
            async with self._lock:
                self._trial = False
            raise
        except Exception:
            async with self._lock:
                self._failures += 1
                self._trial = False
                if self._failures >= self.threshold:
                    self._opened_at = asyncio.get_running_loop().time()
            raise
        else:
            async with self._lock:
                self._failures, self._opened_at, self._trial = 0, None, False
            return result


async def with_retry(make_coro, *, attempts: int = 4, base: float = 0.3,
                     cap: float = 8.0, retry_on: tuple = RETRYABLE):
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await make_coro()
        except asyncio.CancelledError:
            raise
        except retry_on as e:
            last = e
            if attempt == attempts - 1:
                break
            delay = min(cap, base * 2 ** attempt) * (0.5 + random.random() / 2)
            await asyncio.sleep(delay)
    raise last


class ThrottledLLM:
    """Shared across all questions: enforces global rate and concurrency limits."""

    def __init__(self, llm, *, rps: float = 5, max_concurrent: int = 6):
        self._llm = llm
        self._bucket = TokenBucket(rps, capacity=rps)
        self._sem = asyncio.Semaphore(max_concurrent)
        self.calls = 0

    async def complete(self, prompt: str) -> str:
        await self._bucket.acquire()
        async with self._sem:
            self.calls += 1
            return await self._llm.complete(prompt)


# ───────────────────────────── the tools ──────────────────────────────

async def web_search(query: str) -> str:
    await asyncio.sleep(random.uniform(0.4, 1.2))
    return f"3 web results for {query!r}"

async def vector_search(query: str, k: int = 3) -> str:
    await asyncio.sleep(random.uniform(0.1, 0.4))
    return f"top-{k} internal chunks for {query!r}"

async def flaky_pricing_api(sku: str) -> str:
    await asyncio.sleep(0.2)
    if random.random() < 0.8:
        raise ConnectionError("pricing service unavailable")
    return f"price for {sku}: $42"

async def stuck_tool(x: str = "") -> str:
    await asyncio.sleep(60)
    return "never"

TOOLS = {"web_search": web_search, "vector_search": vector_search,
         "flaky_pricing_api": flaky_pricing_api, "stuck_tool": stuck_tool}

BREAKERS = {"flaky_pricing_api": CircuitBreaker(threshold=3, recovery=5.0)}


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)


async def run_tool(call: ToolCall, *, timeout: float = 4.0) -> dict:
    fn = TOOLS.get(call.name)
    if fn is None:
        return {"tool": call.name, "ok": False, "error": "unknown tool"}
    breaker = BREAKERS.get(call.name)
    try:
        async with asyncio.timeout(timeout):
            if breaker is not None:
                result = await breaker.call(lambda: fn(**call.args))
            else:
                result = await fn(**call.args)
        return {"tool": call.name, "ok": True, "result": result}
    except TimeoutError:
        return {"tool": call.name, "ok": False, "error": f"timeout after {timeout}s"}
    except CircuitOpenError as e:
        return {"tool": call.name, "ok": False, "error": f"circuit open ({e})"}
    except asyncio.CancelledError:
        raise
    except Exception as e:
        return {"tool": call.name, "ok": False, "error": f"{type(e).__name__}: {e}"}


async def dispatch(calls: list[ToolCall]) -> list[dict]:
    if not calls:
        return []
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(run_tool(c)) for c in calls]
    return [t.result() for t in tasks]


def plan_tools(step: int, question: str) -> list[ToolCall]:
    """Stands in for parsing the model's tool_use blocks."""
    if step == 0:
        return [ToolCall("web_search", {"query": question}),
                ToolCall("vector_search", {"query": question, "k": 5})]
    if step == 1:
        return [ToolCall("flaky_pricing_api", {"sku": "SKU-1"}),
                ToolCall("web_search", {"query": question + " 2026 outlook"}),
                ToolCall("stuck_tool", {})]
    return []


# ────────────────────────────── the agent ─────────────────────────────

@dataclass
class Report:
    question: str
    answer: str
    steps: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    elapsed: float = 0.0
    reason: str = "completed"


async def research(question: str, llm: ThrottledLLM, *, max_steps: int = 3,
                   budget: float = 25.0) -> Report:
    loop = asyncio.get_running_loop()
    start = loop.time()
    transcript = [f"QUESTION: {question}"]
    rep = Report(question=question, answer="")

    async def think(prompt: str) -> str:
        # count LOGICAL calls for this question; retries are counted globally
        # by ThrottledLLM.calls. Subtracting a shared counter here would be a
        # bug: other questions run concurrently and would inflate the delta.
        rep.llm_calls += 1
        return await with_retry(lambda: llm.complete(prompt))

    try:
        async with asyncio.timeout(budget):
            for step in range(max_steps):
                rep.steps = step + 1
                thought = await think("\n".join(transcript))
                transcript.append(f"THOUGHT[{step}]: {thought}")

                calls = plan_tools(step, question)
                if not calls:
                    break
                rep.tool_calls += len(calls)
                for r in await dispatch(calls):
                    transcript.append(
                        f"OBS[{step}] {r['tool']}: "
                        f"{r['result'] if r['ok'] else 'ERROR ' + r['error']}")

            rep.answer = await think("\n".join(transcript) + "\nFINAL:")
    except TimeoutError:
        rep.reason = "time budget exhausted"
        rep.answer = "PARTIAL: " + " | ".join(transcript[-2:])
    except asyncio.CancelledError:
        rep.reason = "cancelled"
        rep.answer = "CANCELLED: " + " | ".join(transcript[-2:])
        raise
    except Exception as e:
        rep.reason = f"failed ({type(e).__name__})"
        rep.answer = f"ERROR: {e}"
    finally:
        rep.elapsed = loop.time() - start

    return rep


async def research_all(questions: list[str], llm: ThrottledLLM,
                       *, max_parallel: int = 3) -> list[Report]:
    sem = asyncio.Semaphore(max_parallel)
    reports: list[Report] = []

    async def one(q: str) -> None:
        async with sem:
            print(f"  ▶ starting: {q}")
            reports.append(await research(q, llm))
            print(f"  ✔ finished: {q}")

    await asyncio.gather(*(one(q) for q in questions))
    return reports


# ───────────────────────────────  main  ───────────────────────────────

def print_reports(reports: list[Report], elapsed: float) -> None:
    print("\n" + "=" * 72)
    for r in sorted(reports, key=lambda r: r.question):
        print(f"\nQ: {r.question}")
        print(f"   {r.answer[:110]}")
        print(f"   steps={r.steps} llm={r.llm_calls} tools={r.tool_calls} "
              f"{r.elapsed:.1f}s [{r.reason}]")
    print("=" * 72)
    print(f"{len(reports)} questions in {elapsed:.1f}s wall-clock")


async def main() -> None:
    random.seed(0)
    questions = [
        "market size for grid-scale storage",
        "regulatory outlook for hydrogen",
        "cost curve of sodium-ion cells",
        "demand response adoption rates",
        "supply chain risk for rare earths",
    ]
    llm = ThrottledLLM(MockLLM(seed=0, failure_rate=0.2, latency=(0.3, 0.9)),
                       rps=5, max_concurrent=6)

    cancel = asyncio.Event()
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, cancel.set)
    except NotImplementedError:
        pass                                          # Windows

    t0 = time.perf_counter()
    work = asyncio.create_task(research_all(questions, llm), name="research-all")
    watch = asyncio.create_task(cancel.wait(), name="cancel-watch")

    done, pending = await asyncio.wait({work, watch},
                                       return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    elapsed = time.perf_counter() - t0
    if work in done and not work.cancelled():
        print_reports(work.result(), elapsed)
    else:
        print(f"\ncancelled by user after {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
```

**Typical output**

```
  ▶ starting: market size for grid-scale storage
  ▶ starting: regulatory outlook for hydrogen
  ▶ starting: cost curve of sodium-ion cells
  ✔ finished: cost curve of sodium-ion cells
  ▶ starting: demand response adoption rates
  ...
========================================================================

Q: cost curve of sodium-ion cells
   [mock-sonnet] response to: OBS[1] stuck_tool: ERROR timeout after 4.0s
   steps=3 llm=4 tools=5 11.4s [completed]
...
5 questions in 21.3s wall-clock
```

**What to check in your own solution**

- Sequentially this workload is 90+ seconds; yours should be roughly 20.
- The `stuck_tool` should cost exactly 4 s, not 60 — per-tool timeouts working.
- After three `flaky_pricing_api` failures, subsequent calls should report `circuit open` and return in microseconds.
- No `RuntimeWarning: coroutine was never awaited` and no "Task was destroyed but it is pending".
- Ctrl-C at any moment should exit within a second, cleanly.

**Where the limits are enforced — the answer to the paper exercise:**

| Limit | Enforced by | Scope |
|---|---|---|
| 3 questions in parallel | `Semaphore` in `research_all` | global |
| 6 concurrent LLM calls | `Semaphore` in `ThrottledLLM` | global |
| 5 LLM req/s | `TokenBucket` in `ThrottledLLM` | global |
| 4 s per tool | `asyncio.timeout` in `run_tool` | per call |
| 25 s per question | `asyncio.timeout` in `research` | per question |
| 3 steps per question | the `for` loop | per question |
| Failing dependency | `CircuitBreaker` | per tool |

Notice that the two semaphores are created once and shared, while the timeouts are per-operation. **Shared limiters must be shared objects** — a semaphore created inside `research` would limit each question independently and let 3 × 6 = 18 calls through. That off-by-a-factor-of-N bug is extremely common and only shows up under load.

---

## 17. Quick reference card

### Entering async-land
```python
asyncio.run(main())                       # once, at the top of your program
asyncio.Runner()                          # 3.11+, when you need more control
await main()                              # in Jupyter/IPython
```

### Running things concurrently
```python
t = asyncio.create_task(coro(), name="x")         # schedule now, await later
results = await asyncio.gather(*coros)            # ordered results
results = await asyncio.gather(*coros, return_exceptions=True)   # collect failures
async with asyncio.TaskGroup() as tg:             # 3.11+, structured, preferred
    t1 = tg.create_task(coro())
for fut in asyncio.as_completed(coros):           # results as they arrive
    r = await fut
done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
```

### Timeouts and cancellation
```python
async with asyncio.timeout(5): ...                # 3.11+, wraps a block
async with asyncio.timeout_at(deadline): ...      # absolute, loop.time() clock
result = await asyncio.wait_for(coro(), 5)        # wraps one awaitable
task.cancel(); await task                         # request, then confirm
await asyncio.shield(critical())                  # protect from cancellation
except asyncio.CancelledError: ...; raise         # ALWAYS re-raise
```

### Coordination
```python
lock = asyncio.Lock();        async with lock: ...
sem  = asyncio.Semaphore(10); async with sem: ...      # concurrency cap
ev   = asyncio.Event();       await ev.wait(); ev.set()
q    = asyncio.Queue(maxsize=100)                      # ALWAYS set maxsize
await q.put(x); x = await q.get(); q.task_done(); await q.join()
```

### Escaping to threads/processes
```python
await asyncio.to_thread(blocking_fn, arg)                    # blocking I/O
await loop.run_in_executor(process_pool, cpu_fn, arg)        # CPU work
proc = await asyncio.create_subprocess_exec("ls", "-l", stdout=PIPE)
await asyncio.sleep(0)                                       # yield to the loop
```

### Async syntax
```python
async def f(): ...
await x
async with cm: ...          # __aenter__ / __aexit__
async for x in it: ...      # __aiter__ / __anext__
async def gen(): yield x    # async generator
[x async for x in gen()]
@asynccontextmanager        # from contextlib
async with aclosing(gen()) as g: ...
```

### Introspection
```python
asyncio.current_task(); asyncio.all_tasks()
task.done() / .cancelled() / .result() / .exception() / .get_name()
task.print_stack()
asyncio.run(main(), debug=True)          # or PYTHONASYNCIODEBUG=1
loop.slow_callback_duration = 0.05       # blocking-call detector
```

### The rules, compressed
1. Async helps **I/O-bound** work, never CPU-bound.
2. `await` alone is sequential; **launch first, await later.**
3. **Never block the loop** — `time.sleep`, `requests`, heavy loops are forbidden.
4. **Every network call gets a timeout.**
5. **Bound your concurrency** with a semaphore; **bound your queues** with `maxsize`.
6. **Re-raise `CancelledError`**; put cleanup in `finally`.
7. **`TaskGroup` by default**; `gather(return_exceptions=True)` for partial failure.
8. In agents, **tool failures are messages**, not exceptions.

---

## Appendix: where to go next

**Primary documentation (authoritative, and genuinely well written):**
- The `asyncio` chapter of the Python standard library docs — start with "Coroutines and Tasks".
- PEP 492 (`async`/`await` syntax), PEP 525 (async generators), PEP 530 (async comprehensions), PEP 654 (`ExceptionGroup` and `except*`).
- The "Developing with asyncio" page in the docs, which covers debug mode and the concurrency gotchas.

**Libraries you will meet in industry:**
- `httpx` or `aiohttp` — async HTTP clients.
- `FastAPI` / `Starlette` / `uvicorn` — async web services.
- `asyncpg`, `aiosqlite`, `motor`, `redis-py` (async mode) — async database drivers.
- `anyio` and `trio` — alternative structured-concurrency frameworks. Trio's design directly inspired `TaskGroup`; reading its documentation will sharpen your thinking about cancellation even if you never use it.
- `pytest-asyncio`, `AsyncMock` — testing.
- `uvloop` — faster event loop on Linux/macOS.
- `tenacity` — battle-tested retry logic, so you don't maintain your own `with_retry`.

**A note on version drift:** asyncio has changed substantially across 3.7 → 3.12, and a great deal of material online predates `asyncio.run`, `TaskGroup`, `asyncio.timeout`, and `ExceptionGroup`. If a tutorial you find uses `@asyncio.coroutine`, `yield from`, or `loop.run_until_complete` as its main idiom, it's describing a Python that no longer exists — `@asyncio.coroutine` was removed in 3.11. Check the standard library docs, which state the version each feature was added in, rather than trusting blog posts.

**Practice suggestions, in increasing difficulty:**
1. Rewrite a synchronous script you already own to use `httpx.AsyncClient` with a semaphore. Measure before and after.
2. Build a concurrent web crawler with a bounded frontier queue, politeness delays per domain, and depth limiting.
3. Write an async rate-limited wrapper around a real LLM API, with retries, streaming, and a cost tracker.
4. Extend the capstone: add a vector store, persist transcripts, expose it over FastAPI with server-sent events for streaming, and add per-user concurrency quotas.

---

## 18. Glossary

Terms in **bold** are defined elsewhere in this glossary.

**Awaitable** — anything you can `await`: a **coroutine**, a **Task**, or a **Future**. (Also any object implementing `__await__`, which you will rarely write.)

**Backpressure** — the mechanism by which a slow consumer forces a fast producer to slow down, instead of accumulating unbounded work in memory. In asyncio: `asyncio.Queue(maxsize=N)` and `await writer.drain()`.

**Blocking call** — any operation that occupies the thread without yielding to the **event loop**: `time.sleep`, `requests.get`, a long computation. Because there is one thread, a blocking call freezes the entire program. The cardinal sin of async programming.

**Circuit breaker** — a wrapper that stops calling a failing dependency after N consecutive failures, failing fast for a cooldown period. Prevents a partial outage from becoming a total one.

**Concurrency** — multiple operations *in progress* over the same period, interleaved on one thread. Distinct from **parallelism**.

**Cooperative multitasking** — the scheduling model asyncio uses: a task runs until it *voluntarily* yields at an `await`. Contrast with the **preemptive** scheduling of OS threads. Makes reasoning easy, but means one uncooperative task can starve everything.

**Coroutine** — what `async def f()` returns when called. A suspended computation that has not started. Awaiting it runs it. Can only be awaited once.

**Coroutine function** — the `async def` itself, as opposed to the **coroutine** object calling it produces. The distinction matters when writing retry logic: you must pass a factory, not a used coroutine.

**Event loop** — the single-threaded scheduler at asyncio's core. Runs ready tasks, asks the OS which I/O is ready (via `epoll`/`kqueue`/IOCP), and repeats.

**ExceptionGroup** — a container holding several exceptions raised concurrently (Python 3.11+). Raised by **TaskGroup**; caught with `except*`.

**Function colouring** — the property that `async` is contagious: a sync function cannot `await`, so making one deep function async forces every caller up the stack to become async too (§10.6).

**Future** — a low-level placeholder for a result that will exist later. Libraries hand them to you; you rarely construct one. A **Task** is a Future that wraps a coroutine.

**GIL (Global Interpreter Lock)** — CPython's lock ensuring only one thread executes Python bytecode at a time. Released during blocking I/O and inside many C extensions, which is why threads help I/O but not pure-Python computation.

**I/O-bound** — work limited by waiting on something external (network, disk, database). The workload asyncio accelerates. Contrast **CPU-bound**, which it does not.

**Jitter** — deliberate randomness added to retry delays, so that many clients failing simultaneously don't all retry simultaneously (the *thundering herd*).

**Parallelism** — multiple operations executing at the same instant on different cores. Requires `multiprocessing`; asyncio does not provide it.

**Rate limit** vs **concurrency limit** — how many operations *start per unit time* (enforce with a **token bucket**) versus how many are *in flight at once* (enforce with a `Semaphore`). Different constraints; production systems usually need both (§8.3, §15.4).

**Structured concurrency** — the discipline that tasks cannot outlive the block that created them, so errors and cancellation propagate predictably. In asyncio: `asyncio.TaskGroup`.

**Task** — a **coroutine** scheduled to run independently on the **event loop**, created with `asyncio.create_task` or `TaskGroup.create_task`. Unlike a coroutine, it starts running on its own and can be awaited repeatedly.

**Token bucket** — a rate-limiting algorithm: tokens accrue at a fixed rate up to a capacity; each operation consumes one. Permits a bounded burst, then settles to the sustained rate.

**Yield to the loop** — to suspend at an `await` so other tasks can run. `await asyncio.sleep(0)` does it without waiting.
