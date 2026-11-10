# Asyncio for Agentic AI and ML Engineering: A Crash Course

Nov 10, 2026 · @Behi

## How to use this course

This course takes you from zero knowledge of coroutines to building async agent services with FastAPI, MCP and LangGraph in about 90 hours of study (two weeks full-time, or four to five weeks part-time), with an optional 20-hour third week on operations and security. It assumes you can write Python functions, loops and classes, and nothing more. Every module ends with exercises and full solutions, and the course closes with a capstone project and an interview question bank.

### Why asyncio matters for agentic AI and ML engineering jobs

An AI agent spends most of its time waiting: waiting for a large language model (LLM) to answer, for a web search to return, for a database query, for a tool on another server. Asyncio is Python's built-in way to do useful work during those waits. A single Python process using asyncio can keep hundreds of LLM calls, tool calls and user connections in flight at once. Almost every modern agent framework and LLM SDK in Python (the OpenAI and Anthropic SDKs, FastAPI, the MCP SDK, LangGraph) is built on or offers an asyncio interface, so interviewers for agentic AI and ML platform roles expect you to read and write async code fluently.

### Prerequisites

- Python basics: variables, functions, loops, lists, dictionaries, classes, `try`/`except`.
- Using a terminal to run a script.
- Nothing else. Module 0 teaches the intermediate Python features (decorators, generators, context managers, type hints) that asyncio relies on.

### Python version

The course requires Python 3.11 or newer, because it uses `asyncio.TaskGroup` and `asyncio.timeout`, which arrived in 3.11. Python 3.15.0 was released on October 9, 2026, and 3.15.2 is the current patch release as of this writing ([python.org](https://www.python.org/downloads/release/python-3150/)). Use 3.13 or 3.14 if you want the widest library support, or 3.15 if all your libraries already support it. Where a feature needs a version newer than 3.11, the text says so.

### Setup (about 20 minutes)

We use `uv`, a fast package and project manager that is now the common default in industry Python teams. Install it by following the instructions at [docs.astral.sh/uv](https://docs.astral.sh/uv/), then run:

```bash
uv python install 3.14
mkdir asyncio-course && cd asyncio-course
uv init --python 3.14
uv add httpx2 fastapi "uvicorn[standard]" openai anthropic "mcp[cli]" langgraph opentelemetry-sdk
uv add --dev pytest pytest-asyncio
uv run python -c "import sys; print(sys.version)"
```

Run every example in the course with `uv run python file_name.py`. Modules 0 to 9 need no API keys and no internet beyond installing packages. Modules 10 to 13 use real LLM APIs; each one also gives a fake (mock) LLM so you can complete the exercises without paying for API calls.

Library versions move fast in this field. This course was checked against these releases as of November 2026; minor versions newer than these should work unchanged:

| Package | Version tested | Notes |
| --- | --- | --- |
| Python | 3.12.3, 3.13.18, 3.14.10, 3.15.2 | 3.11 is the minimum; differences are in the Python 3.12 guide |
| `httpx2` | 2.16.0 | Maintained successor of `httpx`; `import httpx2` |
| `anthropic` | 2.8.0 | `AsyncAnthropic`; sends API version 2026-10-21; uses httpx2 by default |
| `openai` | 3.49.0 | `AsyncOpenAI`; the Responses API is the primary interface; uses httpx2 |
| `mcp` (MCP Python SDK) | 2.3.0 | `MCPServer` (formerly `FastMCP`) and `Client`; protocol revision 2026-07-28 |
| `langgraph` | 1.2.20 | `StateGraph`, `ainvoke`, `astream`; still depends on the old `httpx` 0.28.1 |
| `fastapi` | 0.162.3, on Starlette 1.10 |  |
| `anyio` | 4.16.0 | Installed with FastAPI, MCP and the SDKs; Module 17 |
| `opentelemetry-sdk` | 1.45.2 | Module 20 |

If you join a team that pins older versions (for example `mcp<2` or `anthropic<1`), the asyncio concepts are identical; only import names and a few signatures differ, and each module points them out.

**What was actually executed.** Every example in Modules 3 to 8, the Python 3.12 guide and Modules 16 to 20 was run, and version-specific features were checked on all four Python versions in the table. The MCP, FastAPI, WebSocket and LangGraph examples ran against the library versions listed. The LLM wrapper from Module 10 and the OpenAI examples were run through the real `anthropic` and `openai` SDKs with `httpx2.MockTransport` standing in for the network, which checks how requests are built and responses parsed, but not model behaviour; no paid API calls were made. The capstone reference solution's 9-test suite passes on Python 3.12 and 3.14 in about 7.5 seconds.

### Study plan

The core path (Modules 0 to 15 and the Python 3.12 guide) takes about 90 hours. Finishing it in two weeks means 6 to 10 hours a day, a bootcamp pace; at 3 to 4 hours a day, plan on four to five weeks. Modules 16 to 20 cover topics many agent roles also expect (subprocesses, AnyIO, WebSockets and shutdown, security, production telemetry) and add about 20 hours.

| Day | Modules | Focus | Hours |
| --- | --- | --- | --- |
| 1 | 0, 1 | Python warm-up, why async exists | 6 |
| 2 | 2, 3 | First coroutines, how the event loop works | 7 |
| 3 | 4 | Running many things at once | 5 |
| 4 | 5 | Timeouts, cancellation, errors | 5 |
| 5 | 6, 7 | Queues, semaphores, async iteration, streaming | 9 |
| 6 | 8, Python 3.12 guide | Blocking code, threads, processes, version differences | 7 |
| 7 | 9, 10 | HTTP with httpx2, retries, async LLM SDKs | 10 |
| 8 | 11, 12 | FastAPI services, MCP servers and clients | 10 |
| 9 | 13 | Agent loops, multi-agent patterns, LangGraph | 6 |
| 10 | 14 | Testing, debugging, production checklist | 5 |
| 11 to 12 | 15 | Capstone project | 12 |
| 13 | Interview bank | Answer every question aloud; redo weak exercises | 6 |
| 14 | Buffer | Catch up, mock interview with a friend | 4 to 6 |
| Optional week 3 | 16 to 20 | Subprocesses, AnyIO, WebSockets, security, telemetry | 20 |

### Course map

| Part | Modules | You will be able to | Minimum Python |
| --- | --- | --- | --- |
| Foundations | 0 to 3 | Read and write coroutines; explain the event loop | 3.11 |
| Concurrency | 4 to 7 | Run, limit, time out, cancel and stream concurrent work | 3.11 (some extras 3.13+) |
| Real systems | 8 to 12 | Mix in blocking code; call HTTP, LLM APIs and MCP servers; serve with FastAPI | 3.11 |
| Agents | 13 to 15 | Build, test and debug agent loops, LangGraph graphs and the capstone service | 3.11 |
| Version differences | Python 3.12 guide | Write code that runs on 3.12 and on 3.13 to 3.15 | 3.12 |
| Operations and safety (optional) | 16 to 20 | Run subprocesses safely, handle shutdown, secure tools, trace and load-test | 3.11 (Unix for signals) |

**Navigating this doc.** It is one long document so that search works across the whole course; use the doc's outline (headings) to jump between modules. Each module ends with Exercises, then Solutions under their own heading: scroll only to the Exercises heading, work for at least 15 minutes, then read on. If you prefer a printed or offline copy, export the doc to PDF or Word.

### Conventions

- **Check yourself** questions appear inside modules. Answer them before reading on; answers follow immediately.
- **Exercises** come at the end of each module, marked by difficulty: (E) easy, (M) medium, (H) hard.
- **Solutions** sit in a separate subsection after the exercises. Try each exercise for at least 15 minutes before you look.
- Code is complete and runnable unless it is marked as a fragment. Type it in rather than copy-pasting; typing builds the habits interviewers look for.
- Timings printed by examples (such as "took 1.00s") will vary slightly on your machine.

## Python 3.12 guide: what changes if your team runs 3.12

Every core concept in this course works on Python 3.12, but ten features the course uses arrived later, and three behaviours differ in ways that bite in production. Python 3.12 has been in security-only maintenance since April 2025 and is supported until October 2028 ([devguide](https://devguide.python.org/versions)), so many companies still run it. Everything in this section was executed on Python 3.12.3.

### Feature availability

| Feature used in the course | Added in | What to do on 3.12 | Modules |
| --- | --- | --- | --- |
| `TaskGroup`, `asyncio.timeout`, `except*`, `asyncio.Runner` | 3.11 | Works | 4, 5, 8 |
| `asyncio.run(main(), loop_factory=...)` | 3.12 | Works (on 3.11, use `asyncio.Runner(loop_factory=...)`) | 8 |
| `asyncio.eager_task_factory` | 3.12 | Works | 4 |
| `Queue.shutdown()` and `QueueShutDown` | 3.13 | Sentinel values (code below) | 6, 7 |
| `async for` over `as_completed`, yielding the original tasks | 3.13 | Plain `for` loop; tag each result with its name (code below) | 4 |
| Free-threaded (no-GIL) build | 3.13 experimental, 3.14 supported | Processes for CPU work | 1, 8 |
| `python -m asyncio ps/pstree`, `asyncio.capture_call_graph()` | 3.14 | `asyncio.all_tasks()` with `task.print_stack()`, or the third-party `py-spy` and `aiomonitor` tools | 14 |
| `concurrent.interpreters` and `InterpreterPoolExecutor` (subinterpreters) | 3.14 | Processes | not covered elsewhere |
| `TaskGroup.cancel()` | 3.15 | Raise a private exception and catch it with `except*` (code below) | 4 |
| `profiling.sampling` (Tachyon) | 3.15 | `py-spy` | 14 |
| UTF-8 as the default text encoding | 3.15 | Always pass `encoding="utf-8"` to `open()` | 15 |

### Behaviours that differ on 3.12

1. **`asyncio.get_event_loop()` still "works".** In a fresh process with no loop, 3.12 emits `DeprecationWarning: There is no current event loop` and creates one; once an earlier `asyncio.run()` has finished, it raises `RuntimeError` instead. On 3.14 and later it always raises `RuntimeError` when no loop is set. Code that passes on 3.12 with warnings hidden breaks on upgrade. Treat that warning as an error in CI (`python -W error::DeprecationWarning`) and use `asyncio.run` or `get_running_loop()`.
2. **`fork` is the default process start method on Linux.** Starting a `ProcessPoolExecutor` after any thread exists (after a single `asyncio.to_thread` call, or inside an SDK that starts background threads) prints `DeprecationWarning: This process ... is multi-threaded, use of fork() may lead to deadlocks in the child`. Pass an explicit context: `ProcessPoolExecutor(mp_context=multiprocessing.get_context("forkserver"))`. This also prepares you for 3.14, where `forkserver` became the default. Under `forkserver` and `spawn`, worker functions and their arguments must be picklable and importable, so lambdas and locally defined functions that happened to work with `fork` now fail.
3. **TaskGroup cancellation edge cases.** Python 3.13 improved how a TaskGroup handles an internal failure and an external cancellation that arrive at the same time, and how it preserves cancellation counts. Ordinary code behaves the same on both versions; if your code nests TaskGroups inside timeouts that can fire together, test on the exact version you deploy.

### What 3.12 added that is worth using

- **Eager tasks.** `loop.set_task_factory(asyncio.eager_task_factory)` makes `create_task` run the new coroutine immediately, up to its first real suspension. Measured on 3.12.3, a child that never awaits completes *before* `create_task` returns. This saves scheduling overhead when many tasks finish without waiting (cache hits), but it changes execution order, so enable it deliberately.
- **`asyncio.run(..., loop_factory=...)`**, for example `asyncio.run(main(), loop_factory=uvloop.new_event_loop)` with the third-party `uvloop` package, a faster event loop popular in web servers.
- **`asyncio.wait()` and `as_completed()` accept generators** of tasks, not only lists.
- **`Task.get_context()`** returns the `contextvars` context a task runs in.
- **`asyncio.wait_for` is implemented on top of `asyncio.timeout`**, so both behave consistently.
- **New generic syntax (PEP 695)**, handy for typed async helpers: `async def first[T](aws: list[Awaitable[T]]) -> T: ...` and `type ToolFn = Callable[..., Awaitable[Any]]`.

### Ported code (tested on 3.12.3)

```python
import asyncio, multiprocessing, sys
from concurrent.futures import ProcessPoolExecutor

async def fake_llm(prompt: str, delay: float = 1.0) -> str:
    await asyncio.sleep(delay)
    return f"answer to {prompt!r}"

# 1. End a TaskGroup early without tg.cancel() (Module 4.2)
class _Done(Exception):
    pass

async def first_good_answer(prompts: list[str]) -> str | None:
    best = None
    async def try_one(p: str):
        nonlocal best
        answer = await fake_llm(p, delay=len(p) * 0.1)
        if best is None:
            best = answer
            raise _Done                 # makes the group cancel the other tasks
    try:
        async with asyncio.TaskGroup() as tg:
            for p in prompts:
                tg.create_task(try_one(p))
    except* _Done:
        pass
    return best

# 2. Close a queue without Queue.shutdown() (Exercise 6.2): one sentinel per consumer
_STOP = object()

async def pipeline() -> list[int]:
    q: asyncio.Queue = asyncio.Queue(maxsize=2)
    out: list[int] = []
    async def producer():
        for i in range(6):
            await q.put(i)
        for _ in range(2):              # one per consumer
            await q.put(_STOP)
    async def consumer():
        while True:
            item = await q.get()
            if item is _STOP:
                return
            await asyncio.sleep(0.05)
            out.append(item)
    async with asyncio.TaskGroup() as tg:
        tg.create_task(producer())
        tg.create_task(consumer())
        tg.create_task(consumer())
    return sorted(out)

# 3. Know which call finished, without async-for as_completed (Module 4.4)
async def tagged(name: str, coro):
    return name, await coro

async def names_in_finish_order() -> list[str]:
    delays = {"slow": 0.3, "mid": 0.2, "fast": 0.1}
    coros = [tagged(n, fake_llm(n, d)) for n, d in delays.items()]
    return [(await f)[0] for f in asyncio.as_completed(coros)]

def square(x: int) -> int:
    return x * x

async def main():
    print(await first_good_answer(["aaaa", "a", "aaa"]))   # answer to 'a', after 0.10s
    print(await pipeline())                                  # [0, 1, 2, 3, 4, 5]
    print(await names_in_finish_order())                     # ['fast', 'mid', 'slow']
    # 4. A process pool that is safe after threads exist (Module 8.3)
    await asyncio.to_thread(lambda: None)
    ctx = multiprocessing.get_context("forkserver" if sys.platform.startswith("linux") else "spawn")
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as pool:
        print(await asyncio.gather(*(loop.run_in_executor(pool, square, i) for i in range(4))))   # [0, 1, 4, 9], no warning

if __name__ == "__main__":
    asyncio.run(main())
```

### Code that runs on every version: feature detection

Libraries that must support 3.12 and 3.13+ at once test for the feature instead of the version number:

```python
import asyncio

HAS_SHUTDOWN = hasattr(asyncio.Queue, "shutdown")
_STOP = object()

async def close_queue(q: asyncio.Queue, n_consumers: int) -> None:
    if HAS_SHUTDOWN:
        q.shutdown()
    else:
        for _ in range(n_consumers):
            await q.put(_STOP)

async def next_item(q: asyncio.Queue):
    """Return the next item, or _STOP once the queue is closed."""
    if HAS_SHUTDOWN:
        try:
            return await q.get()
        except asyncio.QueueShutDown:
            return _STOP
    return await q.get()
```

Consumers then loop with `while (item := await next_item(q)) is not _STOP:`. Feature detection survives backports and alternative Python implementations; comparing `sys.version_info` does not.

### Exercises for 3.12

**P1 (E)** Run the original Exercise 6.2 solution on Python 3.12, read the error, and port it to sentinels. *Solution: the run fails with `AttributeError: 'Queue' object has no attribute 'shutdown'`, wrapped in an `ExceptionGroup` because it happens inside a TaskGroup. The port is snippet 2 above.*

**P2 (M)** In snippet 1, why does catching `_Done` with `except*` stop the other tasks, and what happens if two tasks raise `_Done` in the same loop iteration? *Solution: the first exception makes the TaskGroup cancel its remaining tasks; any task that raises `_Done` before its cancellation lands adds a second `_Done` to the same ExceptionGroup, and `except* _Done` catches all of them, so the result is the same. `best` holds the first answer because the check-and-set has no `await` between the check and the assignment.*

**P3 (M)** Reproduce the fork warning: start a `ProcessPoolExecutor` after `await asyncio.to_thread(...)` with warnings enabled (`python -W always`), then fix it. *Solution: snippet 4.*

**P4 (H)** Your service passes its tests on 3.12. List what could break when the team upgrades to 3.14, and how to catch each before the upgrade. *Solution: `get_event_loop()` without a loop now raises (catch: run CI with `-W error::DeprecationWarning`); event loop policy APIs warn and are slated for removal in 3.16 (catch: the same flag); process pools default to `forkserver`, so unpicklable worker functions fail (catch: set `mp_context` to `forkserver` on 3.12 already); `asyncio` child watcher classes, deprecated in 3.12, are removed (catch: grep for `ChildWatcher`). Add a 3.14 job to CI before switching.*

## Module 0: Python warm-up

Asyncio code leans on five Python features that beginner courses often skip: functions as values, type hints, exceptions with `finally`, context managers and generators, plus decorators. This module teaches just enough of each. If you already know them, do the exercises as a quick check and move on.

### 0.1 Functions are values

In Python a function is an object, like a number or a list. You can store it in a variable, put it in a dictionary, and pass it to another function.

```python
def shout(text):
    return text.upper() + "!"

def whisper(text):
    return text.lower() + "..."

def speak(style, text):
    return style(text)        # call whatever function we were given

print(speak(shout, "hello"))    # HELLO!
print(speak(whisper, "HELLO"))  # hello...

tools = {"shout": shout, "whisper": whisper}   # a "tool registry"
print(tools["shout"]("agents"))                 # AGENTS!
```

Notice the difference between `shout` (the function object) and `shout("hi")` (calling it). This distinction is the single most important idea for understanding coroutines later. An agent's tool registry is exactly the dictionary above: a mapping from a tool name the LLM chooses to a Python function your code calls.

### 0.2 Type hints

Type hints describe what a function expects and returns. Python does not enforce them at runtime, but editors, type checkers such as mypy or pyright, and libraries such as FastAPI and Pydantic read them.

```python
def add(a: int, b: int) -> int:
    return a + b

def first_word(text: str) -> str | None:   # str or None
    parts = text.split()
    return parts[0] if parts else None

scores: dict[str, float] = {"alice": 0.92}
```

You will see `-> str` on async functions too. Later you will meet `Awaitable`, `Coroutine` and `AsyncIterator` from `collections.abc`.

### 0.3 Exceptions and finally

`finally` runs no matter how a block exits: normally, through `return`, or through an exception. Asyncio uses exceptions to deliver cancellation, so cleanup code belongs in `finally`.

```python
def risky(n: int) -> float:
    try:
        print("opening connection")
        return 10 / n
    except ZeroDivisionError:
        print("bad input")
        raise                      # re-raise the same exception
    finally:
        print("closing connection")  # always runs

risky(2)    # opening, closing, returns 5.0
# risky(0)  # opening, bad input, closing, then ZeroDivisionError propagates
```

### 0.4 Context managers: with

A context manager sets something up and guarantees it is torn down. You use one every time you write `with open(...)`. Under the hood, `with` calls `__enter__` at the start and `__exit__` at the end, even if an exception occurs.

```python
import time

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self                  # bound to the name after "as"

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
        print(f"took {self.elapsed:.2f}s")
        return False                 # False means: do not swallow exceptions

with Timer() as t:
    sum(range(10_000_000))
```

The `contextlib.contextmanager` decorator lets you write the same thing as a generator function (see 0.5): code before `yield` is setup, code after is teardown.

```python
from contextlib import contextmanager

@contextmanager
def timer():
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"took {time.perf_counter() - start:.2f}s")

with timer():
    sum(range(10_000_000))
```

In async code you will use `async with` for things like HTTP clients and database sessions (Module 7).

### 0.5 Generators: functions that pause

This is the key stepping stone to coroutines. A normal function runs to completion. A **generator function** contains `yield`; calling it does not run its body. It returns a generator object, and each `next()` runs the body until the next `yield`, then pauses with all local variables intact.

```python
def countdown(n: int):
    print("starting")
    while n > 0:
        yield n          # pause here and hand n to the caller
        n -= 1
    print("done")

g = countdown(3)     # nothing printed yet! g is a generator object
print(next(g))       # starting, then 3
print(next(g))       # 2
print(next(g))       # 1
# next(g)            # prints done, then raises StopIteration

for x in countdown(2):   # for loops call next() and stop at StopIteration
    print(x)
```

Three facts to remember, because coroutines share all of them:

1. Calling the function creates an object; it does not run the code.
2. The object can pause in the middle and resume later, keeping its local variables.
3. Something outside (here, `next()` or a `for` loop) decides when it resumes.

Generators can also *receive* values with `send()`; `x = yield` pauses and later receives whatever is sent in. Module 3 uses this to build a tiny event loop.

### 0.6 Decorators

A decorator is a function that takes a function and returns a new function that usually wraps the original. `@name` above a `def` is shorthand for `func = name(func)`.

```python
import functools

def log_calls(func):
    @functools.wraps(func)          # keep the original name and docstring
    def wrapper(*args, **kwargs):
        print(f"calling {func.__name__} with {args}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper

@log_calls
def add(a, b):
    return a + b

add(2, 3)
```

You will see decorators everywhere in agent code: `@app.get("/")` in FastAPI, `@mcp.tool()` in MCP servers, `@tool` in LangChain. Each one registers or wraps your function.

### Check yourself

1. What does `g = countdown(3)` print? *Answer: nothing. Calling a generator function only creates the generator object.*
2. Why put cleanup in `finally` rather than after the `try` block? *Answer: `finally` also runs when the block exits via an exception or `return`, so the cleanup cannot be skipped.*
3. What is `@log_calls` above `def add` equivalent to? *Answer: `add = log_calls(add)`.*

### Exercises

**0.1 (E)** Write a function `apply_all(funcs, value)` that takes a list of functions and a value and returns a list of each function applied to the value. `apply_all([str.upper, len], "hi")` should return `["HI", 2]`.

**0.2 (E)** Write a generator `fib()` that yields Fibonacci numbers forever (0, 1, 1, 2, 3, 5, ...). Print the first 10 using `next()` or `itertools.islice`.

**0.3 (M)** Write a decorator `retry(times)` that calls the wrapped function again if it raises an exception, up to `times` attempts in total, then re-raises the last exception. Test it on a function that fails randomly with probability 0.7.

**0.4 (M)** Write a context manager class `Indent` with a `print(text)` method that indents text by 2 spaces per level of nesting. Use a class attribute for the current level. This code should print "a" with no indent, "b" with 2 spaces, "c" with 4 spaces and "d" with 2 spaces:

```python
print("a")
with Indent() as ind:
    ind.print("b")
    with Indent() as ind2:
        ind2.print("c")
    ind.print("d")
```

**0.5 (H)** Write a generator `running_average()` that receives numbers through `send()` and yields the average so far after each one. Hint: you must call `next(gen)` once before the first `send()` to advance it to the first `yield` (this is called *priming*).

### Solutions

**0.1**

```python
def apply_all(funcs, value):
    return [f(value) for f in funcs]

print(apply_all([str.upper, len], "hi"))  # ['HI', 2]
```

**0.2**

```python
from itertools import islice

def fib():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

print(list(islice(fib(), 10)))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

The generator is infinite, but it only computes what you ask for. This laziness is why streaming LLM tokens are delivered through generators.

**0.3**

```python
import functools, random

def retry(times):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    print(f"attempt {attempt} failed: {exc}")
            raise last_exc
        return wrapper
    return decorator

@retry(times=5)
def flaky():
    if random.random() < 0.7:
        raise RuntimeError("server busy")
    return "ok"

print(flaky())
```

A decorator that takes arguments needs three layers: `retry(5)` returns `decorator`, which receives `flaky` and returns `wrapper`.

**0.4**

```python
class Indent:
    level = 0                          # shared by all instances

    def __enter__(self):
        Indent.level += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        Indent.level -= 1
        return False

    def print(self, text):
        print("  " * Indent.level + text)

print("a")
with Indent() as ind:
    ind.print("b")
    with Indent() as ind2:
        ind2.print("c")
    ind.print("d")
```

**0.5**

```python
def running_average():
    total, count, average = 0.0, 0, None
    while True:
        value = yield average     # pause; hand out average, receive value
        total += value
        count += 1
        average = total / count

gen = running_average()
next(gen)              # prime: run to the first yield (yields None)
print(gen.send(10))    # 10.0
print(gen.send(20))    # 15.0
print(gen.send(30))    # 20.0
```

The line `value = yield average` does two things: it hands `average` out to the caller and pauses, and when resumed by `send(x)` it evaluates to `x`. Keep this in mind for Module 3.

## Module 1: Why async exists

Asyncio lets one thread make progress on many tasks by switching to another task whenever the current one is waiting. It speeds up programs that wait a lot (I/O-bound), not programs that compute a lot (CPU-bound). Agent code is overwhelmingly I/O-bound, which is why asyncio dominates the field.

### 1.1 The problem: waiting

Here is a program that asks a (simulated) LLM three questions. Each call takes 1 second, the way a real API call might.

```python
import time

def fake_llm(prompt: str) -> str:
    time.sleep(1)                    # pretend we are waiting for the network
    return f"answer to {prompt!r}"

start = time.perf_counter()
for q in ["q1", "q2", "q3"]:
    print(fake_llm(q))
print(f"took {time.perf_counter() - start:.2f}s")   # about 3.00s
```

During those 3 seconds the CPU does almost nothing. The program sends a request, then sits idle until the answer arrives. If the three questions are independent, we should be able to send all three, then wait for all three, finishing in about 1 second. Module 2 shows exactly that.

### 1.2 The restaurant analogy

Imagine a restaurant with one waiter.

- **Synchronous waiter:** takes table 1's order, walks it to the kitchen, stands at the kitchen door until the food is ready, serves it, and only then goes to table 2. Most of the evening is spent standing still.
- **Asynchronous waiter:** takes table 1's order, hands it to the kitchen, and immediately goes to table 2 while the food cooks. When the kitchen rings a bell, the waiter goes back and serves table 1. One waiter, many tables, no standing around.

The asynchronous waiter is still one person doing one thing at a time. The speed comes from never waiting idle. In asyncio, the waiter is the **event loop**, each table is a **task**, and the kitchen bell is an I/O event (a network response arriving).

There is one catch. If a customer asks the waiter to peel 500 potatoes, the waiter is stuck peeling and every other table waits. That is CPU-bound work, and it blocks the async waiter just as badly as the sync one. Module 8 shows how to hand such work to a separate kitchen hand (a thread or process).

### 1.3 Concurrency vs parallelism

These two words are often confused in interviews.

- **Concurrency** means several tasks are *in progress* during the same period, with their steps interleaved. One waiter serving many tables is concurrency.
- **Parallelism** means several tasks are *executing* at the same instant, on different CPU cores. Five waiters each serving a table at the same moment is parallelism.

Asyncio gives you concurrency on a single thread. It does not give you parallelism by itself.

### 1.4 I/O-bound vs CPU-bound

A task is **I/O-bound** when most of its time is spent waiting for input or output: network requests, disk reads, database queries, waiting for a user. It is **CPU-bound** when most of its time is spent computing: matrix multiplication, parsing a huge file, tokenising a corpus, training a model.

| Workload | Bound by | Good fit |
| --- | --- | --- |
| Calling an LLM API | Network I/O | asyncio |
| Running 50 web searches for an agent | Network I/O | asyncio |
| Serving many chat users from one API server | Network I/O | asyncio (FastAPI) |
| Reading many files from cloud storage | Network and disk I/O | asyncio or threads |
| Tokenising a 10 GB corpus | CPU | processes (`multiprocessing`) |
| Training or running a PyTorch model on GPU | GPU and CPU | the framework's own parallelism; call it from async code via a thread (Module 8) |
| Calling a blocking library with no async version | Waiting, but blocks the thread | `asyncio.to_thread` (Module 8) |

### 1.5 Three tools for concurrency in Python

| Tool | How it switches | Parallel on multiple cores? | Typical scale | Main risk |
| --- | --- | --- | --- | --- |
| `asyncio` (coroutines) | The task itself says "I'm waiting" with `await` (cooperative) | No, one thread | Thousands of concurrent connections | One blocking call freezes everything |
| `threading` | The operating system interrupts threads at any moment (preemptive) | Limited by the GIL on the default build (see 1.6) | Tens to hundreds of threads | Race conditions, harder to reason about |
| `multiprocessing` | Separate processes, each with its own interpreter | Yes | Roughly one per CPU core | Memory cost, data must be copied between processes |

Asyncio's big advantage for agent work is that switching happens only at `await` points you can see in the code. Between two `await`s, no other task can interfere with your variables. That makes concurrent code much easier to reason about than threads.

### 1.6 The GIL and free-threaded Python

The standard CPython interpreter has a Global Interpreter Lock (GIL): only one thread runs Python bytecode at a time. Threads still help with I/O, because a thread releases the GIL while it waits for the network, but they do not speed up pure-Python CPU work.

Since Python 3.13 there is an optional **free-threaded** build without the GIL. It became an officially supported (no longer experimental) build in Python 3.14 under PEP 779, though it is still not the default interpreter. The official macOS installers for 3.15 install free-threading support by default ([python.org](https://www.python.org/downloads/release/python-3150/)). Free threading changes the threads-versus-processes trade-off for CPU work, but it does not change when to use asyncio: asyncio remains the standard tool for I/O-heavy code such as agents, because coroutines are far cheaper than threads and switching only at `await` keeps code predictable.

### Check yourself

1. An agent calls an LLM, then 5 search tools, then the LLM again. Is it I/O-bound or CPU-bound? *Answer: I/O-bound; nearly all of its time is spent waiting on network responses.*
2. Can asyncio make a loop that computes prime numbers faster? *Answer: No. CPU-bound work gains nothing from switching tasks; use processes.*
3. What is the difference between concurrency and parallelism? *Answer: concurrency is many tasks in progress, interleaved; parallelism is many tasks executing at the same instant on different cores.*

### Exercises

**1.1 (E)** Classify each as I/O-bound or CPU-bound: (a) downloading 200 web pages; (b) computing embeddings locally with a sentence-transformer model on CPU; (c) sending 1,000 prompts to a hosted LLM API; (d) resizing 10,000 images; (e) waiting for a human to approve an agent's action.

**1.2 (E)** A program makes 10 independent API calls of 2 seconds each. Estimate the total time (a) sequentially, (b) fully concurrently with asyncio, (c) with asyncio but at most 3 calls at a time.

**1.3 (M)** Explain, using the waiter analogy, why calling `time.sleep(5)` inside an async program is a serious bug while `await asyncio.sleep(5)` is fine. (You will see the code in Module 2; answer from the concepts.)

**1.4 (M)** Your team lead says: "We switched to free-threaded Python, so we no longer need asyncio for our agent server." Give two reasons this conclusion is wrong.

### Solutions

**1.1** (a) I/O-bound. (b) CPU-bound (or accelerator-bound with a GPU). (c) I/O-bound. (d) CPU-bound. (e) I/O-bound in the broad sense: the program is waiting on an external event, which asyncio handles well (for example with an `asyncio.Event`, Module 6).

**1.2** (a) 10 × 2 = 20 seconds. (b) About 2 seconds, since all calls wait at the same time. (c) Batches of 3, 3, 3 and 1 means 4 rounds × 2 seconds = about 8 seconds. Module 6 shows how to impose such a limit with `asyncio.Semaphore`; real APIs require it because of rate limits.

**1.3** `time.sleep(5)` is the waiter standing at the kitchen door for 5 seconds: the waiter is the only one who can serve tables, so every table is frozen. `await asyncio.sleep(5)` is the waiter saying "wake me in 5 seconds" and going to serve other tables meanwhile. The first blocks the event loop; the second yields control to it.

**1.4** First, a thread costs far more memory and scheduling overhead than a coroutine, so thousands of concurrent waiting connections are much cheaper with asyncio. Second, the agent server spends its time waiting on I/O, which the GIL never prevented threads from doing; removing the GIL helps CPU-bound threads, not waiting. A third point: the server's stack (FastAPI, async LLM SDKs, MCP) is async-native, and preemptive threads bring race conditions that cooperative `await` points avoid.

## Module 2: Your first coroutines

Three keywords carry all of asyncio: `async def` defines a coroutine function, `await` pauses until something finishes, and `asyncio.run()` starts the event loop that drives everything. By the end of this module you will turn the 3-second program from Module 1 into a 1-second one.

### 2.1 async def creates a coroutine function

```python
import asyncio

async def greet(name: str) -> str:
    return f"hello, {name}"

coro = greet("ada")
print(coro)            # <coroutine object greet at 0x...>
```

Just as with generators (Module 0.5), calling a coroutine function does **not** run its body. It returns a **coroutine object**: a paused computation that something must drive. If you never drive it, Python warns you when it is garbage-collected:

```text
RuntimeWarning: coroutine 'greet' was never awaited
```

That warning almost always means you forgot an `await`. Learn to recognise it; it is the most common async bug.

### 2.2 asyncio.run drives a coroutine

```python
import asyncio

async def greet(name: str) -> str:
    return f"hello, {name}"

result = asyncio.run(greet("ada"))
print(result)          # hello, ada
```

`asyncio.run(coro)` creates an event loop, runs the coroutine until it finishes, returns its result, and closes the loop. A program normally calls `asyncio.run` exactly once, at the top, on a function conventionally called `main`:

```python
async def main() -> None:
    ...  # everything async happens in here

if __name__ == "__main__":
    asyncio.run(main())
```

### 2.3 await waits for an awaitable

Inside an `async def` you can write `await something`. It means: "pause *me* here until `something` finishes, give me its result, and let the event loop run other work in the meantime."

```python
import asyncio

async def fake_llm(prompt: str) -> str:
    await asyncio.sleep(1)            # non-blocking wait
    return f"answer to {prompt!r}"

async def main() -> None:
    answer = await fake_llm("q1")     # runs fake_llm and waits for it
    print(answer)

asyncio.run(main())
```

Things you can `await` are called **awaitables**. There are three kinds you will use:

1. **Coroutine objects**, from calling an `async def` function.
2. **Tasks**, coroutines scheduled to run concurrently (Module 4).
3. **Futures**, low-level placeholders for a result that will arrive later. You rarely create them yourself, but libraries use them internally.

`await` is only legal inside an `async def`. Writing it in a normal function is a `SyntaxError`.

### 2.4 The crucial point: await alone is still sequential

This surprises almost everyone:

```python
import asyncio, time

async def fake_llm(prompt: str) -> str:
    await asyncio.sleep(1)
    return f"answer to {prompt!r}"

async def main() -> None:
    start = time.perf_counter()
    a = await fake_llm("q1")
    b = await fake_llm("q2")
    c = await fake_llm("q3")
    print(a, b, c)
    print(f"took {time.perf_counter() - start:.2f}s")   # about 3.00s!

asyncio.run(main())
```

Still 3 seconds. Each `await` waits for that one call to finish before the next line runs. Async code is not automatically concurrent; you must *ask* for concurrency by starting several things before waiting. The simplest way is `asyncio.gather`:

```python
async def main() -> None:
    start = time.perf_counter()
    a, b, c = await asyncio.gather(
        fake_llm("q1"), fake_llm("q2"), fake_llm("q3")
    )
    print(a, b, c)
    print(f"took {time.perf_counter() - start:.2f}s")   # about 1.00s
```

`gather` schedules all three coroutines at once, waits until all finish, and returns their results in the same order you passed them. All three sleeps overlap, so the total is about 1 second. Module 4 covers `gather` properly, along with the safer modern alternative, `TaskGroup`.

### 2.5 time.sleep vs asyncio.sleep

```python
async def bad():
    time.sleep(1)            # BLOCKS the whole event loop for 1s

async def good():
    await asyncio.sleep(1)   # yields to the loop; others run meanwhile
```

Replace `fake_llm`'s `asyncio.sleep` with `time.sleep` and run the `gather` version: it takes 3 seconds again, because `time.sleep` never gives control back to the loop. The same applies to `requests.get`, a synchronous database driver, or a heavy computation. Inside async code, every slow operation must be either awaited from an async library or moved off the loop (Module 8).

### 2.6 Where you can run async code

- **Scripts:** `asyncio.run(main())`.
- **Jupyter notebooks and IPython:** an event loop is already running, so `asyncio.run` raises `RuntimeError: asyncio.run() cannot be called from a running event loop`. Instead, write `await main()` directly in a cell; notebooks support top-level `await`.
- **The asyncio REPL:** run `python -m asyncio` for an interactive prompt where top-level `await` works. Great for experimenting.
- **Frameworks:** FastAPI, MCP servers and LangGraph run the loop for you; you just write `async def` functions.

### Check yourself

1. What does `fake_llm("x")` return, without `await`? *Answer: a coroutine object; the body has not run.*
2. Three `await` lines in a row each wait 1 second. How long does it take? *Answer: about 3 seconds; consecutive awaits run one after another.*
3. Why does `time.sleep` inside a coroutine ruin concurrency? *Answer: it blocks the single thread, so the event loop cannot run any other task during the sleep.*

### Exercises

**2.1 (E)** Write `async def add(a, b)` that waits 0.5 seconds with `asyncio.sleep` and returns `a + b`. Call it from `main` with `await` and print the result.

**2.2 (E)** This code prints a warning and no answer. Find and fix the bug.

```python
import asyncio

async def fetch() -> str:
    await asyncio.sleep(0.1)
    return "data"

async def main():
    result = fetch()
    print(result)

asyncio.run(main())
```

**2.3 (M)** Write `fake_tool(name, seconds)` that sleeps `seconds` and returns `f"{name} done"`. In `main`, run `fake_tool("search", 2)`, `fake_tool("weather", 1)` and `fake_tool("calc", 0.5)` concurrently with `gather`. Print the results and the total time. Predict the total time and the order of the results before you run it.

**2.4 (M)** Run the `gather` version of 2.3 again with `time.sleep` instead of `asyncio.sleep`. Measure the time and explain the result in two sentences.

**2.5 (H)** Without running it, predict the exact output of this program. Then run it to check.

```python
import asyncio

async def worker(name: str, delay: float) -> str:
    print(f"{name} start")
    await asyncio.sleep(delay)
    print(f"{name} end")
    return name

async def main():
    results = await asyncio.gather(worker("A", 0.3), worker("B", 0.1), worker("C", 0.2))
    print(results)

asyncio.run(main())
```

### Solutions

**2.1**

```python
import asyncio

async def add(a: int, b: int) -> int:
    await asyncio.sleep(0.5)
    return a + b

async def main() -> None:
    print(await add(2, 3))   # 5

asyncio.run(main())
```

**2.2** `fetch()` without `await` creates a coroutine object that is never run, so `print` shows `<coroutine object ...>` and Python warns that it was never awaited. Fix: `result = await fetch()`.

**2.3**

```python
import asyncio, time

async def fake_tool(name: str, seconds: float) -> str:
    await asyncio.sleep(seconds)
    return f"{name} done"

async def main() -> None:
    start = time.perf_counter()
    results = await asyncio.gather(
        fake_tool("search", 2), fake_tool("weather", 1), fake_tool("calc", 0.5)
    )
    print(results)   # ['search done', 'weather done', 'calc done']
    print(f"took {time.perf_counter() - start:.2f}s")   # about 2.00s

asyncio.run(main())
```

Total time is the slowest call (2 s), not the sum (3.5 s). Results come back in the order you passed the coroutines, not the order they finished.

**2.4** It takes about 3.5 seconds. `time.sleep` blocks the event loop's only thread, so each coroutine's sleep must finish before the loop can start the next one, and the three waits add up instead of overlapping.

**2.5** Output:

```text
A start
B start
C start
B end
C end
A end
['A', 'B', 'C']
```

All three start before any sleep finishes, because each one pauses at its first `await`. They end in order of delay (0.1, 0.2, 0.3). The result list keeps the argument order.

## Module 3: How the event loop works

The event loop is a scheduler that runs one task at a time until that task pauses at an `await`, then picks the next task that is ready. Understanding this one sentence explains every asyncio behaviour you will meet, including every bug. This module builds a working toy event loop in about 40 lines so you can see it with no magic.

### 3.1 The loop in plain words

The event loop keeps two collections:

- A **ready queue** of tasks that can run right now.
- A set of **waiting** tasks, each waiting for something: a timer to expire, a network socket to receive data, another task to finish.

It repeats forever:

1. Take the next task from the ready queue and run it until it pauses at an `await` that actually has to wait (or until it finishes).
2. Record what the paused task is waiting for.
3. Ask the operating system: "which of the things we are waiting on are done?" The OS can watch thousands of sockets at once through mechanisms called `epoll` (Linux), `kqueue` (macOS) and IOCP (Windows). If nothing is ready, the loop sleeps until something is.
4. Move every task whose wait is over back into the ready queue.

&#91;embedded content: the event loop cycle · 4 steps\]

The cycle repeats thousands of times per second; step 2 is the only place your code runs, so anything slow there stalls the whole cycle.

Two consequences follow, and you should memorise both:

- **A task runs without interruption between two suspension points.** No other task can run while your code is between `await`s. This is called *cooperative multitasking*: tasks cooperate by giving up control.
- **A task that never suspends starves everyone else.** A long computation or a blocking call (`time.sleep`, `requests.get`) holds the only thread, so the loop cannot run anything else and cannot even notice that responses have arrived.

### 3.2 Build a toy event loop

Recall from Module 0.5 that a generator can pause at `yield` and be resumed by whoever calls `next()`. That is all a scheduler needs. In our toy, a task is a generator that yields the number of seconds it wants to sleep.

```python
import heapq
import time
from collections import deque

T0 = time.monotonic()

def log(msg: str) -> None:
    print(f"{time.monotonic() - T0:4.1f}s  {msg}")

def sleep(seconds: float):
    yield seconds                      # tell the loop: wake me after `seconds`

def task(name: str, delays: list[float]):
    for d in delays:
        log(f"{name} waits {d}s")
        yield from sleep(d)            # delegate to the sleep generator
    log(f"{name} finished")

class ToyLoop:
    def __init__(self):
        self.ready = deque()           # tasks that can run now
        self.sleeping = []             # heap of (wake_time, counter, task)
        self.counter = 0               # tie-breaker so the heap never compares generators

    def spawn(self, gen) -> None:
        self.ready.append(gen)

    def run(self) -> None:
        while self.ready or self.sleeping:
            # 1. wake every sleeper whose time has come
            now = time.monotonic()
            while self.sleeping and self.sleeping[0][0] <= now:
                _, _, gen = heapq.heappop(self.sleeping)
                self.ready.append(gen)
            # 2. nothing ready: the loop itself waits for the next wake-up
            if not self.ready:
                time.sleep(self.sleeping[0][0] - now)
                continue
            # 3. run one ready task until it yields or finishes
            gen = self.ready.popleft()
            try:
                seconds = next(gen)
            except StopIteration:
                continue                # the task is done
            self.counter += 1
            heapq.heappush(self.sleeping, (time.monotonic() + seconds, self.counter, gen))

loop = ToyLoop()
loop.spawn(task("A", [1, 1]))
loop.spawn(task("B", [0.5, 0.5, 0.5]))
loop.run()
```

Output (timings approximate):

```text
 0.0s  A waits 1s
 0.0s  B waits 0.5s
 0.5s  B waits 0.5s
 1.0s  A waits 1s
 1.0s  B waits 0.5s
 1.5s  B finished
 2.0s  A finished
```

Run one after the other, A and B would take 3.5 seconds. Interleaved, they take 2 seconds, the length of the longest task. Notice that the loop calls `time.sleep` only when *no* task is ready; real asyncio does the same thing, except it waits on the OS for sockets *and* timers at once.

### 3.3 From the toy to real asyncio

| Toy loop | Real asyncio |
| --- | --- |
| Generator function with `yield` | `async def` coroutine function |
| `yield from sleep(d)` | `await asyncio.sleep(d)` |
| The task yields a number of seconds | Deep down, the coroutine yields a `Future`; the loop resumes it when the Future gets a result |
| `loop.spawn(gen)` | `asyncio.create_task(coro)` (Module 4) |
| `loop.run()` | `asyncio.run(main())` |
| `time.sleep` when idle | Waiting on `epoll`/`kqueue`/IOCP with a timeout |

The resemblance is not an accident. Before Python 3.5 introduced `async`/`await`, asyncio coroutines literally were generators using `yield from`. That old style (`@asyncio.coroutine`) was removed in Python 3.11, but the machinery underneath is the same idea.

### 3.4 When does await actually pause?

`await` is a *possible* switch point, not a guaranteed one. Control returns to the loop only when the chain of awaits reaches something that genuinely has to wait (a Future without a result yet). Awaiting a coroutine that never waits runs it to completion immediately, like a normal function call:

```python
async def double(x: int) -> int:
    return x * 2            # never suspends

async def main():
    y = await double(21)    # runs straight through; no other task gets a turn
```

If you need to give other tasks a turn inside a long loop, `await asyncio.sleep(0)` yields to the loop exactly once and resumes as soon as possible.

### Check yourself

1. While task A runs the line `x = compute()` (no await), can task B run? *Answer: No. Only one task runs at a time and A has not yielded.*
2. In the toy loop, why does the heap entry contain a counter? *Answer: if two tasks have the same wake time, `heapq` compares the next tuple element; generators cannot be compared, so the counter breaks ties.*
3. Does `await some_coroutine()` always let other tasks run? *Answer: No, only if something inside it actually suspends.*

### Exercises

**3.1 (E)** Add a third task `C` with delays `[0.2] * 5` to the toy loop. Predict when C finishes and the total runtime, then run it.

**3.2 (M)** Extend `ToyLoop` so tasks can `return` a value and `run()` returns a dict mapping each task's name to its return value. Hint: the return value of a generator is stored on the `StopIteration` exception as `exc.value`. You will need to pass a name to `spawn`.

**3.3 (M)** Predict the final value of `counter`, then run it. Explain the result. Then remove the `await asyncio.sleep(0)` line and explain the new result.

```python
import asyncio

counter = 0

async def increment() -> None:
    global counter
    value = counter
    await asyncio.sleep(0)     # a switch point
    counter = value + 1

async def main():
    await asyncio.gather(*(increment() for _ in range(100)))
    print(counter)

asyncio.run(main())
```

**3.4 (H)** Write an asyncio program with a `heartbeat` coroutine that prints "tick" every 0.2 seconds, 10 times, and a `crunch` coroutine that sums `range(30_000_000)` in a plain loop. Run both with `gather`. Observe what happens to the ticks. Then modify `crunch` so it yields every 1,000,000 iterations with `await asyncio.sleep(0)` and compare. Explain what you see.

### Solutions

**3.1** C's five 0.2-second sleeps end at about 1.0 s, so C finishes at about 1.0 s. Total runtime stays about 2.0 s, set by the longest task (A).

**3.2**

```python
class ToyLoop:
    def __init__(self):
        self.ready = deque()
        self.sleeping = []
        self.counter = 0
        self.results = {}

    def spawn(self, name: str, gen) -> None:
        self.ready.append((name, gen))

    def run(self) -> dict:
        while self.ready or self.sleeping:
            now = time.monotonic()
            while self.sleeping and self.sleeping[0][0] <= now:
                _, _, item = heapq.heappop(self.sleeping)
                self.ready.append(item)
            if not self.ready:
                time.sleep(self.sleeping[0][0] - now)
                continue
            name, gen = self.ready.popleft()
            try:
                seconds = next(gen)
            except StopIteration as exc:
                self.results[name] = exc.value   # the generator's return value
                continue
            self.counter += 1
            heapq.heappush(self.sleeping, (time.monotonic() + seconds, self.counter, (name, gen)))
        return self.results

def job(n: int):
    yield from sleep(0.1 * n)
    return n * n

loop = ToyLoop()
for i in range(1, 4):
    loop.spawn(f"job{i}", job(i))
print(loop.run())   # {'job1': 1, 'job2': 4, 'job3': 9}
```

**3.3** With the `sleep(0)`, the result is `1`. Every task reads `counter` (0), then yields; all 100 tasks read 0 before any writes, and each then writes 0 + 1. This is a race condition, a *lost update*. Without the `sleep(0)`, each task runs its read and write with no switch point in between, so the result is `100`. Lesson: shared state is safe across code with no `await`, and unsafe when an `await` sits between reading and writing it. Module 6 introduces `asyncio.Lock` for the cases where you need an `await` in that gap.

**3.4**

```python
import asyncio, time

T0 = time.monotonic()

async def heartbeat() -> None:
    for _ in range(10):
        print(f"{time.monotonic() - T0:4.1f}s tick")
        await asyncio.sleep(0.2)

async def crunch(cooperative: bool) -> int:
    total = 0
    for i in range(30_000_000):
        total += i
        if cooperative and i % 1_000_000 == 0:
            await asyncio.sleep(0)
    return total

async def main():
    await asyncio.gather(heartbeat(), crunch(cooperative=False))

asyncio.run(main())
```

Non-cooperative: you see one tick at 0.0 s, then nothing for the seconds the loop takes, then the remaining ticks resume only after `crunch` finishes. The loop was blocked. Cooperative: ticks appear roughly on time, because `crunch` yields about 30 times. The program is still not faster overall (the CPU work is the same), but the loop stays responsive. In real services, prefer moving such work off the loop entirely (Module 8) over sprinkling `sleep(0)`.

## Module 4: Running things concurrently

A **Task** wraps a coroutine and schedules it on the event loop so it runs concurrently with everything else. Modern code creates tasks inside an `asyncio.TaskGroup`, which guarantees that every task is finished or cancelled when the block ends. `gather`, `as_completed` and `wait` remain useful for specific patterns, and you will meet all of them in agent codebases.

### 4.1 create\_task

```python
import asyncio, time

async def fake_llm(prompt: str, delay: float = 1.0) -> str:
    await asyncio.sleep(delay)
    return f"answer to {prompt!r}"

async def main():
    start = time.perf_counter()
    t1 = asyncio.create_task(fake_llm("q1"))   # starts running at the next switch point
    t2 = asyncio.create_task(fake_llm("q2"))
    print("tasks created; doing other work...")
    a = await t1                                 # wait for each result
    b = await t2
    print(a, b, f"{time.perf_counter() - start:.2f}s")   # about 1.00s

asyncio.run(main())
```

Compare with Module 2.4: awaiting a *coroutine* runs it now and waits; creating a *task* schedules it to run in the background, so both sleeps overlap. Awaiting the task later just collects the result (or re-raises its exception).

Useful task methods: `task.done()`, `task.result()` (only when done), `task.cancel()`, `task.get_name()`, `task.add_done_callback(fn)`. `asyncio.current_task()` returns the task you are running in.

**Keep a reference to every task.** The event loop holds only a *weak* reference to tasks. If you write `asyncio.create_task(log_event())` and throw away the result, the task can be garbage-collected before it finishes and silently disappear. Either await it, put it in a TaskGroup, or store it (see 4.6).

### 4.2 TaskGroup: the default choice (Python 3.11+)

```python
async def main():
    prompts = ["q1", "q2", "q3"]
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(fake_llm(p)) for p in prompts]
    # leaving the block waits for every task
    answers = [t.result() for t in tasks]
    print(answers)
```

A TaskGroup gives **structured concurrency**: tasks cannot outlive the block that created them. Its rules:

- Leaving the `async with` block waits until every task in the group is done.
- If any task raises an exception (other than `CancelledError`), the group cancels all the other tasks, waits for them to finish cancelling, and then raises an `ExceptionGroup` containing every failure. Module 5 shows how to catch it with `except*`.
- If the body of the block itself raises, the tasks are cancelled too.

That last property matters enormously in production: no orphaned LLM calls burning tokens after the request that started them has failed.

**New in Python 3.15:** `tg.cancel()` ends a task group early on purpose, cancelling its remaining tasks and exiting the block without raising. Tested on Python 3.15.2: errors that already happened are still raised, the body of the `async with` block is itself cancelled at its next `await`, and the block then exits normally. Use it when the goal is met, for example once one search result is good enough ([What's new in 3.15](https://docs.python.org/3/whatsnew/3.15.html)).

```python
# Python 3.15+
async def first_good_answer(prompts: list[str]) -> str | None:
    best = None
    async def try_one(p: str):
        nonlocal best
        answer = await fake_llm(p, delay=len(p) * 0.1)
        if best is None:
            best = answer
            tg.cancel()            # stop everyone else
    async with asyncio.TaskGroup() as tg:
        for p in prompts:
            tg.create_task(try_one(p))
    return best
```

### 4.3 gather

`asyncio.gather(*aws)` runs awaitables concurrently and returns their results in argument order. It is concise and very common, but its error handling differs from TaskGroup in ways interviewers like to probe:

| Situation | `gather` (default) | `gather(..., return_exceptions=True)` | `TaskGroup` |
| --- | --- | --- | --- |
| One task raises | First exception propagates to the caller; **other tasks keep running** | Exceptions are returned in the result list like values | Other tasks are cancelled; all errors raised together as an `ExceptionGroup` |
| Caller is cancelled | All children are cancelled | All children are cancelled | All children are cancelled |
| Results | List in argument order | List in argument order, mixing values and exceptions | Read `task.result()` after the block |

Use `gather(..., return_exceptions=True)` when you want partial results, such as "run 10 tools, keep whatever succeeded". Use TaskGroup when the tasks belong together and a failure should stop the whole job.

&#91;embedded content: gather vs TaskGroup when one task fails · measured timeline\]

### 4.4 as\_completed: handle results as they arrive

```python
async def main():
    delays = {"slow": 3, "medium": 2, "fast": 1}
    coros = [fake_llm(name, d) for name, d in delays.items()]
    for next_done in asyncio.as_completed(coros):
        answer = await next_done           # whichever finishes next
        print(answer)                      # fast, then medium, then slow
```

This is how you stream partial results to a user ("3 of 10 sources searched..."). Since Python 3.13, `as_completed` also works with `async for`, and when you pass it tasks it yields those same task objects, so you can tell which one finished:

```python
# Python 3.13+
tasks = [asyncio.create_task(fake_llm(n, d), name=n) for n, d in delays.items()]
async for task in asyncio.as_completed(tasks):
    print(task.get_name(), task.result())
```

### 4.5 wait: fine-grained control

`asyncio.wait(tasks, return_when=...)` returns two sets, `(done, pending)`, and never raises the tasks' exceptions itself. It accepts only tasks and futures, not bare coroutines (passing coroutines raises `TypeError` since Python 3.11).

```python
async def fastest_provider(prompt: str) -> str:
    tasks = {
        asyncio.create_task(fake_llm(f"{prompt} via A", 1.5)),
        asyncio.create_task(fake_llm(f"{prompt} via B", 0.7)),
        asyncio.create_task(fake_llm(f"{prompt} via C", 1.1)),
    }
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()                        # stop paying for the losers
    await asyncio.gather(*pending, return_exceptions=True)   # let them finish cancelling
    return done.pop().result()
```

This is called **request hedging**: send the same request to several providers or replicas and keep the fastest. Other `return_when` values are `FIRST_EXCEPTION` and `ALL_COMPLETED` (the default), and `timeout=` stops waiting without cancelling anything.

### 4.6 Fire-and-forget, done safely

Sometimes you want a task to run in the background without waiting for it, for example to write an audit log of an agent action. Keep a strong reference and drop it when the task ends:

```python
background_tasks: set[asyncio.Task] = set()

def fire_and_forget(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    background_tasks.add(task)                       # strong reference
    task.add_done_callback(background_tasks.discard) # remove when finished
    return task
```

This exact pattern appears in the official asyncio documentation. Remember that exceptions in such tasks are only logged ("Task exception was never retrieved") unless you check them in the callback.

### 4.7 Choosing the right tool

| You want to... | Use |
| --- | --- |
| Run N things that belong together; failure of one should stop all | `TaskGroup` |
| Run N independent things and keep partial results | `gather(..., return_exceptions=True)` |
| Process results in completion order | `as_completed` |
| Take the first result and cancel the rest | `wait(FIRST_COMPLETED)`, or `TaskGroup` + `tg.cancel()` on 3.15+ |
| Run something in the background | `create_task` + stored reference |

An advanced note: Python 3.12 added `asyncio.eager_task_factory`. With it installed (`loop.set_task_factory(asyncio.eager_task_factory)`), a new task starts running immediately inside `create_task` until its first real suspension, which can speed up code that creates many tasks that often finish without waiting (for example cache hits). It changes execution order, so use it only after measuring.

### Check yourself

1. What is the difference between `await coro()` and `t = asyncio.create_task(coro())`? *Answer: the first runs the coroutine now and waits for it; the second schedules it to run concurrently and returns immediately.*
2. With plain `gather`, one of three tasks raises. What happens to the other two? *Answer: they keep running; only the exception reaches the caller.*
3. Why must you keep a reference to a fire-and-forget task? *Answer: the loop holds only a weak reference, so an unreferenced task can be garbage-collected before it finishes.*

### Exercises

**4.1 (E)** Use a TaskGroup to send 5 prompts to `fake_llm` with delays 0.5, 1.0, 0.2, 0.8 and 0.3 seconds. Print all answers in the original order and the total time.

**4.2 (M)** Write `research(topics)` that searches every topic concurrently (simulate with random delays from 0.1 to 1.5 s) and prints a progress line like `[2/5] got results for 'llamas' after 0.62s` as each search finishes. Use `as_completed`.

**4.3 (M)** Predict the output of both versions below, then run them. Explain the difference.

```python
import asyncio

async def ok(name, d):
    await asyncio.sleep(d)
    print(f"{name} finished")
    return name

async def boom():
    await asyncio.sleep(0.1)
    raise ValueError("tool crashed")

async def version_gather():
    try:
        await asyncio.gather(ok("A", 0.5), boom(), ok("B", 1.0))
    except ValueError as e:
        print("caught:", e)
    await asyncio.sleep(1.5)          # give leftovers time to print

async def version_taskgroup():
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(ok("A", 0.5))
            tg.create_task(boom())
            tg.create_task(ok("B", 1.0))
    except* ValueError as eg:
        print("caught:", eg.exceptions)
    await asyncio.sleep(1.5)

asyncio.run(version_gather())
print("----")
asyncio.run(version_taskgroup())
```

**4.4 (M)** Implement request hedging: `hedged(prompt, providers)` where `providers` is a dict of provider name to simulated latency. Return `(winner_name, answer)` from the fastest provider and make sure the others are cancelled. Print a message inside the provider coroutine when it is cancelled (catch `asyncio.CancelledError`, print, and re-raise).

**4.5 (H)** Write a class `BackgroundRunner` with `spawn(coro)`, a property `active` (number of running tasks), and `async def drain(timeout)` that waits up to `timeout` seconds for all background tasks and cancels whatever is left. Any exception raised by a background task must be printed by the runner, not lost.

### Solutions

**4.1**

```python
import asyncio, time

async def fake_llm(prompt: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return f"answer to {prompt!r}"

async def main():
    delays = [0.5, 1.0, 0.2, 0.8, 0.3]
    start = time.perf_counter()
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(fake_llm(f"q{i}", d)) for i, d in enumerate(delays)]
    print([t.result() for t in tasks])
    print(f"took {time.perf_counter() - start:.2f}s")   # about 1.00s

asyncio.run(main())
```

**4.2**

```python
import asyncio, random, time

async def search(topic: str) -> tuple[str, list[str]]:
    await asyncio.sleep(random.uniform(0.1, 1.5))
    return topic, [f"{topic} result {i}" for i in range(3)]

async def research(topics: list[str]) -> dict[str, list[str]]:
    start = time.perf_counter()
    results = {}
    for i, next_done in enumerate(asyncio.as_completed([search(t) for t in topics]), 1):
        topic, hits = await next_done
        results[topic] = hits
        print(f"[{i}/{len(topics)}] got results for {topic!r} after {time.perf_counter() - start:.2f}s")
    return results

asyncio.run(research(["llamas", "rag", "mcp", "vector dbs", "evals"]))
```

**4.3** Gather version:

```text
caught: tool crashed
A finished
B finished
```

The `ValueError` reaches the caller at 0.1 s, but A and B are not cancelled and finish later. TaskGroup version:

```text
caught: (ValueError('tool crashed'),)
```

The group cancels A and B as soon as `boom` fails, so neither prints "finished". The exception arrives wrapped in an `ExceptionGroup`, and `except*` unwraps it (Module 5).

**4.4**

```python
import asyncio

async def call_provider(name: str, latency: float, prompt: str) -> tuple[str, str]:
    try:
        await asyncio.sleep(latency)
        return name, f"{name} says: answer to {prompt!r}"
    except asyncio.CancelledError:
        print(f"{name} cancelled")
        raise                                  # always re-raise CancelledError

async def hedged(prompt: str, providers: dict[str, float]) -> tuple[str, str]:
    tasks = {asyncio.create_task(call_provider(n, lat, prompt)) for n, lat in providers.items()}
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    return done.pop().result()

print(asyncio.run(hedged("hi", {"openai": 1.2, "anthropic": 0.6, "local": 2.0})))
# openai cancelled / local cancelled (order may vary)
# ('anthropic', "anthropic says: answer to 'hi'")
```

If two tasks could finish in the same loop iteration, `done` may hold more than one; `done.pop()` picks an arbitrary one, which is acceptable for hedging.

**4.5**

```python
import asyncio

class BackgroundRunner:
    def __init__(self):
        self._tasks: set[asyncio.Task] = set()

    def spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._on_done)
        return task

    def _on_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            print(f"background task {task.get_name()} failed: {task.exception()!r}")

    @property
    def active(self) -> int:
        return len(self._tasks)

    async def drain(self, timeout: float) -> None:
        if not self._tasks:
            return
        done, pending = await asyncio.wait(set(self._tasks), timeout=timeout)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

async def job(n: float):
    await asyncio.sleep(n)
    if n == 0.2:
        raise RuntimeError("disk full")

async def main():
    runner = BackgroundRunner()
    for n in (0.1, 0.2, 5.0):
        runner.spawn(job(n))
    print("active:", runner.active)     # 3
    await runner.drain(timeout=1.0)     # prints the disk-full failure, cancels the 5 s job
    print("active:", runner.active)     # 0

asyncio.run(main())
```

Calling `task.exception()` in the callback marks the exception as retrieved, so asyncio does not print its own "never retrieved" warning. A web server would call `drain` during shutdown (Module 11).

## Module 5: Timeouts, cancellation and errors

Every network call in production needs a timeout, every task must be cancellable, and every failure must end up somewhere visible. Asyncio implements timeouts *through* cancellation, so learning cancellation properly is the single biggest step from toy code to production agent code.

### 5.1 Timeouts with asyncio.timeout (Python 3.11+)

```python
import asyncio

async def slow_llm(prompt: str) -> str:
    await asyncio.sleep(5)
    return "finally!"

async def main():
    try:
        async with asyncio.timeout(2):
            answer = await slow_llm("hello")
    except TimeoutError:
        answer = "Sorry, the model took too long."
    print(answer)

asyncio.run(main())
```

When the 2 seconds expire, asyncio cancels the work inside the block and, as the block exits, converts that cancellation into a `TimeoutError`. Since Python 3.11, `asyncio.TimeoutError` is the same class as the built-in `TimeoutError`, so catch the built-in one.

Related tools:

- `asyncio.timeout_at(deadline)` takes an absolute deadline in event loop time (`asyncio.get_running_loop().time() + seconds`). Useful for an overall budget across several steps.
- The context manager object has `.when()` and `.reschedule(new_deadline)` to read or move the deadline; `asyncio.timeout(None)` means "no deadline yet".
- `await asyncio.wait_for(coro, timeout=2)` is the older function form. It still works and you will see it in many codebases; in new code prefer `asyncio.timeout`.

### 5.2 How cancellation works

Calling `task.cancel()` does not stop the task instantly. It arranges for an `asyncio.CancelledError` to be *raised inside the task* at the `await` where it is currently paused. The task can run cleanup code, and normally the exception then propagates out and the task ends as "cancelled".

```python
import asyncio

async def stream_tokens():
    try:
        for i in range(100):
            await asyncio.sleep(0.1)
            print("token", i)
    except asyncio.CancelledError:
        print("user hit stop: cleaning up")
        raise                          # re-raise! do not swallow
    finally:
        print("connection closed")      # runs on success, error or cancel

async def main():
    task = asyncio.create_task(stream_tokens())
    await asyncio.sleep(0.35)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("task was cancelled:", task.cancelled())   # True

asyncio.run(main())
```

Three rules every async engineer follows:

1. **Put cleanup in `finally`** (or `async with`), so it runs on cancellation too.
2. **Always re-raise `CancelledError`** if you catch it. Swallowing it breaks timeouts, TaskGroups and server shutdown, because they all rely on cancellation propagating.
3. **`except Exception` does not catch `CancelledError`**, because since Python 3.8 it inherits from `BaseException`. That is deliberate. Avoid bare `except:` and `except BaseException:` unless you re-raise.

Cancellation happens at await points. A task stuck in a CPU loop or a blocking call cannot be cancelled until it next awaits, which is yet another reason never to block the loop.

### 5.3 shield: protect a critical step

`asyncio.shield(aw)` stops cancellation of the *outer* task from cancelling the inner awaitable. The outer `await` still raises `CancelledError`, but the shielded work keeps running.

```python
async def handle_request(state):
    result = await run_agent(state)
    # Even if the client disconnects now, finish writing the conversation to the DB.
    await asyncio.shield(save_conversation(state, result))
```

Use shield sparingly and keep a reference to the inner task if you need its result. A shielded operation that never finishes can block a clean shutdown.

### 5.4 ExceptionGroup and except\* (Python 3.11+)

When several tasks in a TaskGroup fail, you get all the errors at once in an `ExceptionGroup`. The `except*` syntax matches exceptions *inside* the group by type:

```python
import asyncio

async def tool(name: str, fail_with: type[Exception] | None, delay: float):
    await asyncio.sleep(delay)
    if fail_with:
        raise fail_with(f"{name} failed")
    return name

async def main():
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(tool("search", ValueError, 0.1))
            tg.create_task(tool("db", ConnectionError, 0.1))
            tg.create_task(tool("calc", None, 1.0))
    except* ValueError as eg:
        print("bad input:", [str(e) for e in eg.exceptions])
    except* ConnectionError as eg:
        print("network:", [str(e) for e in eg.exceptions])

asyncio.run(main())
# bad input: ['search failed']
# network: ['db failed']
```

Unlike ordinary `except`, several `except*` clauses can run for the same group, one per matching type. Unmatched exceptions are re-raised in a new group. Both tools failed in the same loop iteration, so both errors were collected before `calc` was cancelled.

### 5.5 Agent pattern: turn failures into messages

In an agent, a failing tool usually should not crash the whole run. The LLM can often recover if you tell it what went wrong. A small wrapper gives every tool call a timeout and converts errors into text:

```python
async def safe_tool_call(fn, args: dict, timeout: float = 10.0) -> str:
    try:
        async with asyncio.timeout(timeout):
            return str(await fn(**args))
    except TimeoutError:
        return f"ERROR: tool {fn.__name__} timed out after {timeout}s"
    except Exception as exc:              # does NOT catch CancelledError: good
        return f"ERROR: tool {fn.__name__} failed: {type(exc).__name__}: {exc}"
```

Because `except Exception` leaves `CancelledError` alone, the user can still stop the agent and the server can still shut down cleanly.

### Check yourself

1. What exception do you catch for `asyncio.timeout`? *Answer: the built-in `TimeoutError`.*
2. Why must you re-raise `CancelledError`? *Answer: timeouts, TaskGroups and shutdown depend on cancellation propagating; swallowing it leaves the task running and breaks them.*
3. Can `except Exception` catch `CancelledError`? *Answer: No, it is a `BaseException` subclass since Python 3.8.*

### Exercises

**5.1 (E)** Write `ask_with_fallback(prompt, primary_delay)` that calls a fake primary model (sleeps `primary_delay` seconds) with a 1-second timeout and, on timeout, calls a fake backup model that answers in 0.2 seconds. Test with delays of 0.5 and 3.

**5.2 (M)** Find the bug and fix it. The `main` function should finish in about 1 second but never finishes.

```python
import asyncio

async def worker():
    while True:
        try:
            await asyncio.sleep(10)
        except BaseException:
            print("ignoring error, retrying")

async def main():
    try:
        async with asyncio.timeout(1):
            await worker()
    except TimeoutError:
        print("timed out")

asyncio.run(main())
```

**5.3 (M)** Run three fake tools in a TaskGroup where one raises `KeyError`, one raises `TimeoutError` (use `asyncio.timeout` inside it) and one succeeds. Catch the errors with `except*` and print a summary. Then explain why the successful tool's result may or may not be available afterwards, depending on its delay.

**5.4 (M)** Write `run_with_budget(steps, budget)` where `steps` is a list of `(name, seconds)` simulated agent steps executed one after another. All steps together must finish within `budget` seconds. If the budget runs out, return the names of the completed steps and `"budget exceeded"`. Use a single `asyncio.timeout` around the loop.

**5.5 (H)** Write a coroutine `stream_with_stop(stop_after)` that simulates streaming 20 tokens (0.1 s each), collecting them into a list. A separate task cancels it after `stop_after` seconds. Requirements: the tokens received before the cancel must be saved with a simulated `save(tokens)` that takes 0.3 s, and the save must complete even though the streaming task is being cancelled. Use `shield`.

### Solutions

**5.1**

```python
import asyncio

async def primary(prompt: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return f"primary: {prompt}"

async def backup(prompt: str) -> str:
    await asyncio.sleep(0.2)
    return f"backup: {prompt}"

async def ask_with_fallback(prompt: str, primary_delay: float) -> str:
    try:
        async with asyncio.timeout(1):
            return await primary(prompt, primary_delay)
    except TimeoutError:
        return await backup(prompt)

async def main():
    print(await ask_with_fallback("hi", 0.5))   # primary: hi
    print(await ask_with_fallback("hi", 3))     # backup: hi (after about 1.2s)

asyncio.run(main())
```

**5.2** `except BaseException` catches the `CancelledError` that the timeout sends at 1 s. The worker prints its message, loops and sleeps again. A timeout fires only once, so nothing cancels the worker a second time and the program hangs forever (stop it with Ctrl+C). Fix: catch only what you mean, and let cancellation through:

```python
async def worker():
    while True:
        try:
            await asyncio.sleep(10)
        except Exception:          # CancelledError is not an Exception
            print("ignoring error, retrying")
```

**5.3**

```python
import asyncio

async def lookup():
    await asyncio.sleep(0.1)
    raise KeyError("user_id")

async def slow_api():
    async with asyncio.timeout(0.2):
        await asyncio.sleep(5)

async def calc(delay: float):
    await asyncio.sleep(delay)
    return 42

async def main(calc_delay: float):
    calc_task = None
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(lookup())
            tg.create_task(slow_api())
            calc_task = tg.create_task(calc(calc_delay))
    except* KeyError as eg:
        print("missing keys:", eg.exceptions)
    except* TimeoutError as eg:
        print("timeouts:", eg.exceptions)
    if calc_task.cancelled():
        print("calc was cancelled")
    else:
        print("calc result:", calc_task.result())

asyncio.run(main(calc_delay=0.05))   # missing keys: (KeyError('user_id'),) then calc result: 42
asyncio.run(main(calc_delay=1.0))    # missing keys: ... ; calc was cancelled
```

The `KeyError` at 0.1 s makes the group cancel its remaining tasks. If `calc` already finished (0.05 s), its result is kept; if not (1.0 s), it is cancelled. `slow_api` is cancelled at 0.1 s too, before its own 0.2 s timeout can fire, so the `TimeoutError` clause does not run here. Change `lookup`'s delay to 0.3 and both `except*` clauses run, because `slow_api` now times out first. The point of the exercise: in concurrent code, which errors you see depends on timing, so reason about it explicitly.

**5.4**

```python
import asyncio

async def run_with_budget(steps: list[tuple[str, float]], budget: float):
    completed = []
    try:
        async with asyncio.timeout(budget):
            for name, seconds in steps:
                await asyncio.sleep(seconds)     # one simulated agent step
                completed.append(name)
    except TimeoutError:
        return completed, "budget exceeded"
    return completed, "ok"

print(asyncio.run(run_with_budget([("plan", 0.5), ("search", 1.0), ("write", 1.0)], budget=2.0)))
# (['plan', 'search'], 'budget exceeded')
```

One timeout around the loop is an *overall* budget; a timeout inside the loop would be a *per-step* limit. Production agents usually want both.

**5.5**

```python
import asyncio

async def save(tokens: list[str]) -> None:
    await asyncio.sleep(0.3)
    print(f"saved {len(tokens)} tokens")

async def stream_with_stop(stop_after: float) -> None:
    tokens: list[str] = []

    async def stream():
        try:
            for i in range(20):
                await asyncio.sleep(0.1)
                tokens.append(f"t{i}")
        finally:
            # runs on normal completion AND on cancellation
            save_task = asyncio.create_task(save(list(tokens)))
            await asyncio.shield(save_task)

    task = asyncio.create_task(stream())
    await asyncio.sleep(stop_after)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("stream stopped")

asyncio.run(stream_with_stop(0.45))
# saved 4 tokens
# stream stopped
```

When the stream is cancelled at 0.45 s, four tokens have arrived. The `finally` block starts the save as its own task and awaits it through `shield`; after the save completes, the original `CancelledError` continues to propagate, so `await task` raises and "stream stopped" prints. If a *second* cancellation hit while the `finally` was awaiting, the `await` would raise immediately but the shielded save task would still run to completion. In production you would also hand that save task to a background runner (Exercise 4.5) so the server can wait for it at shutdown.

## Module 6: Coordination primitives

Once many tasks run at the same time, you need ways to limit them, feed them work and make them wait for each other. Asyncio provides `Queue`, `Semaphore`, `Lock`, `Event`, `Condition` and `Barrier`, which mirror the threading versions but are awaited instead of blocking. They are *not* thread-safe; use them only from code running on the event loop.

### 6.1 Semaphore: limit concurrency

LLM providers enforce rate limits, and sending 1,000 requests at once will get most of them rejected with HTTP 429 errors. A semaphore holds a fixed number of permits; `async with sem:` takes one, waiting if none are free, and gives it back on exit.

```python
import asyncio, time

async def fake_llm(prompt: str) -> str:
    await asyncio.sleep(1)
    return f"answer to {prompt}"

async def limited_call(sem: asyncio.Semaphore, prompt: str) -> str:
    async with sem:                       # at most N inside at once
        return await fake_llm(prompt)

async def main():
    sem = asyncio.Semaphore(3)
    start = time.perf_counter()
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(limited_call(sem, f"q{i}")) for i in range(10)]
    print(len(tasks), "answers", f"{time.perf_counter() - start:.2f}s")   # about 4.00s

asyncio.run(main())
```

This is the most-used primitive in agent and data-pipeline code: create all the tasks, let the semaphore throttle them. Create the semaphore inside a running loop (for example in `main`), not at import time.

### 6.2 Queue: producers and consumers

An `asyncio.Queue` passes items between tasks in first-in, first-out order. `await q.put(item)` adds an item, waiting if the queue is full; `await q.get()` removes one, waiting if it is empty. A bounded queue (`maxsize=...`) gives you **backpressure**: a fast producer is automatically slowed down to match slow consumers, so memory does not explode.

The classic worker pool:

```python
import asyncio, random

async def fetch(url: str) -> str:
    await asyncio.sleep(random.uniform(0.1, 0.5))
    return f"<html of {url}>"

async def worker(name: str, q: asyncio.Queue, results: dict) -> None:
    while True:
        url = await q.get()
        try:
            results[url] = await fetch(url)
            print(f"{name} fetched {url}")
        finally:
            q.task_done()                 # tell the queue this item is finished

async def main():
    urls = [f"https://example.com/page{i}" for i in range(12)]
    q: asyncio.Queue[str] = asyncio.Queue()
    for u in urls:
        q.put_nowait(u)
    results: dict[str, str] = {}
    workers = [asyncio.create_task(worker(f"w{i}", q, results)) for i in range(4)]
    await q.join()                        # wait until every item had task_done()
    for w in workers:
        w.cancel()                        # workers loop forever; stop them
    await asyncio.gather(*workers, return_exceptions=True)
    print(len(results), "pages")

asyncio.run(main())
```

**New in Python 3.13: `Queue.shutdown()`.** Calling `q.shutdown()` stops new `put` calls (they raise `asyncio.QueueShutDown`) and lets consumers drain what is left; once the queue is empty, `get` raises `QueueShutDown` too. With `q.shutdown(immediate=True)`, pending items are dropped and `get` raises at once. This replaces the old trick of putting a special "stop" value (a *sentinel*) on the queue:

```python
# Python 3.13+
async def worker(q: asyncio.Queue) -> None:
    while True:
        try:
            item = await q.get()
        except asyncio.QueueShutDown:
            return                        # clean exit
        ...
        q.task_done()
```

Variants: `asyncio.PriorityQueue` (smallest item first, for example `(priority, job)` tuples) and `asyncio.LifoQueue` (last in, first out).

### 6.3 Lock: protect a critical section

Recall the lost-update race of Exercise 3.3. When you must `await` between reading and writing shared state, guard that section with a lock:

```python
lock = asyncio.Lock()

async def increment():
    global counter
    async with lock:
        value = counter
        await asyncio.sleep(0)      # other tasks may run, but none can enter this block
        counter = value + 1
```

If the shared-state code has no `await` inside, you do not need a lock at all (Module 3). Over-locking is a common beginner mistake; under-locking around awaits is a common senior mistake.

### 6.4 Event: wait for a signal

An `Event` is a flag that tasks can wait on. `await event.wait()` blocks until another task calls `event.set()`. This models human-in-the-loop approval, "model loaded" signals and shutdown flags.

```python
async def agent(approved: asyncio.Event):
    print("agent: drafted an email, waiting for approval...")
    try:
        async with asyncio.timeout(5):
            await approved.wait()
        print("agent: approved, sending")
    except TimeoutError:
        print("agent: no approval, discarding draft")

async def human(approved: asyncio.Event):
    await asyncio.sleep(2)          # thinking...
    approved.set()

async def main():
    approved = asyncio.Event()
    async with asyncio.TaskGroup() as tg:
        tg.create_task(agent(approved))
        tg.create_task(human(approved))

asyncio.run(main())
```

### 6.5 Condition and Barrier, briefly

- `asyncio.Condition` combines a lock with waiting for a condition to become true: `async with cond: await cond.wait_for(lambda: len(buffer) >= 10)`, with another task calling `cond.notify_all()` after changing `buffer`. You rarely need it; a Queue or Event is usually simpler.
- `asyncio.Barrier(n)` (Python 3.11+) makes `n` tasks wait until all of them reach `await barrier.wait()`, then releases them together. Useful in tests and simulations.

### 6.6 Summary

| Primitive | Question it answers | Typical agent use |
| --- | --- | --- |
| `Semaphore(n)` | How many may run at once? | Cap concurrent LLM or API calls |
| `Queue` | How do tasks hand work to each other? | Worker pools, multi-stage pipelines, streaming events to a client |
| `Lock` | Who may touch this state right now? | Guarding a shared cache or file across an `await` |
| `Event` | Has something happened yet? | Human approval, shutdown signal, readiness |
| `Condition` | Wait until a condition on shared state holds | Rare; batching |
| `Barrier` | Wait until n tasks arrive | Tests, simulations |

### Check yourself

1. 10 calls of 1 s each run with `Semaphore(3)`. Roughly how long? *Answer: 4 s (rounds of 3, 3, 3, 1).*
2. What does `q.join()` wait for? *Answer: until every item put on the queue has had a matching `task_done()` call.*
3. When do you need an `asyncio.Lock`? *Answer: when an `await` sits between reading and writing shared state.*

### Exercises

**6.1 (E)** Modify the semaphore example to print, every time a call starts, how many calls are currently inside the semaphore. Confirm it never exceeds 3.

**6.2 (M)** Build a pipeline with a bounded queue (`maxsize=2`): one producer creates 10 "documents" quickly (0.05 s each), and two consumers "embed" them slowly (0.4 s each). Print when each document is produced and consumed. Observe and explain the backpressure. Use `Queue.shutdown()` if you are on 3.13+, or a sentinel value otherwise.

**6.3 (M)** Write a rate limiter class `RateLimiter(rate, burst)` that allows at most `rate` acquisitions per second on average, with bursts of up to `burst`, using the token bucket algorithm: the bucket refills continuously at `rate` tokens per second up to `burst`, and each call to `await limiter.acquire()` takes one token, waiting if none is available. Test it with 20 concurrent tasks, `rate=5`, `burst=2`, and print the time of each acquisition.

**6.4 (H)** Request coalescing. Many concurrent users ask the agent for the same expensive embedding at once. Write `Coalescer.get(key, fetch)` so that concurrent calls with the same key share a single in-flight `fetch(key)` call. Then make it robust: if one caller is cancelled, the shared fetch must continue for the other callers. Hint: think about what happens to an awaited task when the awaiting task is cancelled.

### Solutions

**6.1**

```python
inside = 0

async def limited_call(sem: asyncio.Semaphore, prompt: str) -> str:
    global inside
    async with sem:
        inside += 1
        print(f"start {prompt}: {inside} inside")
        try:
            return await fake_llm(prompt)
        finally:
            inside -= 1
```

No lock is needed for `inside`: the increments and decrements have no `await` in the middle.

**6.2**

```python
import asyncio, time

T0 = time.monotonic()
def log(msg): print(f"{time.monotonic() - T0:4.2f}s {msg}")

async def producer(q: asyncio.Queue) -> None:
    for i in range(10):
        await asyncio.sleep(0.05)
        await q.put(f"doc{i}")        # waits whenever the queue already holds 2
        log(f"produced doc{i}")
    q.shutdown()                       # Python 3.13+: no more items

async def consumer(name: str, q: asyncio.Queue) -> None:
    while True:
        try:
            doc = await q.get()
        except asyncio.QueueShutDown:
            return
        await asyncio.sleep(0.4)       # slow embedding
        log(f"{name} embedded {doc}")
        q.task_done()

async def main():
    q: asyncio.Queue[str] = asyncio.Queue(maxsize=2)
    async with asyncio.TaskGroup() as tg:
        tg.create_task(producer(q))
        tg.create_task(consumer("c1", q))
        tg.create_task(consumer("c2", q))

asyncio.run(main())
```

The first few documents are produced quickly, then production slows to about one document per 0.2 s, the combined speed of two consumers at 0.4 s each. The producer is held back by `put` on a full queue: that is backpressure. The whole run takes about 2 s (10 documents ÷ 2 consumers × 0.4 s). This solution needs Python 3.13 or newer: on 3.12 it fails with `AttributeError: 'Queue' object has no attribute 'shutdown'`. On 3.12, use one sentinel value per consumer instead (snippet 2 in the Python 3.12 guide), or the feature-detection helpers shown there.

**6.3**

```python
import asyncio, time

class RateLimiter:
    def __init__(self, rate: float, burst: int):
        self.rate = rate
        self.capacity = burst
        self.tokens = float(burst)
        self.updated = time.monotonic()
        self.lock = asyncio.Lock()       # waiters queue up fairly, in order

    async def acquire(self) -> None:
        async with self.lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                await asyncio.sleep((1 - self.tokens) / self.rate)

async def main():
    limiter = RateLimiter(rate=5, burst=2)
    t0 = time.monotonic()
    async def job(i: int):
        await limiter.acquire()
        print(f"job {i:2d} at {time.monotonic() - t0:.2f}s")
    async with asyncio.TaskGroup() as tg:
        for i in range(20):
            tg.create_task(job(i))

asyncio.run(main())
```

The first two jobs run at 0.00 s (the burst), then one job every 0.2 s, so job 19 runs at about 3.6 s. A semaphore limits how many calls are *in flight*; a rate limiter limits how many *start* per second. Real LLM APIs usually need both, plus a tokens-per-minute budget.

**6.4**

```python
import asyncio

class Coalescer:
    def __init__(self):
        self.inflight: dict[str, asyncio.Task] = {}

    async def get(self, key: str, fetch):
        task = self.inflight.get(key)
        if task is None:
            task = asyncio.create_task(fetch(key))
            self.inflight[key] = task
            task.add_done_callback(lambda _t: self.inflight.pop(key, None))
        # shield: cancelling THIS caller must not cancel the shared fetch
        return await asyncio.shield(task)

calls = 0
async def embed(key: str) -> list[float]:
    global calls
    calls += 1
    await asyncio.sleep(0.5)
    return [0.1, 0.2, 0.3]

async def main():
    c = Coalescer()
    impatient = asyncio.create_task(c.get("hello", embed))
    patient = [asyncio.create_task(c.get("hello", embed)) for _ in range(4)]
    await asyncio.sleep(0.1)
    impatient.cancel()
    print(await asyncio.gather(*patient))
    print("fetch calls:", calls)          # 1

asyncio.run(main())
```

Without `shield`, cancelling the impatient caller would cancel the task it is awaiting, which is the shared fetch, and all four patient callers would get `CancelledError`. When a task is cancelled while awaiting another task, the cancellation is passed on to the awaited task. This is one of the subtlest asyncio behaviours and a favourite senior-level interview question.

## Module 7: Async iteration and context managers

`async with` and `async for` are the async versions of `with` and `for`: they let setup, teardown and "get the next item" steps await I/O. Streaming LLM responses, HTTP clients, database sessions and MCP connections all use them, so you will write them daily.

### 7.1 async with

A normal context manager's `__enter__` and `__exit__` cannot await. An **async context manager** defines `__aenter__` and `__aexit__` as coroutines, and you use it with `async with`:

```python
import asyncio

class FakeConnection:
    async def __aenter__(self):
        await asyncio.sleep(0.1)             # e.g. TCP + TLS handshake
        print("connected")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await asyncio.sleep(0.1)             # e.g. graceful close
        print("closed")
        return False                         # do not swallow exceptions

    async def query(self, q: str) -> str:
        await asyncio.sleep(0.2)
        return f"rows for {q!r}"

async def main():
    async with FakeConnection() as conn:
        print(await conn.query("SELECT 1"))

asyncio.run(main())
```

The easier way to write one is `contextlib.asynccontextmanager` with an async generator, exactly like Module 0.4 but with `async def`:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def connection(url: str):
    print(f"connecting to {url}")
    await asyncio.sleep(0.1)
    conn = {"url": url}
    try:
        yield conn                           # the body of "async with" runs here
    finally:
        await asyncio.sleep(0.1)
        print("closed")                      # runs on success, error and cancellation

async def main():
    async with connection("postgres://db") as conn:
        print("using", conn)
```

When you need to open a variable number of resources (say, one MCP session per configured server), use `contextlib.AsyncExitStack`: `await stack.enter_async_context(cm)` registers each one, and leaving the stack closes them all in reverse order.

### 7.2 async for and async generators

`async for` iterates over an **asynchronous iterator**, whose "get next item" step is awaitable. The easiest way to create one is an **async generator**: an `async def` function that contains `yield`.

```python
import asyncio

async def fake_token_stream(text: str, delay: float = 0.05):
    for word in text.split():
        await asyncio.sleep(delay)       # waiting for the next chunk from the network
        yield word + " "

async def main():
    async for token in fake_token_stream("Asyncio makes streaming responses feel instant"):
        print(token, end="", flush=True)
    print()

asyncio.run(main())
```

This is exactly the shape of every streaming LLM API you will use in Module 10: the SDK gives you an async iterator of chunks (events), and you `async for` over them, forwarding each one to the user as it arrives. Users perceive a streamed answer as much faster because the first words appear after a fraction of a second.

Under the hood, `async for` calls `__aiter__()` once and then `await __anext__()` repeatedly until `StopAsyncIteration` is raised. You can write such a class by hand, but async generators are almost always simpler.

Async comprehensions work too: `words = [t async for t in fake_token_stream("a b c")]`.

### 7.3 Closing async generators properly

If you `break` out of an `async for`, the generator is suspended at its `yield` and its `finally` block has not run yet. It runs only when the generator is closed, which may happen much later during garbage collection, or at loop shutdown. For generators that hold resources (an open HTTP stream), close them explicitly with `contextlib.aclosing` (Python 3.10+):

```python
from contextlib import aclosing

async def stream_from_server():
    try:
        for i in range(1000):
            await asyncio.sleep(0.01)
            yield f"chunk{i}"
    finally:
        print("stream closed")           # must run promptly

async def main():
    async with aclosing(stream_from_server()) as stream:
        async for chunk in stream:
            if chunk == "chunk3":
                break                    # aclosing() closes the generator right here
    print("after loop")
```

Output: `stream closed` prints before `after loop`. Without `aclosing`, the order is not guaranteed. LLM SDK stream objects are async context managers for the same reason, which is why their examples write `async with client.messages.stream(...) as stream:`.

### 7.4 Useful streaming patterns

```python
async def batched(stream, n: int):
    """Group items from an async iterator into lists of size n."""
    batch = []
    async for item in stream:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

async def take(stream, n: int):
    """Yield at most n items, then stop."""
    count = 0
    async for item in stream:
        if count >= n:
            return
        yield item
        count += 1
```

Async generators compose like Unix pipes: `batched(take(fake_token_stream(text), 20), 5)`. Batching matters for ML work: embedding APIs are much cheaper per item when you send 100 texts per request instead of one.

### Check yourself

1. What makes a function an async generator? *Answer: it is defined with `async def` and contains `yield`.*
2. Why wrap a streaming generator in `aclosing`? *Answer: so its cleanup (`finally`) runs immediately when you stop iterating early, instead of at some unpredictable later time.*
3. Which methods does `async with` call? *Answer: `__aenter__` on entry and `__aexit__` on exit, both awaited.*

### Exercises

**7.1 (E)** Write an async generator `ticker(n, interval)` that yields the numbers 0 to n−1, one every `interval` seconds. Consume it with `async for` and also with an async list comprehension.

**7.2 (M)** Write an async context manager `timed(label)` with `@asynccontextmanager` that prints how long its body took, even when the body raises. Test it with a body that awaits a sleep and then raises `ValueError`; confirm the time prints and the `ValueError` still propagates.

**7.3 (M)** Write `stream_until(stream, stop_word)` that consumes an async token stream, prints tokens as they arrive, stops as soon as a token contains `stop_word`, and guarantees the stream's cleanup runs before the function returns. Return the full text received.

**7.4 (H)** Write an async generator `merge(*streams)` that yields items from several async iterators as soon as any of them produces one (fan-in), and finishes when all of them are exhausted. Test it with three `ticker` streams at different speeds. Hint: one task per stream pushing into a shared `asyncio.Queue`.

### Solutions

**7.1**

```python
import asyncio

async def ticker(n: int, interval: float):
    for i in range(n):
        await asyncio.sleep(interval)
        yield i

async def main():
    async for i in ticker(3, 0.2):
        print("tick", i)
    print([i async for i in ticker(5, 0.1)])   # [0, 1, 2, 3, 4]

asyncio.run(main())
```

**7.2**

```python
import asyncio, time
from contextlib import asynccontextmanager

@asynccontextmanager
async def timed(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"{label} took {time.perf_counter() - start:.2f}s")

async def main():
    try:
        async with timed("risky step"):
            await asyncio.sleep(0.3)
            raise ValueError("bad output from model")
    except ValueError as e:
        print("caught:", e)

asyncio.run(main())
# risky step took 0.30s
# caught: bad output from model
```

**7.3**

```python
from contextlib import aclosing

async def stream_until(stream, stop_word: str) -> str:
    received = []
    async with aclosing(stream) as s:
        async for token in s:
            print(token, end="", flush=True)
            received.append(token)
            if stop_word in token:
                break
    print()
    return "".join(received)

async def main():
    text = await stream_until(fake_token_stream("one two STOP three four"), "STOP")
    print(repr(text))    # 'one two STOP '
```

Add a `try/finally` with a print to `fake_token_stream` to watch the cleanup run before `stream_until` returns.

**7.4**

```python
import asyncio

_DONE = object()          # marks the end of one source stream

async def merge(*streams):
    q: asyncio.Queue = asyncio.Queue()

    async def pump(stream):
        try:
            async for item in stream:
                await q.put(item)
        finally:
            await q.put(_DONE)

    tasks = [asyncio.create_task(pump(s)) for s in streams]
    remaining = len(tasks)
    try:
        while remaining:
            item = await q.get()
            if item is _DONE:
                remaining -= 1
            else:
                yield item
    finally:
        for t in tasks:          # consumer stopped early or failed: stop the pumps
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

async def main():
    async def labelled(name, n, interval):
        async for i in ticker(n, interval):
            yield f"{name}{i}"
    async for item in merge(labelled("fast", 6, 0.1), labelled("mid", 3, 0.25), labelled("slow", 2, 0.4)):
        print(item)

asyncio.run(main())
```

This is the core of any UI that shows several agents or tools streaming in parallel. Note the `finally` in `merge`: if the consumer breaks early, the pump tasks must be cancelled, or they would keep running. (On Python 3.13+, you could shut the queue down instead of using the `_DONE` marker, but counting finished sources is still needed to know when all are exhausted.) The third-party `aiostream` library offers a ready-made `merge`.

## Module 8: Mixing sync and async code

Real systems mix async code with blocking libraries, CPU-heavy work and ML models. The rule is simple: nothing slow may run directly on the event loop thread. Blocking I/O goes to a thread with `asyncio.to_thread`, heavy CPU work goes to a process pool, and code in other threads talks to the loop through thread-safe bridges.

### 8.1 Recognise blocking code

Any call that takes noticeable time without `await` blocks the loop. Common culprits in AI codebases:

| Blocking call | Async alternative |
| --- | --- |
| `requests.get(...)` | `httpx.AsyncClient` (Module 9) or `aiohttp` |
| `time.sleep(...)` | `await asyncio.sleep(...)` |
| Sync LLM client (`OpenAI()`, `Anthropic()`) | `AsyncOpenAI()`, `AsyncAnthropic()` (Module 10) |
| `psycopg2`, sync SQLAlchemy sessions | `asyncpg`, `psycopg` async mode, SQLAlchemy `AsyncSession` |
| Sync Redis client | `redis.asyncio` |
| Running a PyTorch/transformers model, tokenising a big batch, parsing a huge JSON or PDF | Run it in a thread or process (below) |
| Reading a large local file | `await asyncio.to_thread(path.read_text)` |

A rule of thumb used in industry: any operation that may hold the loop for more than about 10 to 50 ms in a latency-sensitive service deserves to be moved off the loop.

### 8.2 asyncio.to\_thread for blocking I/O

`await asyncio.to_thread(func, *args, **kwargs)` (Python 3.9+) runs a normal function in a worker thread and lets the loop keep running. It also copies the current `contextvars` context into the thread, so request IDs and tracing context survive.

```python
import asyncio, time

def legacy_search(query: str) -> list[str]:      # a sync library we cannot change
    time.sleep(1)                                  # blocking network call
    return [f"{query} hit {i}" for i in range(3)]

async def main():
    start = time.perf_counter()
    results = await asyncio.gather(*(asyncio.to_thread(legacy_search, q) for q in ["a", "b", "c", "d"]))
    print(len(results), f"{time.perf_counter() - start:.2f}s")   # about 1.00s

asyncio.run(main())
```

The default thread pool has `min(32, os.cpu_count() + 4)` threads, so beyond that many concurrent calls they queue up. Threads work well for I/O because a thread waiting on the network releases the GIL.

### 8.3 Processes for CPU-bound work

On the standard (GIL) build, threads do not speed up pure-Python computation. Use a `ProcessPoolExecutor` through `loop.run_in_executor`:

```python
import asyncio, time
from concurrent.futures import ProcessPoolExecutor

def count_primes(limit: int) -> int:          # pure CPU work
    count = 0
    for n in range(2, limit):
        if all(n % d for d in range(2, int(n ** 0.5) + 1)):
            count += 1
    return count

async def main():
    loop = asyncio.get_running_loop()
    start = time.perf_counter()
    with ProcessPoolExecutor() as pool:
        results = await asyncio.gather(*(loop.run_in_executor(pool, count_primes, 200_000) for _ in range(4)))
    print(results, f"{time.perf_counter() - start:.2f}s")

if __name__ == "__main__":       # required: worker processes re-import this module
    asyncio.run(main())
```

`run_in_executor` passes only positional arguments; use `functools.partial(func, key=value)` for keyword arguments. Arguments and results are pickled to cross process boundaries, so send small inputs (file paths, IDs) rather than giant objects. Always keep the `if __name__ == "__main__":` guard: the default start method is not `fork` on Windows and macOS, and since Python 3.14 not on Linux either (it changed to `forkserver`), so child processes import your module afresh. On Linux with Python 3.12 or 3.13 the default is still `fork`, which can deadlock once any thread exists (even one started by `asyncio.to_thread`); pass `mp_context=multiprocessing.get_context("forkserver")` to the pool, as shown in the Python 3.12 guide.

For GPU models (PyTorch, ONNX Runtime) the heavy work runs in native code that usually releases the GIL, so `asyncio.to_thread(model.forward, batch)` is a common and reasonable choice. Keep one model instance and control concurrency with a semaphore or a batcher (Exercise 8.4) so you do not overload the GPU.

### 8.4 Calling async code from sync code

- **From a script or CLI:** `asyncio.run(main())`.
- **Several calls reusing one loop (Python 3.11+):**

```python
with asyncio.Runner() as runner:
    runner.run(load_index())
    runner.run(answer_question("What is MCP?"))
```

- **From another thread while a loop is running elsewhere:** `asyncio.run_coroutine_threadsafe(coro, loop)` schedules the coroutine on that loop and returns a `concurrent.futures.Future`; call `.result(timeout=...)` on it from the other thread. For plain callbacks, `loop.call_soon_threadsafe(callback, *args)`.
- **From inside a running loop (a notebook, a FastAPI handler):** never call `asyncio.run`. Make your function `async` and `await` it. Patching the loop to allow nesting (the `nest_asyncio` package) hides design problems and is best avoided in production.

Also avoid `asyncio.get_event_loop()` in new code. Since Python 3.14 it raises `RuntimeError` when no loop is set, rather than silently creating one. Inside a coroutine use `asyncio.get_running_loop()`. The event loop *policy* system (`asyncio.set_event_loop_policy` and friends) is deprecated in 3.14 and scheduled for removal in 3.16 ([asyncio policy docs](https://docs.python.org/3/library/asyncio-policy.html)); use `asyncio.run(main(), loop_factory=...)` or `asyncio.Runner(loop_factory=...)` if you need a custom loop such as `uvloop` (`asyncio.run` gained `loop_factory` in 3.12; on 3.11 use `asyncio.Runner`).

### 8.5 Detecting blocking: debug mode

Run with `asyncio.run(main(), debug=True)` or the environment variable `PYTHONASYNCIODEBUG=1`. Debug mode logs any callback or task step that holds the loop longer than `loop.slow_callback_duration` (default 0.1 s), reports coroutines that were never awaited with the line where they were created, and checks for some thread-safety mistakes. Enable it in development and tests.

### Check yourself

1. Your FastAPI service calls `requests.get` inside an `async def` endpoint. What happens under load? *Answer: every request blocks the single event loop thread, so the server handles one request at a time and latency explodes.*
2. Thread or process for running `tiktoken` on 2 million documents? *Answer: a process pool (CPU-bound) on the standard build; batch the documents to reduce pickling overhead.*
3. How do you submit a coroutine to a running loop from another thread? *Answer: `asyncio.run_coroutine_threadsafe(coro, loop)`.*

### Exercises

**8.1 (E)** `fetch_sync(url)` sleeps 1 s with `time.sleep`. Fetch 8 URLs concurrently from async code without changing `fetch_sync`. Total time should be about 1 s.

**8.2 (M)** Time `count_primes(200_000)` run 4 times (a) sequentially, (b) with `asyncio.to_thread`, and (c) with a `ProcessPoolExecutor`. Explain the results. If you have a free-threaded Python installed (`python3.14t`), run (b) again with it and compare.

**8.3 (M)** This program has a hidden blocking call. Run it with debug mode on, find the warning in the output, and fix the bug.

```python
import asyncio, hashlib

def fingerprint(text: str) -> str:
    h = text.encode()
    for _ in range(400_000):
        h = hashlib.sha256(h).digest()
    return h.hex()[:12]

async def handle(doc: str) -> str:
    await asyncio.sleep(0.01)
    return fingerprint(doc)

async def heartbeat():
    for _ in range(5):
        print("alive")
        await asyncio.sleep(0.1)

async def main():
    await asyncio.gather(heartbeat(), *(handle(f"doc{i}") for i in range(4)))

asyncio.run(main(), debug=True)
```

**8.4 (H)** Dynamic batching for model serving. Many concurrent requests call `await batcher.predict(x)`. The batcher collects requests until it has 8 or 20 ms have passed since the first one, runs a blocking `model_predict(batch)` (0.05 s per batch regardless of size) in a thread, and returns each caller its own output. Compare the total time for 64 concurrent requests with and without batching. This is how production inference servers get high GPU utilisation.

**8.5 (M)** A legacy SDK calls your Python callback from its own background thread whenever a document is uploaded. Inside the callback you must run `async def index(doc)` on the application's main event loop and wait for its result. Write the callback and a small demo using `threading.Thread` to simulate the SDK.

### Solutions

**8.1**

```python
import asyncio, time

def fetch_sync(url: str) -> str:
    time.sleep(1)
    return f"body of {url}"

async def main():
    start = time.perf_counter()
    urls = [f"https://site/{i}" for i in range(8)]
    bodies = await asyncio.gather(*(asyncio.to_thread(fetch_sync, u) for u in urls))
    print(len(bodies), f"{time.perf_counter() - start:.2f}s")   # about 1.00s

asyncio.run(main())
```

**8.2** On a standard build with 4+ cores: (a) and (b) take about the same time (say 4 × T), because the GIL lets only one thread run Python bytecode at once; (c) takes about T plus process start-up cost, because each process has its own interpreter and core. On a free-threaded build, (b) also approaches T, since threads can then run Python code in parallel.

**8.3** Debug mode prints a warning like `Executing <Task ...> took 0.4 seconds` (the exact duration depends on your machine), pointing at `handle`, and the "alive" lines stall. `fingerprint` is CPU-bound code called directly in a coroutine. Fix: `return await asyncio.to_thread(fingerprint, doc)`, which keeps the loop responsive. Because the loop inside `fingerprint` is mostly Python bytecode on tiny inputs, threads will not make the four calls finish faster on a standard build; use a process pool for real speed-up.

**8.4**

```python
import asyncio, time

def model_predict(batch: list[str]) -> list[int]:
    time.sleep(0.05)                 # blocking "GPU forward pass", same cost for any batch size
    return [len(x) for x in batch]

class DynamicBatcher:
    def __init__(self, max_batch: int = 8, max_wait: float = 0.02):
        self.max_batch, self.max_wait = max_batch, max_wait
        self.queue: asyncio.Queue = asyncio.Queue()
        self._runner: asyncio.Task | None = None

    def start(self) -> None:
        self._runner = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._runner.cancel()
        await asyncio.gather(self._runner, return_exceptions=True)

    async def predict(self, x: str) -> int:
        fut = asyncio.get_running_loop().create_future()   # a placeholder for this caller's result
        await self.queue.put((x, fut))
        return await fut

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self.queue.get()]                # wait for the first request
            deadline = loop.time() + self.max_wait
            while len(batch) < self.max_batch:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    async with asyncio.timeout(remaining):
                        batch.append(await self.queue.get())
                except TimeoutError:
                    break
            inputs = [x for x, _ in batch]
            try:
                outputs = await asyncio.to_thread(model_predict, inputs)
            except Exception as exc:
                for _, f in batch:
                    if not f.done():
                        f.set_exception(exc)
                continue
            for (_, f), out in zip(batch, outputs):
                if not f.done():                            # the caller may have been cancelled
                    f.set_result(out)

async def main():
    batcher = DynamicBatcher()
    batcher.start()
    start = time.perf_counter()
    results = await asyncio.gather(*(batcher.predict("x" * i) for i in range(64)))
    print(f"batched:   {time.perf_counter() - start:.2f}s", results[:5])   # about 0.4s (8 batches)
    await batcher.stop()

    start = time.perf_counter()
    for i in range(64):
        await asyncio.to_thread(model_predict, ["x" * i])
    print(f"unbatched: {time.perf_counter() - start:.2f}s")                # about 3.2s

asyncio.run(main())
```

`loop.create_future()` creates a bare Future that the batcher fulfils later with `set_result`; each caller awaits its own. Production inference servers such as vLLM and NVIDIA Triton use the same idea (continuous or dynamic batching) at much larger scale.

**8.5**

```python
import asyncio, threading

async def index(doc: str) -> str:
    await asyncio.sleep(0.2)
    return f"indexed {doc}"

def make_callback(loop: asyncio.AbstractEventLoop):
    def on_upload(doc: str) -> None:               # runs in the SDK's thread
        future = asyncio.run_coroutine_threadsafe(index(doc), loop)
        print(future.result(timeout=5))            # blocks this SDK thread only
    return on_upload

async def main():
    callback = make_callback(asyncio.get_running_loop())
    sdk_thread = threading.Thread(target=lambda: [callback(f"doc{i}") for i in range(3)])
    sdk_thread.start()
    await asyncio.to_thread(sdk_thread.join)       # wait without blocking the loop

asyncio.run(main())
```

Never call `future.result()` on the event loop thread itself: it would block the loop that is supposed to produce the result, a deadlock.

## Module 9: Real network I/O with httpx2

For async HTTP in 2026, use **httpx2**, the Pydantic-maintained continuation of the popular `httpx` library, with the same API under a new import name. One shared `httpx2.AsyncClient` per application, explicit timeouts, retries with backoff and jitter, and concurrency limits are the four habits that separate production agent code from demos.

### 9.1 httpx versus httpx2: what happened

`httpx` was the standard async HTTP client for years, and the OpenAI and Anthropic SDKs, FastAPI's test client and the MCP SDK all built on it. Its last release, 0.28.1, came out in December 2024. In May 2026 Pydantic took over stewardship under the name `httpx2`, keeping the original design and API; version 2.16.0 shipped on October 21, 2026 ([httpx2 on PyPI mirror](https://simple-repository.app.cern.ch/project/httpx2)). The major SDKs followed: the Anthropic Python SDK moved to httpx2 in its 1.0 release and OpenAI's in 3.0. The current `anthropic` 2.8.0 and `openai` 3.49.0 both declare httpx2 as a dependency, and Anthropic's [SDK documentation](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python) names it as the default HTTP backend. Starlette 1.2 and later support httpx2 in their test client.

What this means for you:

- Write `import httpx2` in new code. Every `httpx.X` you see in older tutorials is `httpx2.X` now: `AsyncClient`, `Timeout`, `Limits`, `MockTransport`, `ASGITransport`, `HTTPStatusError`.
- httpx2 verifies TLS certificates against the operating system's trust store (through the `truststore` package) instead of the bundled `certifi` list. Minimal Docker images without a CA store can fail TLS handshakes; install the OS `ca-certificates` package or set `SSL_CERT_FILE` ([MCP SDK notes](https://py.sdk.modelcontextprotocol.io/whats-new/)).
- Mocking libraries that patch the old `httpx` module (`respx`, `pytest-httpx`) do not see httpx2 traffic. Prefer `httpx2.MockTransport`, shown below, which needs no extra package. Expect both libraries in one environment for a while: LangChain and LangGraph 1.2 still depend on `httpx` 0.28.1, so a successful `import httpx` tells you nothing about what the SDKs use.
- `aiohttp` remains a popular, fast alternative async client, and both LLM SDKs can use it as a backend.

### 9.2 One shared AsyncClient

```python
import asyncio
import httpx2

async def main():
    timeout = httpx2.Timeout(10.0, connect=3.0)          # 10 s overall per phase, 3 s to connect
    limits = httpx2.Limits(max_connections=50, max_keepalive_connections=10)
    async with httpx2.AsyncClient(base_url="https://httpbin.org", timeout=timeout, limits=limits) as client:
        r = await client.get("/get", params={"q": "asyncio"})
        r.raise_for_status()                              # raise for 4xx/5xx
        print(r.status_code, r.json()["args"])
        r = await client.post("/post", json={"prompt": "hi"})
        print(r.json()["json"])

asyncio.run(main())
```

Create the client once and reuse it. It keeps a **connection pool**: TCP and TLS connections stay open between requests, which saves tens to hundreds of milliseconds per call. Creating a new client per request (a very common beginner mistake) throws that away and can exhaust sockets under load. In a web service, create the client at startup in the lifespan handler (Module 11) and close it at shutdown.

`Timeout` has four phases: `connect`, `read`, `write` and `pool` (waiting for a free connection from the pool). `Limits.max_connections` caps concurrent connections; extra requests wait in the pool, which is a built-in form of concurrency limiting.

### 9.3 Errors

| Exception | Meaning | Usually retry? |
| --- | --- | --- |
| `httpx2.ConnectError` | Could not connect (DNS, refused, TLS) | Yes, a few times |
| `httpx2.TimeoutException` (`ConnectTimeout`, `ReadTimeout`, `PoolTimeout`, ...) | A phase exceeded its timeout | Yes, with care for non-idempotent requests |
| `httpx2.HTTPStatusError` | Raised by `raise_for_status()` for 4xx/5xx | 429 and 5xx yes; other 4xx no |
| `httpx2.TransportError` | Base class of connect, timeout and protocol errors | Depends |
| `httpx2.RequestError` | Base class of every error raised while sending a request | Depends |

A request is **idempotent** if repeating it has the same effect as doing it once (GET, PUT, DELETE). Retrying a non-idempotent POST that charges a credit card can charge twice; APIs solve this with an *idempotency key* header. LLM generation calls are safe to retry from a correctness standpoint but cost money each time.

### 9.4 Retries with exponential backoff and jitter

Transient failures (rate limits, overloaded servers, network blips) are normal at scale. The standard recipe: wait `base × 2^(attempt−1)` seconds, capped, with random **jitter** so that thousands of clients that failed together do not all retry at the same instant. Honour the server's `Retry-After` header when present.

```python
import asyncio, random
import httpx2

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}

async def get_with_retries(client: httpx2.AsyncClient, url: str, *,
                           attempts: int = 5, base: float = 0.5, cap: float = 8.0) -> httpx2.Response:
    for attempt in range(1, attempts + 1):
        delay = None
        try:
            r = await client.get(url)
            if r.status_code not in RETRYABLE_STATUS:
                r.raise_for_status()           # non-retryable 4xx raises immediately
                return r
            error: Exception = httpx2.HTTPStatusError(
                f"retryable status {r.status_code}", request=r.request, response=r)
            retry_after = r.headers.get("retry-after")
            if retry_after and retry_after.isdigit():
                delay = float(retry_after)
        except httpx2.TransportError as exc:  # connect errors, timeouts, protocol errors
            error = exc
        if attempt == attempts:
            raise error
        if delay is None:
            delay = random.uniform(0, min(cap, base * 2 ** (attempt - 1)))   # "full jitter"
        await asyncio.sleep(delay)
    raise AssertionError("unreachable")
```

In practice teams use a library for this, such as `tenacity` or `stamina`, and remember that the official LLM SDKs already retry some errors themselves (the Anthropic SDK retries connection errors, 408, 409, 429 and 5xx twice by default, configurable with `max_retries`; [Anthropic Python SDK docs](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python)). Do not stack your own retries on top of SDK retries without thinking: 3 × 3 attempts is 9 requests.

### 9.5 Streaming responses

For large downloads or server-sent events, stream the body instead of loading it into memory:

```python
async with client.stream("GET", "/stream/5") as r:     # httpbin streams 5 JSON lines
    async for line in r.aiter_lines():
        print(line[:60])
```

Other iterators: `aiter_bytes()`, `aiter_text()`, `aiter_raw()`. This is exactly how LLM SDKs read token streams under the hood.

### 9.6 Testing without the network: MockTransport

```python
import asyncio, json
import httpx2

async def handler(request: httpx2.Request) -> httpx2.Response:
    await asyncio.sleep(0.05)                           # simulate latency
    if request.url.path == "/weather":
        city = request.url.params.get("city", "?")
        return httpx2.Response(200, json={"city": city, "temp_c": 21})
    return httpx2.Response(404, json={"error": "not found"})

async def main():
    transport = httpx2.MockTransport(handler)
    async with httpx2.AsyncClient(transport=transport, base_url="https://api.test") as client:
        r = await client.get("/weather", params={"city": "Toronto"})
        print(r.json())                                   # {'city': 'Toronto', 'temp_c': 21}

asyncio.run(main())
```

Your code under test receives the client as a parameter (dependency injection), so tests pass a mock-backed client and production passes a real one. All exercises below use MockTransport so they run offline.

### Check yourself

1. Why reuse one `AsyncClient` instead of creating one per request? *Answer: it pools connections, avoiding repeated TCP/TLS handshakes and socket exhaustion.*
2. Why add jitter to backoff? *Answer: so clients that failed together do not retry in synchronised waves that overload the server again.*
3. Should you retry a 400 Bad Request? *Answer: no; the request itself is wrong and will fail again.*

### Exercises

**9.1 (E)** Using MockTransport with a random 0.1 to 0.5 s latency, fetch `/item/0` to `/item/29` concurrently with at most 5 requests in flight. Print the total time and confirm the concurrency limit with a counter.

**9.2 (M)** Write a mock handler that returns 503 for the first two requests to `/flaky` and 200 afterwards, and another that returns 429 with `Retry-After: 1` once. Test `get_with_retries` against both, printing each attempt and the wait before it.

**9.3 (M)** A mock endpoint `/events` streams 10 newline-delimited JSON objects like `{"type": "token", "text": "..."}` with a short delay between them, ending with `{"type": "done"}`. Write `async def read_events(client)` that yields parsed events as they arrive and stops at `done`. Hint: give `httpx2.Response` an async generator of bytes as `content`.

**9.4 (H)** Build `ResilientClient`, a wrapper around `AsyncClient` used by an agent's tools. `await rc.get_json(url)` must: wait for a semaphore slot (max 10 in flight), retry transient failures with jittered backoff (max 4 attempts), apply a 2-second per-attempt timeout, respect an optional overall `deadline` in seconds across all attempts, and record metrics (requests, attempts, failures). Test it with a mock that randomly fails 30% of requests with 503 and randomly delays 10% of them by 3 seconds.

### Solutions

**9.1**

```python
import asyncio, random, time
import httpx2

in_flight = 0
peak = 0

async def handler(request: httpx2.Request) -> httpx2.Response:
    global in_flight, peak
    in_flight += 1
    peak = max(peak, in_flight)
    try:
        await asyncio.sleep(random.uniform(0.1, 0.5))
        return httpx2.Response(200, json={"path": request.url.path})
    finally:
        in_flight -= 1

async def main():
    sem = asyncio.Semaphore(5)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler), base_url="https://api.test") as client:
        async def fetch(i: int) -> dict:
            async with sem:
                r = await client.get(f"/item/{i}")
                return r.json()
        start = time.perf_counter()
        results = await asyncio.gather(*(fetch(i) for i in range(30)))
        print(len(results), f"{time.perf_counter() - start:.2f}s", "peak in flight:", peak)  # about 1.8s, peak 5

asyncio.run(main())
```

Thirty requests averaging 0.3 s at 5 at a time take about 30 × 0.3 / 5 = 1.8 s.

**9.2**

```python
import asyncio, time
import httpx2

calls = {"flaky": 0, "limited": 0}

async def handler(request: httpx2.Request) -> httpx2.Response:
    name = request.url.path.strip("/")
    calls[name] += 1
    print(f"{time.monotonic() - T0:4.2f}s attempt {calls[name]} on /{name}")
    if name == "flaky" and calls[name] <= 2:
        return httpx2.Response(503)
    if name == "limited" and calls[name] == 1:
        return httpx2.Response(429, headers={"Retry-After": "1"})
    return httpx2.Response(200, json={"ok": True})

async def main():
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler), base_url="https://api.test") as client:
        print((await get_with_retries(client, "/flaky")).json())
        print((await get_with_retries(client, "/limited")).json())

T0 = time.monotonic()
asyncio.run(main())
```

Expect three attempts on `/flaky` with short random waits (under 0.5 s, then under 1 s), and two attempts on `/limited` exactly 1 s apart because of `Retry-After`.

**9.3**

```python
import asyncio, json
import httpx2

async def event_bytes():
    for i in range(10):
        await asyncio.sleep(0.05)
        yield (json.dumps({"type": "token", "text": f"w{i} "}) + "\n").encode()
    yield (json.dumps({"type": "done"}) + "\n").encode()

async def handler(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, content=event_bytes(), headers={"content-type": "application/x-ndjson"})

async def read_events(client: httpx2.AsyncClient):
    async with client.stream("GET", "/events") as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event["type"] == "done":
                return
            yield event

async def main():
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler), base_url="https://api.test") as client:
        async for event in read_events(client):
            print(event["text"], end="", flush=True)
    print()

asyncio.run(main())
```

**9.4**

```python
import asyncio, random, time
from dataclasses import dataclass
import httpx2

@dataclass
class Metrics:
    requests: int = 0
    attempts: int = 0
    failures: int = 0

class ResilientClient:
    RETRYABLE = {408, 425, 429, 500, 502, 503, 504}

    def __init__(self, client: httpx2.AsyncClient, max_in_flight: int = 10,
                 attempts: int = 4, per_attempt_timeout: float = 2.0):
        self.client = client
        self.sem = asyncio.Semaphore(max_in_flight)
        self.attempts = attempts
        self.per_attempt_timeout = per_attempt_timeout
        self.metrics = Metrics()

    async def get_json(self, url: str, deadline: float | None = None) -> dict:
        self.metrics.requests += 1
        try:
            async with asyncio.timeout(deadline):          # None = no overall deadline
                return await self._get_json_with_retries(url)
        except Exception:
            self.metrics.failures += 1
            raise

    async def _get_json_with_retries(self, url: str) -> dict:
        last_error: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            self.metrics.attempts += 1
            try:
                async with self.sem:                        # hold a slot only while the request runs
                    async with asyncio.timeout(self.per_attempt_timeout):
                        r = await self.client.get(url)
                if r.status_code in self.RETRYABLE:
                    last_error = httpx2.HTTPStatusError(f"{r.status_code}", request=r.request, response=r)
                else:
                    r.raise_for_status()
                    return r.json()
            except (TimeoutError, httpx2.TransportError) as exc:
                last_error = exc
            if attempt < self.attempts:
                await asyncio.sleep(random.uniform(0, min(4.0, 0.2 * 2 ** (attempt - 1))))
        assert last_error is not None
        raise last_error

async def handler(request: httpx2.Request) -> httpx2.Response:
    roll = random.random()
    if roll < 0.10:
        await asyncio.sleep(3)                             # will exceed the 2 s per-attempt timeout
    elif roll < 0.40:
        return httpx2.Response(503)
    await asyncio.sleep(0.05)
    return httpx2.Response(200, json={"url": str(request.url)})

async def main():
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler), base_url="https://api.test") as raw:
        rc = ResilientClient(raw)
        results = await asyncio.gather(*(rc.get_json(f"/doc/{i}", deadline=6.0) for i in range(50)),
                                       return_exceptions=True)
        ok = sum(not isinstance(r, Exception) for r in results)
        print(f"ok={ok}/50", rc.metrics)

asyncio.run(main())
```

Design points an interviewer would look for: the semaphore wraps only the request, not the backoff sleep, so a waiting retry does not hold a slot; `asyncio.timeout(None)` means "no deadline", which keeps one code path; the per-attempt timeout uses `asyncio.timeout` around the call so it covers any transport; and metrics count logical requests separately from attempts. Typical output: 48 to 50 of 50 succeed, with about 1.6 attempts per request.

## Module 10: Async LLM SDKs

Both major LLM SDKs ship an async client (`AsyncAnthropic`, `AsyncOpenAI`) with the same methods as the sync one, just awaited. Everything from Modules 4 to 9 applies directly: one shared client, a semaphore for concurrency, timeouts, streaming with `async for`, and concurrent tool execution. This module also builds a small provider-neutral wrapper and a fake LLM so you can practise without API keys.

### 10.1 Setup and keys

Create API keys in each provider's console and export them; the SDKs read them from the environment:

```bash
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
export ANTHROPIC_MODEL="claude-sonnet-5-5"   # or claude-opus-5-5, claude-haiku-5-5
export OPENAI_MODEL="gpt-5.5"
```

Keep model names in configuration, not in code: they change several times a year. Never commit keys; use environment variables or a secrets manager, and a `.env` file loaded with `python-dotenv` for local work.

### 10.2 Anthropic: AsyncAnthropic

```python
import asyncio, os
from anthropic import AsyncAnthropic

client = AsyncAnthropic()           # reads ANTHROPIC_API_KEY
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")

async def main():
    message = await client.messages.create(
        model=MODEL,
        max_tokens=512,
        system="You are a concise tutor.",
        messages=[{"role": "user", "content": "Explain an event loop in two sentences."}],
    )
    print("".join(b.text for b in message.content if b.type == "text"))
    print(message.usage)            # input and output token counts

asyncio.run(main())
```

Things to know about the current Anthropic Python SDK (2.x), from the official docs ([Python SDK](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python), [API 2026-10-21 migration](https://platform.claude.com/docs/en/api/migration-2026-10-21)) and checked against `anthropic` 2.8.0:

- It sends API version `2026-10-21`, which removed `temperature`, `top_p` and `top_k`. Older tutorials that pass `temperature=0` will fail.
- Requests time out after 10 minutes by default; set `timeout=` on the client or per call (`client.with_options(timeout=30)`).
- It retries connection errors, 408, 409, 429 and 5xx twice by default (`max_retries=`). It raises `anthropic.RateLimitError`, `APITimeoutError`, `APIConnectionError` and `APIStatusError` subclasses.
- It uses httpx2 underneath; you can switch to aiohttp with `http_client=DefaultAioHttpClient()` for heavy concurrency.

**Streaming.** The `messages.stream` helper is an async context manager that accumulates the message for you:

```python
async def stream_answer(question: str) -> None:
    async with client.messages.stream(
        model=MODEL, max_tokens=512,
        messages=[{"role": "user", "content": question}],
    ) as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)
        final = await stream.get_final_message()
    print("\n", final.usage)
```

Alternatively `await client.messages.create(..., stream=True)` returns a lighter async iterator of raw events.

### 10.3 OpenAI: AsyncOpenAI and the Responses API

```python
import asyncio, os
from openai import AsyncOpenAI

client = AsyncOpenAI()              # reads OPENAI_API_KEY
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")

async def main():
    response = await client.responses.create(model=MODEL, input="Explain an event loop in two sentences.")
    print(response.output_text)

    stream = await client.responses.create(model=MODEL, input="Count to five.", stream=True)
    async for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
    print()

asyncio.run(main())
```

The Responses API is OpenAI's primary interface; the older Chat Completions API (`client.chat.completions.create`) is still supported and very common in existing code. Since version 3.0 the OpenAI Python SDK also uses httpx2 ([PyPI](https://pypi.org/project/openai/3.45.0/)).

### 10.4 Many calls at once

The patterns from earlier modules carry over unchanged:

```python
async def ask_many(questions: list[str], max_concurrency: int = 4) -> list[str]:
    sem = asyncio.Semaphore(max_concurrency)
    async def ask(q: str) -> str:
        async with sem:
            msg = await client.messages.create(
                model=MODEL, max_tokens=256, messages=[{"role": "user", "content": q}])
            return "".join(b.text for b in msg.content if b.type == "text")
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(ask(q)) for q in questions]
    return [t.result() for t in tasks]
```

For large offline jobs (evaluations, labelling thousands of documents) both providers also offer **batch APIs** (`client.messages.batches` for Anthropic) that run asynchronously on the provider side, at a discount, and return results within hours. Use live concurrent calls for interactive work and batches for bulk work.

### 10.5 Tool use with concurrent tool execution

Tool use (function calling) is the heart of an agent. You describe tools with JSON Schema; the model replies with one or more `tool_use` blocks; your code runs the tools and sends back `tool_result` blocks; the model continues. When the model asks for several tools in one turn, run them **concurrently**:

```python
import asyncio, json

async def get_weather(city: str) -> dict:
    await asyncio.sleep(0.5)                       # pretend API call
    return {"city": city, "temp_c": 18}

TOOLS = [{
    "name": "get_weather",
    "description": "Current weather for a city.",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}]
REGISTRY = {"get_weather": get_weather}

async def run_tool(block) -> dict:
    try:
        async with asyncio.timeout(10):
            result = await REGISTRY[block.name](**block.input)
        return {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)}
    except Exception as exc:
        return {"type": "tool_result", "tool_use_id": block.id, "content": f"ERROR: {exc}", "is_error": True}

async def weather_agent(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    while True:
        msg = await client.messages.create(model=MODEL, max_tokens=1024, tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": msg.content})
        tool_blocks = [b for b in msg.content if b.type == "tool_use"]
        if not tool_blocks:
            return "".join(b.text for b in msg.content if b.type == "text")
        results = await asyncio.gather(*(run_tool(b) for b in tool_blocks))   # concurrent!
        messages.append({"role": "user", "content": list(results)})

# asyncio.run(weather_agent("Compare the weather in Toronto, Paris and Tokyo."))
```

If the model requests three weather lookups, they take about 0.5 s together instead of 1.5 s. `run_tool` never raises for ordinary errors (its `except Exception` leaves cancellation alone), so `gather` is safe here and the model sees each failure as an error result it can react to. OpenAI's Responses API works the same way with different field names: the model emits `function_call` items with a `call_id`, and you reply with `function_call_output` items.

### 10.6 A provider-neutral wrapper and a fake LLM

Production teams wrap SDKs behind a small interface so they can switch providers, add caching and logging in one place, and test without the network. Save this as `llm.py`; Modules 11 to 15 import it.

```python
# llm.py
from __future__ import annotations
import asyncio, itertools, os
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

@dataclass
class AssistantTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    content: list = field(default_factory=list)    # provider-native blocks, sent back as-is

class AnthropicLLM:
    def __init__(self, model: str | None = None, max_concurrency: int = 4):
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic()
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        self.sem = asyncio.Semaphore(max_concurrency)

    async def complete(self, messages: list[dict], tools: list[dict] | None = None,
                       system: str | None = None, max_tokens: int = 1024) -> AssistantTurn:
        kwargs: dict = {"model": self.model, "max_tokens": max_tokens, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if system:
            kwargs["system"] = system
        async with self.sem:
            msg = await self.client.messages.create(**kwargs)
        text = "".join(b.text for b in msg.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, b.input) for b in msg.content if b.type == "tool_use"]
        return AssistantTurn(text, calls, list(msg.content))

    async def stream(self, messages: list[dict], system: str | None = None) -> AsyncIterator[str]:
        kwargs: dict = {"model": self.model, "max_tokens": 1024, "messages": messages}
        if system:
            kwargs["system"] = system
        async with self.sem:
            async with self.client.messages.stream(**kwargs) as s:
                async for text in s.text_stream:
                    yield text

class FakeLLM:
    """Offline stand-in. `script(messages) -> AssistantTurn` decides each reply."""
    _ids = itertools.count(1)

    def __init__(self, script: Callable[[list[dict]], AssistantTurn], latency: float = 0.3):
        self.script = script
        self.latency = latency
        self.calls = 0

    @staticmethod
    def tool_turn(*calls: tuple[str, dict], text: str = "") -> AssistantTurn:
        tcs = [ToolCall(f"toolu_{next(FakeLLM._ids)}", name, args) for name, args in calls]
        content = ([{"type": "text", "text": text}] if text else []) + [
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in tcs]
        return AssistantTurn(text, tcs, content)

    @staticmethod
    def text_turn(text: str) -> AssistantTurn:
        return AssistantTurn(text, [], [{"type": "text", "text": text}])

    async def complete(self, messages, tools=None, system=None, max_tokens=1024) -> AssistantTurn:
        self.calls += 1
        await asyncio.sleep(self.latency)
        return self.script(messages)

    async def stream(self, messages, system=None) -> AsyncIterator[str]:
        turn = await self.complete(messages)
        for word in turn.text.split(" "):
            await asyncio.sleep(0.03)
            yield word + " "
```

Both classes have the same `complete` and `stream` methods. That shared shape is a **protocol** in Python's typing sense (duck typing): any object with those methods works, so your agent code does not care which one it gets.

### Check yourself

1. Why put the model name in an environment variable? *Answer: model names change frequently and differ per environment; configuration changes should not need code changes.*
2. The model requests 4 tools in one turn, each taking 1 s. How long should your agent take to run them? *Answer: about 1 s, by running them concurrently.*
3. Why does `run_tool` return an error result instead of raising? *Answer: so one failing tool does not crash the turn, and the model can see the error and recover.*

### Exercises

Each exercise works with `AnthropicLLM` if you have a key, or with `FakeLLM` otherwise.

**10.1 (E)** Ask 6 questions with at most 2 concurrent requests. Print each answer's first 60 characters and the total time. With `FakeLLM(latency=1.0)`, predict the time first.

**10.2 (M)** Measure **time to first token** (TTFT) and total time for a streamed answer. TTFT is the key latency metric users feel in chat products.

**10.3 (M)** Hedge between two "providers" (two `FakeLLM` instances with latencies 0.8 and 0.4, or Anthropic versus OpenAI): send the same prompt to both, return the first answer and the provider name, and cancel the other.

**10.4 (H)** Using `FakeLLM`, script a two-turn tool conversation: turn 1 requests `search("asyncio")`, `search("trio")` and `get_time()`; after receiving the tool results, turn 2 answers in text. Implement `run_turn_tools(turn, registry)` that executes all tool calls concurrently with a 2-second timeout each and returns the `tool_result` blocks. Make `get_time` deliberately slow (3 s) and confirm the agent still finishes, with an error result for it.

### Solutions

**10.1**

```python
import asyncio, time
from llm import FakeLLM

async def main():
    llm = FakeLLM(lambda msgs: FakeLLM.text_turn(f"Answer to: {msgs[-1]['content']}"), latency=1.0)
    sem = asyncio.Semaphore(2)
    async def ask(q: str) -> str:
        async with sem:
            turn = await llm.complete([{"role": "user", "content": q}])
            return turn.text
    start = time.perf_counter()
    answers = await asyncio.gather(*(ask(f"question {i}") for i in range(6)))
    for a in answers:
        print(a[:60])
    print(f"{time.perf_counter() - start:.2f}s")     # about 3.00s: 3 rounds of 2

asyncio.run(main())
```

The wrapper already has a semaphore inside `AnthropicLLM`; with a real client you could rely on that instead.

**10.2**

```python
import asyncio, time
from llm import FakeLLM

async def measure(llm, question: str) -> None:
    start = time.perf_counter()
    ttft = None
    pieces = []
    async for piece in llm.stream([{"role": "user", "content": question}]):
        if ttft is None:
            ttft = time.perf_counter() - start
        pieces.append(piece)
    total = time.perf_counter() - start
    print(f"TTFT {ttft:.2f}s, total {total:.2f}s, {len(pieces)} chunks")

llm = FakeLLM(lambda m: FakeLLM.text_turn("word " * 40), latency=0.4)
asyncio.run(measure(llm, "hi"))       # TTFT about 0.43s, total about 1.6s
```

**10.3**

```python
import asyncio
from llm import FakeLLM

async def hedge(prompt: str, providers: dict) -> tuple[str, str]:
    async def call(name, llm):
        turn = await llm.complete([{"role": "user", "content": prompt}])
        return name, turn.text
    tasks = [asyncio.create_task(call(n, p)) for n, p in providers.items()]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        return done.pop().result()
    finally:
        for t in tasks:
            t.cancel()                         # no-op for finished tasks
        await asyncio.gather(*tasks, return_exceptions=True)

providers = {
    "slow": FakeLLM(lambda m: FakeLLM.text_turn("slow answer"), latency=0.8),
    "fast": FakeLLM(lambda m: FakeLLM.text_turn("fast answer"), latency=0.4),
}
print(asyncio.run(hedge("hi", providers)))   # ('fast', 'fast answer')
```

The `finally` block guarantees cleanup even if the caller of `hedge` is itself cancelled. Hedging doubles cost, so production systems usually send the second request only if the first has not answered within, say, the 95th-percentile latency.

**10.4**

```python
import asyncio, json, time
from llm import FakeLLM, AssistantTurn

async def search(query: str) -> list[str]:
    await asyncio.sleep(0.5)
    return [f"{query} result {i}" for i in range(2)]

async def get_time() -> str:
    await asyncio.sleep(3)                     # too slow on purpose
    return time.strftime("%H:%M")

REGISTRY = {"search": search, "get_time": get_time}

async def run_turn_tools(turn: AssistantTurn, registry: dict, timeout: float = 2.0) -> list[dict]:
    async def one(call):
        try:
            async with asyncio.timeout(timeout):
                result = await registry[call.name](**call.arguments)
            return {"type": "tool_result", "tool_use_id": call.id, "content": json.dumps(result)}
        except TimeoutError:
            return {"type": "tool_result", "tool_use_id": call.id,
                    "content": f"ERROR: {call.name} timed out after {timeout}s", "is_error": True}
        except Exception as exc:
            return {"type": "tool_result", "tool_use_id": call.id,
                    "content": f"ERROR: {type(exc).__name__}: {exc}", "is_error": True}
    return list(await asyncio.gather(*(one(c) for c in turn.tool_calls)))

def script(messages: list[dict]) -> AssistantTurn:
    last = messages[-1]
    if isinstance(last["content"], str):       # first turn: the user's question
        return FakeLLM.tool_turn(("search", {"query": "asyncio"}), ("search", {"query": "trio"}), ("get_time", {}))
    errors = sum(1 for b in last["content"] if b.get("is_error"))
    return FakeLLM.text_turn(f"Here is my comparison of asyncio and trio. ({errors} tool failed.)")

async def main():
    llm = FakeLLM(script, latency=0.2)
    messages = [{"role": "user", "content": "Compare asyncio and trio, and tell me the time."}]
    start = time.perf_counter()
    while True:
        turn = await llm.complete(messages)
        messages.append({"role": "assistant", "content": turn.content})
        if not turn.tool_calls:
            break
        messages.append({"role": "user", "content": await run_turn_tools(turn, REGISTRY)})
    print(turn.text, f"({time.perf_counter() - start:.2f}s)")   # about 2.4s

asyncio.run(main())
```

The two searches and the time lookup run together; the turn takes 2.0 s (the timeout), not 0.5 + 0.5 + 3 s, and the model is told that `get_time` failed. This loop is the core of the agent you will build in Module 13.

## Module 11: FastAPI for agent services

FastAPI is the most common way to put an agent behind an HTTP API in Python. It runs on an ASGI server (uvicorn) that owns the event loop; your `async def` endpoints run as tasks on it. Three things matter for agent services: create shared clients once in the **lifespan**, never block inside `async def`, and **stream** model output with server-sent events while cleaning up when the client disconnects.

### 11.1 async def or def?

```python
from fastapi import FastAPI
import asyncio, time

app = FastAPI()

@app.get("/async-ok")
async def async_ok():
    await asyncio.sleep(1)       # yields: thousands of these can wait at once
    return {"ok": True}

@app.get("/sync-ok")
def sync_ok():
    time.sleep(1)                # FastAPI runs plain def endpoints in a thread pool
    return {"ok": True}

@app.get("/async-BAD")
async def async_bad():
    time.sleep(1)                # blocks the event loop: every other request freezes
    return {"ok": False}
```

FastAPI runs a plain `def` endpoint in a worker thread pool, so blocking code there is tolerable (the pool is limited, by default to 40 threads through AnyIO). An `async def` endpoint runs on the loop itself, so it must only await. The worst option is the third: blocking code inside `async def`. If you must call a blocking library from an async endpoint, wrap it in `await asyncio.to_thread(...)`.

### 11.2 A complete agent service

Save as `app.py` next to `llm.py` from Module 10:

```python
# app.py
import asyncio, json, os
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

import httpx2
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llm import AnthropicLLM, FakeLLM

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # startup: create long-lived, shared resources once
    app.state.http = httpx2.AsyncClient(timeout=httpx2.Timeout(10.0))
    if os.environ.get("ANTHROPIC_API_KEY"):
        app.state.llm = AnthropicLLM(max_concurrency=8)
    else:
        app.state.llm = FakeLLM(lambda m: FakeLLM.text_turn("Hello from the fake model, streaming word by word."))
    try:
        yield                                   # the app serves requests here
    finally:
        await app.state.http.aclose()           # shutdown: release resources

app = FastAPI(lifespan=lifespan)

def get_llm(request: Request):
    return request.app.state.llm

LLM = Annotated[object, Depends(get_llm)]

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, llm: LLM):
    try:
        async with asyncio.timeout(60):
            turn = await llm.complete([{"role": "user", "content": body.message}])
    except TimeoutError:
        raise HTTPException(status_code=504, detail="model timed out")
    return ChatResponse(reply=turn.text)

@app.post("/chat/stream")
async def chat_stream(body: ChatRequest, llm: LLM):
    async def events():
        try:
            async for piece in llm.stream([{"role": "user", "content": body.message}]):
                yield f"data: {json.dumps({'delta': piece})}\n\n"   # one SSE event
            yield "data: [DONE]\n\n"
        finally:
            print("stream closed")               # also runs if the client disconnects
    return StreamingResponse(events(), media_type="text/event-stream")
```

Run it and try it:

```bash
uv run uvicorn app:app --reload
curl -s localhost:8000/chat -H 'content-type: application/json' -d '{"message": "hi"}'
curl -N localhost:8000/chat/stream -H 'content-type: application/json' -d '{"message": "hi"}'
```

Key ideas in this file:

- **Lifespan.** The `lifespan` async context manager runs once at startup (before `yield`) and once at shutdown (after). This is where connection pools, LLM clients, model weights and background workers are created and closed. Never create an `AsyncClient` per request.
- **Dependencies.** `Depends(get_llm)` hands each endpoint the shared client. In tests you replace it with `app.dependency_overrides`, which is how you test agents without real API calls.
- **Server-sent events (SSE).** SSE is a simple text protocol over HTTP: each event is a line starting `data:` followed by a blank line. Browsers read it with `EventSource`; most chat UIs use it to show tokens as they arrive.
- **Disconnects.** When a user closes the tab mid-stream, Starlette stops the streaming generator, by cancelling the task that drives it or by closing the generator, depending on the server and version. A cancellation arrives at the current `await` (Module 5), the `finally` block runs, and the upstream LLM stream is closed too, so you stop paying for tokens nobody will read.

### 11.3 Production notes

- **Concurrency per process.** One uvicorn worker is one event loop on one core. Scale out with several workers (`uvicorn app:app --workers 4`, or several containers behind a load balancer). Cap concurrent LLM calls per process with a semaphore (our `AnthropicLLM` has one).
- **Timeouts at every layer.** Load balancer, server, endpoint (`asyncio.timeout`), SDK client and tool calls should each have a limit, from longest on the outside to shortest on the inside.
- **Background work.** `fastapi.BackgroundTasks` runs a function after the response is sent, in the same process, which is fine for small jobs such as logging. Long agent runs belong in a task queue (Celery, Arq, a cloud queue) or a durable workflow engine, so a server restart does not lose them.
- **Graceful shutdown.** On shutdown, uvicorn stops accepting connections, waits for in-flight requests, then runs the lifespan's shutdown code. Drain your own background tasks there (Exercise 4.5's `drain`).

### 11.4 Testing async endpoints

Use httpx2 with `ASGITransport` to call the app in-process, and override the LLM dependency:

```python
# test_app.py
import httpx2, pytest
from app import app, get_llm
from llm import FakeLLM

@pytest.fixture
def fake_llm():
    llm = FakeLLM(lambda m: FakeLLM.text_turn("pong"), latency=0.01)
    app.dependency_overrides[get_llm] = lambda: llm
    yield llm
    app.dependency_overrides.clear()

@pytest.mark.asyncio
async def test_chat(fake_llm):
    transport = httpx2.ASGITransport(app=app)
    async with httpx2.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/chat", json={"message": "ping"})
    assert r.status_code == 200
    assert r.json() == {"reply": "pong"}
    assert fake_llm.calls == 1
```

`ASGITransport` does not run the lifespan, which is fine here because the override replaces the only state the endpoint needs. When a test needs the lifespan, use FastAPI's `TestClient` as a context manager (`with TestClient(app) as client:`), which runs startup and shutdown. Module 14 covers pytest-asyncio. One limitation to know: `ASGITransport` collects the *entire* response before returning it (measured: with a 3-second SSE stream, the first event arrived after 3.08 s), so it cannot test streaming latency or client disconnects. For those, run a real uvicorn server in the test (Exercise 14.3).

### Check yourself

1. Where should a shared `httpx2.AsyncClient` be created in a FastAPI app? *Answer: in the lifespan handler, stored on `app.state`, and closed at shutdown.*
2. A `def` endpoint calls `time.sleep(1)`. Does it block other requests? *Answer: no, FastAPI runs `def` endpoints in a thread pool; but the pool is limited, so very many concurrent calls still queue.*
3. What happens to a streaming generator when the client disconnects? *Answer: it is stopped (cancelled or closed); cleanup in its `finally` runs.*

### Exercises

**11.1 (E)** Run `app.py` with the fake LLM. Call `/chat` 20 times concurrently from an async script using `httpx2.AsyncClient` and `asyncio.gather`. How long does it take with the fake latency of 0.3 s? Why?

**11.2 (M)** Write `sse_client.py` that posts to `/chat/stream`, parses SSE `data:` lines, prints deltas as they arrive, reports time to first token, and stops after the first 3 events. Check the server log for "stream closed" and explain why it appears even though the stream had not finished.

**11.3 (M)** Add three endpoints that each "call a slow legacy API" for 1 s: one `async def` with `time.sleep`, one plain `def` with `time.sleep`, one `async def` with `await asyncio.to_thread(time.sleep, 1)`. Fire 10 concurrent requests at each and record the total time. Explain the three results.

**11.4 (H)** Add `POST /research` that takes `{"topic": ...}`, runs three async tools concurrently (simulate `web_search`, `wiki_lookup` and `news` with different latencies, one of which sometimes takes 10 s), applies a 3-second budget to the whole fan-out, and returns every tool's status (`ok`, `timeout` or `error`) and result. Write a pytest test that runs offline.

### Solutions

**11.1**

```python
import asyncio, time
import httpx2

async def main():
    async with httpx2.AsyncClient(base_url="http://localhost:8000") as client:
        start = time.perf_counter()
        rs = await asyncio.gather(*(client.post("/chat", json={"message": f"hi {i}"}) for i in range(20)))
        print({r.status_code for r in rs}, f"{time.perf_counter() - start:.2f}s")   # {200} about 0.35s

asyncio.run(main())
```

About 0.3 s plus overhead, because all 20 requests await the fake model's `asyncio.sleep` concurrently on the server's single event loop. With the real `AnthropicLLM(max_concurrency=8)`, the semaphore would cap the model calls at 8 at a time.

**11.2**

```python
import asyncio, json, time
import httpx2

async def main():
    async with httpx2.AsyncClient(base_url="http://localhost:8000", timeout=None) as client:
        start = time.perf_counter()
        seen = 0
        async with client.stream("POST", "/chat/stream", json={"message": "hello"}) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    break
                if seen == 0:
                    print(f"TTFT {time.perf_counter() - start:.2f}s")
                print(json.loads(payload)["delta"], end="", flush=True)
                seen += 1
                if seen == 3:
                    break                     # leaving the block closes the connection
        print()

asyncio.run(main())
```

Leaving `client.stream(...)` early closes the TCP connection. The server notices the disconnect and cancels the generator, so its `finally` block prints "stream closed". This is the behaviour you want: the server stops generating (and paying for) tokens nobody will read.

**11.3** Typical results with 10 concurrent requests: the `async def` + `time.sleep` endpoint takes about 10 s, because each request blocks the single loop for 1 s in turn (and every other endpoint, including `/health`, freezes meanwhile). The plain `def` endpoint takes about 1 s, because the 10 calls run in 10 pool threads. The `to_thread` endpoint also takes about 1 s for the same reason. Lesson: inside `async def`, blocking work must be awaited through a thread or replaced with an async library.

**11.4**

```python
# research.py: add to app.py, or import it there with app.include_router(router)
import asyncio, random
from typing import Any
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

async def web_search(topic: str) -> list[str]:
    await asyncio.sleep(0.5)
    return [f"{topic}: web hit {i}" for i in range(3)]

async def wiki_lookup(topic: str) -> str:
    await asyncio.sleep(0.8)
    return f"{topic} is a topic with a long history."

async def news(topic: str) -> list[str]:
    await asyncio.sleep(10 if random.random() < 0.5 else 0.3)   # sometimes very slow
    return [f"Breaking: {topic}"]

TOOLS = {"web_search": web_search, "wiki_lookup": wiki_lookup, "news": news}

class ResearchRequest(BaseModel):
    topic: str

class ToolOutcome(BaseModel):
    status: str                    # "ok" | "timeout" | "error"
    result: Any = None
    error: str | None = None

@router.post("/research")
async def research(body: ResearchRequest) -> dict[str, ToolOutcome]:
    tasks = {name: asyncio.create_task(fn(body.topic)) for name, fn in TOOLS.items()}
    done, pending = await asyncio.wait(tasks.values(), timeout=3.0)   # overall budget
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    out = {}
    for name, t in tasks.items():
        if t in pending:
            out[name] = ToolOutcome(status="timeout")
        elif t.exception() is not None:
            out[name] = ToolOutcome(status="error", error=repr(t.exception()))
        else:
            out[name] = ToolOutcome(status="ok", result=t.result())
    return out
```

```python
# test_research.py
import httpx2, pytest
from fastapi import FastAPI
import research

@pytest.mark.asyncio
async def test_research_reports_timeouts(monkeypatch):
    async def slow_news(topic):
        import asyncio
        await asyncio.sleep(10)
    monkeypatch.setitem(research.TOOLS, "news", slow_news)
    app = FastAPI()
    app.include_router(research.router)
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/research", json={"topic": "mcp"})
    body = r.json()
    assert body["web_search"]["status"] == "ok"
    assert body["news"]["status"] == "timeout"
```

`asyncio.wait(..., timeout=)` is used rather than a TaskGroup because we want partial results: a TaskGroup would cancel everything on the first error. The test takes 3 s because of the budget; Module 14 shows how to make the budget configurable so tests run fast.

## Module 12: MCP servers and clients

The Model Context Protocol (MCP) is an open standard for connecting AI applications to tools and data: an **MCP server** exposes tools, resources and prompts, and an **MCP client** inside an agent or app (Claude, IDEs, your own code) discovers and calls them. The official Python SDK is async from top to bottom. Version 2 is the current stable line: `FastMCP` was renamed `MCPServer`, and a single `Client` object replaced v1's three nested layers ([What's new in v2](https://py.sdk.modelcontextprotocol.io/whats-new/)).

### 12.1 Concepts in one minute

| Term | Meaning |
| --- | --- |
| Host | The application the user interacts with (a chat app, an IDE, your agent service) |
| Client | The component inside the host that talks to one MCP server |
| Server | A process or service that exposes capabilities |
| Tool | A function the model can call, described by a name, description and JSON Schema |
| Resource | Read-only data addressed by a URI, such as a file or a database row |
| Prompt | A reusable prompt template the server offers |
| Transport | How messages travel: **stdio** (the client launches the server as a subprocess) or **Streamable HTTP** (the server is a web service) |

The current protocol revision, 2026-07-28, makes requests stateless: no connection handshake and no session, so any replica behind a load balancer can answer. SDK v2 serves both the new revision and older 2025-era clients from the same server automatically.

### 12.2 A server

```python
# research_server.py
import asyncio
from mcp.server import MCPServer

mcp = MCPServer("research-tools", instructions="Search the notes before answering.")

NOTES = {
    "asyncio": "asyncio is Python's standard library for cooperative concurrency.",
    "mcp": "MCP is an open protocol connecting AI apps to tools and data.",
    "taskgroup": "asyncio.TaskGroup provides structured concurrency since Python 3.11.",
}

@mcp.tool()
async def search_notes(query: str, limit: int = 3) -> list[str]:
    """Search the course notes and return matching snippets."""
    await asyncio.sleep(0.3)                      # pretend this is a vector-database query
    hits = [text for key, text in NOTES.items() if query.lower() in key or query.lower() in text.lower()]
    return hits[:limit]

@mcp.tool()
async def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())

if __name__ == "__main__":
    mcp.run(transport="streamable-http")          # serves http://localhost:8000/mcp by default
```

The decorator turns each function into a tool. The tool's name is the function name, the description is the docstring, and the input schema is generated from the type hints, so write good docstrings and precise types: they are the documentation the model reads. Tools may be `async def` (preferred for I/O) or plain `def`; in SDK v2 a plain `def` tool runs on a worker thread so it cannot block the server's event loop.

Run it with `uv run python research_server.py`, or `uv run mcp run research_server.py --transport streamable-http`. For a server that a desktop host launches over stdio, call `mcp.run()` with no arguments.

Errors: if a tool raises, the client receives a result with `is_error=True`. In v2 only the message of a `ToolError` (from `mcp.server.mcpserver.exceptions`) reaches the model; other exceptions are reported as a generic "Error executing tool" message, with the traceback in the server log. That protects you from leaking internal details to the model.

### 12.3 A client

```python
# client_demo.py
import asyncio
from mcp import Client

async def main():
    async with Client("http://localhost:8000/mcp") as client:
        listing = await client.list_tools()
        for tool in listing.tools:
            print(tool.name, "-", tool.description)
        result = await client.call_tool("search_notes", {"query": "asyncio"})
        print("error?", result.is_error)
        print(result.structured_content)          # machine-readable result
        print(result.content[0].text)             # text form, which is what models usually see

asyncio.run(main())
```

`async with Client(...)` connects and negotiates the protocol version; leaving the block closes the connection. `Client` also accepts stdio server parameters to launch a subprocess, or, in tests, **the server object itself**, which runs everything in memory with no network:

```python
from research_server import mcp as server

async def test_in_memory():
    async with Client(server) as client:
        r = await client.call_tool("word_count", {"text": "one two three"})
        assert r.structured_content == {"result": 3}
```

In v2 every attribute on protocol types is snake\_case in Python (`is_error`, `input_schema`, `structured_content`); the JSON on the wire is still camelCase. If you read v1 tutorials you will see `isError` and `inputSchema`. Simple return values such as `int` or `list[str]` are wrapped as `{"result": ...}` in the structured content; a `dict` or Pydantic model is returned as-is.

### 12.4 Bridging MCP tools into an LLM agent

An agent discovers tools at runtime from its MCP servers and passes them to the model. Converting between the two formats takes a few lines:

```python
from llm import ToolCall

def mcp_tools_to_llm_tools(listing) -> list[dict]:
    return [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
            for t in listing.tools]

async def call_mcp_tool(client, call: ToolCall, timeout: float = 30.0) -> dict:
    try:
        async with asyncio.timeout(timeout):
            result = await client.call_tool(call.name, call.arguments)
        text = "\n".join(c.text for c in result.content if c.type == "text")
        return {"type": "tool_result", "tool_use_id": call.id, "content": text, "is_error": bool(result.is_error)}
    except Exception as exc:
        return {"type": "tool_result", "tool_use_id": call.id, "content": f"ERROR: {exc}", "is_error": True}
```

With several servers, open them all with `contextlib.AsyncExitStack` (Module 7.1) so they are all closed on exit, prefix tool names with the server name to avoid collisions, and keep a routing table from tool name to client. Exercise 12.4 builds this.

### 12.5 Async lessons specific to MCP

- **One client per server, shared.** A `Client` handles concurrent `call_tool` calls; you do not need one per request. Create clients at agent or app startup (for example in a FastAPI lifespan) and reuse them.
- **Tools run concurrently on the server.** Several calls from one or many clients are served concurrently, as long as your tool functions do not block. Use async libraries in tools, or plain `def` for blocking code (v2 runs it on a thread).
- **Timeouts on both sides.** Wrap `call_tool` in `asyncio.timeout` in the client, and bound your server-side work too. In SDK v2 a client request that times out also cancels the handler on the server.
- **Long-lived server resources go in a lifespan.** `MCPServer("name", lifespan=...)` takes an async context manager like FastAPI's. In v2 the streamable HTTP lifespan runs once at startup and its state is shared by every request.

### Check yourself

1. What are the two standard MCP transports? *Answer: stdio and Streamable HTTP.*
2. Where does an MCP tool's input schema come from when you use `@mcp.tool()`? *Answer: the function's type hints; the description comes from its docstring.*
3. How do you test an MCP server without a network? *Answer: pass the server object to `Client(server)`, which connects in memory.*

### Exercises

**12.1 (E)** Write an MCP server with tools `add(a: float, b: float) -> float` and `async slow_echo(text: str, seconds: float = 1.0) -> str`. Test both with an in-memory `Client`.

**12.2 (M)** Call `slow_echo` 10 times concurrently (1 s each) through one in-memory client. Measure the total time. Then change `slow_echo` to use `time.sleep` inside an `async def` and measure again. Finally make it a plain `def` with `time.sleep`. Explain all three numbers.

**12.3 (M)** Connect the tool loop from Exercise 10.4 to MCP: discover tools from `research_server` with `list_tools`, give them to a `FakeLLM` script that requests `search_notes("asyncio")` and `search_notes("mcp")` in one turn, execute both concurrently through `call_mcp_tool`, and let the second turn summarise the tool results.

**12.4 (H)** Write `MCPRouter`, an async context manager that connects to several servers given as `{name: target}` (a URL or an in-memory server object), skips any server that fails to connect (logging a warning), exposes a combined tool list with names prefixed `name__tool`, and routes `call(prefixed_name, args)` to the right client. Then answer: why is it a trap to connect to all servers concurrently by entering each client inside a separate TaskGroup task?

### Solutions

**12.1**

```python
# calc_server.py
import asyncio
from mcp.server import MCPServer

mcp = MCPServer("calc")

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@mcp.tool()
async def slow_echo(text: str, seconds: float = 1.0) -> str:
    """Echo text back after a delay."""
    await asyncio.sleep(seconds)
    return text
```

```python
import asyncio
from mcp import Client
from calc_server import mcp

async def main():
    async with Client(mcp) as client:
        print((await client.call_tool("add", {"a": 2, "b": 3})).structured_content)        # {'result': 5.0}
        print((await client.call_tool("slow_echo", {"text": "hi", "seconds": 0.2})).content[0].text)  # hi

asyncio.run(main())
```

**12.2**

```python
import asyncio, time
from mcp import Client
from calc_server import mcp

async def main():
    async with Client(mcp) as client:
        start = time.perf_counter()
        await asyncio.gather(*(client.call_tool("slow_echo", {"text": str(i)}) for i in range(10)))
        print(f"{time.perf_counter() - start:.2f}s")

asyncio.run(main())
```

With `await asyncio.sleep`: about 1 s, because the server runs the ten calls concurrently. With `time.sleep` inside `async def`: about 10 s, because each call blocks the event loop that serves all of them (in the in-memory case, it is even the same loop as the client). With a plain `def` and `time.sleep`: about 1 s again, because SDK v2 runs sync tools on worker threads. Same lesson as Module 11.3, in a different framework.

**12.3**

```python
import asyncio
from mcp import Client
from llm import FakeLLM, AssistantTurn
from research_server import mcp as server
# mcp_tools_to_llm_tools and call_mcp_tool from section 12.4

def script(messages: list[dict]) -> AssistantTurn:
    last = messages[-1]
    if isinstance(last["content"], str):
        return FakeLLM.tool_turn(("search_notes", {"query": "asyncio"}), ("search_notes", {"query": "mcp"}))
    snippets = " | ".join(block["content"] for block in last["content"])
    return FakeLLM.text_turn(f"Summary based on the notes: {snippets}")

async def main():
    llm = FakeLLM(script, latency=0.1)
    async with Client(server) as client:
        tools = mcp_tools_to_llm_tools(await client.list_tools())
        messages = [{"role": "user", "content": "What are asyncio and MCP?"}]
        while True:
            turn = await llm.complete(messages, tools=tools)
            messages.append({"role": "assistant", "content": turn.content})
            if not turn.tool_calls:
                break
            results = await asyncio.gather(*(call_mcp_tool(client, c) for c in turn.tool_calls))
            messages.append({"role": "user", "content": list(results)})
    print(turn.text)

asyncio.run(main())
```

Swap `FakeLLM(script)` for `AnthropicLLM()` and the same loop runs against a real model, which now chooses the tools itself from the descriptions.

**12.4**

```python
import asyncio, logging
from contextlib import AsyncExitStack
from mcp import Client

log = logging.getLogger("router")
SEP = "__"

class MCPRouter:
    def __init__(self, targets: dict[str, object]):
        self.targets = targets
        self.clients: dict[str, Client] = {}
        self.routes: dict[str, tuple[str, str]] = {}    # prefixed name -> (server, tool)
        self.tools: list[dict] = []
        self._stack = AsyncExitStack()

    async def _connect(self, name: str, target) -> None:
        try:
            async with asyncio.timeout(10):
                client = await self._stack.enter_async_context(Client(target))
                listing = await client.list_tools()
        except Exception as exc:
            log.warning("skipping MCP server %s: %r", name, exc)
            return
        self.clients[name] = client
        for t in listing.tools:
            prefixed = f"{name}{SEP}{t.name}"
            self.routes[prefixed] = (name, t.name)
            self.tools.append({"name": prefixed, "description": t.description or "", "input_schema": t.input_schema})

    async def __aenter__(self) -> "MCPRouter":
        await self._stack.__aenter__()
        try:
            for name, target in self.targets.items():   # sequential, in THIS task (see below)
                await self._connect(name, target)
        except BaseException:
            await self._stack.aclose()
            raise
        return self

    async def __aexit__(self, *exc_info):
        return await self._stack.__aexit__(*exc_info)

    async def call(self, prefixed_name: str, args: dict):
        server, tool = self.routes[prefixed_name]
        return await self.clients[server].call_tool(tool, args)

async def main():
    from calc_server import mcp as calc
    from research_server import mcp as research
    async with MCPRouter({"calc": calc, "research": research, "broken": "http://localhost:1/mcp"}) as router:
        print([t["name"] for t in router.tools])
        r = await router.call("calc__add", {"a": 1, "b": 2})
        print(r.structured_content)
        # tool calls to different servers CAN run concurrently:
        a, b = await asyncio.gather(router.call("calc__add", {"a": 1, "b": 1}),
                                    router.call("research__word_count", {"text": "a b c"}))

asyncio.run(main())
```

The broken server is logged and skipped (after at most 10 s), and the other two keep working. Why connect sequentially? The MCP SDK is built on AnyIO, and many async libraries built on AnyIO or on `asyncio.TaskGroup` hold *task-scoped* state inside their context managers (cancel scopes, task groups). Such a context manager must be exited by the same task that entered it. If each `Client` were entered inside a child task of a TaskGroup and then closed later by the parent task through the exit stack, AnyIO raises `RuntimeError: Attempted to exit cancel scope in a different task than it was entered in` (reproduced in Module 17). Two correct ways to get concurrent connection: accept sequential connection at startup (usually fine, since it happens once), or run each client inside its own long-lived task that enters the context, signals readiness with an `asyncio.Event`, serves requests from a queue, and exits the context itself at shutdown. Calling tools concurrently on already-connected clients, as at the end of `main`, is always fine.

## Module 13: Building agents

An agent is an LLM in a loop: it reads the conversation, either answers or requests tools, your code runs the tools (concurrently), appends the results, and the loop repeats until the model answers or a limit is hit. Every concept in this course shows up in that loop: tasks, timeouts, cancellation, semaphores, queues and streaming. This module builds a production-style loop from scratch, then adds multi-agent patterns and the same ideas in LangGraph.

### 13.1 The agent loop

A production loop needs more than the toy in Module 10.5:

- **Stopping conditions:** a final answer, a maximum number of steps, and an overall deadline. Without them a confused model can loop forever and burn money.
- **Concurrent tool execution** with a timeout per tool, and errors returned to the model as text.
- **Output limits:** truncate huge tool outputs so they do not overflow the context window.
- **Events** for observability and UIs: "step 2 started", "calling search", "final answer".
- **Cancellation safety:** if the user hits stop, everything in flight is cancelled cleanly.

&#91;embedded content: the agent loop · 5 steps, 1 decision\]

Every pass through the loop checks the step limit and the overall deadline; the code below implements exactly this picture.

Save as `agent.py`:

```python
# agent.py
import asyncio, json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from llm import ToolCall

ToolFn = Callable[..., Awaitable[Any]]

@dataclass
class AgentConfig:
    max_steps: int = 8
    deadline_s: float = 60.0
    tool_timeout_s: float = 15.0
    max_tool_output_chars: int = 4000

@dataclass
class AgentResult:
    answer: str
    steps: int
    stopped_because: str              # "final_answer" | "max_steps" | "deadline"
    messages: list = field(default_factory=list)

class Agent:
    def __init__(self, llm, tools: dict[str, ToolFn], tool_specs: list[dict], *,
                 system: str | None = None, config: AgentConfig | None = None,
                 events: asyncio.Queue | None = None):
        self.llm = llm
        self.tools = tools
        self.tool_specs = tool_specs
        self.system = system
        self.config = config or AgentConfig()
        self.events = events

    async def _emit(self, kind: str, **data) -> None:
        if self.events is not None:
            await self.events.put({"type": kind, **data})

    async def _run_tool(self, call: ToolCall) -> dict:
        fn = self.tools.get(call.name)
        if fn is None:
            content, is_error = f"ERROR: unknown tool {call.name!r}", True
        else:
            try:
                async with asyncio.timeout(self.config.tool_timeout_s):
                    result = await fn(**call.arguments)
                content = result if isinstance(result, str) else json.dumps(result, default=str)
                is_error = False
            except TimeoutError:
                content, is_error = f"ERROR: {call.name} timed out after {self.config.tool_timeout_s}s", True
            except Exception as exc:                 # never catches CancelledError
                content, is_error = f"ERROR: {type(exc).__name__}: {exc}", True
        content = content[: self.config.max_tool_output_chars]
        await self._emit("tool_result", name=call.name, is_error=is_error)
        return {"type": "tool_result", "tool_use_id": call.id, "content": content, "is_error": is_error}

    async def run(self, user_message: str) -> AgentResult:
        messages: list[dict] = [{"role": "user", "content": user_message}]
        steps = 0
        try:
            async with asyncio.timeout(self.config.deadline_s):
                while steps < self.config.max_steps:
                    steps += 1
                    await self._emit("step", n=steps)
                    turn = await self.llm.complete(messages, tools=self.tool_specs, system=self.system)
                    messages.append({"role": "assistant", "content": turn.content})
                    if not turn.tool_calls:
                        await self._emit("final", text=turn.text)
                        return AgentResult(turn.text, steps, "final_answer", messages)
                    for c in turn.tool_calls:
                        await self._emit("tool_call", name=c.name, args=c.arguments)
                    results = await asyncio.gather(*(self._run_tool(c) for c in turn.tool_calls))
                    messages.append({"role": "user", "content": list(results)})
                return AgentResult("Stopped: step limit reached.", steps, "max_steps", messages)
        except TimeoutError:
            return AgentResult("Stopped: time limit reached.", steps, "deadline", messages)
```

A runnable demo with the fake model:

```python
# demo_agent.py
import asyncio
from agent import Agent
from llm import FakeLLM

async def search(query: str) -> list[str]:
    await asyncio.sleep(0.4)
    return [f"{query}: result {i}" for i in range(3)]

import ast, operator

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}

def safe_eval(expression: str, max_len: int = 100) -> float:
    """Evaluate arithmetic without eval(): walk the syntax tree, allow only numbers and + - * / **."""
    if len(expression) > max_len:
        raise ValueError("expression too long")
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
                raise ValueError("exponent too large")     # stops 9**9**9**9 from freezing the loop
            return _OPS[type(node.op)](left, right)
        raise ValueError(f"unsupported expression: {type(node).__name__}")
    return ev(ast.parse(expression, mode="eval"))

async def calculator(expression: str) -> float:
    return safe_eval(expression)

TOOLS = {"search": search, "calculator": calculator}
SPECS = [
    {"name": "search", "description": "Web search.", "input_schema": {"type": "object",
        "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculator", "description": "Evaluate arithmetic.", "input_schema": {"type": "object",
        "properties": {"expression": {"type": "string"}}, "required": ["expression"]}},
]

def script(messages):
    n_assistant = sum(1 for m in messages if m["role"] == "assistant")
    if n_assistant == 0:
        return FakeLLM.tool_turn(("search", {"query": "python 3.15 release"}),
                                 ("calculator", {"expression": "2026 - 1991"}))
    return FakeLLM.text_turn("Python 3.15 shipped in October 2026; Python is 35 years old.")

async def main():
    events: asyncio.Queue = asyncio.Queue()
    agent = Agent(FakeLLM(script, latency=0.2), TOOLS, SPECS, events=events)

    async def printer():
        while True:
            print("event:", await events.get())

    printer_task = asyncio.create_task(printer())
    result = await agent.run("When did 3.15 ship, and how old is Python?")
    await asyncio.sleep(0)                 # let the printer drain the last event
    printer_task.cancel()
    print(result.stopped_because, "after", result.steps, "steps:", result.answer)

asyncio.run(main())
```

Note the separation: the agent *produces* events into a queue; who *consumes* them (a printer, a FastAPI SSE stream, a tracing system) is somebody else's concern. That producer/consumer split (Module 6.2) is how real agent UIs show live progress.

**A lesson from an earlier draft of this course.** The first version of this demo used `eval()` on input restricted to digits, spaces and `+-*/().`. That looks safe, but `*` allows `**`, and the model (or a prompt-injected document) can send `9**9**9**9`. Run on Python 3.12.3, that single tool call froze the event loop: the heartbeat task stopped printing and even a 1-second `asyncio.timeout` never fired, because a timeout is delivered by the very loop that was blocked. Treat every tool argument as untrusted input, never `eval` it, bound the work a tool can do, and remember that CPU-heavy tool code needs a separate process if it must be killable: a thread started by `to_thread` cannot be cancelled.

### 13.2 Multi-agent patterns with asyncio

Most "multi-agent" systems are a handful of concurrency patterns you already know:

| Pattern | What happens | Asyncio building blocks |
| --- | --- | --- |
| Orchestrator and workers | A lead agent splits a task into subtasks; worker agents run concurrently; the lead merges results | `TaskGroup` or `gather(return_exceptions=True)`, `Semaphore` |
| Parallel guardrails | A cheap safety or relevance check runs at the same time as the main generation; a failed check cancels it | `create_task`, `wait(FIRST_COMPLETED)`, `cancel` |
| Hedging and fallbacks | Same request to two models; take the first or fall back on timeout | `wait(FIRST_COMPLETED)`, `asyncio.timeout` |
| Pipeline | Stages connected by queues (retrieve, then rerank, then generate) | `Queue` with `maxsize`, worker tasks |
| Evaluator loop | A generator agent drafts; an evaluator scores; repeat until good enough or out of budget | A `while` loop with a step limit and `asyncio.timeout` |
| Human in the loop | Pause until a person approves | `Event`, `Future`, timeout |

### 13.3 The same ideas in LangGraph

LangGraph models an agent as a graph: **nodes** are functions (sync or `async def`) that read the shared **state** and return updates; **edges** decide what runs next. Nodes that are ready in the same step run concurrently, and LangGraph's async API (`ainvoke`, `astream`) runs them as asyncio tasks. Version 1.x is the stable line (1.2 as of this writing; [LangChain changelog](https://docs.langchain.com/oss/python/releases/changelog/index.html)).

A map-reduce research graph, with three research steps running in parallel:

```python
# research_graph.py
import asyncio, operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

class State(TypedDict):
    question: str
    subquestions: list[str]
    findings: Annotated[list[str], operator.add]   # reducer: parallel updates are concatenated
    answer: str

async def plan(state: State) -> dict:
    await asyncio.sleep(0.1)                         # an LLM call in real life
    q = state["question"]
    return {"subquestions": [f"{q}: history", f"{q}: current state", f"{q}: criticisms"]}

def fan_out(state: State) -> list[Send]:
    # one Send per subquestion: each starts a "research" run with its own input
    return [Send("research", {"question": sq}) for sq in state["subquestions"]]

async def research(task: dict) -> dict:
    await asyncio.sleep(1.0)                         # search + LLM summarise
    return {"findings": [f"finding about {task['question']}"]}

async def write(state: State) -> dict:
    return {"answer": " | ".join(sorted(state["findings"]))}

builder = StateGraph(State)
builder.add_node("plan", plan)
builder.add_node("research", research)
builder.add_node("write", write)
builder.add_edge(START, "plan")
builder.add_conditional_edges("plan", fan_out, ["research"])
builder.add_edge("research", "write")
builder.add_edge("write", END)
graph = builder.compile()

async def main():
    result = await graph.ainvoke({"question": "asyncio"})
    print(result["answer"])                          # about 1.1 s: the three research runs overlap
    async for update in graph.astream({"question": "MCP"}, stream_mode="updates"):
        print(update)                                # one dict per finished node

if __name__ == "__main__":
    asyncio.run(main())
```

Things to notice:

- `Annotated[list[str], operator.add]` is a **reducer**: when three parallel `research` runs each return `{"findings": [...]}`, LangGraph concatenates the lists instead of letting the last one win. Without a reducer, parallel writes to the same key raise an error.
- `Send` is LangGraph's fan-out primitive: a dynamic number of parallel branches, each with its own input.
- `write` runs only after all `research` branches finish, because LangGraph executes in **supersteps** (all nodes ready at the same time run together, then the next step starts). It is the graph version of awaiting a TaskGroup.
- Persistence: `builder.compile(checkpointer=InMemorySaver())` (from `langgraph.checkpoint.memory`) saves state after each step, keyed by `config={"configurable": {"thread_id": "..."}}`, which enables conversation memory, human-in-the-loop interrupts and resuming after a crash (with a database-backed checkpointer).

For a ready-made tool-calling agent on top of LangGraph, LangChain 1.x provides `create_agent`. Other popular agent frameworks (the OpenAI Agents SDK, Anthropic's Claude Agent SDK, Pydantic AI) are also asyncio-native; once you understand this course, learning any of them is a matter of reading its docs.

### Check yourself

1. Name three stopping conditions every production agent loop needs. *Answer: a final answer, a maximum number of steps, and an overall deadline (plus, often, a token or cost budget).*
2. Why does the agent push events to a queue instead of printing them? *Answer: to decouple producing progress from consuming it, so the same agent can feed a terminal, an SSE stream or a tracer.*
3. What does a reducer do in LangGraph? *Answer: it defines how updates to the same state key are combined, which is required when parallel nodes write to that key.*

### Exercises

**13.1 (M)** Write a `FakeLLM` script that always requests a tool and never answers. Run the `Agent` with `max_steps=5` and confirm it stops with `max_steps`. Then make the tool take 2 s, set `deadline_s=3`, and confirm it stops with `deadline`. How many steps ran in each case?

**13.2 (M)** Add a `POST /agent/stream` endpoint to the FastAPI app from Module 11 that runs the agent and streams its events to the client as SSE. Run the agent as a task, forward events from the queue until a `final` event, and make sure the agent task is cancelled if the client disconnects.

**13.3 (H)** Parallel guardrail. Write `guarded_answer(question)` that starts `generate(question)` (2 s) and `check_safety(question)` (0.5 s, returns `False` for questions containing "exploit") concurrently. If the check fails, cancel the generation immediately and return a refusal. If the check passes, return the generation's answer when it finishes. Measure that a blocked question returns in about 0.5 s and an allowed one in about 2 s, not 2.5 s.

**13.4 (M)** Extend the LangGraph research graph with a `review` node after `research`: if fewer than 4 findings exist and fewer than 2 review rounds have happened, route back to a node that issues one more research `Send`; otherwise go to `write`. Track the round count in the state.

**13.5 (H)** Orchestrator and workers. Write `orchestrate(task, n_workers)` that asks a fake planner for up to 8 subtasks, runs a worker agent (an `Agent` with a fake LLM) for each, at most 3 at a time, tolerates individual worker failures, and returns a merged report listing successes and failures. Compare its wall time to running the workers sequentially.

### Solutions

**13.1**

```python
import asyncio
from agent import Agent, AgentConfig
from llm import FakeLLM

async def slow_tool(x: int = 0) -> str:
    await asyncio.sleep(2)
    return "done"

async def fast_tool(x: int = 0) -> str:
    return "done"

always_tool = lambda messages: FakeLLM.tool_turn(("t", {"x": 1}))
spec = [{"name": "t", "description": "a tool", "input_schema": {"type": "object", "properties": {}}}]

async def main():
    a = Agent(FakeLLM(always_tool, latency=0.05), {"t": fast_tool}, spec, config=AgentConfig(max_steps=5))
    r = await a.run("go")
    print(r.stopped_because, r.steps)          # max_steps 5

    b = Agent(FakeLLM(always_tool, latency=0.05), {"t": slow_tool}, spec, config=AgentConfig(deadline_s=3))
    r = await b.run("go")
    print(r.stopped_because, r.steps)          # deadline 2

asyncio.run(main())
```

With the slow tool, step 1 ends at about 2.05 s; step 2 starts and is still waiting on its tool when the 3 s deadline cancels it, so two steps ran.

**13.2**

```python
# add to app.py
import asyncio, json
from fastapi.responses import StreamingResponse
from agent import Agent
from demo_agent import TOOLS, SPECS

def sse_event(data: dict) -> str:
    return f"data: {json.dumps(data, default=str)}\n\n"

@app.post("/agent/stream")
async def agent_stream(body: ChatRequest, llm: LLM):
    events: asyncio.Queue = asyncio.Queue()
    agent = Agent(llm, TOOLS, SPECS, events=events)

    async def sse():
        task = asyncio.create_task(agent.run(body.message))
        try:
            while True:
                if task.done():
                    while not events.empty():          # drain what is left, without waiting
                        yield sse_event(events.get_nowait())
                    result = task.result()             # re-raises if the agent crashed
                    yield sse_event({"type": "end", "reason": result.stopped_because})
                    break
                getter = asyncio.create_task(events.get())
                await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)
                if getter.done():
                    yield sse_event(getter.result())
                else:
                    getter.cancel()                    # the agent finished first; loop drains
        finally:
            task.cancel()                              # client disconnected or we finished
            await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(sse(), media_type="text/event-stream")
```

Waiting on both the next event *and* the agent task handles agents that stop through `max_steps` or `deadline` without a `final` event. The drain step matters: when the agent task finishes, events it queued just before returning may not have been read yet, and a naive version that only waits for "whichever finishes first" would drop them. Cancelling a pending `Queue.get` is safe; no item is lost. The `finally` block makes disconnects safe: the agent and all its in-flight tool calls are cancelled.

**13.3**

```python
import asyncio, time

async def generate(q: str) -> str:
    await asyncio.sleep(2)
    return f"Detailed answer to {q!r}"

async def check_safety(q: str) -> bool:
    await asyncio.sleep(0.5)
    return "exploit" not in q.lower()

async def guarded_answer(q: str) -> str:
    gen = asyncio.create_task(generate(q))
    try:
        if not await check_safety(q):
            return "Sorry, I can't help with that."
        return await gen
    finally:
        gen.cancel()                                  # no-op if it already finished
        await asyncio.gather(gen, return_exceptions=True)

async def main():
    for q in ["how do I write an exploit", "how does asyncio work"]:
        start = time.perf_counter()
        answer = await guarded_answer(q)
        print(f"{time.perf_counter() - start:.2f}s  {answer}")   # 0.50s refusal / 2.00s answer

asyncio.run(main())
```

The generation starts immediately and runs *while* the check runs, so an allowed request costs max(2, 0.5) = 2 s, not 2.5 s. This pattern is widely used to keep guardrails from adding latency.

**13.4**

```python
class State(TypedDict):
    question: str
    subquestions: list[str]
    findings: Annotated[list[str], operator.add]
    rounds: int
    answer: str

async def review(state: State) -> dict:
    return {"rounds": state.get("rounds", 0) + 1}

def after_review(state: State):
    if len(state["findings"]) < 4 and state["rounds"] < 2:
        return [Send("research", {"question": f"{state['question']}: follow-up {state['rounds']}"})]
    return "write"

builder = StateGraph(State)
builder.add_node("plan", plan)
builder.add_node("research", research)
builder.add_node("review", review)
builder.add_node("write", write)
builder.add_edge(START, "plan")
builder.add_conditional_edges("plan", fan_out, ["research"])
builder.add_edge("research", "review")
builder.add_conditional_edges("review", after_review, ["research", "write"])
builder.add_edge("write", END)
graph = builder.compile()
```

First round: 3 findings, so review routes back for one follow-up; second review sees 4 findings and routes to `write`. The `rounds < 2` guard is the graph's equivalent of `max_steps`: never build a loop without one.

**13.5**

```python
import asyncio, random, time
from agent import Agent, AgentConfig
from llm import FakeLLM

async def plan_subtasks(task: str) -> list[str]:
    await asyncio.sleep(0.2)
    return [f"{task} / part {i}" for i in range(8)]

def make_worker() -> Agent:
    def script(messages):
        if random.random() < 0.15:
            raise RuntimeError("model overloaded")       # simulate a provider failure
        return FakeLLM.text_turn(f"Report on {messages[0]['content']}")
    return Agent(FakeLLM(script, latency=1.0), {}, [], config=AgentConfig(deadline_s=5))

async def orchestrate(task: str, max_parallel: int = 3) -> dict:
    subtasks = await plan_subtasks(task)
    sem = asyncio.Semaphore(max_parallel)
    async def work(sub: str) -> str:
        async with sem:
            result = await make_worker().run(sub)
            return result.answer
    outcomes = await asyncio.gather(*(work(s) for s in subtasks), return_exceptions=True)
    report = {"succeeded": [], "failed": []}
    for sub, out in zip(subtasks, outcomes):
        if isinstance(out, BaseException):
            report["failed"].append(f"{sub}: {out!r}")
        else:
            report["succeeded"].append(out)
    return report

async def main():
    start = time.perf_counter()
    report = await orchestrate("survey async Python")
    print(len(report["succeeded"]), "ok,", len(report["failed"]), "failed",
          f"in {time.perf_counter() - start:.2f}s")     # about 3.2s (3 rounds of 1s + planning)

asyncio.run(main())
```

Sequentially the 8 workers would take about 8.2 s. `gather(return_exceptions=True)` is the right tool because each worker is independent and partial results are valuable; a TaskGroup would cancel the remaining workers on the first failure.

## Module 14: Testing, debugging and production pitfalls

Async code is tested with pytest plus the pytest-asyncio plugin, kept fast by injecting fakes and short time limits, and made deterministic by controlling ordering with events instead of sleeps. For debugging, asyncio's debug mode, task introspection (`python -m asyncio ps` and `pstree`, new in Python 3.14) and a short list of well-known pitfalls cover most real incidents.

### 14.1 pytest-asyncio

Install `pytest` and `pytest-asyncio` (1.4 is current; [configuration docs](https://pytest-asyncio.readthedocs.io/en/stable/reference/configuration.html)). By default the plugin runs in *strict* mode, where each async test needs `@pytest.mark.asyncio`. Most projects switch on *auto* mode in `pyproject.toml` so every `async def test_...` just works:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
```

```python
# test_basics.py  (auto mode)
import asyncio, pytest

async def add_later(a: int, b: int) -> int:
    await asyncio.sleep(0.01)
    return a + b

async def test_add_later():
    assert await add_later(2, 3) == 5

async def test_timeout_raises():
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await asyncio.sleep(1)

@pytest.fixture
async def queue_with_items():          # async fixtures work in auto mode
    q: asyncio.Queue[int] = asyncio.Queue()
    for i in range(3):
        await q.put(i)
    yield q

async def test_queue(queue_with_items):
    assert queue_with_items.qsize() == 3
```

Each test gets a fresh event loop by default, so state does not leak between tests. Do not create asyncio primitives (locks, queues, clients) at module import time in code you test: create them inside the code path or in fixtures.

Two older patterns you will meet in existing code: the `event_loop` fixture (removed in pytest-asyncio 1.0) and the third-party `anyio` pytest plugin (`@pytest.mark.anyio`), which FastAPI's documentation uses and which also works well.

### 14.2 Fast, deterministic async tests

- **Inject dependencies.** Pass the LLM client, HTTP client and time limits into your functions or classes. Tests pass `FakeLLM`, `httpx2.MockTransport` clients and limits of milliseconds instead of seconds.
- **Do not sleep to wait for ordering.** `await asyncio.sleep(0.1)` "to let the other task start" makes tests slow and flaky. Use an `asyncio.Event` that the code under test sets at the interesting moment, or a fake whose method blocks on an event you control.
- **Measure concurrency, do not guess.** Count in-flight calls inside a fake (Exercise 9.1) and assert the peak.
- **Test cancellation explicitly.** Start the coroutine as a task, wait until it reaches the interesting point, cancel it, and assert that cleanup ran.

```python
async def test_cleanup_runs_on_cancel():
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def stream():
        try:
            started.set()
            await asyncio.sleep(3600)          # "waiting for tokens"
        finally:
            cleaned.set()

    task = asyncio.create_task(stream())
    await started.wait()                       # deterministic: no sleeps
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set()
```

### 14.3 Reading asyncio's warnings

| Message | What it means | Usual fix |
| --- | --- | --- |
| `RuntimeWarning: coroutine 'x' was never awaited` | A coroutine object was created and discarded | Add `await`, or wrap it in a task |
| `Task exception was never retrieved` | A task failed and nobody awaited it or read its exception | Await the task, use a TaskGroup, or check `task.exception()` in a done callback |
| `Task was destroyed but it is pending!` | A task was garbage-collected or the loop closed while it was still running | Keep a reference; cancel and await tasks at shutdown |
| `Executing <Task ...> took 0.4 seconds` (debug mode) | Something blocked the loop | Move blocking work to a thread or process |
| `RuntimeError: asyncio.run() cannot be called from a running event loop` | `asyncio.run` inside a notebook or async framework | `await` the coroutine instead |
| `RuntimeError: ... attached to a different loop` | A lock, queue or client created on one loop is used on another (often in tests) | Create such objects inside the running loop |
| `RuntimeError: There is no current event loop` | `asyncio.get_event_loop()` with no loop (3.14+) | Use `asyncio.run`, or `get_running_loop()` inside coroutines |

### 14.4 Seeing what tasks are doing

- **In code:** `asyncio.all_tasks()` returns every unfinished task; `task.get_name()` and `task.get_coro()` tell you what each is. Name important tasks: `asyncio.create_task(coro, name="agent-run-42")`.
- **Python 3.14+, from outside the process:** `python -m asyncio ps <PID>` lists the tasks of a running program, and `python -m asyncio pstree <PID>` shows which task is awaiting which, as a tree. This is how you debug a hung agent in production without restarting it. Inside the program, `asyncio.print_call_graph()` prints the same await chain for the current task.
- **Python 3.15+:** the new `profiling.sampling` profiler (Tachyon) can attach to a running process with negligible overhead, has an `--async-aware` option that reconstructs stacks per task, and a `dump` mode that prints every thread's or task's stack once, useful for hung processes ([What's new in 3.15](https://docs.python.org/3/whatsnew/3.15.html)).
- **Debug mode** (Module 8.5) in development and CI: `PYTHONASYNCIODEBUG=1`.

### 14.5 Context variables for request-scoped data

Global variables are shared by every task, so they cannot hold "the current request ID". `contextvars.ContextVar` can: each task gets a *copy* of the context that was current when it was created, so values set in a request handler are visible in every task and `to_thread` call it starts, but not in other requests.

```python
import asyncio, contextvars, logging

request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id.get()
        return True

logging.basicConfig(format="%(request_id)s %(message)s", level=logging.INFO)
logging.getLogger().addFilter(RequestIdFilter())

async def tool_call(name: str):
    await asyncio.sleep(0.1)
    logging.info("tool %s finished", name)        # logs the right request id

async def handle(rid: str):
    request_id.set(rid)
    async with asyncio.TaskGroup() as tg:
        tg.create_task(tool_call("search"))
        tg.create_task(tool_call("calc"))

async def main():
    await asyncio.gather(handle("req-A"), handle("req-B"))

asyncio.run(main())
```

Tracing libraries (OpenTelemetry, which both the MCP SDK and the Anthropic SDK can emit spans to) rely on the same mechanism to connect an LLM call to the request that caused it.

### 14.6 Production checklist for async agent services

| Area | Check |
| --- | --- |
| Blocking | No blocking calls in `async def`; heavy CPU in processes; debug mode clean in CI |
| Clients | One shared HTTP/LLM/MCP client per process, created in a lifespan, closed on shutdown |
| Timeouts | Every network call bounded; overall deadline per agent run; nested limits shrink inward |
| Retries | Only transient errors, with jittered backoff, capped, not stacked on SDK retries |
| Concurrency | Semaphores sized to provider rate limits, divided across processes; bounded queues for backpressure |
| Cancellation | `CancelledError` never swallowed; cleanup in `finally`/`async with`; disconnects cancel work |
| Tasks | No orphan tasks: TaskGroups or tracked background tasks drained at shutdown |
| Subprocesses | No shell with model input; own process group; group killed and reaped on timeout and cancellation (Module 16) |
| Shutdown | SIGTERM handled (uvicorn lifespan or your own handler); exec-form `CMD`; init process in containers (Module 18) |
| Errors | Tool errors returned to the model as text; unexpected errors logged with context |
| Limits | Max steps, token or cost budget per run, truncated tool outputs |
| Tool safety | Arguments treated as untrusted; no `eval`; URL guard with redirect checks; approval for side effects; least-privilege credentials (Module 19) |
| Observability | Request IDs in logs via contextvars; traces per LLM and tool call; TTFT, latency percentiles and queue time (Modules 14, 20) |
| Testing | Fake LLM and mock transports; deterministic tests for timeouts and cancellation; a real server for streaming tests |
| Versions | CI runs on the oldest and newest Python you support, with `-W error::DeprecationWarning` (Python 3.12 guide) |

### Check yourself

1. What does `asyncio_mode = "auto"` do? *Answer: pytest-asyncio treats every async test and fixture as asyncio without needing a marker.*
2. Why avoid `asyncio.sleep` to order events in tests? *Answer: it makes tests slow and flaky; events give exact control.*
3. How do you see the task tree of a hung process on Python 3.14? *Answer: `python -m asyncio pstree <PID>`.*

### Exercises

**14.1 (E)** Turn the solutions to Exercises 5.1, 6.3 and 9.2 into pytest tests in auto mode. Make each test run in under 0.5 s by injecting shorter delays.

**14.2 (M)** Write a deterministic test proving that the `Coalescer` from Exercise 6.4 calls `fetch` exactly once for 10 concurrent callers, and that cancelling one caller does not cancel the others. Use an `asyncio.Event` inside a fake `fetch` to hold it open, instead of sleeping.

**14.3 (M)** Write a test for the `/chat/stream` endpoint from Module 11 that reads only the first event and closes the stream, then asserts that the fake LLM's stream was closed. Hint: give the fake an `aclosed` flag set in a `finally`.

**14.4 (M)** This program leaks tasks. Find the leak by printing `asyncio.all_tasks()` at the end of `main`, explain it, and fix it.

```python
import asyncio

async def heartbeat():
    while True:
        await asyncio.sleep(0.1)

async def handle_request(i: int):
    asyncio.create_task(heartbeat())        # "keep the connection alive"
    await asyncio.sleep(0.2)
    return i

async def main():
    await asyncio.gather(*(handle_request(i) for i in range(5)))
    print("done")

asyncio.run(main())
```

**14.5 (H)** On Python 3.14 or later, run the agent demo from Module 13 with a slow tool (sleep 60 s) and, from another terminal, inspect it with `python -m asyncio ps <PID>` and `python -m asyncio pstree <PID>` (find the PID with `ps aux | grep demo_agent`). Describe the tree you see and map each level to the code.

### Solutions

**14.1** (one of the three, the others follow the same shape)

```python
# test_fallback.py
import asyncio

async def primary(prompt: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return f"primary: {prompt}"

async def backup(prompt: str) -> str:
    return f"backup: {prompt}"

async def ask_with_fallback(prompt: str, primary_delay: float, timeout: float = 1.0) -> str:
    try:
        async with asyncio.timeout(timeout):
            return await primary(prompt, primary_delay)
    except TimeoutError:
        return await backup(prompt)

async def test_uses_primary_when_fast():
    assert await ask_with_fallback("hi", primary_delay=0.01, timeout=0.1) == "primary: hi"

async def test_falls_back_on_timeout():
    assert await ask_with_fallback("hi", primary_delay=1.0, timeout=0.05) == "backup: hi"
```

The only change from the original solution is the `timeout` parameter: making limits injectable is what makes async code testable.

**14.2**

```python
# test_coalescer.py
import asyncio, pytest
from coalescer import Coalescer        # the class from Exercise 6.4

async def test_single_fetch_and_cancel_isolation():
    release = asyncio.Event()
    calls = 0

    async def fetch(key: str) -> str:
        nonlocal calls
        calls += 1
        await release.wait()             # hold the fetch open until we say so
        return f"value for {key}"

    c = Coalescer()
    callers = [asyncio.create_task(c.get("k", fetch)) for _ in range(10)]
    await asyncio.sleep(0)               # let every caller reach its await (one loop pass)
    callers[0].cancel()
    release.set()
    results = await asyncio.gather(*callers, return_exceptions=True)

    assert calls == 1
    assert isinstance(results[0], asyncio.CancelledError)
    assert results[1:] == ["value for k"] * 9
```

One `sleep(0)` is acceptable here: it is not waiting for time to pass, only for the already-ready tasks to take one step, which is deterministic.

**14.3**

```python
# test_stream.py  (asyncio_mode = "auto")
import asyncio, socket
import httpx2, pytest, uvicorn
from app import app, get_llm

class ClosingFake:
    def __init__(self):
        self.produced = 0
        self.aclosed = asyncio.Event()
    async def stream(self, messages, system=None):
        try:
            for i in range(10_000):
                await asyncio.sleep(0.001)
                self.produced += 1
                yield f"tok{i} "
        finally:
            self.aclosed.set()

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

@pytest.fixture
async def live_server():
    port = free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())       # a real server on the test's event loop
    while not server.started:
        await asyncio.sleep(0.01)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    await task

async def test_disconnect_closes_upstream(live_server):
    fake = ClosingFake()
    app.dependency_overrides[get_llm] = lambda: fake
    try:
        async with httpx2.AsyncClient(base_url=live_server) as client:
            async with client.stream("POST", "/chat/stream", json={"message": "hi"}) as r:
                async for line in r.aiter_lines():
                    if line.startswith("data: "):
                        break                        # read one event, then hang up
        async with asyncio.timeout(2):
            await fake.aclosed.wait()                # the server stopped the generator...
        assert fake.produced < 10_000                # ...long before it would have finished
    finally:
        app.dependency_overrides.clear()
```

Why a real server? The first draft of this solution used `httpx2.ASGITransport`, and it passed, but for the wrong reason: that transport buffers the whole response, so the fake ran to completion and set `aclosed` normally; no disconnect was ever tested. A test that cannot fail is worse than no test. Against a real uvicorn server (verified with FastAPI 0.162.3), the first event arrives after about 0.04 s and the generator is closed one event after the client hangs up. `lifespan="off"` skips the app's startup code because the dependency override supplies the only state the endpoint needs; drop it if your test needs the lifespan. The whole test runs in well under a second.

**14.4** `all_tasks()` shows five `heartbeat` tasks still pending (plus `main`). Each request starts a heartbeat that loops forever and nobody stops it; when `asyncio.run` finishes, it cancels them, but in a long-running server they would accumulate forever, a memory and CPU leak. They also risk being garbage-collected because no reference is kept. Fix: tie the heartbeat's lifetime to the request with a TaskGroup and stop it when the work is done:

```python
async def handle_request(i: int):
    async with asyncio.TaskGroup() as tg:
        hb = tg.create_task(heartbeat())
        await asyncio.sleep(0.2)
        hb.cancel()                          # work finished: stop the heartbeat
    return i
```

Cancelling a child task with `cancel()` does not count as a failure of the group. On Python 3.15+ you could also call `tg.cancel()` once the work is done.

**14.5** Here is real output from Python 3.14.10 for a stripped-down agent whose `run_agent` task opens a TaskGroup with two slow tools named `tool-search` and `tool-db` (the Module 13 demo shows the same shape, with `gather` in place of the TaskGroup):

```text
$ python -m asyncio pstree 1234
└── (T) Task-1
    └──  main hung.py:9
        └── (T) agent-run-42
            └──  run_agent hung.py:5
                └──  TaskGroup.__aexit__ .../asyncio/taskgroups.py:72
                    └──  TaskGroup._aexit .../asyncio/taskgroups.py:121
                        ├── (T) tool-search
                        │   └──  slow_tool hung.py:3
                        │       └──  sleep .../asyncio/tasks.py:716
                        └── (T) tool-db
                            └──  slow_tool hung.py:3
                                └──  sleep .../asyncio/tasks.py:716
```

Each `(T)` line is a task and the lines under it are its await chain, innermost last. Read it top-down: `main` awaits the `agent-run-42` task, which waits in the TaskGroup's exit for its two tool tasks, each of which is parked in `asyncio.sleep`. In a hung production agent, the deepest line under each task tells you exactly which external call it is waiting on. `python -m asyncio ps <PID>` prints the same tasks as a flat table with an "awaiter" column. On Python 3.15, `python -m profiling.sampling dump --async-aware <PID>` prints equivalent per-task stacks. On Linux both tools must run as the same user as the target process, with permission to trace it (ptrace). The named tasks are why this output is readable; unnamed ones show as `Task-2`, `Task-3`, and so on.

## Module 15: Capstone project, an async research agent service

The capstone combines every module into one portfolio-ready service: a FastAPI app that takes a research question, plans subquestions, runs worker agents concurrently with MCP tools, streams progress to the client, and returns a synthesised answer within a deadline. Budget 10 to 12 hours. It runs fully offline with `FakeLLM`, and against a real model when `ANTHROPIC_API_KEY` is set.

### 15.1 Specification

**Endpoints**

| Endpoint | Behaviour |
| --- | --- |
| `GET /health` | Returns `{"status": "ok"}` |
| `POST /research` | Body `{"question": str}`. Returns `{"answer", "subquestions", "workers"}` where `workers` lists each subquestion's status (`ok`, `failed`) and stop reason |
| `POST /research/stream` | Same input. Streams SSE events: `plan`, `worker_started`, `tool_call`, `worker_done`, `answer`, `end` |

**Required behaviour**

1. A planner LLM call produces 2 to 5 subquestions.
2. One worker `Agent` (Module 13) per subquestion, at most 3 running at once, each with its own step limit, deadline and tool timeout.
3. Workers use tools discovered at startup from at least one MCP server through `MCPRouter` (Exercise 12.4).
4. One worker failing must not fail the request; the final answer is synthesised from the successful workers.
5. The whole run has an overall deadline (default 90 s). On expiry, `/research` returns HTTP 504 and the stream sends `end` with reason `deadline`.
6. A client disconnect on the stream cancels all in-flight work.
7. Every request gets a request ID (from an `X-Request-ID` header or a new UUID) that appears in every log line, via `contextvars`.
8. An audit record of each run is written in the background without delaying the response, and pending writes are drained at shutdown.
9. Shared resources (LLM client, MCP clients) are created once in the lifespan and closed at shutdown.
10. A pytest suite runs offline in under 10 seconds.

**Suggested layout**

```text
capstone/
  llm.py              # Module 10.6
  agent.py            # Module 13.1
  mcp_router.py       # Exercise 12.4
  background.py       # Exercise 4.5 (BackgroundRunner)
  research_server.py  # Module 12.2 MCP server
  orchestrator.py     # plan, workers, synthesis
  service.py          # FastAPI app
  tests/
    test_orchestrator.py
    test_service.py
  pyproject.toml
```

### 15.2 Milestones

1. **Orchestrator (3 h).** Write `orchestrator.py` and test it with `FakeLLM` and an in-memory MCP server. Prove with a test that 5 workers at 1 s each finish in about 2 s with a limit of 3.
2. **Service (2 h).** Lifespan, `/health`, `/research`, settings injected through a dependency.
3. **Streaming (2 h).** `/research/stream` with the event-forwarding pattern from Exercise 13.2, including the drain step and cancellation on disconnect.
4. **Operations (2 h).** Request IDs in logs, background audit log, graceful shutdown, HTTP 504 on deadline.
5. **Tests and polish (2 h).** Offline tests for success, partial failure, deadline and disconnect; a README with a sequence diagram and how to run it with a real model.
6. **Stretch goals.** Swap in `AnthropicLLM` and a second MCP server over Streamable HTTP; add a token budget per run; add an in-memory cache with request coalescing (Exercise 6.4) for repeated subquestions; add OpenTelemetry tracing; run two uvicorn workers and explain what state is per process.

### 15.3 Grading rubric

Score yourself out of 100, or swap with a study partner.

| Criterion | Points | Full marks means |
| --- | --- | --- |
| Correct concurrency | 20 | Workers overlap, limit of 3 respected, measured in a test |
| Timeouts and limits | 15 | Overall deadline, per-worker deadline, tool timeouts, step limits all present and tested |
| Cancellation safety | 15 | Disconnect cancels everything; no `CancelledError` swallowed; no orphan tasks at shutdown |
| Error handling | 10 | Partial failures reported, not fatal; tool errors returned to the model as text |
| Resource management | 10 | Clients created once in the lifespan, closed at shutdown; background work drained |
| Streaming | 10 | Correct SSE format; events not lost at the end of a run |
| Observability | 5 | Request IDs in all logs; named tasks |
| Tests | 10 | Offline, deterministic, under 10 s total |
| Code quality | 5 | Type hints, small functions, no blocking calls in `async def` |

### 15.4 Reference solution: orchestrator

Try the milestones yourself first. This is one good solution, not the only one.

```python
# orchestrator.py
import asyncio, logging
from dataclasses import dataclass, field

from agent import Agent, AgentConfig
from llm import AssistantTurn, FakeLLM

log = logging.getLogger("orchestrator")

PLAN_PROMPT = "PLAN: List 2 to 5 short research subquestions, one per line, for: {q}"
WORKER_PROMPT = "You are a research worker. Use the tools, then answer concisely with facts."
SYNTH_PROMPT = "SYNTHESISE: Answer the question {q!r} using these findings:\n{findings}"

@dataclass
class Settings:
    max_parallel_workers: int = 3
    run_deadline_s: float = 90.0
    worker: AgentConfig = field(default_factory=lambda: AgentConfig(max_steps=6, deadline_s=45, tool_timeout_s=15))

def mcp_tool_fns(router) -> dict:
    """Adapt MCPRouter tools to the Agent's {name: async fn(**kwargs)} format."""
    def make(name: str):
        async def call(**kwargs):
            result = await router.call(name, kwargs)
            text = "\n".join(c.text for c in result.content if c.type == "text")
            if result.is_error:
                raise RuntimeError(text)
            return text
        return call
    return {t["name"]: make(t["name"]) for t in router.tools}

async def emit(events: asyncio.Queue | None, **event) -> None:
    if events is not None:
        await events.put(event)

async def plan(llm, question: str) -> list[str]:
    turn = await llm.complete([{"role": "user", "content": PLAN_PROMPT.format(q=question)}])
    lines = [line.strip("-*0123456789. ").strip() for line in turn.text.splitlines()]
    subqs = [line for line in lines if line]
    return subqs[:5] or [question]

async def run_research(question: str, llm, router, settings: Settings,
                       events: asyncio.Queue | None = None) -> dict:
    subqs = await plan(llm, question)
    await emit(events, type="plan", subquestions=subqs)
    tools = mcp_tool_fns(router)
    sem = asyncio.Semaphore(settings.max_parallel_workers)

    async def worker(i: int, sq: str):
        async with sem:
            await emit(events, type="worker_started", i=i, question=sq)
            agent = Agent(llm, tools, router.tools, system=WORKER_PROMPT,
                          config=settings.worker, events=events)
            result = await agent.run(sq)
            await emit(events, type="worker_done", i=i, stopped_because=result.stopped_because)
            return result

    # If this coroutine is cancelled (deadline, disconnect), gather cancels every worker.
    outcomes = await asyncio.gather(*(worker(i, sq) for i, sq in enumerate(subqs)), return_exceptions=True)
    workers, findings = [], []
    for sq, out in zip(subqs, outcomes):
        if isinstance(out, asyncio.CancelledError):
            raise out                              # never report cancellation as a worker failure
        if isinstance(out, BaseException):
            log.warning("worker failed for %r: %r", sq, out)
            workers.append({"question": sq, "status": "failed", "error": repr(out)})
        else:
            workers.append({"question": sq, "status": "ok", "stopped_because": out.stopped_because})
            findings.append(f"- {sq}: {out.answer}")
    synth = await llm.complete([{"role": "user", "content":
                                 SYNTH_PROMPT.format(q=question, findings="\n".join(findings) or "(none)")}])
    await emit(events, type="answer", text=synth.text)
    return {"answer": synth.text, "subquestions": subqs, "workers": workers}

def fake_script(messages: list[dict]) -> AssistantTurn:
    """Offline model: plans, makes one tool call per worker, then answers."""
    first = messages[0]["content"]
    if first.startswith("PLAN:"):
        return FakeLLM.text_turn("asyncio basics\nTaskGroup semantics\nMCP overview")
    if first.startswith("SYNTHESISE:"):
        return FakeLLM.text_turn("Synthesised answer from " + str(first.count("\n- ")) + " findings.")
    if len(messages) == 1:                    # worker, first turn: call the search tool
        return FakeLLM.tool_turn(("research__search_notes", {"query": first.split()[0]}))
    return FakeLLM.text_turn(f"Findings for {first!r}: " + str(messages[-1]["content"][0]["content"])[:80])
```

### 15.5 Reference solution: service

```python
# service.py
import asyncio, contextvars, json, logging, os, uuid
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Annotated, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import research_server
from background import BackgroundRunner
from llm import AnthropicLLM, FakeLLM
from mcp_router import MCPRouter
from orchestrator import Settings, fake_script, run_research

request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id.get()
        return True

handler = logging.StreamHandler()
handler.addFilter(RequestIdFilter())
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(request_id)s %(name)s %(message)s",
                    handlers=[handler])
log = logging.getLogger("service")

class RequestIdMiddleware:
    """Pure ASGI middleware: the whole request, including a streamed body, runs inside
    __call__, so the context variable stays set until the last byte is sent."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        incoming = dict(scope["headers"]).get(b"x-request-id", b"").decode()
        rid = incoming or uuid.uuid4().hex[:12]
        token = request_id.set(rid)

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                message["headers"] = [*message.get("headers", []), (b"x-request-id", rid.encode())]
            await send(message)
        try:
            await self.app(scope, receive, send_with_id)
        finally:
            request_id.reset(token)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with AsyncExitStack() as stack:
        if os.environ.get("ANTHROPIC_API_KEY"):
            app.state.llm = AnthropicLLM(max_concurrency=8)
        else:
            app.state.llm = FakeLLM(fake_script, latency=0.2)
        app.state.router = await stack.enter_async_context(MCPRouter({"research": research_server.mcp}))
        app.state.settings = Settings()
        app.state.background = BackgroundRunner()
        try:
            yield
        finally:
            await app.state.background.drain(timeout=5)      # finish audit writes

app = FastAPI(lifespan=lifespan)
app.add_middleware(RequestIdMiddleware)

def get_deps(request: Request):
    s = request.app.state
    return s.llm, s.router, s.settings, s.background

Deps = Annotated[tuple, Depends(get_deps)]

class ResearchRequest(BaseModel):
    question: str

AUDIT_PATH = os.environ.get("AUDIT_LOG", "audit.log")

def write_audit_sync(record: dict) -> None:
    with open(AUDIT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

async def write_audit(record: dict) -> None:
    await asyncio.to_thread(write_audit_sync, record)         # file I/O off the loop

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/research")
async def research(body: ResearchRequest, deps: Deps):
    llm, router, settings, background = deps
    log.info("research started: %s", body.question)
    try:
        async with asyncio.timeout(settings.run_deadline_s):
            result = await run_research(body.question, llm, router, settings)
    except TimeoutError:
        background.spawn(write_audit({"id": request_id.get(), "question": body.question, "status": "deadline"}))
        raise HTTPException(status_code=504, detail="research deadline exceeded")
    background.spawn(write_audit({"id": request_id.get(), "question": body.question, "status": "ok"}))
    log.info("research finished")
    return result

def sse(data: dict) -> str:
    return f"data: {json.dumps(data, default=str)}\n\n"

@app.post("/research/stream")
async def research_stream(body: ResearchRequest, deps: Deps):
    llm, router, settings, background = deps
    events: asyncio.Queue = asyncio.Queue()

    async def run() -> dict:
        async with asyncio.timeout(settings.run_deadline_s):
            return await run_research(body.question, llm, router, settings, events)

    async def stream():
        task = asyncio.create_task(run(), name=f"research-{request_id.get()}")
        try:
            while True:
                if task.done():
                    while not events.empty():
                        yield sse(events.get_nowait())
                    try:
                        task.result()
                        reason = "ok"
                    except TimeoutError:
                        reason = "deadline"
                    yield sse({"type": "end", "reason": reason})
                    break
                getter = asyncio.create_task(events.get())
                await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)
                if getter.done():
                    yield sse(getter.result())
                else:
                    getter.cancel()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            log.info("stream closed")

    return StreamingResponse(stream(), media_type="text/event-stream")
```

Why plain ASGI middleware rather than `@app.middleware("http")`? The decorator builds Starlette's `BaseHTTPMiddleware`, which runs your endpoint in a separate task and relays the response body through a stream. That design has known pitfalls with streaming responses and context variables. A plain ASGI middleware is a few more lines, but the whole request, including every byte of a streamed body, runs inside its `__call__`, so the request ID set there is visible everywhere and is reset only when the response is complete. The background audit task also gets the ID, because `create_task` copies the current context.

### 15.6 Reference solution: tests

```python
# tests/test_orchestrator.py   (pytest.ini: asyncio_mode = auto)
import time
import research_server
from agent import AgentConfig
from llm import FakeLLM
from mcp_router import MCPRouter
from orchestrator import Settings, fake_script, run_research

FAST = Settings(max_parallel_workers=3, run_deadline_s=5,
                worker=AgentConfig(max_steps=4, deadline_s=2, tool_timeout_s=1))

class CountingLLM(FakeLLM):
    """A FakeLLM that records the peak number of concurrent calls."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_flight = self.peak = 0
    async def complete(self, messages, tools=None, system=None, max_tokens=1024):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            return await super().complete(messages, tools, system, max_tokens)
        finally:
            self.in_flight -= 1

async def test_happy_path():
    async with MCPRouter({"research": research_server.mcp}) as router:
        result = await run_research("asyncio", FakeLLM(fake_script, latency=0.01), router, FAST)
    assert len(result["subquestions"]) == 3
    assert all(w["status"] == "ok" for w in result["workers"])
    assert result["answer"] == "Synthesised answer from 3 findings."

async def test_partial_failure_is_not_fatal():
    def flaky(messages):
        if messages[0]["content"] == "TaskGroup semantics":
            raise RuntimeError("provider overloaded")
        return fake_script(messages)
    async with MCPRouter({"research": research_server.mcp}) as router:
        result = await run_research("asyncio", FakeLLM(flaky, latency=0.01), router, FAST)
    assert sorted(w["status"] for w in result["workers"]) == ["failed", "ok", "ok"]

async def test_workers_run_concurrently_with_limit():
    llm = CountingLLM(fake_script, latency=0.2)
    async with MCPRouter({"research": research_server.mcp}) as router:
        start = time.perf_counter()
        await run_research("asyncio", llm, router, FAST)
        elapsed = time.perf_counter() - start
    assert llm.peak == 3                   # three workers' model calls overlapped
    assert elapsed < 2.0                   # measured 1.11 s; with a limit of 1 it takes 2.51 s

async def test_limit_of_one_serialises_workers():
    llm = CountingLLM(fake_script, latency=0.01)
    one = Settings(max_parallel_workers=1, run_deadline_s=5, worker=FAST.worker)
    async with MCPRouter({"research": research_server.mcp}) as router:
        await run_research("asyncio", llm, router, one)
    assert llm.peak == 1
```

```python
# tests/test_service.py
import asyncio, json, socket
import httpx2, pytest, uvicorn
from fastapi.testclient import TestClient
import service
from orchestrator import Settings

@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(service, "AUDIT_PATH", str(tmp_path / "audit.log"))
    return tmp_path / "audit.log"

def test_research_endpoint_and_request_id(offline):
    with TestClient(service.app) as client:            # "with" runs the lifespan
        r = client.post("/research", json={"question": "asyncio"}, headers={"x-request-id": "test-1"})
    assert r.status_code == 200
    assert r.headers["x-request-id"] == "test-1"
    assert r.json()["answer"].startswith("Synthesised")
    audit = [json.loads(line) for line in offline.read_text().splitlines()]
    assert audit == [{"id": "test-1", "question": "asyncio", "status": "ok"}]   # drained at shutdown

def test_generated_request_id():
    with TestClient(service.app) as client:
        r = client.get("/health")
    assert len(r.headers["x-request-id"]) == 12

def test_deadline_returns_504():
    with TestClient(service.app) as client:
        service.app.state.settings = Settings(run_deadline_s=0.05)
        r = client.post("/research", json={"question": "asyncio"})
    assert r.status_code == 504

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

@pytest.fixture
async def live_server():                                 # streaming needs a real server (Module 11.4)
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(service.app, host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    await task

async def read_events(client, limit=None) -> list[dict]:
    events = []
    async with client.stream("POST", "/research/stream", json={"question": "asyncio"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
                if limit and len(events) >= limit:
                    break
    return events

async def test_stream_full_run(live_server):
    async with httpx2.AsyncClient(base_url=live_server, timeout=10) as client:
        events = await read_events(client)
    types = [e["type"] for e in events]
    assert types[0] == "plan" and types[-2:] == ["answer", "end"]
    assert events[-1]["reason"] == "ok"
    assert types.count("worker_done") == 3

async def test_stream_disconnect_cancels_work(live_server):
    async with httpx2.AsyncClient(base_url=live_server, timeout=10) as client:
        events = await read_events(client, limit=1)       # hang up after the plan event
    assert events[0]["type"] == "plan"
    await asyncio.sleep(0.5)                               # a full run takes about 1.1 s
    leftover = [t for t in asyncio.all_tasks() if t.get_name().startswith("research-")]
    assert leftover == []                                  # the research task was cancelled
```

Result on Python 3.12.3 and 3.14.10: `9 passed` in about 7.5 seconds. The live-server tests share the test's event loop with uvicorn, which is why `asyncio.all_tasks()` can see the server's research tasks. To finish the rubric, add tests for a worker that hits its own deadline and for a slow MCP tool that triggers `tool_timeout_s`.

A quick load test shows what asyncio buys you here. With a small async load generator (Exercise 20.3) against the service running in uvicorn with the fake LLM, 40 requests gave:

| Concurrent clients | Throughput | Median latency | 95th percentile |
| --- | --- | --- | --- |
| 1 | 0.9 requests/s | 1.11 s | 1.11 s |
| 10 | 8.9 requests/s | 1.12 s | 1.17 s |
| 40 | 33.9 requests/s | 1.16 s | 1.17 s |

Throughput grows almost linearly with concurrency while latency stays flat, because one event loop overlaps all the waiting. With a real model, provider rate limits and the semaphore in `AnthropicLLM` would cap this curve.

### 15.7 Presenting it in interviews

Put the project on GitHub with a README covering the architecture (a diagram of planner, workers, MCP router and SSE stream), the concurrency limits and why you chose them, the failure modes you handled (partial failures, deadlines, disconnects, slow tools), and the test strategy. Interviewers for agentic AI roles consistently probe exactly these topics, and a project that demonstrates them is stronger evidence than any certificate.

## Module 16: Subprocesses and code-execution tools

Agents constantly run other programs: a code interpreter, `git`, a linter, a test suite, a headless browser. `asyncio.create_subprocess_exec` runs them without blocking the loop, but the hard part is cleanup. A cancelled or timed-out task does **not** stop the child process unless you kill it yourself, and killing the child does not kill *its* children unless you kill the whole process group. Everything in this module was run on Python 3.12.3 on Linux.

### 16.1 Running a command

```python
import asyncio

async def main():
    proc = await asyncio.create_subprocess_exec(
        "python3", "-c", "print('hi')",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate()          # waits for exit, reads both pipes fully
    print(proc.returncode, out.decode())         # 0 hi

asyncio.run(main())
```

Pass the program and its arguments as separate strings. Never build a shell command from model output with `asyncio.create_subprocess_shell` or `sh -c`: a model, or a document it read, can inject `; rm -rf ~`. Use `communicate()` rather than `wait()` when you pipe output, because a child that fills the pipe buffer blocks until someone reads it, and `wait()` alone never reads.

Five one-second commands run concurrently take about 1 second in total, exactly like coroutines (measured: 1.01 s).

### 16.2 Timeouts, cancellation and process groups

Two measured facts drive the design:

- A naive coroutine that awaits `proc.wait()` and is then cancelled leaves the child **still running** (measured: the `sleep` process survived its cancelled task).
- `proc.kill()` kills only the direct child. A shell that started two background `sleep` commands was killed, and both `sleep` processes **kept running** (measured: 2 orphans).

The fix is to start the child in its own session (and therefore its own process group) and kill the whole group on *any* exit path, including cancellation:

```python
import asyncio, os, signal

async def run(cmd: list[str], timeout: float = 10.0) -> tuple[int, str, str]:
    """Run a command without a shell, with a timeout; always kill and reap the process group."""
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        start_new_session=True)                    # own session and process group (POSIX)
    try:
        async with asyncio.timeout(timeout):
            out, err = await proc.communicate()
    except BaseException:                          # timeout OR cancellation of the caller
        try:
            os.killpg(proc.pid, signal.SIGKILL)    # kill the child and all its descendants
        except ProcessLookupError:
            pass                                   # it already exited
        await proc.wait()                          # reap the child: no zombie left behind
        raise
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")
```

Catching `BaseException` here is correct because the handler re-raises: it exists only to guarantee cleanup, whether the cause was a `TimeoutError` or a `CancelledError`. With this helper, both measured leaks disappear.

**Containers and zombies.** When a process's parent dies, the orphan is re-parented to PID 1, which is expected to reap it. In a container whose PID 1 is your Python app, nothing reaps orphans, and killed grandchildren linger as `<defunct>` zombie entries (observed while testing this module). Run containers with an init process: `docker run --init`, or `tini` as the entrypoint, or the equivalent setting in your orchestrator.

**Platforms.** `start_new_session` and `os.killpg` are POSIX-only. On Windows, the default event loop (Proactor) supports subprocesses; use `creationflags=subprocess.CREATE_NEW_PROCESS_GROUP` and a job object, or a library such as `psutil` to walk and kill the process tree. Python 3.12 deprecated asyncio's "child watcher" classes and 3.14 removed them; with `asyncio.run` you never need them.

### 16.3 Streaming output

For long-running commands, read output as it arrives instead of waiting for the end, so an agent UI can show progress:

```python
proc = await asyncio.create_subprocess_exec(
    "sh", "-c", "for i in 1 2 3; do echo line$i; sleep 0.1; done",
    stdout=asyncio.subprocess.PIPE)
async for line in proc.stdout:          # a StreamReader is an async iterator of lines
    print("got", line.decode().strip())
await proc.wait()                       # safe here: we have already read stdout to the end
```

### 16.4 A code-execution tool, and why it is not a sandbox

```python
import os, sys, tempfile

async def run_python(code: str, timeout: float = 5.0, max_output: int = 2000) -> dict:
    """Agent tool: run Python in a separate process. Isolates crashes and hangs, NOT attacks."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "snippet.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            rc, out, err = await run([sys.executable, "-I", path], timeout=timeout)  # -I: isolated mode
        except TimeoutError:
            return {"ok": False, "error": f"timed out after {timeout}s"}
    return {"ok": rc == 0, "exit_code": rc, "stdout": out[:max_output], "stderr": err[-max_output:]}
```

Measured: `print(sum(range(10)))` returns `45`, and `while True: pass` is killed after the timeout while the event loop stays responsive, which is exactly the protection the in-process `eval` of Module 13 lacked. But the child runs as your user, with your files, network and environment. Real isolation for model-written code needs a container with no network and resource limits, a gVisor or Firecracker microVM, or a hosted sandbox service. Use the subprocess as the *mechanism*, and the sandbox as the *boundary*.

### Check yourself

1. A task awaiting `proc.wait()` is cancelled. Is the child process stopped? *Answer: no; you must kill it yourself in the cancellation path.*
2. Why kill the process group rather than the process? *Answer: killing a process does not kill its children; they keep running as orphans.*
3. Why not `create_subprocess_shell(model_output)`? *Answer: shell metacharacters in the output become extra commands (command injection).*

### Exercises

**16.1 (E)** Run `python3 --version` and `git --version` concurrently and print both outputs.

**16.2 (M)** Write `run_many(cmds, limit, timeout)` that runs a list of commands with at most `limit` at once and a per-command timeout, returning one result per command with `rc` or `"timeout"`. Test it with four `sleep 1`, one `echo hi` and one `sleep 10`, `limit=4`, `timeout=1.5`. Predict the total time.

**16.3 (M)** Reproduce the leak: start `sleep 31` with `create_subprocess_exec`, await `proc.wait()` in a task, cancel the task, and count surviving `sleep` processes with `pgrep -x sleep`. Then use `run` from 16.2 and count again.

**16.4 (H)** Give the Module 13 `Agent` a `run_python` tool. Script a `FakeLLM` that requests two calls in one turn, `while True: pass` and `print(2 + 2)`, and then answers with both results. Confirm the agent finishes with `final_answer` and that the first result is a timeout error.

### Solutions

**16.1**

```python
async def main():
    results = await asyncio.gather(run(["python3", "--version"]), run(["git", "--version"]))
    for rc, out, err in results:
        print(rc, (out or err).strip())
```

**16.2**

```python
async def run_many(cmds: list[list[str]], limit: int = 4, timeout: float = 5.0) -> list[dict]:
    sem = asyncio.Semaphore(limit)
    async def one(cmd):
        async with sem:
            try:
                rc, out, err = await run(cmd, timeout=timeout)
                return {"cmd": cmd, "rc": rc, "out": out.strip()}
            except TimeoutError:
                return {"cmd": cmd, "rc": None, "error": "timeout"}
    return await asyncio.gather(*(one(c) for c in cmds))
```

Measured 2.51 s: the four one-second sleeps fill the four slots for 1 s, then `echo` and `sleep 10` run, and `sleep 10` is killed at 1.5 s. Result codes: `[0, 0, 0, 0, 0, None]`.

**16.3** The naive version leaves 1 `sleep 31` running after the cancel; with `run`, 0 remain. Remember to clean up after the experiment with `pkill -x sleep`.

**16.4**

```python
from agent import Agent, AgentConfig
from llm import FakeLLM

SPEC = [{"name": "run_python", "description": "Run Python code and return its output.",
         "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}]

def script(messages):
    if len(messages) == 1:
        return FakeLLM.tool_turn(("run_python", {"code": "while True: pass"}),
                                 ("run_python", {"code": "print(2 + 2)"}))
    return FakeLLM.text_turn(" | ".join(r["content"] for r in messages[-1]["content"]))

async def run_python_tool(code: str) -> dict:
    return await run_python(code, timeout=1.0)

async def main():
    agent = Agent(FakeLLM(script, latency=0.05), {"run_python": run_python_tool}, SPEC,
                  config=AgentConfig(tool_timeout_s=5))
    result = await agent.run("compute")
    print(result.stopped_because, result.answer)
```

Measured output: `final_answer`, followed by `{"ok": false, "error": "timed out after 1.0s"} | {"ok": true, "exit_code": 0, "stdout": "4\n", "stderr": ""}`. The tool's own 1-second limit is shorter than the agent's 5-second tool timeout, so the tool reports a clean error instead of being cancelled. Set inner limits shorter than outer ones.

## Module 17: AnyIO, trio and cancel scopes

AnyIO is a library that offers structured concurrency (task groups and cancel scopes) on top of either asyncio or trio. You may never import it, but it runs underneath much of your stack: FastAPI and Starlette, httpx2, the MCP SDK and both the Anthropic and OpenAI SDKs depend on it (anyio 4.16.0 was installed alongside them while testing this course). Its cancellation rules differ from asyncio's in two ways that cause real bugs, including one in an early draft of Module 12. All examples ran with `asyncio.run`, showing that AnyIO code runs on the asyncio backend.

### 17.1 Task groups and timeouts in AnyIO

```python
import anyio

async def worker(n: int, results: list) -> None:
    await anyio.sleep(0.1 * n)
    results.append(n)

async def main():
    results: list[int] = []
    async with anyio.create_task_group() as tg:
        for n in (3, 1, 2):
            tg.start_soon(worker, n, results)     # returns nothing: collect results yourself
    print(results)                                # [1, 2, 3]

    with anyio.move_on_after(0.05) as scope:      # a timeout that does not raise
        await anyio.sleep(1)
    print(scope.cancelled_caught)                 # True

    try:
        with anyio.fail_after(0.05):              # a timeout that raises TimeoutError on exit
            await anyio.sleep(1)
    except TimeoutError:
        print("timed out")
```

| Concept | asyncio | AnyIO |
| --- | --- | --- |
| Structured task group | `asyncio.TaskGroup`, `tg.create_task(coro)` returns a Task | `anyio.create_task_group()`, `tg.start_soon(fn, *args)` returns nothing |
| Timeout that raises | `async with asyncio.timeout(s)` | `with anyio.fail_after(s)` |
| Timeout that does not raise | catch `TimeoutError` | `with anyio.move_on_after(s)`; check `scope.cancelled_caught` |
| Cancel a scope from inside | `tg.cancel()` (3.15+) | `tg.cancel_scope.cancel()` |
| Cancellation exception | `asyncio.CancelledError` | `anyio.get_cancelled_exc_class()` (asyncio's on the asyncio backend) |
| Run blocking code | `asyncio.to_thread(fn)` | `await anyio.to_thread.run_sync(fn)` |

### 17.2 Edge-triggered versus level-triggered cancellation

In asyncio, `task.cancel()` delivers one `CancelledError`. If the code swallows it, the next `await` succeeds and the task keeps running (*edge-triggered*). In AnyIO, a cancelled scope stays cancelled: every `await` inside it raises again until the code leaves the scope (*level-triggered*). Measured:

```python
import asyncio, anyio

async def swallow_asyncio():
    try:
        await asyncio.sleep(10)
    except asyncio.CancelledError:
        pass                                  # bad practice, for the demo
    await asyncio.sleep(0.1)                  # succeeds
    return "asyncio: kept running after swallowing"

async def swallow_anyio():
    with anyio.move_on_after(0.05):
        try:
            await anyio.sleep(10)
        except anyio.get_cancelled_exc_class():
            pass                              # swallowed...
        try:
            await anyio.sleep(0.1)            # ...but the scope is still cancelled
        except anyio.get_cancelled_exc_class():
            print("anyio: second await was cancelled too")
            raise

async def main():
    t = asyncio.create_task(swallow_asyncio())
    await asyncio.sleep(0.05)
    t.cancel()
    print(await t)                            # asyncio: kept running after swallowing
    await swallow_anyio()                     # anyio: second await was cancelled too

asyncio.run(main())
```

Level-triggered cancellation is more forgiving of sloppy code, which is why trio and AnyIO chose it. In asyncio code, the rule from Module 5 stands: always re-raise `CancelledError`.

### 17.3 The same-task rule

An AnyIO cancel scope, and therefore any task group or any library object built on one (an MCP `Client`, an httpx2 streaming response, Starlette internals), must be exited by the same task that entered it. Measured: entering a task group in one task and exiting it from another raises

```text
RuntimeError: Attempted to exit cancel scope in a different task than it was entered in
```

That is why Exercise 12.4 connects to MCP servers sequentially in one task. When you do need concurrent setup, give each resource an *owner task* that enters the context, signals readiness, waits for a stop signal and exits the context itself. Other tasks may call the resource's methods freely; only entering and exiting are tied to the owner. Exercise 17.3 builds this.

### 17.4 Where trio fits

trio is an alternative async runtime built around structured concurrency from the start; its design inspired asyncio's TaskGroup and AnyIO itself. Few agent libraries require it, and AnyIO lets libraries support both. Learn asyncio first, recognise AnyIO idioms when you read library code, and treat trio as an option for projects that choose it.

### Check yourself

1. Why does AnyIO matter if you only write asyncio code? *Answer: FastAPI, httpx2, the MCP SDK and the LLM SDKs run on it, so its cancellation rules apply inside them.*
2. What does level-triggered cancellation mean? *Answer: a cancelled scope keeps cancelling every await inside it until the code leaves the scope.*
3. What must be true of the task that exits an AnyIO cancel scope? *Answer: it must be the same task that entered it.*

### Exercises

**17.1 (E)** Port Exercise 4.1 (five `fake_llm` calls in a TaskGroup) to AnyIO, keeping the results in the original order even though `start_soon` returns nothing.

**17.2 (M)** Without running it, predict what `swallow_asyncio` and `swallow_anyio` print and why. Then run them.

**17.3 (H)** Write `OwnedClient`, which runs one MCP `Client` inside its own task, and `connect_all(targets)`, which connects to several servers concurrently, skips any that fail, and closes everything cleanly. Test it with two in-memory servers and one bad URL.

### Solutions

**17.1**

```python
import anyio

async def fake_llm(prompt: str, delay: float) -> str:
    await anyio.sleep(delay)
    return f"answer to {prompt!r}"

async def main():
    delays = [0.5, 1.0, 0.2, 0.8, 0.3]
    results: list[str | None] = [None] * len(delays)
    async def one(i: int, d: float):
        results[i] = await fake_llm(f"q{i}", d)     # write into the slot for this index
    async with anyio.create_task_group() as tg:
        for i, d in enumerate(delays):
            tg.start_soon(one, i, d)
    print(results)                                   # in original order, after about 1.0 s

anyio.run(main)          # or asyncio.run(main()): both work
```

**17.2** `swallow_asyncio` returns normally: asyncio delivered one cancellation and the swallowed exception left the task running. `swallow_anyio` prints that the second await was cancelled too, because the scope remains cancelled. The `move_on_after` scope then absorbs the cancellation, and `main` continues.

**17.3**

```python
import asyncio
from mcp import Client

class OwnedClient:
    """Runs one MCP Client inside its own task, so the client's context manager is
    entered and exited by the same task, while other tasks call its methods freely."""
    def __init__(self, target, connect_timeout: float = 10.0):
        self.target, self.connect_timeout = target, connect_timeout
        self.client: Client | None = None
        self._ready, self._stop = asyncio.Event(), asyncio.Event()
        self._error: BaseException | None = None
        self._task: asyncio.Task | None = None

    async def _own(self) -> None:
        try:
            async with Client(self.target) as client:     # entered in THIS task...
                self.client = client
                self._ready.set()
                await self._stop.wait()
        except Exception as exc:
            self._error = exc
        finally:
            self._ready.set()                              # ...and exited in this task too

    async def start(self) -> "OwnedClient":
        self._task = asyncio.create_task(self._own())
        async with asyncio.timeout(self.connect_timeout):
            await self._ready.wait()
        if self._error is not None:
            raise self._error
        return self

    async def close(self) -> None:
        self._stop.set()
        if self._task:
            await self._task

async def connect_all(targets: dict) -> dict[str, OwnedClient]:
    owners = {name: OwnedClient(t, connect_timeout=5) for name, t in targets.items()}
    results = await asyncio.gather(*(o.start() for o in owners.values()), return_exceptions=True)
    connected = {}
    for (name, owner), res in zip(owners.items(), results):
        if isinstance(res, BaseException):
            print(f"skipping {name}: {type(res).__name__}")
            await owner.close()
        else:
            connected[name] = owner
    return connected
```

Measured with the `research_server` from Module 12, a one-tool calculator server and `http://127.0.0.1:1/mcp`: the broken server is skipped (it fails with an `ExceptionGroup` from AnyIO's task group), the two others connect concurrently in 0.04 s, concurrent `call_tool` calls on both return `{'result': 3.0}` and `{'result': 3}`, and `await asyncio.gather(*(o.close() for o in connected.values()))` shuts down without errors.

## Module 18: Graceful shutdown and WebSockets

Production processes are stopped all the time: deployments, autoscaling, Ctrl+C. Whether in-flight agent work is cleaned up depends on which signal arrives and whether you handle it. And when users need to interrupt an agent mid-answer, or the app streams audio both ways, server-sent events are not enough and you need WebSockets. Everything here was run on Python 3.12.3 on Linux with FastAPI 0.162.3 and uvicorn 0.54.0.

### 18.1 What each signal does to `asyncio.run`

| Signal | Typical sender | Default behaviour (measured) |
| --- | --- | --- |
| SIGINT | Ctrl+C in a terminal | `asyncio.run` cancels the main task, `finally` blocks run, then `KeyboardInterrupt` reaches the top |
| SIGTERM | Docker, Kubernetes, systemd, `kill` | **No cleanup at all**: the process dies immediately with exit code −15 |
| SIGKILL | The orchestrator after the grace period | Cannot be caught; nothing runs |

The SIGTERM row surprises people: the signal every orchestrator sends first is the one Python does not turn into cleanup. Handle it explicitly:

```python
import asyncio, signal

async def worker(name: str) -> None:
    try:
        while True:
            await asyncio.sleep(0.1)          # real work: consume a queue, poll a source...
    finally:
        print(f"{name}: cleanup ran", flush=True)

async def main() -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)   # Unix only
    async with asyncio.TaskGroup() as tg:
        workers = [tg.create_task(worker(f"w{i}")) for i in range(2)]
        await stop.wait()
        print("signal received: shutting down", flush=True)
        for w in workers:
            w.cancel()                           # each worker's finally runs
    print("clean exit", flush=True)

asyncio.run(main())
```

Measured for both SIGTERM and SIGINT: `signal received: shutting down`, both cleanup lines, `clean exit`, exit code 0. On Windows, `add_signal_handler` raises `NotImplementedError`; rely on `KeyboardInterrupt` there.

### 18.2 Servers and containers

- **uvicorn handles SIGINT and SIGTERM for you.** It stops accepting connections, waits for in-flight requests, then runs your lifespan shutdown code. `--timeout-graceful-shutdown N` caps that wait. Do not add your own signal handlers inside a FastAPI app; put cleanup in the lifespan.
- **Know your grace period.** Kubernetes sends SIGTERM, waits `terminationGracePeriodSeconds` (30 s by default), then sends SIGKILL. Long agent runs that cannot finish in that window need checkpointing (Module 20.6) rather than a longer timeout.
- **Signals must reach Python.** In a Dockerfile, use the exec form `CMD ["uvicorn", "app:app"]`. The shell form `CMD uvicorn app:app` runs `/bin/sh -c`, which becomes PID 1 and may not forward SIGTERM to your process.
- **Reap orphans.** Run containers with an init process (`docker run --init` or `tini`), for the zombie reason explained in Module 16.2.

### 18.3 WebSockets for interruptible agents

SSE (Module 11) is one-way: server to client. A WebSocket is a single connection both sides can write to at any time, so the user can send "stop" or a new prompt while tokens are still streaming. The key design is that the **reader loop never waits for generation**: generation runs as its own task, which the reader can cancel.

```python
# ws_app.py
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

async def fake_tokens(prompt: str):
    for i in range(50):
        await asyncio.sleep(0.02)
        yield f"tok{i} "

@app.websocket("/ws/chat")
async def chat(ws: WebSocket):
    await ws.accept()
    generation: asyncio.Task | None = None

    async def generate(prompt: str):
        try:
            async for tok in fake_tokens(prompt):
                await ws.send_json({"type": "delta", "text": tok})
            await ws.send_json({"type": "done"})
        except asyncio.CancelledError:
            await ws.send_json({"type": "stopped"})
            raise

    try:
        while True:                                    # the reader: always listening
            msg = await ws.receive_json()
            if msg["type"] == "prompt":
                if generation and not generation.done():
                    generation.cancel()                # a new prompt replaces the old answer
                generation = asyncio.create_task(generate(msg["text"]))
            elif msg["type"] == "stop" and generation:
                generation.cancel()
    except WebSocketDisconnect:
        pass                                           # the client went away
    finally:
        if generation:
            generation.cancel()
            await asyncio.gather(generation, return_exceptions=True)
```

Test it with FastAPI's `TestClient`, whose `websocket_connect` works without a running server:

```python
# test_ws.py
from fastapi.testclient import TestClient
from ws_app import app

def test_stop_mid_stream():
    with TestClient(app) as client, client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "prompt", "text": "hi"})
        got = [ws.receive_json() for _ in range(3)]
        ws.send_json({"type": "stop"})
        while (m := ws.receive_json())["type"] == "delta":
            got.append(m)
        assert m["type"] == "stopped"
        assert 3 <= len(got) < 50

def test_full_answer():
    with TestClient(app) as client, client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "prompt", "text": "hi"})
        msgs = []
        while (m := ws.receive_json())["type"] != "done":
            msgs.append(m)
        assert len(msgs) == 50
```

Both tests pass (measured: 2 passed in 1.22 s). WebSockets also carry realtime voice agents, where audio frames flow in both directions concurrently; the same reader-plus-task structure applies.

### Check yourself

1. What happens to `finally` blocks when an asyncio program receives SIGTERM with no handler? *Answer: they do not run; the process is killed immediately.*
2. Why must the WebSocket reader not await the generation? *Answer: while it waits, it cannot receive the user's "stop" message.*
3. Where does cleanup belong in a FastAPI app? *Answer: in the lifespan's shutdown code, which uvicorn runs on SIGTERM and SIGINT.*

### Exercises

**18.1 (M)** Add SIGTERM handling to the worker pool of Module 6.2: on the signal, stop accepting new URLs, let in-flight fetches finish (up to 2 seconds), then cancel the rest and exit 0. Test it by sending the signal from a Python `subprocess.Popen`.

**18.2 (M)** Extend the WebSocket endpoint so a new `prompt` arriving mid-answer sends `stopped` for the old answer before the new deltas begin, and write a test proving it.

**18.3 (E)** A teammate's Dockerfile ends with `CMD python main.py`. Pods take the full 30-second grace period to stop and never log shutdown messages. Explain why and fix it.

### Solutions

**18.1**

```python
async def main():
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, stop.set)
    q: asyncio.Queue[str] = asyncio.Queue()
    for u in urls:
        q.put_nowait(u)
    results: dict[str, str] = {}
    workers = [asyncio.create_task(worker(f"w{i}", q, results)) for i in range(4)]
    done_waiting = asyncio.create_task(q.join())
    stop_waiting = asyncio.create_task(stop.wait())
    await asyncio.wait({done_waiting, stop_waiting}, return_when=asyncio.FIRST_COMPLETED)
    if stop.is_set():
        while not q.empty():                     # stop accepting new work
            q.get_nowait()
            q.task_done()
        try:
            async with asyncio.timeout(2):       # let in-flight fetches finish
                await q.join()
        except TimeoutError:
            pass
    for t in (*workers, done_waiting, stop_waiting):
        t.cancel()
    await asyncio.gather(*workers, done_waiting, stop_waiting, return_exceptions=True)
```

The drain step empties the queue but still calls `task_done()` for each discarded item, so `join()` then waits only for fetches already in progress. To test it, start the script with `subprocess.Popen`, sleep briefly, call `p.send_signal(signal.SIGTERM)` and assert `p.wait(timeout=5) == 0`.

**18.2** The endpoint above already cancels the old task when a prompt arrives, and the old task's `except CancelledError` branch sends `stopped` before the new task starts sending. A test:

```python
def test_new_prompt_replaces_old():
    with TestClient(app) as client, client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "prompt", "text": "first"})
        ws.receive_json()
        ws.send_json({"type": "prompt", "text": "second"})
        types = []
        while (m := ws.receive_json())["type"] != "done":
            types.append(m["type"])
        assert "stopped" in types
        assert types.index("stopped") < len(types) - 1   # new deltas follow the stop
```

**18.3** The shell form wraps the command in `/bin/sh -c`, so the shell is PID 1 and the SIGTERM sent to the pod goes to the shell, which does not forward it to Python. Nothing reacts until SIGKILL arrives after 30 seconds. Fix: use the exec form `CMD ["python", "main.py"]`, add a SIGTERM handler (18.1) or move cleanup into a FastAPI lifespan, and consider `--init` or `tini` so orphans are reaped too.

## Module 19: Security for tool-using agents

An agent turns text into actions, and some of that text comes from attackers: a web page it reads, a document a user uploads, an email it summarises. Instructions hidden in such content can steer the model into calling your tools with hostile arguments, an attack called **prompt injection**. You cannot reliably prevent injection with prompting alone, so engineers defend the *tools*: every tool argument is untrusted input, every tool's work is bounded, and dangerous actions need a human. OWASP maintains a Top 10 list for LLM applications that is worth reading alongside this module. All code here was run on Python 3.12.3.

### 19.1 Bound every tool

The Module 13 calculator showed what one unbounded tool call does: `9**9**9**9` froze the event loop so completely that even a timeout could not fire. For every tool, decide:

| Bound | How |
| --- | --- |
| Time | `asyncio.timeout` per call (Module 5); a subprocess and process-group kill for CPU work (Module 16) |
| CPU and memory | Separate process or container with limits; never `eval`/`exec` model output in your server process |
| Output size | Truncate before returning to the model (the Agent's `max_tool_output_chars`) |
| Concurrency | A semaphore per tool or per downstream service (Module 6) |
| Reach | Allowlists of hosts, paths, tables or commands; read-only credentials where possible |
| Side effects | Human approval for anything that sends, pays, deletes or deploys (19.3) |

### 19.2 Server-side request forgery (SSRF)

A "fetch this URL" tool is the most common agent tool and the most commonly abused. Injected text asks the agent to fetch `http://169.254.169.254/latest/meta-data/` (the cloud metadata service, which can expose credentials) or `http://localhost:8000/admin`. Check every URL, including the address it resolves to:

```python
import asyncio, ipaddress, socket
from urllib.parse import urlsplit

class BlockedURL(ValueError):
    pass

async def check_url(url: str, allowed_ports=(80, 443)) -> None:
    """Reject non-HTTP schemes, unusual ports, and hosts that resolve to private, loopback,
    link-local (cloud metadata) or reserved addresses."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise BlockedURL(f"scheme {parts.scheme!r} not allowed")
    if not parts.hostname:
        raise BlockedURL("no host")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    if port not in allowed_ports:
        raise BlockedURL(f"port {port} not allowed")
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(parts.hostname, port, type=socket.SOCK_STREAM)  # non-blocking DNS
    except socket.gaierror as exc:
        raise BlockedURL(f"cannot resolve {parts.hostname}") from exc
    for *_, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if not ip.is_global:                       # private, loopback, link-local, reserved...
            raise BlockedURL(f"{parts.hostname} resolves to non-public address {ip}")
```

Measured, it blocks `http://169.254.169.254/...`, `http://localhost:8000/admin` (port), `http://10.0.0.5/`, `file:///etc/passwd`, `http://[::1]/`, `https://example.com:8443/` and the disguised loopback `http://0x7f000001/`, which resolves to 127.0.0.1. Note the async DNS lookup: `socket.getaddrinfo` would block the loop, while `loop.getaddrinfo` runs it in a thread for you.

Two gaps remain, and a production guard closes both:

- **Redirects.** A public URL can answer `302 Location: http://169.254.169.254/`. httpx2 does not follow redirects unless you ask it to, so follow them yourself and check every hop (Exercise 19.1).
- **DNS rebinding.** An attacker's DNS can return a public address when you check and a private one when the HTTP client connects a moment later. Close the gap by connecting to the exact IP you checked, or by sending all agent traffic through an egress proxy that enforces the same rules.

### 19.3 Human approval for side effects

```python
class ApprovalGate:
    """Side-effecting tools wait here until a person approves or rejects."""
    def __init__(self):
        self.pending: dict[str, asyncio.Future] = {}

    async def request(self, action_id: str, timeout: float) -> bool:
        fut = asyncio.get_running_loop().create_future()
        self.pending[action_id] = fut
        try:
            async with asyncio.timeout(timeout):
                return await fut
        except TimeoutError:
            return False                         # no answer means no
        finally:
            self.pending.pop(action_id, None)

    def decide(self, action_id: str, approved: bool) -> None:
        fut = self.pending.get(action_id)
        if fut and not fut.done():
            fut.set_result(approved)
```

A UI lists `gate.pending` and calls `decide` from an HTTP or WebSocket handler. The default when nobody answers is *no*. In a multi-process deployment the pending approvals must live in a shared store, and long waits call for durable execution (Module 20.6) rather than an open Future.

### 19.4 Secrets, tool output and MCP servers

- **Secrets never enter the prompt.** API keys, database passwords and session tokens stay in your code's configuration; tools receive them from the server side. Anything in the context window can be echoed back to a user or sent to another tool.
- **Least privilege per tool.** Give each tool its own narrowly scoped credential (read-only database role, a single bucket prefix) so that a hijacked call can do little damage.
- **Tool output is data, not instructions.** Truncate it and label it clearly as untrusted content in the prompt. This reduces, but does not eliminate, injection; the bounds above are what actually protect you.
- **MCP servers are code you are trusting.** Connect only to servers you trust, pin their versions, review tool descriptions (they are text the model reads and can themselves carry injected instructions, sometimes called *tool poisoning*), and apply the same bounds to MCP tools as to local ones.

### Check yourself

1. Why defend tools rather than rely on the prompt to resist injection? *Answer: no prompt reliably stops injected instructions; bounding what tools can do limits the damage whatever the model is told.*
2. Why does `check_url` use `loop.getaddrinfo`? *Answer: the blocking `socket.getaddrinfo` would stall the event loop during DNS lookups.*
3. What should an approval gate do when nobody answers? *Answer: treat it as a rejection.*

### Exercises

**19.1 (M)** Write `fetch_safe(client, url, max_redirects=5)` that calls `check_url` before every request and follows redirects manually. Test it with an `httpx2.MockTransport` whose `/start` path redirects to `http://169.254.169.254/latest/meta-data/`. Use a public IP literal such as `http://93.184.215.14/start` as the starting URL, so the test needs no DNS.

**19.2 (M)** Write `requires_approval(gate, fn, timeout)` that wraps an async tool so each call waits for approval and raises `PermissionError` if rejected or unanswered. Wire it into an `Agent` for a `send_email` tool and show both outcomes.

**19.3 (E)** For the `run_python` tool of Module 16, list one concrete enforcement for each row of the table in 19.1.

### Solutions

**19.1**

```python
import httpx2

async def fetch_safe(client: httpx2.AsyncClient, url: str, max_redirects: int = 5) -> httpx2.Response:
    for _ in range(max_redirects + 1):
        await check_url(url)
        r = await client.get(url)                 # httpx2 does not follow redirects by default
        if r.is_redirect:
            url = str(r.next_request.url) if r.next_request else r.headers["location"]
            continue
        return r
    raise BlockedURL("too many redirects")

async def handler(request: httpx2.Request) -> httpx2.Response:
    if request.url.path == "/start":
        return httpx2.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})
    return httpx2.Response(200, text="ok")

async def main():
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        try:
            await fetch_safe(client, "http://93.184.215.14/start")
        except BlockedURL as e:
            print("blocked ->", e)   # blocked -> 169.254.169.254 resolves to non-public address 169.254.169.254
```

**19.2**

```python
import json

def requires_approval(gate: ApprovalGate, fn, timeout: float = 30.0):
    async def wrapped(**kwargs):
        action_id = f"{fn.__name__}:{json.dumps(kwargs, sort_keys=True)}"
        if not await gate.request(action_id, timeout=timeout):
            raise PermissionError(f"{fn.__name__} was not approved")
        return await fn(**kwargs)
    wrapped.__name__ = fn.__name__
    return wrapped

async def send_email(to: str, body: str) -> str:
    return f"sent to {to}"

async def main():
    gate = ApprovalGate()
    guarded = requires_approval(gate, send_email, timeout=0.5)
    async def approver():                          # stands in for a person clicking "approve"
        await asyncio.sleep(0.05)
        for action_id in list(gate.pending):
            gate.decide(action_id, True)
    approver_task = asyncio.create_task(approver())
    print(await guarded(to="a@b.c", body="hi"))    # sent to a@b.c
    await approver_task
    try:
        await guarded(to="x@y.z", body="spam")     # nobody answers
    except PermissionError as e:
        print(e)                                   # send_email was not approved
```

Register `guarded` in the Agent's tool dictionary under `send_email`. Because the Agent converts exceptions into error results (Module 13), a rejection reaches the model as `ERROR: PermissionError: send_email was not approved` and the run continues.

**19.3** Time: the 5-second subprocess timeout with a process-group kill. CPU and memory: run inside a container with CPU and memory limits, or set `resource.setrlimit` limits in a small wrapper script that the subprocess runs first. Output: the existing `max_output` truncation. Concurrency: a `Semaphore(2)` around `run_python`. Reach: a container with no network and a read-only filesystem except a scratch directory. Side effects: none should be possible once reach is restricted, which is the point of the container.

## Module 20: Production at scale

A working agent service becomes a production one when you can see what it is doing, cap what it spends, know how it behaves under load and survive restarts mid-run. Each of these builds directly on asyncio mechanics you already know: context variables carry traces, code without `await` makes budgets race-free, semaphores and event loops shape load curves. Code in 20.1 to 20.3 was run with opentelemetry-sdk 1.45.2 on Python 3.12.3; 20.4 to 20.6 are design guidance.

### 20.1 Tracing with OpenTelemetry

A **trace** is a tree of **spans**, one per operation (an agent run, each model call, each tool call), with timings and attributes. OpenTelemetry stores the current span in a context variable, so a span opened in a request handler automatically becomes the parent of spans in every task that handler creates (Module 14.5). Measured with two agent runs interleaved by `gather`, every tool span was attached to the correct run:

```python
import asyncio
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

exporter = InMemorySpanExporter()                # in production: an OTLP exporter to your backend
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("agent")

async def tool(name: str, seconds: float):
    with tracer.start_as_current_span(f"tool.{name}") as span:
        span.set_attribute("tool.name", name)
        await asyncio.sleep(seconds)

async def agent_run(request_id: str):
    with tracer.start_as_current_span("agent.run", attributes={"request.id": request_id}):
        async with asyncio.TaskGroup() as tg:     # child tasks inherit the current span
            tg.create_task(tool("search", 0.1))
            tg.create_task(tool("db", 0.05))

async def main():
    await asyncio.gather(agent_run("req-A"), agent_run("req-B"))
    spans = exporter.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}
    for s in spans:
        if s.name.startswith("tool."):
            parent = by_id[s.parent.span_id]
            print(s.name, "->", parent.attributes["request.id"])   # each tool under its own run

asyncio.run(main())
```

In production, use `BatchSpanProcessor` with an OTLP exporter rather than the in-memory one, and record at least: model name, input and output tokens, stop reason, tool name, tool status, and request ID. Both the Anthropic SDK and the MCP SDK depend on `opentelemetry-api`, and several tracing vendors offer drop-in instrumentation for them.

### 20.2 Budgets

An agent that loops is an agent that spends. Give every run a budget of model calls, tokens or money, and enforce it in one place:

```python
class BudgetExceeded(RuntimeError):
    pass

class Budget:
    """Shared token budget for one run; safe without a lock because charge() never awaits."""
    def __init__(self, max_tokens: int):
        self.max_tokens, self.used = max_tokens, 0
    def charge(self, tokens: int) -> None:
        if self.used + tokens > self.max_tokens:
            raise BudgetExceeded(f"token budget exceeded ({self.used + tokens} > {self.max_tokens})")
        self.used += tokens
```

Measured: four concurrent workers each charging 60 tokens five times against a budget of 1,000 stopped at 960 tokens used, with all four workers receiving the error at their next charge. No lock is needed because the check and the update run with no `await` between them (Module 3). With the real SDK, charge `message.usage.input_tokens + message.usage.output_tokens` after each call.

### 20.3 Load testing

Never guess how many concurrent users a process handles; measure it. A load generator is itself a nice asyncio program:

```python
# loadgen.py
import asyncio, statistics, time
import httpx2

async def load_test(url: str, total: int, concurrency: int, body: dict) -> dict:
    latencies: list[float] = []
    errors = 0
    sem = asyncio.Semaphore(concurrency)
    async with httpx2.AsyncClient(timeout=60, limits=httpx2.Limits(max_connections=concurrency)) as client:
        async def one():
            nonlocal errors
            async with sem:
                start = time.perf_counter()
                try:
                    r = await client.post(url, json=body)
                    r.raise_for_status()
                    latencies.append(time.perf_counter() - start)
                except httpx2.HTTPError:
                    errors += 1
        wall = time.perf_counter()
        await asyncio.gather(*(one() for _ in range(total)))
        wall = time.perf_counter() - wall
    qs = statistics.quantiles(latencies, n=100) if len(latencies) > 1 else [0.0] * 99
    return {"requests": total, "errors": errors, "throughput_rps": round(total / wall, 1),
            "p50_s": round(qs[49], 3), "p95_s": round(qs[94], 3)}
```

Against the capstone service, this produced the flat-latency, linear-throughput table in Module 15.6. For larger tests, dedicated tools such as Locust or k6 add ramp-up schedules and reporting. Always look at the 95th or 99th percentile, not just the median: tail latency is what users complain about.

### 20.4 Rate limits across processes

A semaphore or token bucket (Module 6) limits one process. With 8 uvicorn workers across 4 pods, each has its own, so the real ceiling is 32 times the per-process value. Options, in increasing complexity: divide the provider limit by the number of processes and set each semaphore accordingly; route model calls through a gateway service that owns the limits; or keep a shared token bucket in Redis. Whatever you choose, still handle 429 responses with backoff (Module 9), because the provider's view of your usage is the one that counts.

### 20.5 Caching and deduplication

Request coalescing (Exercise 6.4) removes duplicate in-flight calls. A response cache keyed on a hash of the model name, prompt and tool definitions removes repeated calls across time, for deterministic use cases such as classification or extraction. Providers also offer prompt caching, which makes a long, repeated prompt prefix (system prompt, tool definitions, a large document) cheaper and faster to process on later calls; check your provider's documentation for how to mark the cacheable prefix.

### 20.6 Durable execution for long runs

An agent run that takes minutes or waits hours for human approval should not live only in a coroutine: a deployment restart kills it (Module 18). Persist state after each step so a run can resume. LangGraph's checkpointers (Module 13.3) do this at the graph level with a `thread_id`. Workflow engines such as Temporal make it a platform feature, replaying a workflow's recorded history to rebuild its state after a crash. Either way, design tools to be idempotent (Module 9.3), because a resumed run may repeat the step that was in progress when the process died.

### Check yourself

1. How does a span opened in a request handler become the parent of spans in tasks the handler creates? *Answer: OpenTelemetry keeps the current span in a context variable, and each new task copies the current context.*
2. Why is the `Budget` class safe without a lock? *Answer: `charge` never awaits, so no other task can run between its check and its update.*
3. Why is a per-process semaphore not a global rate limit? *Answer: every process has its own semaphore, so the effective limit multiplies by the number of processes.*

### Exercises

**20.1 (M)** Add tracing to the Module 13 `Agent` without editing it: write a `TracedLLM` wrapper and a `traced_tool` decorator, then run a scripted agent with one working and one failing tool. Check that the failing tool's span has status `ERROR` and that all spans share the `agent.run` parent.

**20.2 (M)** Write `BudgetedLLM(llm, max_calls)` that raises `BudgetExceeded` once a run has used its calls, and show that it stops an agent whose fake model requests a tool forever, before the agent's own `max_steps` limit.

**20.3 (M)** Run `loadgen.py` against the capstone at concurrency 1, 10 and 40. Explain why throughput rises while latency stays flat, and predict what changes when `AnthropicLLM(max_concurrency=8)` replaces the fake model.

### Solutions

**20.1**

```python
import functools

class TracedLLM:
    def __init__(self, llm):
        self.llm = llm
    async def complete(self, messages, tools=None, system=None, max_tokens=1024):
        with tracer.start_as_current_span("llm.complete") as span:
            span.set_attribute("llm.messages", len(messages))
            turn = await self.llm.complete(messages, tools=tools, system=system, max_tokens=max_tokens)
            span.set_attribute("llm.tool_calls", len(turn.tool_calls))
            return turn

def traced_tool(fn):
    @functools.wraps(fn)
    async def wrapper(**kwargs):
        with tracer.start_as_current_span(f"tool.{fn.__name__}") as span:
            try:
                return await fn(**kwargs)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                raise
    return wrapper
```

Measured span tree for one run with `search` (works) and `broken` (raises): `agent.run`, then `llm.complete`, `tool.search`, `tool.broken` (status `ERROR`) and a second `llm.complete`, all with `agent.run` as their parent, and the agent still ends with `final_answer`. Wrapping instead of editing is the same adapter idea as `llm.py`.

**20.2**

```python
class BudgetedLLM:
    def __init__(self, llm, max_calls: int):
        self.llm, self.max_calls, self.calls = llm, max_calls, 0
    async def complete(self, *args, **kwargs):
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"budget of {self.max_calls} model calls used up")
        self.calls += 1
        return await self.llm.complete(*args, **kwargs)
```

With a fake model that always requests a tool and `max_calls=3`, the run raises `BudgetExceeded: budget of 3 model calls used up` on the fourth call, before the default `max_steps=8`. In a service, catch it at the endpoint and return the partial result with a clear status.

**20.3** Measured with the fake model: throughput 0.9, 8.9 and 33.9 requests per second, median latency 1.11 to 1.16 s. Each request spends its time awaiting sleeps that stand in for model and tool calls, so one event loop overlaps 40 requests as easily as one. With `AnthropicLLM(max_concurrency=8)`, at most 8 model calls run at once per process: throughput stops growing at the level the semaphore and the provider's rate limit allow, and latency at concurrency 40 rises as requests queue for the semaphore. That queueing time is a metric worth exporting.

## Interview question bank

These 46 questions reflect what agentic AI and ML platform interviews ask about async Python, from screening questions to senior system design. Answer each out loud in under two minutes before reading the model answer; the module in brackets is where to revise.

### Fundamentals

**Q1. What is the difference between concurrency and parallelism, and which does asyncio give you?** (Module 1)

Concurrency is several tasks in progress over the same period with interleaved steps; parallelism is several tasks executing at the same instant on different cores. Asyncio provides concurrency on a single thread. For parallel CPU work you add processes (or threads on a free-threaded build).

**Q2. When is asyncio the right tool, and when is it not?** (Module 1)

It is right for I/O-bound workloads with many concurrent waits: API calls, LLM calls, web servers, crawlers, agents. It is wrong for CPU-bound work, which blocks the loop; use a process pool there, called from async code through `run_in_executor`.

**Q3. What does calling an `async def` function return?** (Module 2)

A coroutine object. The body does not run until the coroutine is awaited, wrapped in a task, or passed to `asyncio.run`. Forgetting the `await` produces a "coroutine was never awaited" warning.

**Q4. Explain what the event loop does.** (Module 3)

It keeps a queue of ready tasks and a set of waiting ones. It runs one ready task until that task suspends at an `await` that genuinely has to wait, asks the OS (epoll, kqueue, IOCP) which I/O is ready and which timers expired, moves the corresponding tasks back to the ready queue, and repeats.

**Q5. Is every `await` a point where other tasks can run?** (Module 3)

No. Control returns to the loop only when the await chain reaches something that actually has to wait, such as a pending Future. Awaiting a coroutine that never suspends runs it straight through. `await asyncio.sleep(0)` forces exactly one yield.

**Q6. Why does `time.sleep(1)` inside a coroutine hurt, while `await asyncio.sleep(1)` does not?** (Module 2)

`time.sleep` blocks the only thread, so no other task can run and the loop cannot even notice incoming I/O. `asyncio.sleep` registers a timer and yields to the loop, which runs other tasks until the timer fires.

**Q7. Can two asyncio tasks race on shared state?** (Modules 3, 6)

Yes, but only across an `await`. Code between two awaits runs atomically with respect to other tasks. If you read state, await, then write based on what you read, another task can interleave: a lost update. Guard such sections with `asyncio.Lock`, or restructure to avoid the await in between.

### Tasks and structured concurrency

**Q8. What is the difference between `await coro()` and `asyncio.create_task(coro())`?** (Module 4)

The first runs the coroutine now and waits for it. The second schedules it to run concurrently and returns a Task immediately; you await the task later to get its result or exception.

**Q9. Why must you keep a reference to tasks you create?** (Module 4)

The loop keeps only weak references to tasks, so an unreferenced task can be garbage-collected mid-execution and silently disappear. Store it, await it, or create it in a TaskGroup.

**Q10. Compare `gather` and `TaskGroup` on error handling.** (Module 4)

With plain `gather`, the first exception propagates to the caller but the other tasks keep running; with `return_exceptions=True`, exceptions come back as values. A `TaskGroup` cancels the remaining tasks on the first failure, waits for them, and raises all failures together in an `ExceptionGroup`. Use TaskGroup when tasks belong together; use `gather(return_exceptions=True)` for independent work where partial results matter.

**Q11. What is structured concurrency and why does it matter for agents?** (Module 4)

Tasks cannot outlive the scope that created them: leaving a TaskGroup block guarantees every child is finished or cancelled. For agents this prevents orphaned LLM or tool calls that keep running, and costing money, after the request that started them failed or was abandoned.

**Q12. How do you take the first of several results and stop the rest?** (Modules 4, 10)

`asyncio.wait(tasks, return_when=FIRST_COMPLETED)`, then cancel the pending tasks and await them. On Python 3.15+ a TaskGroup task can call `tg.cancel()` once the goal is reached. This is the hedging pattern used across LLM providers or replicas.

**Q13. How do you process results in the order they finish?** (Module 4)

`asyncio.as_completed`, either as a plain loop of awaitables or, since Python 3.13, with `async for`, which yields the original task objects when you pass tasks.

### Cancellation and timeouts

**Q14. How does task cancellation work?** (Module 5)

`task.cancel()` requests that `CancelledError` be raised inside the task at its current await. The task can clean up in `finally`, and normally the exception propagates and the task ends cancelled. A task stuck in CPU work or a blocking call cannot be cancelled until it next awaits.

**Q15. Why must you never swallow `CancelledError`?** (Module 5)

Timeouts, TaskGroups, `wait_for`, server shutdown and client disconnect handling all depend on cancellation propagating. Swallowing it leaves the task running and breaks those mechanisms, often as a hang. Since Python 3.8 `CancelledError` derives from `BaseException`, so `except Exception` does not catch it; avoid bare `except:`.

**Q16. How do you put a timeout on async work?** (Module 5)

`async with asyncio.timeout(seconds):` (Python 3.11+), catching the built-in `TimeoutError`. It is implemented through cancellation of the work inside the block. `asyncio.wait_for` is the older function form. For a budget across several steps, wrap the whole sequence or use `asyncio.timeout_at` with an absolute deadline.

**Q17. What does `asyncio.shield` do, and what does it not do?** (Module 5)

It prevents cancellation of the outer task from cancelling the inner awaitable, so critical work such as saving state can finish. It does not stop the outer `await` from raising `CancelledError`, and a shielded operation that never finishes can block shutdown.

**Q18. A coroutine is awaiting a shared task, and the coroutine gets cancelled. What happens to the shared task?** (Module 6)

It is cancelled too: cancelling a task that is waiting on another task or future propagates the cancellation to it. If other callers share that task (request coalescing), wrap the await in `asyncio.shield`.

### Coordination and backpressure

**Q19. How do you limit concurrent LLM calls to respect a rate limit?** (Module 6)

An `asyncio.Semaphore(n)` around each call caps calls in flight. To cap calls started per second, add a token-bucket rate limiter. Real providers also limit tokens per minute, so production systems often combine all three, plus retries with backoff on 429 responses.

**Q20. What is backpressure and how do you get it in asyncio?** (Module 6)

Backpressure slows a fast producer to the speed of its consumers so memory does not grow without bound. A bounded `asyncio.Queue(maxsize=n)` provides it: `await q.put()` waits while the queue is full.

**Q21. How do worker pools shut down cleanly?** (Module 6)

Classic: `await q.join()` until all items are processed, then cancel the workers and await them. Since Python 3.13, `q.shutdown()` makes `get` raise `QueueShutDown` once the queue is drained, so workers can exit their loops cleanly.

### Mixing sync and async

**Q22. You must call a blocking library from async code. What are your options?** (Module 8)

`await asyncio.to_thread(fn, ...)` for blocking I/O, `loop.run_in_executor(process_pool, fn, ...)` for CPU-bound work, or replace the library with an async one. Never call it directly inside `async def`.

**Q23. How do you call async code from another thread?** (Module 8)

`asyncio.run_coroutine_threadsafe(coro, loop)` returns a `concurrent.futures.Future`; the other thread calls `.result(timeout=...)`. Never call `.result()` on the loop's own thread, which deadlocks.

**Q24. In FastAPI, what is the difference between a `def` and an `async def` endpoint?** (Module 11)

FastAPI runs `def` endpoints in a thread pool, so blocking code there is tolerable within the pool's size. `async def` endpoints run on the event loop, so they must never block; blocking code there stalls every request in the process.

**Q25. How would you find what is blocking your event loop?** (Modules 8, 14)

Enable debug mode (`PYTHONASYNCIODEBUG=1` or `asyncio.run(..., debug=True)`), which logs steps that take longer than 100 ms. On Python 3.14+, `python -m asyncio ps/pstree <PID>` shows what each task awaits; on 3.15+, the `profiling.sampling` profiler can attach to the process with async-aware stacks.

### Production agent engineering

**Q26. Walk me through a production agent loop.** (Module 13)

Call the model with the conversation and tool specs; if it returns tool calls, execute them concurrently with per-tool timeouts, convert failures into error results, truncate large outputs, append results, and loop. Stop on a final answer, a step limit or an overall deadline, and emit events for observability. All of it must be cancellation-safe so a user stop cancels in-flight calls.

**Q27. Why return tool errors to the model instead of raising?** (Modules 5, 10)

The model can often recover by retrying with different arguments or using another tool. Raising would abort the whole run for a recoverable problem. Unexpected internal errors are still logged with full context.

**Q28. Where should HTTP and LLM clients be created in a web service, and why?** (Modules 9, 11)

Once, in the application lifespan, and closed at shutdown. Clients hold connection pools; creating one per request wastes TCP and TLS handshakes and can exhaust sockets under load.

**Q29. How do you stream LLM tokens to a browser, and what happens on disconnect?** (Module 11)

Iterate the SDK's async stream and yield server-sent events (`data: ...` lines) from a `StreamingResponse` generator. On disconnect the server stops the generator; cleanup in its `finally` closes the upstream LLM stream so you stop paying for unread tokens.

**Q30. Describe a retry policy for LLM API calls.** (Module 9)

Retry only transient failures (connection errors, timeouts, 429, 5xx), with exponential backoff, random jitter and a cap, honouring `Retry-After`. Bound total attempts and total time, and remember the SDKs already retry by default, so avoid stacking retries multiplicatively.

**Q31. How do you keep a request ID in logs across many concurrent tasks?** (Module 14)

Store it in a `contextvars.ContextVar` at the start of the request. Each task created afterwards gets a copy of the current context, as does `asyncio.to_thread`, so a logging filter can read the right ID for every log line.

**Q32. What changed in the Python async HTTP ecosystem in 2026?** (Module 9)

The original `httpx` stopped receiving releases after 0.28.1 in December 2024, and Pydantic now maintains it as `httpx2`, with the same API under a new import name. The Anthropic and OpenAI Python SDKs moved to httpx2, which verifies TLS against the OS trust store, so mocks that patch `httpx` and minimal containers without CA certificates need attention.

**Q33. How do you test async agent code without calling real APIs?** (Modules 10, 14)

Inject dependencies: a fake LLM with scripted turns, `httpx2.MockTransport` for HTTP, in-memory MCP clients (`Client(server)`), FastAPI `dependency_overrides`, and short configurable limits. Use events rather than sleeps for ordering, and pytest-asyncio in auto mode.

### Code-reading puzzles

**Q34. What does this print, and how long does it take?**

```python
async def f(x):
    await asyncio.sleep(x)
    return x

async def main():
    a = f(1)
    b = asyncio.create_task(f(2))
    print(await a, await b)
```

It prints `1 2` after about 2 seconds. The task for `f(2)` starts running at the first suspension, which is inside `await a`, so both sleeps overlap.

**Q35. What is wrong here?**

```python
async def handler(urls):
    for url in urls:
        asyncio.create_task(fetch_and_store(url))
    return "queued"
```

The tasks are not referenced, so they may be garbage-collected mid-run; their exceptions are never observed; and nothing limits how many run at once. Store them in a tracked set with a done callback that logs failures, bound concurrency with a semaphore, or hand the work to a proper job queue.

**Q36. Why might this never finish?**

```python
async def worker():
    while True:
        try:
            await do_work()
        except BaseException:
            log.exception("retrying")
```

`except BaseException` swallows `CancelledError`, so cancelling the worker (for a timeout or shutdown) only triggers another iteration. Catch `Exception` instead.

**Q37. This test passes alone but fails in the full suite with "attached to a different loop". Why?**

```python
limiter = asyncio.Semaphore(5)   # module level
```

The semaphore binds to the first event loop that uses it, and pytest-asyncio gives each test a fresh loop. Create such objects inside the code under test or in fixtures.

### System design

**Q38. Design a service that answers 1,000 concurrent chat users with streaming, using an LLM API limited to 200 concurrent requests.** (Modules 6, 11)

Several async workers behind a load balancer; per-process semaphores sized so their sum stays under 200, or a shared limiter (for example Redis-based) for exact global limits. Queue with a timeout and a clear "busy" response when saturated, stream tokens over SSE, cancel upstream calls on disconnect, and track queue time, TTFT and error rates.

**Q39. Design a pipeline that embeds 10 million documents with a hosted embedding API.** (Modules 6, 7, 9)

Stream documents from storage with an async generator, batch them (for example 100 per request), use a bounded queue between reading, embedding and writing stages, cap concurrent requests with a semaphore plus a rate limiter, retry transient errors with backoff, and checkpoint progress so a crash resumes rather than restarts. Consider the provider's batch API if latency does not matter.

**Q40. Design a multi-agent research system with a 60-second SLA.** (Modules 13, 15)

A planner splits the question; workers run concurrently under a semaphore, each with its own step limit and deadline shorter than the overall one; tools come from MCP servers through a router; partial results are synthesised if some workers fail or time out; progress streams to the client; and an overall `asyncio.timeout` enforces the SLA. Parallel guardrails run alongside generation, and everything is traced by request ID.

### Operations, safety and version differences

**Q41. A task running a subprocess is cancelled. What happens to the child process, and how do you make cleanup reliable?** (Module 16)

Nothing happens to it: cancelling the awaiting task does not stop the child. Start the child with `start_new_session=True`, and in an exception handler that also catches cancellation, kill the whole process group with `os.killpg`, await `proc.wait()` to reap it, then re-raise. Killing only the direct child leaves its own children running.

**Q42. Your agent service drops in-flight work on every deployment. What is happening?** (Module 18)

The orchestrator sends SIGTERM, and Python's default response to SIGTERM is immediate termination with no `finally` blocks run. Under uvicorn, put cleanup in the lifespan (uvicorn handles the signal); in workers you run yourself, install a handler with `loop.add_signal_handler` that sets an event and drains. Also check that the Dockerfile uses the exec form so the signal reaches Python, and checkpoint runs that cannot finish within the grace period.

**Q43. How does cancellation in AnyIO differ from asyncio, and why should an asyncio developer care?** (Module 17)

Asyncio delivers one `CancelledError`; AnyIO cancel scopes are level-triggered and keep cancelling every await until the scope exits. AnyIO also requires a cancel scope to be exited by the task that entered it. FastAPI, httpx2, the MCP SDK and the LLM SDKs run on AnyIO, so these rules apply inside them; entering an MCP client in one task and closing it from another fails.

**Q44. An agent has a "fetch URL" tool. What can go wrong and how do you guard it?** (Module 19)

Prompt injection can make it fetch internal addresses such as the cloud metadata service or admin endpoints (SSRF). Allow only http and https on standard ports, resolve the host asynchronously and reject any non-global address, re-check every redirect hop yourself, and defend against DNS rebinding by connecting to the checked IP or using an egress proxy. Bound time and response size too.

**Q45. Which asyncio features would you avoid if the service must also run on Python 3.12?** (Python 3.12 guide)

`Queue.shutdown` and `async for` over `as_completed` (3.13), `python -m asyncio ps/pstree`, call-graph introspection and subinterpreters (3.14), and `TaskGroup.cancel()` (3.15). Use sentinels, tagged results and a private exception caught with `except*` instead, or test for the feature with `hasattr`. Also pass an explicit `forkserver` context to process pools, because 3.12 still defaults to `fork` on Linux.

**Q46. How do you trace one user request through an agent that fans out into many tasks?** (Modules 14, 20)

Open a span (and set a request-ID context variable) at the start of the request. Both live in `contextvars`, and every task copies the context current at its creation, so spans opened in tool and model calls attach to the request's span automatically. Export the spans with OpenTelemetry, recording model, token counts, tool name and status on each.

## Appendix

### A. Asyncio cheat sheet

| I want to... | Write | Since |
| --- | --- | --- |
| Run the program | `asyncio.run(main())` | 3.7 |
| Run several coroutines with one loop | `with asyncio.Runner() as r: r.run(a()); r.run(b())` | 3.11 |
| Wait without blocking | `await asyncio.sleep(s)` |  |
| Yield to other tasks once | `await asyncio.sleep(0)` |  |
| Start work in the background | `t = asyncio.create_task(coro(), name="x")` (keep `t`) | 3.7 |
| Run tasks that belong together | `async with asyncio.TaskGroup() as tg: tg.create_task(...)` | 3.11 |
| End a task group early, on purpose | `tg.cancel()` | 3.15 |
| Run independent work, keep partial results | `await asyncio.gather(*aws, return_exceptions=True)` |  |
| Handle results as they finish | `for f in asyncio.as_completed(aws): await f` or `async for t in asyncio.as_completed(tasks)` | async for: 3.13 |
| First result wins | `done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)` |  |
| Timeout | `async with asyncio.timeout(s):` and catch `TimeoutError` | 3.11 |
| Absolute deadline | `async with asyncio.timeout_at(loop.time() + s):` | 3.11 |
| Protect a step from outer cancellation | `await asyncio.shield(task)` |  |
| Catch errors from a TaskGroup | `except* ValueError as eg:` | 3.11 |
| Limit concurrency | `async with asyncio.Semaphore(n):` |  |
| Mutual exclusion across an await | `async with asyncio.Lock():` |  |
| Signal between tasks | `ev = asyncio.Event(); await ev.wait(); ev.set()` |  |
| Producer and consumer | `q = asyncio.Queue(maxsize=n); await q.put(x); await q.get()` |  |
| Shut a queue down | `q.shutdown()`; consumers catch `asyncio.QueueShutDown` | 3.13 |
| Async context manager | `@contextlib.asynccontextmanager` + `async with` |  |
| Async generator | `async def g(): yield x`, consumed by `async for` |  |
| Close an async generator promptly | `async with contextlib.aclosing(gen) as g:` | 3.10 |
| Several async resources | `async with contextlib.AsyncExitStack() as stack:` |  |
| Blocking I/O from async code | `await asyncio.to_thread(fn, *args)` | 3.9 |
| CPU work from async code | `await loop.run_in_executor(ProcessPoolExecutor(), fn, arg)` |  |
| Submit a coroutine from another thread | `asyncio.run_coroutine_threadsafe(coro, loop).result()` |  |
| Get the loop inside a coroutine | `asyncio.get_running_loop()` | 3.7 |
| Request-scoped values | `contextvars.ContextVar` | 3.7 |
| Debug mode | `PYTHONASYNCIODEBUG=1` or `asyncio.run(main(), debug=True)` |  |
| Inspect a running process's tasks | `python -m asyncio ps <PID>` / `pstree <PID>` | 3.14 |

| Operations and safety (Modules 16 to 20) | Write |
| --- | --- |
| Run a command without a shell | `await asyncio.create_subprocess_exec(*argv, stdout=PIPE, stderr=PIPE, start_new_session=True)` |
| Kill a command and its children | `os.killpg(proc.pid, signal.SIGKILL); await proc.wait()` |
| Stream a command's output | `async for line in proc.stdout:` |
| React to SIGTERM (Unix) | `loop.add_signal_handler(signal.SIGTERM, stop.set)` |
| AnyIO task group | `async with anyio.create_task_group() as tg: tg.start_soon(fn, arg)` |
| AnyIO timeout without raising | `with anyio.move_on_after(s) as scope:`, then check `scope.cancelled_caught` |
| Non-blocking DNS lookup | `await loop.getaddrinfo(host, port)` |
| Check an address is public | `ipaddress.ip_address(ip).is_global` |
| Span around an operation | `with tracer.start_as_current_span("tool.search"):` |

### B. Version notes

| Version | Change relevant to this course |
| --- | --- |
| 3.5 | `async` / `await` syntax |
| 3.7 | `asyncio.run`, `create_task`, `get_running_loop`, `contextvars` |
| 3.8 | `CancelledError` becomes a `BaseException` subclass; `python -m asyncio` REPL |
| 3.9 | `asyncio.to_thread` |
| 3.10 | `contextlib.aclosing`; asyncio primitives no longer take a `loop` argument |
| 3.11 | `TaskGroup`, `timeout`, `Runner`, `Barrier`, `ExceptionGroup` and `except*`; `@asyncio.coroutine` removed; `asyncio.wait` rejects bare coroutines |
| 3.12 | `eager_task_factory`; `asyncio.run(loop_factory=...)`; `wait` and `as_completed` accept generators; `Task.get_context()`; `wait_for` built on `timeout`; child watchers deprecated; `get_event_loop()` warns when it would create a loop |
| 3.13 | `Queue.shutdown`; `async for` over `as_completed`; TaskGroup cancellation edge cases improved; experimental free-threaded build |
| 3.14 | `python -m asyncio ps/pstree`, `capture_call_graph()`; `get_event_loop()` raises without a loop; event loop policies deprecated (removal planned for 3.16); child watchers removed; free-threaded build supported; `forkserver` default start method on Linux; `concurrent.interpreters` and `InterpreterPoolExecutor` |
| 3.15 | `TaskGroup.cancel()`; `profiling.sampling` (Tachyon) with `dump --async-aware`; UTF-8 default encoding; macOS installers include free-threading support |

Every feature in the 3.12 to 3.15 rows was checked by running Python 3.12.3, 3.13.18, 3.14.10 and 3.15.2 side by side.

### C. Library landscape (November 2026)

| Need | Common choice | Notes |
| --- | --- | --- |
| HTTP client | `httpx2`, `aiohttp` | httpx2 is the maintained successor of httpx ([PyPI mirror](https://simple-repository.app.cern.ch/project/httpx2)) |
| Web framework | FastAPI on Starlette and uvicorn | Async-native; lifespan, dependencies, streaming |
| LLM SDKs | `anthropic` 2.x, `openai` 3.x | Both async, both on httpx2 ([Anthropic migration](https://platform.claude.com/docs/en/api/migration-2026-10-21)) |
| Tool protocol | MCP Python SDK 2.x | `MCPServer`, `Client` ([What's new in v2](https://py.sdk.modelcontextprotocol.io/whats-new/)) |
| Agent orchestration | LangGraph 1.x; also OpenAI Agents SDK, Claude Agent SDK, Pydantic AI | All asyncio-based |
| Databases | `asyncpg`, SQLAlchemy `AsyncSession`, `redis.asyncio` |  |
| Retries | `tenacity`, `stamina` | Mind SDK built-in retries |
| Testing | `pytest` + `pytest-asyncio` 1.x, or the `anyio` pytest plugin | `asyncio_mode = "auto"` |
| Alternative async runtimes | `trio`, `anyio` (runs on asyncio or trio) | MCP SDK and httpx2 are built on anyio |

### D. Glossary

- **Awaitable:** anything usable with `await`: a coroutine, a Task or a Future.
- **Backpressure:** slowing producers to match consumers, for example with a bounded queue.
- **Cooperative multitasking:** tasks switch only where they choose to (`await`), unlike preemptive threads.
- **Coroutine:** the paused computation created by calling an `async def` function.
- **Event loop:** the scheduler that runs tasks and waits for I/O and timers.
- **Future:** a low-level placeholder for a result that will be set later.
- **Hedging:** sending duplicate requests and keeping the fastest.
- **Idempotent:** safe to repeat with the same effect.
- **Jitter:** randomness added to retry delays to avoid synchronised retries.
- **MCP:** Model Context Protocol, the open standard for connecting AI apps to tools and data.
- **SSE:** server-sent events, a simple HTTP streaming format used for token streaming.
- **Structured concurrency:** child tasks cannot outlive the scope that created them.
- **Task:** a coroutine scheduled on the event loop to run concurrently.
- **TTFT:** time to first token, the latency users feel in streaming chat.

### E. Further reading

- The official asyncio documentation: [docs.python.org/3/library/asyncio.html](https://docs.python.org/3/library/asyncio.html), especially "Coroutines and Tasks", "Synchronization Primitives" and "Developing with asyncio".
- What's new in Python [3.14](https://docs.python.org/3/whatsnew/3.14.html) and [3.15](https://docs.python.org/3/whatsnew/3.15.html), asyncio sections.
- [Anthropic Python SDK documentation](https://platform.claude.com/docs/zh-CN/cli-sdks-libraries/sdks/python.md) and the [OpenAI Python library](https://pypi.org/project/openai/3.45.0/).
- [MCP Python SDK documentation](https://py.sdk.modelcontextprotocol.io/whats-new/).
- [LangChain and LangGraph changelog](https://docs.langchain.com/oss/python/releases/changelog/index.html).
- [pytest-asyncio configuration](https://pytest-asyncio.readthedocs.io/en/stable/reference/configuration.html).
- Nathaniel J. Smith, "Notes on structured concurrency, or: Go statement considered harmful" (2018), the essay behind TaskGroups.

### F. Sources checked for this edition

Facts about versions and recent changes were verified on November 10, 2026 against these pages:

- [Python 3.15.0 release page](https://www.python.org/downloads/release/python-3150/) and [What's new in Python 3.15](https://docs.python.org/3/whatsnew/3.15.html)
- [asyncio event loop policy docs](https://docs.python.org/zh-cn/3/library/asyncio-policy.html) (deprecations in 3.14)
- [httpx2 package page (PyPI mirror)](https://simple-repository.app.cern.ch/project/httpx2)
- [Anthropic Python SDK 1.0 and httpx2 write-up](https://dev.to/mr_manushukla/anthropic-python-sdk-10-dropped-httpx-on-20-august-2026-what-breaks-in-your-mocks-and-tracing-dgn)
- [Anthropic Python SDK docs](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python) and [Migrating to API version 2026-10-21](https://platform.claude.com/docs/en/api/migration-2026-10-21)
- [OpenAI Python library on PyPI](https://pypi.org/project/openai/3.45.0/)
- [MCP Python SDK: What's new in v2](https://py.sdk.modelcontextprotocol.io/whats-new/)
- [LangChain and LangGraph changelog](https://docs.langchain.com/oss/python/releases/changelog/index.html)
- [pytest-asyncio configuration reference](https://pytest-asyncio.readthedocs.io/en/stable/reference/configuration.html)

How this edition was verified: the code was run, not just reviewed. Version-specific asyncio features were tested on Python 3.12.3, 3.13.18, 3.14.10 and 3.15.2, including `TaskGroup.cancel()`, `python -m asyncio ps/pstree` and `python -m profiling.sampling dump --async-aware` against a live process. Library behaviour was tested against httpx2 2.16.0, anthropic 2.8.0, openai 3.49.0, mcp 2.3.0, FastAPI 0.162.3 with Starlette 1.10, uvicorn 0.54.0, LangGraph 1.2.20, AnyIO 4.16.0 and opentelemetry-sdk 1.45.2. The LLM SDK calls used `httpx2.MockTransport` instead of the network, so they confirm request and response handling, not model output. Python version support dates come from the [Python devguide](https://devguide.python.org/versions), and the Anthropic SDK details from its [English documentation](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python).
