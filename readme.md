# Tracing Auto Instrumentation

Tracing Auto Instrumentation allows you to easily instrument popular LLM frameworks for tracing your LLM application. It is built with and on top of the LastMile Rag Debugger: [https://rag.lastmileai.dev/](https://rag.lastmileai.dev/).

## Examples

Supported Frameworks and Libraries:

| Framework/Library                                                                                     | Example Link                                                                                                  |
| ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| [OpenAI](https://github.com/lastmile-ai/eval-cookbook/blob/main/tutorials/distributed_tracing.ipynb)     | [OpenAI Example](https://github.com/lastmile-ai/eval-cookbook/blob/main/tutorials/distributed_tracing.ipynb)     |
| [IBM](https://github.com/lastmile-ai/eval-cookbook/blob/main/tutorials/distributed_tracing.ipynb)        | [IBM Example](https://github.com/lastmile-ai/eval-cookbook/blob/main/tutorials/distributed_tracing.ipynb)        |
| [LangChain](https://github.com/lastmile-ai/eval-cookbook/blob/main/tutorials/distributed_tracing.ipynb)  | [LangChain Example](https://github.com/lastmile-ai/eval-cookbook/blob/main/tutorials/distributed_tracing.ipynb)  |
| [LLamaIndex](https://github.com/lastmile-ai/eval-cookbook/blob/main/tutorials/distributed_tracing.ipynb) | [LLamaIndex Example](https://github.com/lastmile-ai/eval-cookbook/blob/main/tutorials/distributed_tracing.ipynb) |

## Getting Started

Getting started is easy. Simply choose the framework you want to instrument and follow the instructions below. If you want to instrument multiple frameworks, you can install `all`.

```shell
pip install "tracing-auto-instrumentation[all]"
```

### OpenAI

```shell
pip install "tracing-auto-instrumentation[openai]"
```

```python
import openai
from tracing_auto_instrumentation.openai import wrap_openai
from lastmile_eval.rag.debugger.tracing.sdk import get_lastmile_tracer

tracer = get_lastmile_tracer(
    tracer_name="OpenAI Function Calling",
)
client = wrap_openai(openai.OpenAI(), tracer)
```

### LangChain

```shell
pip install "tracing-auto-instrumentation[langchain]"
```

```python
import langchain
from tracing_auto_instrumentation.langchain import LangChainInstrumentor

# Create an instance of LangChainInstrumentor and instrument the code
instrumentor = LangChainInstrumentor(project_name="Plan-and-Execute Example")
instrumentor.instrument()
```

### LLamaIndex

```shell
pip install "tracing-auto-instrumentation[llamaindex]"
```

```python
import llama_index.core

from tracing_auto_instrumentation.llama_index import LlamaIndexCallbackHandler

llama_index.core.global_handler = LlamaIndexCallbackHandler(
    project_name="LlamaIndex with Paul Graham",
)
```

### IBM

```shell
pip install "tracing-auto-instrumentation[ibm]"
```

```python
# todo
```
