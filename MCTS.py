"""
title: 基于蒙特卡洛搜索树的迭代式检索增强
author: 王宇峰
date: 2025-03-01
version: 1.0
license: MIT
description: 基于蒙特卡洛搜索树的迭代式检索增强，通过多次模拟与评估不断改进答案。
requirements: fastapi, openai, rich, asyncio, uuid
"""


### 导入了 FastAPI、日志、随机数、数学、异步、JSON、正则表达式等标准库，同时引入了 Open-Webui 框架中的常量、路由和应用实例。
import logging
import random
import math
import asyncio
import json
import re
import uuid

from typing import List, Optional, AsyncGenerator, Callable, Awaitable, Generator, Iterator, Union
from open_webui.constants import TASKS
from openai import OpenAI


# ==============================================================================
"""
default_max_children：每个节点最多扩展多少个子节点。
default_exploration_weight：探索因子，用于平衡探索和利用（通常用近似√2值 1.414）。
default_max_iterations 和 default_max_simulations：分别定义整个 MCTS 搜索的迭代次数和每次迭代中模拟的次数。
default_thoughts：在扩展节点时调用 LLM 思考的次数。
"""
name = "mcts"
default_max_children = 2
default_exploration_weight = 1.414
default_max_iterations = 2
default_max_simulations = 2
default_thoughts = 2

# ==============================================================================
"""
thoughts_prompt：请求 LLM 给出如何改进当前答案的建议，要求回复为一句话。
eval_answer_prompt：要求 LLM 对一个答案给出评分（1 到 10 分），以便后续选择最佳答案。
analyze_prompt：对一次迭代进行分析，总结哪些部分有效、哪些需要改进。
update_prompt：根据批评意见更新答案，要求 LLM 给出修改后的答案。
initial_prompt：初始回答模板，直接让 LLM 回答问题。
"""

thoughts_prompt = """
<instruction>
Give a suggestion on how this answer can be improved.
WRITE ONLY AN IMPROVEMENT SUGGESTION AND NOTHING ELSE.
YOUR REPLY SHOULD BE A SINGLE SENTENCE.
</instruction>

<question>
{question}
</question>

<draft>
{answer}
</draft>
""".strip()

eval_answer_prompt = """
Given the following text:
"{answer}"

How well does it answers this question:
"{question}"

Rate the answer from 1 to 10, where 1 is completely wrong or irrelevant and 10 is a perfect answer.
Reply with a single number between 1 and 10 only. Do not write anything else, it will be discarded.
THINK CAREFULLY AND USE BEST PRACTICES.
""".strip()

analyze_prompt = """
Iteration Analysis:

Original question: {question}
Best answer found: {best_answer}
Best score achieved: {best_score}

Analyze this iteration of the thought process. Consider the following:
1. What aspects of the best answer made it successful?
2. What patterns or approaches led to higher-scoring thoughts?
3. Were there any common pitfalls or irrelevant tangents in lower-scoring thoughts?
4. How can the thought generation process be improved for the next iteration?

Provide a concise analysis and suggest one specific improvement strategy for the next iteration.
""".strip()

update_prompt = """
<instruction>
Your task is to read the question and the answer below, then analyse the given critique.
When you are done - think about how the answer can be improved based on the critique.
WRITE A REVISED ANSWER THAT ADDRESSES THE CRITIQUE. DO NOT WRITE ANYTHING ELSE.
</instruction>
<question>
{question}
</question>
<draft>
{answer}
</draft>
<critique>
{improvements}
</critique>
""".strip()

initial_prompt = """
<instruction>
Answer the question below. Do not pay attention to, unexpected casing, punctuation or accent marks.
</instruction>

<question>
{question}
</question>
"""

# ==============================================================================


def setup_logger():
  logger = logging.getLogger(__name__)
  if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.set_name(name)
    formatter = logging.Formatter(
      "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
  return logger

logger = setup_logger()


# ==============================================================================

mods = [
  "capitalize",
  "diacritic",
  "leetspeak",
  "remove_vowel",
]


def modify_text(text, percentage):
  if percentage == 0:
    return text, {}

  if not text:
    return "", {}  # 当输入为空时，返回空字符串和空映射

  if not 0 <= percentage <= 100:
    raise ValueError("Percentage must be between 0 and 100")

  words = text.split()
  chars = list(text)
  num_chars_to_modify = max(1, int(len(chars) * (percentage / 100)))
  indices_to_modify = random.sample(range(len(chars)), num_chars_to_modify)
  word_mapping = {}

  for idx in indices_to_modify:
    modification = random.choice(mods)

    current_length = 0
    for word_idx, word in enumerate(words):
      if current_length <= idx < current_length + len(word):
        original_word = word
        word_start_idx = current_length
        break
      current_length += len(word) + 1  # +1 表示空格
    else:
      continue

    if modification == "capitalize":
      chars[idx] = chars[idx].swapcase()
    elif modification == "diacritic":
      if chars[idx].isalpha():
        diacritics = ["̀", "́", "̂", "̃", "̈", "̄", "̆", "̇", "̊", "̋"]
        chars[idx] = chars[idx] + random.choice(diacritics)
    elif modification == "leetspeak":
      leetspeak_map = {
        "a": "4", "e": "3", "i": "1", "o": "0",
        "s": "5", "t": "7", "b": "8", "g": "9", "l": "1",
      }
      chars[idx] = leetspeak_map.get(chars[idx].lower(), chars[idx])
    elif modification == "remove_vowel":
      if chars[idx].lower() in "aeiou":
        chars[idx] = ""

    modified_word = "".join(chars[word_start_idx:word_start_idx + len(original_word)])

    if modified_word != original_word:
      # 采用复合键（添加位置信息）避免因多个相同修改而引起映射冲突
      mapping_key = f"{word_start_idx}_{modified_word}"
      word_mapping[mapping_key] = original_word

  modified_text = "".join(chars)
  return modified_text, word_mapping


# ==============================================================================


def escape_mermaid(text):
  return text.replace('"', "&quot;").replace("(", "&#40;").replace(")", "&#41;")


"""
每个节点保存当前的“回答”内容、父节点、子节点列表，以及搜索过程中累计的访问次数和评估得分。
"""
class Node:
  id: str
  content: str
  parent: Optional["Node"]
  max_children: int
  children: List["Node"]
  visits: int
  value: float

  # 随机生成一个 4 字母的 ID，设置内容、父节点、探索权重和最大子节点数，同时初始化访问次数和累积分数为 0。
  def __init__(self, **kwargs):
    # 使用 UUID 确保节点 ID 的唯一性，防止重复冲突
    self.id = str(uuid.uuid4())
    self.content = kwargs.get("content")
    self.parent = kwargs.get("parent")
    self.exploration_weight = kwargs.get(
      "exploration_weight", default_exploration_weight
    )
    self.max_children = kwargs.get("max_children", default_max_children)
    self.children = []
    self.visits = 0
    self.value = 0

  # 将一个新的答案（子节点）添加到当前节点下，并建立父子关系。
  def add_child(self, child: "Node"):
    child.parent = self
    self.children.append(child)
    return child

  # 如果当前节点的子节点数已达到预设上限，则认为它已被充分扩展。
  def fully_expanded(self):
    return len(self.children) >= self.max_children

  # 利用 UCT（上置信界）公式计算当前节点的探索值，既考虑节点本身的平均得分，也考虑探索未充分访问的节点。
  def uct_value(self):
    epsilon = 1e-6
    # 使用 (parent.visits + 1) 避免对数函数传入 0，确保数值稳定
    parent_visits = self.parent.visits if self.parent else 1
    return self.value / (self.visits +
                         epsilon) + self.exploration_weight * math.sqrt(
                           math.log(parent_visits) /
                           (self.visits + epsilon)
                         )

  # 递归生成当前节点及其所有子节点的 Mermaid 图描述，便于实时可视化搜索树状态。
  def mermaid(self, offset=0, selected=None):
    padding = " " * offset
    msg = f"{padding}{self.id}({self.id}:{self.visits} - {escape_mermaid(self.content[:25])})\n"

    if selected == self.id:
      msg += f"{padding}style {self.id} stroke:#0ff\n"

    for child in self.children:
      msg += child.mermaid(offset + 4, selected)
      msg += f"{padding}{self.id} --> {child.id}\n"

    return msg

  # 递归寻找访问次数最多（即最有可能正确）的答案。
  def best_child(self):
    if not self.children:
      return self

    return max(self.children, key=lambda child: child.visits).best_child()


"""
该类封装了 MCTS 算法的核心逻辑，通过以下步骤来生成和评估答案：
`选择（select）：从根节点开始沿着子节点路径前进，直到到达叶节点。
`扩展（expand）：对未充分扩展的叶节点生成新的子节点（即调用 LLM “思考”以生成新的答案变化）。
`模拟（simulate）：通过 LLM 对当前节点的答案进行评估，获得一个分数。
`反向传播（back propagate）：将评估得分沿树向上传播，更新每个节点的访问次数和累计得分。
"""
class MCTS:
  question: str
  root: Node
  llm: "Pipeline"
  selected: Optional[Node]
  exploration_weight: float

  def __init__(self, **kwargs):
    self.question = kwargs.get("question")
    self.root = kwargs.get("root")
    self.llm = kwargs.get("llm")
    self.selected = None
    self.exploration_weight = kwargs.get(
      "exploration_weight", default_exploration_weight
    )

  # 从根节点出发，一直选择 UCT 值最高的子节点，直至到达叶节点。
  async def select(self):
    logger.debug("Selecting node...")
    node = self.root
    # 如果当前节点未完全扩展，则停止选择，直接返回该叶节点进行扩展
    while node.fully_expanded() and node.children:
      node = self.uct_select(node)
    return node

  """
  对给定节点生成若干“思考”过程：
  调用 LLM 得到一个改进建议（通过 generate_thought）。
  利用建议更新当前答案（update_approach），形成新内容。
  为每次改进创建一个新子节点，并添加到当前节点下。
  最后返回随机挑选的一个子节点，作为扩展后的节点。
  """
  async def expand(self, node):
    logger.debug(f"Expanding node {node.id}...")
    await self.llm.progress(f"正在对节点 {node.id} 进行深度思考...")

    for _ in range(random.randint(default_thoughts, default_thoughts + 1)):
      await self.llm.emit_replace(self.mermaid(node))
      await self.llm.emit_message(f"思考: ")
      thought = await self.llm.generate_thought(node.content)
      await self.llm.emit_message(f"\n\n---\n\n解决:\n")

      new_content = await self.llm.update_approach(node.content, thought)
      child = Node(content=new_content, parent=node)
      node.add_child(child)

    return random.choice(node.children)

  # 调用 LLM 对当前节点（即某个答案）的质量进行评估，返回一个评分值。
  async def simulate(self, node):
    logger.debug(f"Simulating node {node.id}...")
    await self.llm.progress(f"正在对节点 {node.id} 进行打分......")
    await self.llm.emit_replace(self.mermaid())

    return await self.llm.evaluate_answer(node.content)

  # 从叶节点开始，逐级向上传播，将模拟评价的分数累加到每个节点，并增加访问次数，便于后续比较。
  def backpropagate(self, node, score):
    logger.debug(f"Backpropagating from {node.id}...")
    while node:
      node.visits += 1
      node.value += score
      node = node.parent

  # 在当前节点的所有子节点中，根据 UCT 值（综合考虑分数与探索）选择最佳的下一个节点。
  def uct_select(self, node):
    logger.debug(f"Selecting uct {node.id}...")
    return max(node.children, key=lambda child: child.uct_value())

  # 递归寻找访问次数最多（即最有可能正确）的答案。
  def best_child(self):
    return self.root.best_child()

  """
  过程：
  对每次模拟：
  选择一个叶节点。
  若未完全扩展则进行扩展（生成新思考）。
  对扩展后的节点进行模拟评价。
  将评价分数反向传播更新整棵树。
  """
  async def search(self, num_simulations):
    logger.debug("Starting search...")

    for _ in range(num_simulations):
      leaf = await self.select()
      self.selected = leaf
      if not leaf.fully_expanded():
        leaf = await self.expand(leaf)
      score = await self.simulate(leaf)
      self.backpropagate(leaf, score)

    return self.selected

  # 根据当前 MCTS 树生成一个 Mermaid 格式的图形描述，用于在前端界面上动态展示搜索过程。
  def mermaid(self, selected=None):
    return f"""
```mermaid
graph LR
{self.root.mermaid(0, selected.id if selected else self.selected.id)}
```
"""


# ==============================================================================

EventEmitter = Callable[[dict], Awaitable[None]]

### 从请求中提取模型标识符和用户问题，并返回当前可用的模型列表。
class Pipeline:
  # 通过事件发射器（__current_event_emitter__）逐步更新进度、发送“思考”过程、生成的 Mermaid 图以及最终答案，
  # 使用户可以直观地看到系统如何逐步改进回答。
  __current_event_emitter__: EventEmitter
  __current_node__: Node
  __question__: str
  __model__: str

  def __init__(self):
    self.client = None
    self.type = "manifold"
    self.name = "基于蒙特卡洛搜索树的迭代式检索增强"

  # 管道启动时立即执行的函数
  async def on_startup(self):
    logger.info("管道启动中...")
    self.client = OpenAI(
      api_key="Empty",  # 请更新为实际的 OpenAI API key
      base_url="http://172.18.117.251:8000/v1"
    )

  # 管道关闭的时候立即执行的函数
  async def on_shutdown(self):
    logger.info("管道正在关闭...")
    # 此处可以添加资源清理逻辑，如关闭连接、释放缓存等


  # 从请求中提取模型标识符和用户问题，并返回当前可用的模型列表。
  def resolve_model(self, body: dict) -> str:
    model_id = body.get("model")
    if not model_id:
        raise ValueError("请求正文中缺少模型标识符")
    parts = model_id.split(".")
    if len(parts) < 2:
        raise ValueError("模型标识符格式错误")
    without_pipe = ".".join(parts[1:])
    return without_pipe.replace(f"{name}-", "")


  # 从请求中提取模型标识符和用户问题，并返回当前可用的模型列表。
  def resolve_question(self, body: dict) -> str:
    messages = body.get("messages")
    # 检查 messages 是否存在且为列表
    if not messages or not isinstance(messages, list):
        raise ValueError("消息字段缺失或不是列表")
    last_message = messages[-1]
    # 检查最后一条消息是否为字典且包含 content 键
    if not isinstance(last_message, dict) or "content" not in last_message:
        raise ValueError("最后一条消息的格式不正确")
    return last_message.get("content", "").strip()


  """
  根据请求体获取问题，并调用 modify_text（这里设置的修改百分比为 0，即不做修改）。
  调用 stream_prompt_completion 以 initial_prompt 得到初始回答，作为 MCTS 树的根节点内容。
  构建一个 MCTS 实例，并进行多次（默认 default_max_iterations）搜索，每次内部模拟次数为 default_max_simulations。
  每轮搜索后调用 evaluate_answer 得到当前最佳答案的评分，最终选出整体最佳答案。
  通过 emit_replace 和 emit_message 将生成的 Mermaid 图和最终答案传递给前端界面。
  最后调用 done 表示流程结束。
  """

  async def pipe(
          self,
          user_message: str,
          model_id: str,
          messages: List[dict],
          body: dict
  ) -> Union[str, Generator, Iterator]:

    logger.debug(f"管道 {self.name} 收到请求: user_message={user_message}, model_id={model_id}")

    # 将传入参数赋值给实例变量，便于后续调用
    self.__current_event_emitter__ = body.get("event_emitter")
    self.__model__ = model_id
    self.__question__ = user_message

    # 针对特定任务（如标题生成）的处理
    if body.get("task") == TASKS.TITLE_GENERATION:
      content = await self.get_completion(model_id, messages)
      return f"{self.name}: {content}"

    # 对问题文本进行预处理
    question, mapping = modify_text(user_message, 0)
    logger.debug(f"Question: {question}")

    best_answer = None
    best_score = -float("inf")

    # 初始思路准备阶段
    await self.progress("正在进行初步思考...")
    initial_reply = await self.stream_prompt_completion(initial_prompt, question=question)

    # 初始化 MCTS 根节点与搜索树
    root = Node(content=initial_reply)
    mcts = MCTS(root=root, llm=self)

    logger.debug("Starting MCTS...")
    for i in range(default_max_iterations):
      logger.debug(f"Iteration {i + 1}/{default_max_iterations}...")
      await mcts.search(default_max_simulations)
      logger.debug(mcts.mermaid())

      best_child = mcts.best_child()
      score = await self.evaluate_answer(best_child.content)
      if score > best_score:
        best_score = score
        best_answer = best_child.content

    # 向前端传递最终生成的最佳答案与搜索树图谱
    await self.emit_replace(mcts.mermaid(best_child))
    await self.emit_message(f"{best_answer}")
    await asyncio.sleep(0.2)
    await self.done()

    return ""

  async def progress(
    self,
    message: str,
  ):
    logger.debug(f"Progress: {message}")
    await self.emit_status("info", message, False)

  async def done(self,):
    await self.emit_status("info", "Fin.", True)

  async def emit_message(self, message: str):
    await self.__current_event_emitter__(
      {
        "type": "message",
        "data": {
          "content": message
        }
      }
    )

  async def emit_replace(self, message: str):
    await self.__current_event_emitter__(
      {
        "type": "replace",
        "data": {
          "content": message
        }
      }
    )

  async def emit_status(self, level: str, message: str, done: bool):
    await self.__current_event_emitter__(
      {
        "type": "status",
        "data":
          {
            "status": "complete" if done else "in_progress",
            "level": level,
            "description": message,
            "done": done,
          },
      }
    )

  # get_streaming_completion、get_message_completion、get_completion 方法
  # 封装了调用 LLM 接口的逻辑，通过 ollama.generate_openai_chat_completion 模拟 HTTP 请求。
  async def get_streaming_completion(self, model: str, messages) -> AsyncGenerator[str, None]:
    response = await self.call_openai_endpoint_function({
      "model": model,
      "messages": messages,
      "stream": True
    })

    async for chunk in response:
      if "choices" in chunk:
        for choice in chunk["choices"]:
          content = choice.get("message", {}).get("content", "")
          if content:
            yield content

  async def get_message_completion(self, model: str, content):
    async for chunk in self.get_streaming_completion(
      model, [{
        "role": "user",
        "content": content
      }]
    ):
      yield chunk

  async def get_completion(self, model: str, messages):
    response = await self.call_openai_endpoint_function({
        "model": model,
        "messages": messages,
        "stream": False
    })
    # Extract the full response content from the first choice
    return response["choices"][0]["message"]["content"]

  # 访问 OpenAI 的接口
  async def call_openai_endpoint_function(self, payload):
    try:
        if payload.get("stream", False):
            response = await self.client.chat.completions.create(
                model=payload["model"],
                messages=payload["messages"],
                stream=True
            )
        else:
            response = await self.client.chat.completions.create(
                model=payload["model"],
                messages=payload["messages"],
                stream=False
            )
        return response
    except Exception as e:
        logger.error(f"Error calling OpenAI endpoint: {str(e)}")
        # 根据需求：这里可以重试、返回默认值或直接抛出异常
        raise



  async def stream_prompt_completion(self, prompt, **format_args):
    try:
        formatted_prompt = prompt.format(**format_args)
    # 在格式化模板字符串前增加 try/except 捕获，确保当格式化参数缺失时能够记录错误并及时处理，防止运行时错误。
    except KeyError as e:
        logger.error(f"Missing parameter in prompt formatting: {e}")
        raise

    complete = ""
    async for chunk in self.get_message_completion(self.__model__, formatted_prompt):
        complete += chunk
        await self.emit_message(chunk)
    return complete

  # 调用 thoughts_prompt 模板生成对当前答案的改进思考。
  async def generate_thought(self, answer):
    return await self.stream_prompt_completion(
      thoughts_prompt, answer=answer, question=self.__question__
    )

  # 用于分析本轮迭代结果
  async def analyze_iteration(self, best_answer, best_score):
    return await self.stream_prompt_completion(
      analyze_prompt,
      question=self.__question__,
      best_answer=best_answer,
      best_score=best_score
    )

  # 根据批评更新答案，使得整个搜索过程能够不断自我改进。
  async def update_approach(self, answer, improvements):
    return await self.stream_prompt_completion(
      update_prompt,
      question=self.__question__,
      answer=answer,
      improvements=improvements
    )

  # 调用 eval_answer_prompt 模板，请 LLM 给出一个 1 到 10 的评分，用于后续反向传播和选择最佳答案。
  async def evaluate_answer(self, answer):
    result = await self.stream_prompt_completion(
      eval_answer_prompt,
      answer=answer,
      question=self.__question__,
    )
    try:
      score = re.search(r"\d+", result).group()
      return int(score)
    except AttributeError:
      logger.error(f"AnswerEval: unable to parse \"{result[:100]}\"")
      return 0

  # 获取模型的返回结果的文本内容
  def get_response_content(self, response):
    try:
      return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
      logger.error(
        f"ResponseError: unable to extract content from \"{response[:100]}\""
      )
      return ""

  def get_chunk_content(self, chunk):
    chunk_str = chunk.decode("utf-8")
    if chunk_str.startswith("data: "):
      chunk_str = chunk_str[6:]

    chunk_str = chunk_str.strip()

    if chunk_str == "[DONE]" or not chunk_str:
      return

    try:
      chunk_data = json.loads(chunk_str)
      if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
        delta = chunk_data["choices"][0].get("delta", {})
        if "content" in delta:
          yield delta["content"]
    except json.JSONDecodeError:
      logger.error(f"ChunkDecodeError: unable to parse \"{chunk_str[:100]}\"")
