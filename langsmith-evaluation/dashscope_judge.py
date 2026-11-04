#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepEval 判官适配层：用 DashScope 的 qwen3.8-max-0902 替代 OpenAI

背景
----
DeepEval 内置模型覆盖 OpenAI / Azure / Ollama / vLLM / LocalModel 等，但不含 DashScope。
官方允许子类化 DeepEvalBaseLLM 接管判官模型，因此这里把 DashScope 的 OpenAI 兼容接口
包装成 DeepEval 可用的模型对象，无需 OpenAI 账号。

关键实现点
----------
1. `generate(prompt, schema=None)`：schema 由 DeepEval 内部通过
   `generate_with_schema()` 传入（GEval / TaskCompletion 等指标依赖结构化输出），
   这里用 json_object 模式 + pydantic 校验解析，失败时退化为从文本中截取 JSON。
2. `a_generate` 必须实现：`evaluate()` 批量模式走异步路径，只实现同步方法会退化或报错。
3. 统一 temperature=0、超时与重试，保证判官输出可复现；日志只打印模型名与 base_url，
   不打印 api_key。

用法
----
    from dashscope_judge import get_judge
    metric = AnswerRelevancyMetric(threshold=0.6, model=get_judge())
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional, Type

from deepeval.models.base_model import DeepEvalBaseLLM
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

# 判官默认 deepseek-v4.1-flash-1017（deepeval 全量评估实际所用模型）；
# qwen3.8-max-0902 的免费额度已耗尽（403 Free quota exhausted），保留说明仅作历史记录
DEFAULT_MODEL = os.getenv("JUDGE_MODEL", "deepseek-v4.1-flash-1017")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class DashScopeJudge(DeepEvalBaseLLM):
    """把 DashScope 的 OpenAI 兼容接口包装成 DeepEval 判官"""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        timeout: float = 300.0,
        max_retries: int = 4,
    ) -> None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("缺少 DASHSCOPE_API_KEY 环境变量，判官无法初始化")

        # 先赋值再调父类：DeepEvalBaseLLM.__init__ 内部会立刻调用 self.load_model()，
        # 顺序颠倒会导致 _api_key / _sync_client 尚未初始化（AttributeError）
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self._sync_client: Optional[OpenAI] = None
        self._async_client: Optional[AsyncOpenAI] = None
        self._api_key = api_key

        super().__init__(model=model_name)

        # 不打印 api_key，只打印模型与端点，便于排查配置
        print(f"[判官] 模型={self.model_name}  endpoint={BASE_URL}")

    # ---------- 客户端 ----------

    def _client(self) -> OpenAI:
        if self._sync_client is None:
            self._sync_client = OpenAI(
                api_key=self._api_key,
                base_url=BASE_URL,
                timeout=self.timeout,
                max_retries=0,  # 重试由本类控制，避免双层重试放大延迟
            )
        return self._sync_client

    def _aclient(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=BASE_URL,
                timeout=self.timeout,
                max_retries=0,
            )
        return self._async_client

    def load_model(self, *args: Any, **kwargs: Any) -> OpenAI:
        """DeepEval 约定的模型加载钩子"""
        return self._client()

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return self.model_name

    # ---------- 结构化输出解析 ----------

    @staticmethod
    def _build_messages(prompt: str, schema: Optional[Type[BaseModel]]) -> list[dict[str, str]]:
        messages = [{"role": "user", "content": prompt}]
        if schema is not None:
            # json_object 模式要求提示中出现 "json"，否则部分服务端会拒绝
            messages[0]["content"] = (
                f"{prompt}\n\n"
                "请严格按以下 JSON Schema 输出，只返回一个 JSON 对象，不要包含解释文字或代码块标记。\n"
                f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
            )
        return messages

    @staticmethod
    def _parse_schema(content: str, schema: Type[BaseModel]) -> BaseModel:
        """先尝试直接解析，失败则从文本中截取首个 JSON 对象"""
        try:
            return schema.model_validate_json(content)
        except Exception:
            match = re.search(r"\{.*\}", content, re.S)
            if match:
                return schema.model_validate(json.loads(match.group(0)))
            raise

    # ---------- 同步 ----------

    def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None) -> Any:
        kwargs: dict[str, Any] = {}
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self._client().chat.completions.create(
                    model=self.model_name,
                    messages=self._build_messages(prompt, schema),
                    temperature=0,
                    **kwargs,
                )
                content = response.choices[0].message.content or ""
                return self._parse_schema(content, schema) if schema is not None else content
            except Exception as exc:  # 网络抖动 / 限流 / 结构化解析失败
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2 * (attempt + 1))

        raise RuntimeError(f"判官调用失败（已重试 {self.max_retries} 次）: {last_error}")

    # ---------- 异步 ----------

    async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None) -> Any:
        import asyncio

        kwargs: dict[str, Any] = {}
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = await self._aclient().chat.completions.create(
                    model=self.model_name,
                    messages=self._build_messages(prompt, schema),
                    temperature=0,
                    **kwargs,
                )
                content = response.choices[0].message.content or ""
                return self._parse_schema(content, schema) if schema is not None else content
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))

        raise RuntimeError(f"判官异步调用失败（已重试 {self.max_retries} 次）: {last_error}")


_JUDGE: Optional[DashScopeJudge] = None


def get_judge(model_name: str = DEFAULT_MODEL) -> DashScopeJudge:
    """进程内单例，避免每个指标重复建客户端"""
    global _JUDGE
    if _JUDGE is None or _JUDGE.model_name != model_name:
        _JUDGE = DashScopeJudge(model_name=model_name)
    return _JUDGE
