#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
纯文本 JSON 判官包装器

背景：qwen3.8-max-0902 的免费额度耗尽后，剩下的 deepseek / glm 模型在
LangChain 的 `with_structured_output`（function calling / response_format=json_schema）
路径上不稳定：实测 glm-5.3 报 `list indices must be integers`、
deepseek-v4.1-flash 报 400、deepseek-v4-pro 解析失败被计 0 分。

做法：子类化 ChatOpenAI，把 `with_structured_output` 换成"让模型输出纯 JSON 文本，
再自己解析"，从而绕开厂商的结构化输出实现差异。schema 只用于告知模型要输出的字段。
"""

import json
import re
from typing import Any, Optional

from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI


def _extract_json(text: str) -> Optional[dict]:
    """从模型输出里抠出第一个 JSON 对象；失败则用正则兜底取 score"""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    match = re.search(r'"?score"?\s*[:=]\s*(-?\d+(?:\.\d+)?)', text)
    if match:
        return {"score": float(match.group(1))}
    return None


def _normalize_score(value: Any) -> Optional[float]:
    """把分数压到 0-1；模型偶尔会给 0-100"""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score > 1.0:
        score = score / 100.0 if score <= 100.0 else 1.0
    return max(0.0, min(1.0, score))


class JsonTextJudge(ChatOpenAI):
    """用纯文本 JSON 输出替代厂商结构化输出的判官模型"""

    def with_structured_output(self, schema, **kwargs):
        llm = self
        schema_text = json.dumps(schema, ensure_ascii=False) if isinstance(schema, dict) else str(schema)

        def _run(messages, config=None):
            if hasattr(messages, "to_messages"):
                messages = messages.to_messages()
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]

            instruction = (
                "\n\n请严格按要求只输出一个 JSON 对象，不要代码块、不要解释。"
                f"\n必须符合该 JSON Schema：{schema_text}"
                "\n如果给出连续分数，请使用 0 到 1 之间的小数。"
            )
            payload = list(messages) + [{"role": "user", "content": instruction}]

            response = llm.invoke(payload, config=config)
            data = _extract_json(str(response.content))
            if not data:
                raise ValueError(f"判官未返回可解析的 JSON：{str(response.content)[:120]}")

            if "score" in data:
                normalized = _normalize_score(data["score"])
                if normalized is None:
                    raise ValueError(f"判官返回的 score 非法：{data.get('score')}")
                data["score"] = normalized
            return data

        return RunnableLambda(_run)


def make_judge(model: str, timeout: float = 180.0, temperature: float = 0.0):
    """构造判官实例（走 DashScope OpenAI 兼容接口）"""
    import os

    return JsonTextJudge(
        model=model,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=temperature,
        timeout=timeout,
    )
