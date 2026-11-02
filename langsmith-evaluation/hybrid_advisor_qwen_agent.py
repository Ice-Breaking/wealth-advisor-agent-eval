#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
混合智能体 - 财富管理投顾AI助手（Qwen-Agent 版）

基于 Qwen-Agent 框架实现的混合型智能体，结合反应式与深思熟虑两种处理模式：
- 反应式：快速响应简单查询，支持上证指数等工具调用
- 深思熟虑：通过深度投资分析工具完成复杂理财规划
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool, register_tool

# 配置 DashScope
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')
dashscope.timeout = 60

# ==================== 示例客户画像 ====================

SAMPLE_CUSTOMER_PROFILES = {
    "customer1": {
        "customer_id": "C10012345",
        "risk_tolerance": "平衡型",
        "investment_horizon": "中期",
        "financial_goals": ["退休规划", "子女教育金"],
        "investment_preferences": ["ESG投资", "科技行业"],
        "portfolio_value": 1500000.0,
        "current_allocations": {
            "股票": 0.40,
            "债券": 0.30,
            "现金": 0.10,
            "另类投资": 0.20
        }
    },
    "customer2": {
        "customer_id": "C10067890",
        "risk_tolerance": "进取型",
        "investment_horizon": "长期",
        "financial_goals": ["财富增长", "资产配置多元化"],
        "investment_preferences": ["新兴市场", "高成长行业"],
        "portfolio_value": 3000000.0,
        "current_allocations": {
            "股票": 0.65,
            "债券": 0.15,
            "现金": 0.05,
            "另类投资": 0.15
        }
    }
}

# 当前会话绑定的客户画像，供工具内部读取
_current_customer_profile: Dict[str, Any] = SAMPLE_CUSTOMER_PROFILES["customer1"]

# ==================== 提示模板 ====================

DATA_COLLECTION_PROMPT = """你是一个财富管理投顾AI助手的数据收集模块。基于以下用户查询，确定需要收集哪些市场和财务数据进行深入分析。

用户查询: {user_query}

客户信息:
{customer_profile}

请确定需要收集的数据类型，例如:
- 资产类别表现数据
- 经济指标
- 行业趋势
- 历史回报率
- 风险指标
- 税收信息
- 其他相关数据

以JSON格式返回结果，包含以下字段:
- required_data_types: 需要收集的数据类型列表
- data_sources: 建议的数据来源列表
- collected_data: 模拟收集的数据（为简化示例，请生成合理的模拟数据）
"""

ANALYSIS_PROMPT = """你是一个财富管理投顾AI助手的分析引擎。请根据收集的数据对用户的投资情况进行深入分析。

用户查询: {user_query}

客户信息:
{customer_profile}

市场数据:
{market_data}

请提供全面的投资分析，包括:
1. 当前市场状况评估
2. 客户投资组合分析
3. 个性化投资建议
4. 风险评估
5. 预期结果和回报预测

以JSON格式返回分析结果，包含以下字段:
- market_assessment: 市场评估
- portfolio_analysis: 投资组合分析
- recommendations: 投资建议列表
- risk_analysis: 风险分析
- expected_outcomes: 预期结果
"""

RECOMMENDATION_PROMPT = """你是一个财富管理投顾AI助手。请根据深入分析结果，为客户准备最终的咨询建议。

用户查询: {user_query}

客户信息:
{customer_profile}

分析结果:
{analysis_results}

请提供专业、个性化且详细的投资建议，语言应友好易懂，避免过多专业术语。建议应包括:
1. 总体投资策略
2. 具体行动步骤
3. 资产配置建议
4. 风险管理策略
5. 时间框架
6. 预期收益
7. 后续跟进计划

返回格式应为自然语言文本，适合直接呈现给客户。
"""


def _call_llm(prompt: str) -> str:
    """调用通义千问模型"""
    response = dashscope.Generation.call(
        model='qwen-flash',
        messages=[{'role': 'user', 'content': prompt}],
        result_format='message',
    )
    if response.status_code != 200:
        raise RuntimeError(f"LLM 调用失败: {response.message}")
    return response.output.choices[0].message.content


def _parse_json_from_text(text: str) -> Dict[str, Any]:
    """从 LLM 返回文本中提取 JSON"""
    text = text.strip()
    if text.startswith('```'):
        lines = text.split('\n')
        text = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        raise ValueError("未找到有效的 JSON 内容")
    return json.loads(text[start:end + 1])


def build_system_prompt(customer_profile: Dict[str, Any]) -> str:
    """构建包含客户画像的系统提示"""
    profile_str = json.dumps(customer_profile, ensure_ascii=False, indent=2)
    return f"""你是财富管理投顾AI助手，采用混合智能体架构，能够根据问题复杂度选择不同处理方式。

当前客户画像:
{profile_str}

## 处理模式

1. **反应式处理**（简单/紧急查询）
   - 适用：指数行情、概念解释、账户信息等
   - 方式：直接回答，或调用 query_shanghai_index 工具查询上证指数

2. **深思熟虑处理**（复杂分析查询）
   - 适用：投资组合优化、退休规划、长期理财方案等
   - 方式：必须调用 deep_investment_analysis 工具，传入用户原始问题

## 工具使用规则

- 用户询问上证指数、大盘行情时，调用 query_shanghai_index
- 用户需要深度投资分析或长期规划时，调用 deep_investment_analysis，参数 user_query 为用户完整问题
- 回答应专业、清晰，并充分考虑当前客户的风险偏好与投资目标
"""


# ==================== 工具注册 ====================

@register_tool('query_shanghai_index')
class QueryShanghaiIndexTool(BaseTool):
    """上证指数实时查询工具"""
    description = '查询上证指数最新行情数据，适用于大盘走势、指数点位等简单查询'
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        name = "上证指数"
        price = "3125.62"
        change = "6.32"
        pct = "0.20"
        result = f"{name} 当前点位: {price}，涨跌: {change}，涨跌幅: {pct}%"
        print('query_shanghai_index result=', result)
        return result


@register_tool('deep_investment_analysis')
class DeepInvestmentAnalysisTool(BaseTool):
    """深度投资分析工具，整合数据收集、分析与建议生成"""
    description = (
        '对复杂投资问题进行深度分析，包括市场评估、组合分析、风险评估和个性化建议。'
        '适用于投资组合优化、退休规划、长期理财方案等需要深思熟虑的查询。'
    )
    parameters = [{
        'name': 'user_query',
        'type': 'string',
        'description': '用户的完整投资咨询问题',
        'required': True,
    }]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        user_query = args['user_query']
        customer_profile = _current_customer_profile
        profile_str = json.dumps(customer_profile, ensure_ascii=False, indent=2)

        print('[DEBUG] 进入深度分析: 数据收集')
        data_prompt = DATA_COLLECTION_PROMPT.format(
            user_query=user_query,
            customer_profile=profile_str,
        )
        data_result = _parse_json_from_text(_call_llm(data_prompt))
        market_data = data_result.get('collected_data', {})

        print('[DEBUG] 进入深度分析: 投资分析')
        analysis_prompt = ANALYSIS_PROMPT.format(
            user_query=user_query,
            customer_profile=profile_str,
            market_data=json.dumps(market_data, ensure_ascii=False, indent=2),
        )
        analysis_results = _parse_json_from_text(_call_llm(analysis_prompt))

        print('[DEBUG] 进入深度分析: 生成建议')
        recommend_prompt = RECOMMENDATION_PROMPT.format(
            user_query=user_query,
            customer_profile=profile_str,
            analysis_results=json.dumps(analysis_results, ensure_ascii=False, indent=2),
        )
        return _call_llm(recommend_prompt)


# ==================== 助手初始化 ====================

def init_agent_service(customer_id: str = "customer1") -> Assistant:
    """初始化财富管理投顾助手"""
    global _current_customer_profile
    customer_profile = SAMPLE_CUSTOMER_PROFILES.get(
        customer_id, SAMPLE_CUSTOMER_PROFILES["customer1"]
    )
    _current_customer_profile = customer_profile

    llm_cfg = {
        'model': 'qwen-flash',
        'timeout': 60,
        'retry_count': 3,
    }
    function_list = ['query_shanghai_index', 'deep_investment_analysis']

    bot = Assistant(
        llm=llm_cfg,
        name='财富管理投顾助手',
        description='提供智能化财富管理咨询，支持快速响应与深度投资分析',
        system_message=build_system_prompt(customer_profile),
        function_list=function_list,
    )
    print("助手初始化成功！")
    return bot


def _extract_final_response(messages: List[Dict[str, Any]]) -> str:
    """从对话消息中提取最终助手回复"""
    for msg in reversed(messages):
        if msg.get('role') == 'assistant':
            content = msg.get('content', '')
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                texts = [item.get('text', '') for item in content if isinstance(item, dict)]
                combined = '\n'.join(t for t in texts if t)
                if combined.strip():
                    return combined.strip()
    return ""


def _detect_processing_mode(messages: List[Dict[str, Any]]) -> str:
    """根据工具调用记录判断处理模式"""
    for msg in messages:
        if msg.get('role') == 'assistant':
            fn_call = msg.get('function_call')
            if isinstance(fn_call, dict):
                name = fn_call.get('name', '')
                if name == 'deep_investment_analysis':
                    return 'deliberative'
                if name == 'query_shanghai_index':
                    return 'reactive'
        if msg.get('role') == 'function':
            name = msg.get('name', '')
            if name == 'deep_investment_analysis':
                return 'deliberative'
            if name == 'query_shanghai_index':
                return 'reactive'
    return 'reactive'


def run_wealth_advisor(user_query: str, customer_id: str = "customer1") -> Dict[str, Any]:
    """运行财富顾问智能体并返回结果（供程序化调用）"""
    customer_profile = SAMPLE_CUSTOMER_PROFILES.get(
        customer_id, SAMPLE_CUSTOMER_PROFILES["customer1"]
    )
    initial_state = {
        "user_query": user_query,
        "customer_profile": customer_profile,
        "processing_mode": None,
        "final_response": None,
        "error": None,
    }

    try:
        bot = init_agent_service(customer_id)
        messages = [{'role': 'user', 'content': user_query}]
        all_messages = list(messages)

        for response in bot.run(messages):
            all_messages.extend(response)

        final_response = _extract_final_response(all_messages)
        processing_mode = _detect_processing_mode(all_messages)

        return {
            **initial_state,
            "processing_mode": processing_mode,
            "final_response": final_response or "未能生成响应，请重试。",
        }
    except Exception as e:
        error_msg = str(e)
        print(f"捕获异常: {error_msg}")
        return {
            **initial_state,
            "error": f"执行过程中发生错误: {error_msg}",
            "final_response": "很抱歉，处理您的请求时出现了问题。",
        }


# ==================== 交互入口 ====================

def app_tui(customer_id: str = "customer1"):
    """终端交互模式"""
    try:
        bot = init_agent_service(customer_id)
        profile = SAMPLE_CUSTOMER_PROFILES.get(customer_id, SAMPLE_CUSTOMER_PROFILES["customer1"])
        print(f"当前客户: {profile['risk_tolerance']} 投资者 ({profile['customer_id']})")
        print("输入 quit 退出\n")

        messages = []
        while True:
            try:
                query = input('请输入咨询问题: ').strip()
                if query.lower() in ('quit', 'exit', 'q'):
                    break
                if not query:
                    print('问题不能为空')
                    continue

                messages.append({'role': 'user', 'content': query})
                print("正在处理...\n")

                for response in bot.run(messages):
                    for msg in response:
                        role = msg.get('role', '')
                        content = msg.get('content', '')
                        if role == 'assistant' and content:
                            print(f"助手: {content}\n")
                    messages.extend(response)
            except Exception as e:
                print(f"处理请求时出错: {str(e)}")
    except Exception as e:
        print(f"启动终端模式失败: {str(e)}")


def app_gui(customer_id: str = "customer1"):
    """Web 图形界面模式"""
    try:
        print("正在启动 Web 界面...")
        bot = init_agent_service(customer_id)
        chatbot_config = {
            'prompt.suggestions': [
                '今天上证指数的表现如何？',
                '请解释一下什么是ETF？',
                '根据当前市场情况，我应该如何调整投资组合以应对可能的经济衰退？',
                '考虑到我的退休目标，请评估我当前的投资策略并提供优化建议。',
            ]
        }
        print("Web 界面准备就绪，正在启动服务...")
        WebUI(bot, chatbot_config=chatbot_config).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")


if __name__ == '__main__':
    # 默认启动 Web 图形界面
    app_gui()
