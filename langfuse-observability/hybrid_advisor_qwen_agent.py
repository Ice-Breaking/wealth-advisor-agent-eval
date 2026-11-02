#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
混合智能体（Hybrid Agent）- 财富管理投顾AI助手

基于 qwen-agent 实现的混合型智能体，结合反应式架构的即时响应能力和深思熟虑架构的长期规划能力，
通过智能协调动态切换处理模式，提供智能化财富管理咨询服务。

三层架构：
1. 底层（反应式）：即时响应客户查询，提供快速反馈
2. 中层（协调）：评估任务类型和优先级，动态选择处理模式
3. 顶层（深思熟虑）：进行复杂的投资分析和长期财务规划
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool, register_tool

# 设置API密钥
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
dashscope.api_key = DASHSCOPE_API_KEY
dashscope.timeout = 30

# 示例客户画像数据
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

# ====== 工具定义 ======

@register_tool('query_shanghai_index')
class QueryShanghaiIndexTool(BaseTool):
    """
    上证指数实时查询工具（模拟版），返回固定的行情数据
    """
    description = '查询上证指数的最新行情数据，包括当前点位、涨跌和涨跌幅'
    parameters = []

    def call(self, params: str, **kwargs) -> str:
        """执行上证指数查询"""
        # 解析参数（即使没有参数也需要处理）
        try:
            if params:
                import json
                args = json.loads(params)
        except:
            pass
        
        # 直接返回模拟数据，避免外部API不可用导致报错
        name = "上证指数"
        price = "3125.62"
        change = "6.32"
        pct = "0.20"
        result = f"{name} 当前点位: {price}，涨跌: {change}，涨跌幅: {pct}%"
        print('工具调用结果:', result)
        return result

# ====== 系统提示词 ======

def get_system_prompt(customer_profile: Dict[str, Any]) -> str:
    """根据客户画像生成系统提示词"""
    customer_info = json.dumps(customer_profile, ensure_ascii=False, indent=2)
    
    system_prompt = f"""你是一个专业的财富管理投顾AI助手，具备混合智能处理能力。

## 客户信息
{customer_info}

## 你的能力

你具备两种处理模式，会根据查询类型自动选择：

### 1. 反应式处理模式（Reactive Mode）
适用于需要快速响应的查询，如：
- 市场状况查询（如"今天上证指数如何？"）
- 账户信息查询（如"我的投资组合中科技股占比是多少？"）
- 产品信息查询（如"请解释一下什么是ETF？"）
- 简单的数据查询

处理方式：
- 直接回答用户问题
- 如需市场数据，使用 query_shanghai_index 工具查询
- 提供简洁、准确的回答
- 可提供相关的关键数据点和建议操作

### 2. 深思熟虑处理模式（Deliberative Mode）
适用于需要深度分析的查询，如：
- 投资组合优化建议
- 长期理财规划
- 风险评估和应对策略
- 资产配置调整建议
- 市场趋势分析和投资策略

处理方式：
- 首先分析当前市场状况
- 评估客户的投资组合
- 考虑客户的风险承受能力、投资期限和财务目标
- 提供全面的投资分析，包括：
  * 市场评估
  * 投资组合分析
  * 个性化投资建议
  * 风险评估
  * 预期结果和回报预测
- 提供专业、详细、个性化的建议

## 工作流程

1. **评估阶段**：分析用户查询，判断查询类型
   - 如果是紧急/简单查询 → 使用反应式处理
   - 如果是分析性查询 → 使用深思熟虑处理

2. **处理阶段**：
   - 反应式：快速响应，必要时调用工具
   - 深思熟虑：进行深度分析，提供全面建议

3. **响应阶段**：根据处理结果，生成专业、友好的回答

## 回答要求

- 语言友好易懂，避免过多专业术语
- 提供具体、可操作的建议
- 考虑客户的个性化需求
- 对于分析性查询，提供：
  * 总体投资策略
  * 具体行动步骤
  * 资产配置建议
  * 风险管理策略
  * 时间框架
  * 预期收益
  * 后续跟进计划

## 工具使用

- query_shanghai_index: 查询上证指数行情，当用户询问市场指数时使用

请根据用户查询，智能选择处理模式，提供专业的财富管理咨询服务。"""
    
    return system_prompt

# ====== 初始化智能体 ======

def init_wealth_advisor_agent(customer_profile: Dict[str, Any]) -> Assistant:
    """初始化财富顾问智能体"""
    llm_cfg = {
        'model': 'qwen-flash',
        'timeout': 30,
        'retry_count': 3,
    }
    
    try:
        agent = Assistant(
            llm=llm_cfg,
            name='财富管理投顾AI助手',
            description='混合智能体财富管理咨询服务',
            system_message=get_system_prompt(customer_profile),
            function_list=['query_shanghai_index'],
        )
        print("财富顾问智能体初始化成功！")
        return agent
    except Exception as e:
        print(f"智能体初始化失败: {str(e)}")
        raise

# ====== 运行智能体 ======

def run_wealth_advisor(user_query: str, customer_id: str = "customer1") -> Dict[str, Any]:
    """运行财富顾问智能体并返回结果"""
    
    # 获取客户画像
    customer_profile = SAMPLE_CUSTOMER_PROFILES.get(customer_id, SAMPLE_CUSTOMER_PROFILES["customer1"])
    
    # 初始化智能体
    agent = init_wealth_advisor_agent(customer_profile)
    
    # 准备消息
    messages = [{'role': 'user', 'content': user_query}]
    
    try:
        # 运行智能体并收集响应
        response_messages = []
        for response in agent.run(messages):
            response_messages = response  # 每次迭代都会更新为最新的响应消息列表
        
        # 提取最终响应
        final_response = ""
        for msg in response_messages:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                if isinstance(content, str):
                    final_response += content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get('text'):
                                final_response += item['text']
                            elif item.get('content'):
                                final_response += item['content']
                        elif isinstance(item, str):
                            final_response += item
        
        return {
            "user_query": user_query,
            "customer_profile": customer_profile,
            "final_response": final_response if final_response else "未能生成响应",
            "response_messages": response_messages
        }
    except Exception as e:
        error_msg = str(e)
        print(f"执行过程中发生错误: {error_msg}")
        return {
            "user_query": user_query,
            "customer_profile": customer_profile,
            "error": f"执行过程中发生错误: {error_msg}",
            "final_response": "很抱歉，处理您的请求时出现了问题。"
        }

# ====== GUI 模式 ======

def app_gui(customer_id: str = "customer1"):
    """图形界面模式，提供 Web 图形界面"""
    try:
        print("正在启动 Web 界面...")
        
        # 获取客户画像
        customer_profile = SAMPLE_CUSTOMER_PROFILES.get(customer_id, SAMPLE_CUSTOMER_PROFILES["customer1"])
        
        # 初始化智能体
        agent = init_wealth_advisor_agent(customer_profile)
        
        # 配置聊天界面，列举典型财富管理查询问题
        chatbot_config = {
            'prompt.suggestions': [
                '今天上证指数的表现如何？',
                '我的投资组合中科技股占比是多少？',
                '根据当前市场情况，我应该如何调整投资组合以应对可能的经济衰退？',
                '考虑到我的退休目标，请评估我当前的投资策略并提供优化建议。',
                '我想为子女准备教育金，请帮我设计一个10年期的投资计划。',
            ]
        }
        
        print(f"客户类型: {customer_profile['risk_tolerance']} 投资者")
        print("Web 界面准备就绪，正在启动服务...")
        
        # 启动 Web 界面
        WebUI(
            agent,
            chatbot_config=chatbot_config
        ).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")
        print("请检查网络连接和 API Key 配置")

# ====== TUI 模式 ======

def app_tui():
    """终端交互模式，提供命令行交互界面"""
    try:
        print("=== 混合智能体 - 财富管理投顾AI助手 ===\n")
        print("使用模型：qwen-flash")
        print("框架：qwen-agent\n")
        print("-"*50 + "\n")
        
        # 选择客户
        customer_id = "customer1"  # 默认客户
        customer_choice = input("选择客户 (1: 平衡型投资者, 2: 进取型投资者，默认1): ").strip()
        if customer_choice == "2":
            customer_id = "customer2"
        
        # 获取客户画像
        customer_profile = SAMPLE_CUSTOMER_PROFILES.get(customer_id, SAMPLE_CUSTOMER_PROFILES["customer1"])
        
        # 初始化智能体
        agent = init_wealth_advisor_agent(customer_profile)
        
        print(f"\n客户类型: {customer_profile['risk_tolerance']} 投资者")
        print("已就绪，请输入您的问题（输入 'quit' 或 'exit' 退出）\n")
        
        # 对话历史
        messages = []
        while True:
            try:
                # 获取用户输入
                query = input('用户问题: ').strip()
                
                # 退出命令
                if query.lower() in ['quit', 'exit', '退出']:
                    print("再见！")
                    break
                
                # 输入验证
                if not query:
                    print('问题不能为空！')
                    continue
                
                # 构建消息
                messages.append({'role': 'user', 'content': query})
                
                print("正在处理您的请求...")
                
                # 运行助手并处理响应
                response = []
                for response in agent.run(messages):
                    print('助手回复:', response)
                
                messages.extend(response)
                print()  # 空行分隔
                
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"处理请求时出错: {str(e)}")
                print("请重试或输入新的问题\n")
                
    except Exception as e:
        print(f"启动终端模式失败: {str(e)}")

# ====== 主函数 ======

if __name__ == "__main__":
    # 运行模式选择
    print("=== 混合智能体 - 财富管理投顾AI助手 ===\n")
    print("请选择运行模式:")
    print("1. Web 图形界面 (GUI) - 推荐")
    print("2. 终端交互界面 (TUI)")
    
    mode_choice = input("\n请输入选项数字(1-2，默认1): ").strip()
    
    if mode_choice == "2":
        # 终端交互模式
        app_tui()
    else:
        # Web 图形界面模式（默认）
        # 选择客户
        customer_id = "customer1"  # 默认客户
        customer_choice = input("\n选择客户 (1: 平衡型投资者, 2: 进取型投资者，默认1): ").strip()
        if customer_choice == "2":
            customer_id = "customer2"
        
        app_gui(customer_id)

