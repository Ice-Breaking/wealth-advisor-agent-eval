#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
混合智能体（Hybrid Agent）- 财富管理投顾AI助手（集成 LangFuse 监测）

基于 qwen-agent 实现的混合型智能体，结合反应式架构的即时响应能力和深思熟虑架构的长期规划能力，
通过智能协调动态切换处理模式，提供智能化财富管理咨询服务。

集成 LangFuse 用于 LLM 调用监测、追踪和调试。

LangFuse 配置：
1. 安装：pip install langfuse
2. 设置环境变量：
   - LANGFUSE_PUBLIC_KEY: LangFuse 公钥
   - LANGFUSE_SECRET_KEY: LangFuse 私钥
   - LANGFUSE_BASE_URL: LangFuse 服务地址（可选，默认 https://cloud.langfuse.com）

三层架构：
1. 底层（反应式）：即时响应客户查询，提供快速反馈
2. 中层（协调）：评估任务类型和优先级，动态选择处理模式
3. 顶层（深思熟虑）：进行复杂的投资分析和长期财务规划
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
# 注意：WebUI 只在 GUI 模式下按需导入（qwen_agent.gui 依赖 gradio/modelscope_studio，
# 命令行调用不需要，缺失时不应阻塞模块加载）

# ====== LangFuse 导入和初始化 ======

# 尝试导入 LangFuse（可选，用于LLM调用监测）
try:
    from langfuse import Langfuse, observe, get_client
    HAS_LANGFUSE = True
except ImportError:
    HAS_LANGFUSE = False
    observe = None
    get_client = None
    print("  [提示] LangFuse 未安装，LLM调用监测功能将不可用")
    print("        安装命令: pip install langfuse")

# 初始化 LangFuse（如果可用）
langfuse_client = None
if HAS_LANGFUSE:
    # 抑制 OpenTelemetry 导出错误日志（避免网络超时等错误干扰主流程）
    otel_logger = logging.getLogger("opentelemetry")
    otel_logger.setLevel(logging.CRITICAL)  # 只显示严重错误
    
    langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_base_url = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    
    if langfuse_secret_key and langfuse_public_key:
        try:
            # 移除密钥中的引号（如果用户不小心加了引号）
            langfuse_secret_key = langfuse_secret_key.strip('"\'')
            langfuse_public_key = langfuse_public_key.strip('"\'')
            
            langfuse_client = Langfuse(
                secret_key=langfuse_secret_key,
                public_key=langfuse_public_key,
                host=langfuse_base_url,
                flush_at=1,  # 每次事件后立即发送（默认是批量发送）
                flush_interval=1,  # 每秒检查一次（默认是10秒）
                # 跨区域访问（如 US 区域）延迟较高，默认 5 秒容易读超时导致事件丢弃
                timeout=30,
            )
            print(f"  [提示] LangFuse 监测已启用 (Host: {langfuse_base_url})")
            print(f"  [提示] LangFuse 导出错误将被静默处理，不影响主流程")
        except Exception as e:
            print(f"  [警告] LangFuse 初始化失败: {e}")
            langfuse_client = None
    else:
        print("  [提示] LangFuse 密钥未配置，跳过监测")
        print("        要启用监测，请设置环境变量:")
        print("        - LANGFUSE_PUBLIC_KEY")
        print("        - LANGFUSE_SECRET_KEY")

# 设置API密钥
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
dashscope.api_key = DASHSCOPE_API_KEY
dashscope.timeout = 30

# 被测模型：可通过 AGENT_MODEL 环境变量切换
# 默认 deepseek-v4-pro-0813（额度耗尽史：qwen 系、flash-1017 均 403）
AGENT_MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-pro-0813")

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

# ====== LangFuse 追踪包装类 ======

class TracedAssistant(Assistant):
    """带 LangFuse 追踪的 Assistant 包装类"""
    
    def __init__(self, *args, customer_id: str = "customer1", 
                 customer_profile: Dict[str, Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.customer_id = customer_id
        self.customer_profile = customer_profile or {}
    
    def run(self, messages, **kwargs):
        """重写 run 方法，添加 LangFuse 追踪"""
        print("[DEBUG] TracedAssistant.run() 被调用")
        print(f"[DEBUG] langfuse_client={langfuse_client}, observe={observe}")
        
        if langfuse_client and observe:
            try:
                # 提取用户查询用于记录
                user_query = ""
                if messages:
                    if isinstance(messages, list) and len(messages) > 0:
                        last_msg = messages[-1]
                        if isinstance(last_msg, dict):
                            user_query = last_msg.get('content', '')
                        elif isinstance(last_msg, str):
                            user_query = last_msg
                    elif isinstance(messages, str):
                        user_query = messages
                
                print(f"[DEBUG] 提取的用户查询: {user_query[:100] if user_query else '空'}")
                
                # 使用 observe 装饰器追踪（LangFuse 3.x）
                @observe(name="gui_query", as_type="generation")
                def _traced_run():
                    print("[DEBUG] _traced_run() 函数开始执行")
                    
                    # 设置 input 和 metadata
                    input_set = False
                    if get_client:
                        try:
                            langfuse = get_client()
                            print(f"[DEBUG] get_client() 返回: {langfuse}")
                            if langfuse:
                                input_value = user_query if user_query else str(messages)[:500]
                                print(f"[DEBUG] 准备设置 input: {input_value[:100]}")
                                langfuse.update_current_generation(
                                    input=input_value,
                                    model=AGENT_MODEL,
                                    metadata={
                                        "customer_id": self.customer_id,
                                        "risk_tolerance": self.customer_profile.get("risk_tolerance", "unknown"),
                                        "investment_horizon": self.customer_profile.get("investment_horizon", "unknown"),
                                        "portfolio_value": self.customer_profile.get("portfolio_value", 0),
                                        "query_length": len(user_query) if user_query else 0,
                                        "mode": "gui"
                                    }
                                )
                                input_set = True
                                print("[DEBUG] input 和 metadata 设置成功")
                        except Exception as e:
                            print(f"[DEBUG] 设置 input/metadata 时出错: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # 调用原始的 run 方法（直接调用父类方法，避免 super() 在装饰器中的问题）
                    print("[DEBUG] 准备调用 Assistant.run()")
                    result_generator = Assistant.run(self, messages, **kwargs)
                    print("[DEBUG] Assistant.run() 返回生成器")
                    
                    # 收集所有响应
                    all_responses = []
                    try:
                        print("[DEBUG] 开始收集响应...")
                        for idx, response in enumerate(result_generator):
                            print(f"[DEBUG] 收到响应 #{idx+1}: {type(response)}, 长度: {len(str(response))}")
                            all_responses.append(response)
                        print(f"[DEBUG] 响应收集完成，共 {len(all_responses)} 个响应")
                    except Exception as e:
                        print(f"[DEBUG] 收集响应时出错: {e}")
                        import traceback
                        traceback.print_exc()
                        # 如果出错，记录错误到 LangFuse
                        if get_client:
                            try:
                                langfuse = get_client()
                                if langfuse:
                                    langfuse.update_current_span(
                                        output=None,
                                        level="ERROR",
                                        status_message=str(e)
                                    )
                                    langfuse.flush()
                                    print(f"[DEBUG] 错误已记录到 LangFuse: {e}")
                            except Exception as e2:
                                print(f"[DEBUG] 记录错误到 LangFuse 时失败: {e2}")
                        raise
                    
                    # 提取最终响应内容
                    final_output = ""
                    print(f"[DEBUG] 开始提取最终响应，all_responses 长度: {len(all_responses)}")
                    if all_responses:
                        last_response = all_responses[-1]
                        print(f"[DEBUG] 最后一个响应类型: {type(last_response)}")
                        print(f"[DEBUG] 最后一个响应内容预览: {str(last_response)[:200]}")
                        
                        if isinstance(last_response, list):
                            for msg_idx, msg in enumerate(last_response):
                                print(f"[DEBUG] 消息 #{msg_idx}: {type(msg)}, role={msg.get('role') if isinstance(msg, dict) else 'N/A'}")
                                if isinstance(msg, dict) and msg.get('role') == 'assistant':
                                    content = msg.get('content', '')
                                    print(f"[DEBUG] 助手消息内容类型: {type(content)}")
                                    if isinstance(content, str):
                                        final_output += content
                                    elif isinstance(content, list):
                                        for item in content:
                                            if isinstance(item, dict) and item.get('text'):
                                                final_output += item['text']
                                            elif isinstance(item, str):
                                                final_output += item
                        elif isinstance(last_response, str):
                            final_output = last_response
                    
                    print(f"[DEBUG] 提取的最终输出长度: {len(final_output)}")
                    print(f"[DEBUG] 最终输出预览: {final_output[:200] if final_output else '空'}")
                    
                    # 设置 output
                    output_set = False
                    if get_client:
                        try:
                            langfuse = get_client()
                            if langfuse:
                                output_value = final_output if final_output else "响应已生成"
                                print(f"[DEBUG] 准备设置 output: {output_value[:100]}")
                                langfuse.update_current_generation(
                                    output=output_value,
                                    metadata={
                                        "response_length": len(final_output),
                                        "response_count": len(all_responses)
                                    }
                                )
                                output_set = True
                                print("[DEBUG] output 设置成功")
                                langfuse.flush()
                                print("[DEBUG] LangFuse flush() 完成")
                        except Exception as e:
                            print(f"[DEBUG] 设置 output 时出错: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    print(f"[DEBUG] _traced_run() 完成，input_set={input_set}, output_set={output_set}")
                    # 返回所有响应（作为列表返回，observe 装饰器会自动处理）
                    return all_responses
                
                # 调用追踪函数并返回生成器
                print("[DEBUG] 准备调用 _traced_run()")
                all_responses = _traced_run()
                print(f"[DEBUG] _traced_run() 返回，响应数量: {len(all_responses) if isinstance(all_responses, list) else 'N/A'}")
                
                # 返回生成器（重新生成）
                for response in all_responses:
                    yield response
                
            except Exception as e:
                print(f"[DEBUG] LangFuse 追踪外层异常: {e}")
                import traceback
                traceback.print_exc()
                print(f"  [警告] LangFuse 追踪失败: {e}，继续执行")
                # 本方法是生成器，必须用 yield from 转发父类产出；
                # 若写 return，返回的生成器对象会被丢弃，表现为无响应
                yield from super().run(messages, **kwargs)
        else:
            print("[DEBUG] LangFuse 未启用，使用原始 run 方法")
            yield from super().run(messages, **kwargs)

# ====== 初始化智能体 ======

def init_wealth_advisor_agent(customer_profile: Dict[str, Any], 
                              customer_id: str = "customer1",
                              enable_tracing: bool = True) -> Assistant:
    """初始化财富顾问智能体"""
    llm_cfg = {
        'model': AGENT_MODEL,
        # 走 DashScope 的 OpenAI 兼容接口：qwen3.8 等新模型只在兼容接口提供，
        # 原生 dashscope SDK 会报 "url error"；model_server 以 http 开头时
        # qwen_agent 会自动选择 TextChatAtOAI（OpenAI 协议）
        'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': DASHSCOPE_API_KEY,
        # qwen_agent 只识别 generate_cfg，旧的 timeout/retry_count 键会被静默忽略
        'generate_cfg': {'max_retries': 3},
    }
    
    try:
        # 如果启用追踪且 LangFuse 可用，使用包装类
        if enable_tracing and langfuse_client and observe:
            agent = TracedAssistant(
                llm=llm_cfg,
                name='财富管理投顾AI助手',
                description='混合智能体财富管理咨询服务',
                system_message=get_system_prompt(customer_profile),
                function_list=['query_shanghai_index'],
                customer_id=customer_id,
                customer_profile=customer_profile
            )
        else:
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

# ====== 运行智能体（带 LangFuse 追踪） ======

def _run_agent_internal(agent: Assistant, messages: List[Dict[str, Any]], 
                       customer_id: str, customer_profile: Dict[str, Any]) -> Dict[str, Any]:
    """运行智能体的内部实现"""
    start_time = time.time()
    
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
    
    elapsed_time = time.time() - start_time
    
    return {
        "user_query": messages[0].get('content', '') if messages else '',
        "customer_profile": customer_profile,
        "final_response": final_response if final_response else "未能生成响应",
        "response_messages": response_messages,
        "elapsed_time": elapsed_time
    }

def run_wealth_advisor(user_query: str, customer_id: str = "customer1") -> Dict[str, Any]:
    """运行财富顾问智能体并返回结果（带 LangFuse 追踪）"""
    
    # 获取客户画像
    customer_profile = SAMPLE_CUSTOMER_PROFILES.get(customer_id, SAMPLE_CUSTOMER_PROFILES["customer1"])
    
    # 初始化智能体（GUI 模式启用追踪）
    agent = init_wealth_advisor_agent(customer_profile, customer_id=customer_id, enable_tracing=True)
    
    # 准备消息
    messages = [{'role': 'user', 'content': user_query}]
    
    # LangFuse 追踪
    if langfuse_client and observe:
        try:
            @observe(name="wealth_advisor_query")
            def _traced_run():
                # 更新 metadata
                if get_client:
                    try:
                        langfuse = get_client()
                        if langfuse:
                            langfuse.update_current_span(
                                metadata={
                                    "model": AGENT_MODEL,
                                    "customer_id": customer_id,
                                    "risk_tolerance": customer_profile.get("risk_tolerance", "unknown"),
                                    "investment_horizon": customer_profile.get("investment_horizon", "unknown"),
                                    "portfolio_value": customer_profile.get("portfolio_value", 0),
                                    "query_length": len(user_query),
                                    "query_preview": user_query[:100]  # 只记录前100个字符
                                }
                            )
                    except Exception:
                        pass  # metadata 设置失败不影响主流程
                
                result = _run_agent_internal(agent, messages, customer_id, customer_profile)

                # 输出 metadata 必须在 span 内部更新：_traced_run 返回后 span 已结束，
                # 在外部调用 update_current_span 会因无 active span 而静默失效
                if get_client:
                    try:
                        langfuse = get_client()
                        if langfuse:
                            final_response = result.get("final_response", "") or ""
                            langfuse.update_current_span(
                                output=final_response,
                                metadata={
                                    "response_length": len(final_response),
                                    "elapsed_time": result.get("elapsed_time", 0),
                                    "has_error": bool(result.get("error"))
                                }
                            )
                    except Exception:
                        pass

                return result

            return _traced_run()
            
        except Exception as e:
            print(f"  [警告] LangFuse 追踪失败: {e}，继续执行")
            return _run_agent_internal(agent, messages, customer_id, customer_profile)
    else:
        return _run_agent_internal(agent, messages, customer_id, customer_profile)

# ====== GUI 模式 ======

def app_gui(customer_id: str = "customer1"):
    """图形界面模式，提供 Web 图形界面"""
    try:
        print("正在启动 Web 界面...")
        
        # 获取客户画像
        customer_profile = SAMPLE_CUSTOMER_PROFILES.get(customer_id, SAMPLE_CUSTOMER_PROFILES["customer1"])
        
        # 初始化智能体（GUI 模式启用追踪）
        agent = init_wealth_advisor_agent(customer_profile, customer_id=customer_id, enable_tracing=True)
        
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

        # 启动 Web 界面（GUI 依赖在此按需导入）
        from qwen_agent.gui import WebUI

        WebUI(
            agent,
            chatbot_config=chatbot_config
        ).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")
        print("请检查网络连接和 API Key 配置")
    finally:
        # 确保 LangFuse 事件落盘
        _flush_langfuse()

def _flush_langfuse():
    """确保 LangFuse 事件落盘"""
    try:
        if HAS_LANGFUSE and get_client:
            langfuse = get_client()
            if langfuse:
                langfuse.flush()
    except Exception:
        pass

# ====== 主函数 ======

if __name__ == "__main__":
    print("=== 混合智能体 - 财富管理投顾AI助手 ===\n")
    print("使用模型：Qwen-Turbo")
    print("框架：qwen-agent")
    if langfuse_client:
        print("监测：LangFuse 已启用\n")
    else:
        print("监测：LangFuse 未启用\n")
    print("-"*50 + "\n")
    
    # 选择客户
    customer_id = "customer1"  # 默认客户
    customer_choice = input("选择客户 (1: 平衡型投资者, 2: 进取型投资者，默认1): ").strip()
    if customer_choice == "2":
        customer_id = "customer2"
    
    try:
        # 启动 Web 图形界面模式
        app_gui(customer_id)
    finally:
        # 确保 LangFuse 事件落盘
        _flush_langfuse()

