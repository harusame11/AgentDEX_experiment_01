# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import random
import time
import json
import asyncio
import subprocess
import argparse
import logging
from tavily import TavilyClient
import tiktoken
from transformers import AutoTokenizer

# --- 关键修改：从 LLM_API 导入配置和函数 ---
from LLM_API import get_llm_response, MODEL_MAPPING

logging.disable(logging.CRITICAL)

# 全局变量占位
MODEL_NAME = None
my_output_dir = None
MAX_ROUNDS = None
MODEL_TYPE = None
TOOL_PRICING = None
# 加载工具定义
with open('tools.json') as f:
    raw_tools = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
# --- 搜索客户端初始化 (只在这里用) ---
os.environ["TAVILY_API_KEY"] = "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"
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

# 工具定义字典
ALL_TOOLS = {
    "enhance_reasoning": {'model': ["reasoner-1", "reasoner-2", "reasoner-3"]},
    "answer": {'model': ["answer-math-1", "answer-math-2", "answer-1", "answer-2", "answer-3", "answer-4"]},
    "search": {"model": ["search-1", "search-2", "search-3"]},
}

def cut_seq(seq,l):
    if len(seq)==0:
        return {
            'effective_length': 0,
            'string_after_cut': ''
        }
    token_ids = tokenizer(seq)['input_ids']
    rs = tokenizer.batch_decode(token_ids[-l:], skip_special_tokens=True)
    return {
        'effective_length': len(token_ids),
        'string_after_cut': ''.join(rs)
    }

def call_tool(arguments):
    """
    重构后的工具执行函数。
    """
    tool_name = arguments['tool']
    
    # ---------------------------
    # 工具 1: Enhance Reasoning (写代码并执行)
    # ---------------------------
    if tool_name == 'enhance_reasoning':
        prompt = arguments['context_str'].strip() + '\n\n'
        prompt += f"Question: {arguments['problem']}\nInstead of directly answering the question, please write additional python code that will give intermidiate results after execution. Wrap the code within ```python and ```. The code should be self-contained with all the import and initialization."
        
        # 调用 API 生成代码
        response = get_llm_response(
            model_alias=arguments['model'],
            messages=[{"role": "user", "content": prompt}],
            temperature=1
        )
        
        if isinstance(response, str): # API 报错
            arguments['generated_code'] = ''
            arguments['exec_result'] = f"Error generating code: {response}"
            return arguments

        content = response.choices[0].message.content
        try:
            generated_code = content.split('```python')[-1].split('```')[0].strip()
        except:
            generated_code = ""
            
        if not generated_code:
            arguments['generated_code'] = ""
            arguments['exec_result'] = "No code found in response."
            return arguments

        # 本地执行代码
        arguments['generated_code'] = generated_code
        code_path = os.path.join(arguments['cur_output_dir'], f'exec_code_{arguments["id"]}.py')
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(generated_code)
            
        try:
            exec_result = subprocess.run(
                ['python', code_path], 
                timeout=30, 
                capture_output=True, 
                text=True
            )
            if exec_result.stdout and len(exec_result.stdout.strip()) > 0:
                final_output = exec_result.stdout
            elif exec_result.stderr and len(exec_result.stderr.strip()) > 0:
                final_output = f"Execution Error:\n{exec_result.stderr}"
            else:
                final_output = "Code executed successfully but printed nothing (stdout is empty)."

            arguments['exec_result'] = final_output
            with open(os.path.join(arguments['cur_output_dir'],f'exec_out_{arguments["id"]}.txt'),'w') as f:
                f.write(final_output)
        except Exception as e:
            arguments['exec_result'] = f"Execution Error: {str(e)}"
            
        return arguments

    # ---------------------------
    # 工具 2: Search (搜索)
    # ---------------------------
    elif tool_name == 'search':
        prompt = arguments['context_str'].strip()+'\n\n'
        prompt += f"Question: {arguments['problem']}\nInstead of directly answering the question, please write a query to search for a piece of relevant and missing information. The query should be a few key words about the information to search or a short sentence. Wrap the query within <query> and </query>."        
        # 修正：之前这里拼写错误写成了 get_llm_responsel
        response = get_llm_response(
            model_alias=arguments['model'],  
            messages=[{"role": "user", "content": prompt}]
        )
        
        query = arguments['problem']
        if not isinstance(response, str):
            content = response.choices[0].message.content 
            # 简单尝试提取 <query>，如果模型没遵循，就用全文
            if "<query>" in content:
                query = content.split('<query>')[-1].split('</query>')[0]
            else:
                query = content

        # 调用 Tavily
        try:
            search_result = tavily_client.search(query=query[:300], max_results=10)
            contents = [res['content'] for res in search_result['results']]
        except Exception as e:
            print(f"Search API Error: {e}")
            contents = []

        arguments['query'] = query
        arguments['search_results_data'] = contents
        return arguments

    # ---------------------------
    # 工具 3: Answer (最终回答)
    # ---------------------------
    elif tool_name == 'answer':
        prompt = arguments['context_str'].strip() + '\n\nProblem:\n' + arguments['problem']
        
        response = get_llm_response(
            model_alias=arguments['model'],
            messages=[
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        if isinstance(response, str):
            print(f"!!! Answer API Error: {response}") # 在控制台打印具体错误
            # 或者将其写入 arguments 以便在 json 中看到
            arguments['pred'] = ''
            arguments['response'] = f'!!! Answer API Error: {response}'
            arguments['correctness'] = False
            return arguments

        response_str = response.choices[0].message.content
        arguments['response'] = response_str
        
        if '\\boxed{' in response_str:
            pred = response_str.split('\\boxed{')[-1].split('}')[:-1]
            pred = '}'.join(pred).strip()
        else:
            pred = ""
        arguments['pred'] = pred

        # 判分逻辑
        reference = arguments['answer']
        if pred.lower() == str(reference).lower():
            correctness = True
        else:
            eval_prompt = (
                f"Question: {arguments['problem']}\n"
                f"Student Answer: {pred}\n"
                f"Reference Answer: {reference}\n"
                "Assume reference is correct. Is student answer correct? Output <correct>True</correct> or <correct>False</correct>."
            )
            judge_resp = get_llm_response("answer-1", [{"role": "user", "content": eval_prompt}])
            
            if isinstance(judge_resp, str):
                correctness = False
            else:
                judge_content = judge_resp.choices[0].message.content
                if "<correct>True</correct>" in judge_content:
                    correctness = True
                else:
                    correctness = False

        arguments['correctness'] = correctness
        return arguments

    return arguments

# ---------------------------
# 并发调度器 (保持不变)
# ---------------------------
import contextlib
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Tuple, Any, Callable

async def run_all(
    task_list: Iterable[Tuple[Callable[[Any], Any], Any]],
    concurrency: int = 2,
    progress: bool = False,
    return_exceptions: bool = False,
):
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(concurrency)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        async def run_one(idx: int, func: Callable, arg: Any):
            async with sem:
                if asyncio.iscoroutinefunction(func):
                    res = await func(arg)
                else:
                    res = await loop.run_in_executor(executor, func, arg)
                return idx, res, None

        task_list = list(task_list)
        tasks = [asyncio.create_task(run_one(i, f, a)) for i, (f, a) in enumerate(task_list)]
        results = [None] * len(tasks)

        if progress:
            from tqdm import tqdm
            pbar = tqdm(total=len(tasks))
        else:
            pbar = None

        try:
            for fut in asyncio.as_completed(tasks):
                idx, res, err = await fut
                if err is None:
                    results[idx] = res
                else:
                    if return_exceptions:
                        results[idx] = err
                    else:
                        for t in tasks: t.cancel()
                        with contextlib.suppress(Exception):
                            await asyncio.gather(*tasks, return_exceptions=True)
                        raise err
                if pbar: pbar.update(1)
        finally:
            if pbar: pbar.close()
        return results


def run_single(e):
    doc_list = []
    code_list = []
    attempt_list = []
    problem = e['question']
    user_problem = problem
    answer = e['answer']
    all_tool_calls = []
    final_correct = False
    all_tool_responses = {}
    used_tools = []
    all_message_responses = {}
    
    for step in range(MAX_ROUNDS):
        cur_output_dir = os.path.join(my_output_dir,f"step_{step}")
        if not os.path.isdir(os.path.join(cur_output_dir,'tool_return')):
            try:
                os.makedirs(os.path.join(cur_output_dir,'tool_return'))
            except:
                pass
        tools = []
        doc_str = ''
        for doc_idx, doc in enumerate(doc_list):
            doc_str += f"Doc {doc_idx+1}: {doc[:1200]} ...\n\n"
        code_str = ''
        for code_idx, code_piece in enumerate(code_list):
            code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
        attempt_str = ''
        for attempt_idx, attempt in enumerate(attempt_list):
            attempt_str += f"Attempt{attempt_idx+1} answer by {attempt['model']}: {attempt['answer']}\n"
        str_cut = cut_seq(seq=attempt_str,l=8000)
        attempt_str = str_cut['string_after_cut']
        if not attempt_str.startswith('Attempt') and len(attempt_str)>0:
            attempt_str = 'Attempt answer: '+attempt_str
        str_cut = cut_seq(seq=code_str+attempt_str,l=12000)
        code_attempt_str = str_cut['string_after_cut']
        code_attempt_str_len = str_cut['effective_length']
        if not code_attempt_str.startswith('```') and len(code_attempt_str)>0:
            code_attempt_str = '```\n'+code_attempt_str
        doc_flag = False
        problem_length = len(tokenizer(problem)['input_ids'])
        if code_attempt_str_len<27000-problem_length:
            if code_attempt_str:
                context_str = cut_seq(seq=doc_str+"\npython code and execution outputs:\n"+code_attempt_str,l=27000-problem_length)
            else:
                context_str = cut_seq(seq=doc_str,l=27000-problem_length)
            context_str = context_str['string_after_cut']
            if len(doc_str)>0:
                doc_flag = True
                context_str = 'Documents:\n'+context_str
        else:
            context_str = code_attempt_str
        removed_tool = None
        if len(used_tools)>1 and used_tools[-1]==used_tools[-2]:
            updated_tools = []
            removed_tool = used_tools[-1]
            for t in tools:
                if t['function']['name']!=used_tools[-1]:
                    updated_tools.append(t)
        else:
            updated_tools = tools
        cur_tool_set = [t['function']['name'] for t in updated_tools]

        # 2. 调用 Orchestrator
        chat = [
            {"role": "system", "content":"You are good at using tools"},
            {"role": "user", "content": f"Problem: {problem}\n\n{context_str}\n\nChoose an appropriate tool."}
        ]
        
        response = get_llm_response(
            model_alias='orchestrator-1', 
            messages=chat, 
            tools=raw_tools, 
            temperature=0.2,
            return_raw_response=True,
            max_length=12000
        )
        
        if isinstance(response, str):
            continue # 出错重试

        tool_calls = response.choices[0].message.tool_calls
        # 记录 模型的思考全过程
        cache_tool_calls = []
        if tool_calls:
            for one_tool_call in tool_calls:
                t_name = one_tool_call.function.name
                try:
                    t_args = json.loads(one_tool_call.function.arguments)
                except:
                    t_args = {} # 解析失败留空，防止报错
                
                cache_tool_calls.append({
                    'tool_name': t_name,
                    'tool_arguments': t_args
                })
        
        # 构造 message_dict 并存入总记录
        message_dict = {
            'content': response.choices[0].message.content, # 模型的思考过程文本
            'tool_calls': cache_tool_calls                  # 解析后的工具调用列表
        }
        all_message_responses[f"turn_{step}_message"] = message_dict
        # -----------------------------------------------------------
        if not tool_calls or len(tool_calls) == 0:
            # 没调工具，可能是想直接回答，或者出错了。这里简单continue
            continue
            
        # 3. 解析工具
        tool_call_list = []
        cur_tool_calls = []
        

        for one_tool_call in tool_calls:
            tool_name = one_tool_call.function.name
            try:
                tool_arguments = json.loads(one_tool_call.function.arguments)
            except:
                continue
            
            if tool_name not in ALL_TOOLS:
                continue
                
            tool_call_item = {
                'name': tool_name,
                'arguments': tool_arguments
            }
            cur_tool_calls.append(tool_call_item)
            expert_model_to_call = tool_arguments.get('model') 
            # 准备执行参数
            call_tool_argument = {
                'tool': tool_name,
                'model': expert_model_to_call,
                'context_str': context_str,
                'cur_output_dir': cur_output_dir,
                'problem': user_problem,
                'answer': answer,
                'id': e.get('id', 'unknown'),
                'eid': e.get('eid', 0)
            }
            # 构造执行函数对
            tool_call_list.append([call_tool, call_tool_argument])
            
            used_tools.append(tool_name)
            # 只要有一个是 answer，就只执行这一个
            if tool_name == 'answer':
                break
        
        all_tool_calls.append(cur_tool_calls)
        
        if len(tool_call_list) == 0:
            continue

        # 4. 执行工具
        # run_all 是异步的，run_single 是同步的，这里用 asyncio.run 桥接
        # 注意：因为 tool_call_list 是 [[func, arg], ...]，符合 run_all 要求
        cur_responses = asyncio.run(run_all(tool_call_list))
        all_message_responses[f"turn_{step}_message"] = message_dict
        all_tool_responses[f"turn_{step}_response"] = cur_responses
        
        # 5. 处理结果
        finish_flag = False
        for cur_response in cur_responses:
            if not cur_response: continue
            
            if cur_response['tool'] == 'enhance_reasoning':
                # [关键修改 2] 移除 len(...) > 0 的判断
                # 无论结果是什么，都必须加入历史记录，否则模型会无限重复
                code_content = cur_response.get('generated_code', '')
                exec_output = cur_response.get('exec_result', 'No result returned')
                
                # 只有当真的有代码时才记录
                if code_content:
                    code_list.append({'code': code_content, 'output': exec_output})
                    print(f"📝 Step {step}: Code execution recorded. Output length: {len(exec_output)}")
            
            elif cur_response['tool'] == 'search':
                for one_doc in cur_response.get('search_results_data', [])[::-1]:
                    if one_doc not in doc_list:
                        doc_list.append(one_doc)
            
            elif cur_response['tool'] == 'answer':
                final_correct = cur_response.get('correctness', False)
                finish_flag = True
                break
        
        if finish_flag:
            break

    return_dict = {
        'id': e['id'],
        'problem': problem,
        'all_tool_calls': all_tool_calls,
        'all_tool_responses': all_tool_responses,
        'all_message_response':all_message_responses,
        'answer': answer,
        'correct': final_correct
    }
    
    # 结果写入
    if not os.path.exists(my_output_dir):
        os.makedirs(my_output_dir, exist_ok=True)
    with open(os.path.join(my_output_dir, f"{e.get('id', 'unknown')}.json"), 'w') as f:
        json.dump(return_dict, f, indent=2)
    return return_dict

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str) # 默认值
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--example_file_path', type=str)
    parser.add_argument('--max_rounds', type=int, default=20) # 减少轮数省钱
    parser.add_argument('--basic_tools', action='store_true')
    args = parser.parse_args()

    # 从 LLM_API 导入的 MODEL_MAPPING 用于这里的逻辑
    if args.basic_tools:
        keys = list(MODEL_MAPPING.keys())
        for k in keys:
            MODEL_MAPPING[k] = args.model_name

    MODEL_NAME = args.model_name
    my_output_dir = args.output_dir
    MAX_ROUNDS = args.max_rounds
    
    if not os.path.isdir(os.path.join(my_output_dir,'answer_cache')):
        os.makedirs(os.path.join(my_output_dir,'answer_cache'), exist_ok=True)
   
    # 读取题目
    with open(args.example_file_path) as f:
        lines = f.readlines()
    examples = []
    for eid,l in enumerate(lines):
        if not l.strip(): continue
        raw_example = json.loads(l)
        raw_example['eid'] = eid
        # 确保有 id 字段，否则报错
        if 'id' not in raw_example:
            raw_example['id'] = f"test_{eid}"
        examples.append([run_single, raw_example])

    # 运行
    print(f"Starting evaluation on {len(examples)} examples...")
    tool_call_results = asyncio.run(run_all(examples, concurrency=2)) # 提高并发
    print("Evaluation finished.")