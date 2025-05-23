from tqdm import tqdm
import math
import tiktoken
import argparse
from utils import *
# from compress_G.py import *
import random
from cot_prompt_list import main_path_select_prompt
from subgraph_utilts import *
from collections import defaultdict
import pickle
import json
import os 
import os
import json
import asyncio
import os
import json
import sqlite3
import time

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import ast

LLM_model = "gpt"
# LLM_model = "google"
Num_run_LLM = 0
Global_depth = 3
Global_error_status = ""
reasoning_input_token_length = 0

def change_depth(depth = 3):
    global Global_depth
    Global_depth = depth
    print(f"Global depth is changed to {Global_depth}")
def changemode(name = "gpt3"):  
    global LLM_model
    LLM_model = name
    print(f"LLM model is changed to {LLM_model}")
def increment(num = 1):
    global Num_run_LLM
    Num_run_LLM += num
def input_error(error_message="format error, "):
    global Global_error_status
    if error_message not in Global_error_status:
        Global_error_status += error_message
def input_token_length(length):
    global reasoning_input_token_length
    reasoning_input_token_length += length


def inital_num():
    global Num_run_LLM
    Num_run_LLM = 0
    global Global_error_status
    Global_error_status = ""
    global reasoning_input_token_length
    
def display_LLM_calls():
    global Num_run_LLM
    print(f"LLM calls time: {Num_run_LLM}")
    return Num_run_LLM

def display_error_status():
    global Global_error_status
    print(f"Error status: {Global_error_status}")
    return Global_error_status
def display_input_token_length():
    global reasoning_input_token_length
    print(f"input_token_length: {reasoning_input_token_length}")
    return reasoning_input_token_length

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
def extract_top_list(text):
    """Extracts the top 3 paths from a given text.

    Args:
    text: The input text containing the top_list dictionary.

    Returns:
    A list of the top 3 paths.
    """

    # 找到 top_list 的内容
    # match = re.search(r'top_list:\{([^}]*)\}', text)
    match = re.search(r'ist:\s*\{([^}]+)\}', text)
    if not match:
        # match = re.search(r'list:\s*\{([^}]+)\}', text)
        return []
    
    top_list_str = match.group(1)
    
    # 提取所有的数字
    numbers = re.findall(r'\b\d+\b', top_list_str)
    
    # 转换为 int 并返回
    return list(map(int, numbers))

    match = re.search(r'top_entity:\s*\{([^}]+)\}', text)
    if match is None:
        return []

    exploration_required_content = match.group(1)

    # 提取所有以 "m." 开头的实体 ID
    entity_ids = re.findall(r'm\.\w+', exploration_required_content)
    return entity_ids

def extract_entities(text):
    # 使用正则表达式匹配 entities: 后面的花括号中的内容
    match = re.search(r"entities:\s*{([^}]*)}", text)
    if match:
        entities_str = match.group(1)
        # 分割字符串并去除多余的空格
        entities_list = [entity.strip() for entity in entities_str.split(",")]
        return entities_list
    return []


def extract_possible_entities(text):
    # 使用正则表达式匹配 entities: 后面的花括号中的内容
    # 修改正则表达式以匹配花括号或双引号内的全部内容
    # 更新正则表达式以确保能匹配包括花括号和逗号在内的所有内容
    print ("planing extract_possible_entities")
    pattern = r'edicted:\s*(?:"(.*?)"|{(.*?)})'
    matches = re.findall(pattern, text)
    
    results = []
    for match in matches:
        # 处理从正则表达式捕获的每个组
        combined = ' '.join([m.strip() for m in match if m])
        # 处理中英文逗号作为分隔符
        # split_items = [item.strip() for item in re.split(r'[,]', combined)]
        entities_list = [entity.strip() for entity in combined.split(",")]
        results.extend(entities_list)
        
    print ("extract_possible_entities", results)
    return results


def extract_unique_entities_from_backet(text):
    # 使用正则表达式提取所有 {} 中的内容
    entities = re.findall(r'\{(.*?)\}', text)
    
    # 使用 set 去重，然后转换回 list
    unique_entities = list(set(entities))
    
    return unique_entities
# # 示例用法
# text = "Answer: top_list:{Path:2, Path:1, Path:3} Explanation: ..."
# result = extract_top_list(text)
# print(result)  # 输出：['2', '1', '3']


def beam_path_select(question, split_answer,data, to_be_deleted_path, prompt_formate = main_path_select_prompt):
    to_be_deleted_path_prompt = ""
    prunned_path_number = []
    LLM_run_time = 0

    for i in range(1,len(to_be_deleted_path)+1):
        # print("to_be_deleted_path:", to_be_deleted_path[i])
        to_be_deleted_path_prompt += f"Path {i}: " + to_be_deleted_path[i-1] + "\n"

        if i % 40 == 0 or i == len(to_be_deleted_path):
            # print("first section")
            # print(to_be_deleted_path_prompt)
            # prompt_Cot = prompt_formate + "\nQ: " + question +"\nThe question also writen as: "+ data["machine_question"] + "?"+ "  and  "+ data["webqsp_question"] + "\n"
            prompt_Cot = prompt_formate + "\nQ: " + question +f"\n {split_answer[2:]}"
            
            prompt_Cot += f"\nThe entity list is: \n" + to_be_deleted_path_prompt
            try:
                # print(f"Q: {question}\n ")
                # print("answer is ", data["answer"])
                # print(prompt_Cot)

                question_answer = run_LLM(prompt_Cot, LLM_model)
                LLM_run_time += 1
                
                # question_answer = run_LLM(prompt_Cot, LLM_model)
                # print(f"GPT3.5-A\n: {question_answer}")

                prunned_path_number += extract_top_list(question_answer)
                print(prunned_path_number)
                to_be_deleted_path_prompt = ""


                # question_answer = run_LLM(prompt_Cot, args.model_path)
                # print(f"llama-A: {question_answer}")
            except ValueError as e:
                print(f"Error: {e}")
    if len(prunned_path_number) == 0:
        print(question_answer)
        print("LLM reasonning error, requsting again")
        time.sleep(120)
        input_error()

        # increment(-1)
        prunned_path_number = beam_path_select(question, split_answer,data, to_be_deleted_path, prompt_formate)
    answer_list = []
    increment(LLM_run_time)

    for i in prunned_path_number:
        i_1 = int(i)
        # if i_1<= 0 or i_1 > len(to_be_deleted_path):
        #     prunned_path_number.remove(i)
        if i_1>0 and i_1<= len(to_be_deleted_path):
            answer_list.append(i_1)
    return answer_list


def beam_path_expand_select(question, split_answer, data, to_be_deleted_path, existing_path, total_id_to_name_dict,query_sentence,prompt_formate=explored_path_select_prompt, test_size_ori = 40,max_token_length=16000):
    prunned_path_number = []
    llm_run_time = 0
    i = 0
    if LLM_model == "gpt4":
        max_token_length = 8192    
    while i < len(to_be_deleted_path):
        test_size = test_size_ori
        while test_size >= 5:
            end_index = i + test_size
            if end_index > len(to_be_deleted_path):
                end_index = len(to_be_deleted_path)
            to_be_deleted_path_prompt = ""
            for idx in range(i, end_index):
                to_be_deleted_path_prompt += f"Candidate Edge {idx+1}: " + to_be_deleted_path[idx] + "\n"
            # num_token_size = num_tokens_from_string(to_be_deleted_path_prompt, "cl100k_base")
            # if num_token_size <= max_token_length:
                # 构建提示并发送请求
            prompt_Cot = prompt_formate + "\nQ: " + question + f"\n {split_answer}"
            existing_path_prompt = ""
            for j in range(len(existing_path)):
                existing_path_prompt += f"existing_path {j+1}: " + existing_path[j] + "\n"
            prompt_Cot += existing_path_prompt

            prompt_Cot += f"\n\nThe Candidate Edge list is: \n" + to_be_deleted_path_prompt
            
            num_token_size = num_tokens_from_string(to_be_deleted_path_prompt, "cl100k_base")
            if num_token_size <= max_token_length:
                
                try:
                    question_answer = run_LLM(prompt_Cot, LLM_model)
                    llm_run_time += 1
                    prunned_path_number += extract_top_list(question_answer)
                except ValueError as e:
                    print(f"Error: {e}")
                i = end_index  # 更新索引
                break  # 退出当前test_size循环
            else:
                print(f"Token size exceeds {max_token_length}, size-10")
            if test_size <=5:
                print("由于token大小限制无法继续处理。")
                break
            test_size -= 5
        
    increment(llm_run_time)
    answer_list = []
    for num in prunned_path_number:
        num_int = int(num)
        if num_int > 0 and num_int <= len(to_be_deleted_path):
            answer_list.append(num_int)
    new_returned_answer = [to_be_deleted_path[i-1] for i in answer_list]

    if len(new_returned_answer) == 0:
        print(question_answer)
        print("LLM推理错误正在重新请求")
        time.sleep(5)
        input_error("select error, anwer formate error, refuse answering\n")
        new_returned_answer = beam_path_expand_select(question, split_answer, data, to_be_deleted_path, existing_path,total_id_to_name_dict,query_sentence, prompt_formate,30)

    return new_returned_answer

    
def extract_head_tail(data, num):
    head3_set = set()
    end3_set = set()

    # 使用正则表达式匹配大括号内的内容
    for item in data:
        matches = re.findall(r'\{[^{}]*\}', item)
        
        # 如果找到的匹配项少于三个，就继续到下一个项
        if len(matches) < 3:
            continue
        
        # 提取前三个和后三个大括号区块
        head3 = ' - '.join(matches[:num])
        end3 = ' - '.join(matches[-(num):])
        
        # 将提取出的结果添加到对应集合
        head3_set.add(head3)
        end3_set.add(end3)
    
    # 将集合转换为列表返回
    return list(head3_set), list(end3_set)

def Beam_search_step1(query_sentence, NL_CoT_all_paths, total_id_to_name_dict, top_k_value = 80):
    print('''
        +++++++++++++++++++++   Path beam search: step1 - Fuzzy Reduction  +++++++++++++++++++++++          
        ''')
    if not NL_CoT_all_paths:
        return []
    if not isinstance(NL_CoT_all_paths[0], str): 
        NL_path_tobe_del = format_paths_to_natural_language_id_with_name(NL_CoT_all_paths,total_id_to_name_dict)
    else:
        NL_path_tobe_del = NL_CoT_all_paths
    if len(NL_path_tobe_del) >= top_k_value:
        # 加载模型
        # model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        model = SentenceTransformer('msmarco-distilbert-base-tas-b')
        # model = SentenceTransformer('all-mpnet-base-v2')
        print("none-LLM model loaded")
        # 准备数据
        candidate_sentences = NL_path_tobe_del

        # 编码句子
        candidate_embeddings = model.encode(candidate_sentences, batch_size=64, show_progress_bar=True)
        query_embedding = model.encode([query_sentence])

        # 计算相似度
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]

        # 获取最相似的前20个句子的索引
        top_k = top_k_value
        top_k_indices = similarities.argsort()[-top_k:][::-1]

        # 输出结果
        print(f"finish obtained the top {top_k} sentences.\n") 
        NL_path_tobe_del = [candidate_sentences[i] for i in top_k_indices]
    return NL_path_tobe_del

def Beam_search_step2(NL_CoT_all_paths, version, total_id_to_name_dict, query_sentence, top_k_value = 40):

    print('''
        +++++++++++++++++++++   Path beam search: step2 - Branch Reduction   +++++++++++++++++++++
        ''')    
    intial_hop = 0
    head_3, end_3 = extract_head_tail(NL_CoT_all_paths,(3+intial_hop*2))
    while len(NL_CoT_all_paths) > top_k_value:
        head_3, end_3 = extract_head_tail(NL_CoT_all_paths,(3+intial_hop*2))
        print("number of head_3:", len(head_3))  
        print("number of end_3:", len(end_3))  
        print("test_initial_hop:", intial_hop)


        clean_head_first = True
        if len(head_3) > len(end_3):
            tem = head_3
            head_3 = end_3
            end_3 = tem
            clean_head_first = False
    
        # NL_tobe_del = format_paths_to_natural_language_id_with_name(head_3,total_id_to_name_dict)
        NL_tobe_del = head_3
        if len(head_3) < 40 and len(NL_CoT_all_paths) >= 40 and intial_hop < Global_depth+1:

            intial_hop = intial_hop+1
            continue
        token_length = num_tokens_from_string("".join(NL_tobe_del), "cl100k_base")
        input_token_length(token_length)
        if version == 1:
            NL_tobe_del = beam_path_select(question, split_answer,data, NL_tobe_del, main_path_select_prompt)

        else :
            NL_tobe_del = beam_path_expand_select(question, split_answer,data, NL_tobe_del, final_entity_path, total_id_to_name_dict, query_sentence)


        # New_after_del_v2 = []
        # for i in range (1, len(NL_tobe_del)+1):
        #     if i in revise_prunned_path_number:
        #         New_after_del_v2.append(NL_tobe_del[i-1])
        # NL_tobe_del = New_after_del_v2

        print("number of NL_CoT_all_paths, before remove:", len(NL_CoT_all_paths))
        NL_CoT_all_paths = [i for i in NL_CoT_all_paths if any(j in i for j in NL_tobe_del)]
        print("number of NL_CoT_all_paths, after remove:", len(NL_CoT_all_paths))
        


        # NL_tobe_del = format_paths_to_natural_language_id_with_name(end_3,total_id_to_name_dict)
        # for i in NL_tobe_del:
        #     if i not in NL_CoT_all_paths:
        #         NL_tobe_del.remove(i)
        head_3, end_3 = extract_head_tail(NL_CoT_all_paths,(3+intial_hop*2))
        print("number of head_3:", len(head_3))  
        print("number of end_3:", len(end_3)) 
        if clean_head_first:
            NL_tobe_del = end_3
        else:
            NL_tobe_del = head_3

        if version == 1:
            NL_tobe_del = beam_path_select(question, split_answer,data, NL_tobe_del, main_path_select_prompt)
        else :
            NL_tobe_del = beam_path_expand_select(question, split_answer,data, NL_tobe_del, final_entity_path, total_id_to_name_dict, query_sentence)

        # New_after_del_v2 = []
        # for i in range (1, len(NL_tobe_del)+1):
        #     if i in revise_prunned_path_number:
        #         New_after_del_v2.append(NL_tobe_del[i-1])
        # NL_tobe_del = New_after_del_v2

        print("number of NL_CoT_all_paths, before remove:", len(NL_CoT_all_paths))
        NL_CoT_all_paths = [i for i in NL_CoT_all_paths if any(j in i for j in NL_tobe_del)]
        print("number of NL_CoT_all_paths, after remove:", len(NL_CoT_all_paths))
        NL_CoT_all_paths = list(set(NL_CoT_all_paths))
        if intial_hop > Global_depth+1:
            break
        intial_hop = intial_hop+1
    return NL_CoT_all_paths

def Beam_search_step3(NL_CoT_all_paths, version, total_id_to_name_dict, query_sentence, top_k_value = 3):
    
    print('''
        +++++++++++++++++++++   Path beam search: step3 - Path Selection     +++++++++++++++++++++
        ''')
    while len(NL_CoT_all_paths) > top_k_value:
        # prunned_path_number = beam_path_select(question, split_answer,data, NL_CoT_all_paths)
        token_length = num_tokens_from_string("".join(NL_CoT_all_paths), "cl100k_base")
        input_token_length(token_length)
        if version == 1:
            prunned_path_number = beam_path_select(question, split_answer,data, NL_CoT_all_paths, main_path_select_prompt)
        else :
            NL_CoT_all_paths = beam_path_expand_select(question, split_answer,data, NL_CoT_all_paths, final_entity_path, total_id_to_name_dict, query_sentence)
        
        # token_length = num_tokens_from_string("".join(NL_CoT_all_paths), "cl100k_base")
        # input_token_length(token_length)
        # # print("prunned_path_number v1:", prunned_path_number)
        # prunned_path_number.sort()
        # # NL_CoT_all_paths_temp = [NL_CoT_all_paths[i-1] for i in prunned_path_number]
        # NL_CoT_all_paths_temp = []
        # for i in prunned_path_number:
        #     if i> 0 and i <= len(NL_CoT_all_paths):
        #         NL_CoT_all_paths_temp.append(NL_CoT_all_paths[i-1])
        
        # NL_CoT_all_paths = NL_CoT_all_paths_temp
    return NL_CoT_all_paths
def Beam_search(question, split_answer,query_sentence, data, total_id_to_name_dict, CoT_all_paths, final_entity_path = [], version = 1):

    if not CoT_all_paths:
        return []
    if not isinstance(CoT_all_paths[0], str): 
        NL_CoT_all_paths = format_paths_to_natural_language_id_with_name(CoT_all_paths,total_id_to_name_dict)
        
    else:
        NL_CoT_all_paths = CoT_all_paths
    
    original_path = NL_CoT_all_paths

    if using_beam_step1_only:
        print("test with only step1 - fuzzy reduction")
        NL_CoT_all_paths = Beam_search_step1(query_sentence, NL_CoT_all_paths, total_id_to_name_dict, 3)
        return NL_CoT_all_paths
    else:
        NL_CoT_all_paths = Beam_search_step1(query_sentence, NL_CoT_all_paths, total_id_to_name_dict, 80)

    
    initial_path = NL_CoT_all_paths

    if using_beam_step1_2:
        print("test with only step1+2 - branch reduction")
        NL_CoT_all_paths = Beam_search_step2(NL_CoT_all_paths, version, total_id_to_name_dict,query_sentence, 3)

        # NL_CoT_all_paths = Beam_search_step2(NL_CoT_all_paths, version, 3)

        return NL_CoT_all_paths
    
    if using_beam_step1_3:
        print("test with only step1+3 - path selection")
        NL_CoT_all_paths = Beam_search_step3(NL_CoT_all_paths, version, total_id_to_name_dict,query_sentence, 3)
        return NL_CoT_all_paths

    ##
    ## full version
    ##
    print("test with step1+2+3 - full version")

    NL_CoT_all_paths = Beam_search_step2(NL_CoT_all_paths, version, total_id_to_name_dict,query_sentence, 20)
    NL_CoT_all_paths = Beam_search_step3(NL_CoT_all_paths, version, total_id_to_name_dict,query_sentence, 3)

    return NL_CoT_all_paths


def CoT_entity_expand_inchain(question, split_answer, data, main_path = None):

    prompt_Cot = cot_prompt_in_chain + "\nQ:\nQuestion: " + question  + "?"+ "\n"
    prompt_Cot += "Main Topic Entities: " + str(data['topic_entity']) + "\n" + f"\n {split_answer}\n"



    if main_path:
        main_path_prompt = ""
        for i in range(1, len(main_path)+1):
            main_path_prompt += f"Path {i}: " + main_path[i-1] + "\n"
        prompt_Cot += main_path_prompt
    try:
        question_answer = run_LLM(prompt_Cot, LLM_model)
        increment()
        # print("prompt_Cot:", prompt_Cot)
        print(f"A: {question_answer}")

        GPT_entity = extract_possible_entities(question_answer)
        CoT_list = extract_cots_as_strings(question_answer)
        if len(CoT_list) == 0:
            print("LLM reasonning error, requsting again")
            print(question_answer)
            time.sleep(120)
            input_error()

            increment(-1)
        
            return CoT_entity_expand_inchain(question, split_answer, data, main_path = None)
        return GPT_entity, CoT_list
    except ValueError as e:
        print(f"Error: {e}")
        return [], question_answer
    
def check_n_explor(question, split_question, data, topic_entity_path, CoT_entity_path, prompt_formate):

    prompt_Cot = prompt_formate + "\nQ: " + question +f"\n {split_question[2:]}"
    # + "\n"
    

    if len(topic_entity_path) > 0:
        # while len(topic_entity_path) > 3:
        #     # print("involved_path:", involved_path)
        #     prunned_path_number = beam_path_expand_select(question, split_answer,data, topic_entity_path, [])
        #     topic_entity_path = [topic_entity_path[i-1] for i in prunned_path_number]

        prompt_topic_entity_path = "\nTopic entity path: \n" 
        for i in range(1, len(topic_entity_path)+1):
            prompt_topic_entity_path += f"Path{i}: " + topic_entity_path[i-1] + "\n"
        prompt_Cot += prompt_topic_entity_path
    
    if len(CoT_entity_path) > 0:
        prompt_CoT_entity_path = "\nSupplementary Edges: \n"
        for i in range(1, len(CoT_entity_path)+1):
            prompt_CoT_entity_path += f"Edge{i}: " + CoT_entity_path[i-1] + "\n"
        prompt_Cot += prompt_CoT_entity_path
    try:
        # print(f"Q: {question}\n ")
        # print("answer is ", data["answer"])
        # print(prompt_Cot)

        question_answer = run_LLM(prompt_Cot, LLM_model,0)
        increment()
        
        # print(f"A: {question_answer}")
        # exit()
        # GPT_entity = extract_entities(question_answer)
        return question_answer

    except ValueError as e:
        print(f"Error: {e}")
        return []

def check_n_explor_v4(question, data,  split_answer, related_edge, CoT, prompt_formate):

    prompt_Cot = prompt_formate + "\nQ:\n Question: " + question+ "?\n"
    prompt_Cot += "Main Topic Entities: \n" + str(data['topic_entity']) + "\n"
    
    
    # if len(explored_entity) > 1:
    prompt_topic_entity_path = ""
    if len(split_answer) > 0:
        prompt_topic_entity_path += f"\n {split_answer}"
    
    prompt_related_edge_path = ""
    if len(related_edge) > 0:
        prompt_related_edge_path += "\nRelated edge: \n"
        for i in range(1, len(related_edge)+1):
            prompt_related_edge_path += f"Related Path:{i} " + related_edge[i-1] + "\n\n"
    prompt_CoT_entity_path = ""
    if len(CoT) > 0:
        prompt_CoT_entity_path += "\nLLM_generated CoT: \n"
        for i in range(1, len(CoT)+1):
            prompt_CoT_entity_path += f"CoT Path:{i} " + CoT[i-1] + "\n\n"

    prompt_Cot = prompt_Cot + prompt_topic_entity_path +prompt_related_edge_path+ prompt_CoT_entity_path

    
    # prompt_Cot += prompt_topic_entity_path +prompt_related_edge_path+ prompt_CoT_entity_path
    try:
        # print(f"Q: {question}\n ")
        # print("answer is ", data["answer"])
        # print(prompt_Cot)
        print("-------- start summarization ---------")
        question_answer = run_LLM(prompt_Cot, LLM_model)
        increment()
        # print(f"A: {question_answer}")

        # GPT_entity = extract_entities(question_answer)
        return question_answer
    except ValueError as e:
        print(f"Error: {e}")
        return []


def extract_entities_from_strings(paths):
    # 创建一个字典来存储实体 ID 和名称
    entities_dict = {}

    # 正则表达式用于匹配实体
    # entity_pattern = r'\{(m\.\w+): ([^,}]+)'
    entity_pattern = r'(m\.\w+): ([^,}]+)'

    # 遍历每个路径字符串
    for path in paths:
        # 使用正则表达式找到所有匹配的实体
        entities = re.findall(entity_pattern, path)
        
        # 将实体添加到字典中
        for entity_id, name in entities:
            entities_dict[entity_id] = name.strip()

    return entities_dict


def get_name_to_id(name1, total_id_to_name_dict):
    ids = []
    for id, name in total_id_to_name_dict.items():
        if name == name1:
            ids += [id]
    return ids

def extract_cots_as_strings(text):
    # Use regular expression to find all occurrences of lines that start with "CoT" followed by a number and a colon.
    cot_patterns = re.findall(r'CoT*\d+: .*', text)
    
    # Initialize a list to hold the contents of each CoT found.
    cots = []
    
    # Iterate over each CoT pattern found in the text.
    for cot in cot_patterns:
        # Extract the entire line following "CoT<digit>: "
        cot_content = re.search(r'CoT*\d+: (.*)', cot)
        if cot_content:
            cots.append(cot_content.group(1))
    
    return cots
def extract_path_length_from_text(text):
    back_up = text
    tokens = re.split(r'\s*-\s*', text.strip())
    # 计算路径长度
    path_length = (len(tokens) - 1) // 2

    match = re.search(r'cot\s*:\s*(.*)', back_up, re.IGNORECASE)
    match2 = re.search(r'cot\s*:\s*(.*)', back_up, re.IGNORECASE)
    match3 = re.search(r'cot\s*:\s*(.*)', back_up, re.IGNORECASE)

    # 输出结果
    if match:
        thinking_cot_line = match.group(1).strip()
        # print('提取的文本是：')
        # print(thinking_cot_line)
    else:
        print('cannot find the cot line')

    return path_length, thinking_cot_line

import re

def extract_split_questions(text):
    # 将文本按行分割
    lines = text.strip().split('\n')
    questions = []

    for line in lines:
        # 去除行中的所有空格
        line_no_spaces = line.replace(' ', '')
        # 检查行中是否包含 'split'（忽略大小写）
        if re.search(r'split', line_no_spaces, re.IGNORECASE):
            # 使用 ':' 分割，提取问题部分
            parts = line.split(':', 1)
            if len(parts) > 1:
                question = parts[1].strip()
                questions.append(question)
            else:
                # 如果没有 ':'，整个行作为问题
                questions.append(line.strip())

    return questions

def extract_entities_from_sentence(sentence):
    words = sentence.split()
    entities = []
    inside_entity = False
    current_entity = ''
    for word in words:
        if word.startswith('"') and word.endswith('"'):
            # 单词被双引号包围，直接提取
            entities.append(word.strip('"'))
        elif word.startswith('"'):
            # 实体开始
            inside_entity = True
            current_entity = word.lstrip('"') + ' '
        elif word.endswith('"'):
            # 实体结束
            current_entity += word.rstrip('"')
            entities.append(current_entity.strip())
            current_entity = ''
            inside_entity = False
        elif inside_entity:
            # 实体内部
            current_entity += word + ' '
    return entities

def find_top_similar_entities(
    entity_id_to_name,
    query_sentence,
    top_k=3,
    topic_exsiting=[],
    sbert_model=None,
    ner_pipeline=None,
    device='cuda'  # 新增参数
):
    """
    Finds the top_k entities from entity_id_to_name that are most similar to the topic entities extracted from query_sentence.

    Parameters:
    - entity_id_to_name (dict): A dictionary mapping entity IDs to entity names.
    - query_sentence (str): The input sentence containing the topic entities.
    - top_k (int): The number of top similar entities to retrieve (default is 3).
    - sbert_model (SentenceTransformer, optional): Pre-loaded SentenceTransformer model. If None, a default model will be loaded.
    - ner_pipeline (transformers.pipeline, optional): Pre-loaded NER pipeline. If None, a default English NER model will be loaded.
    - device (str or int): The device to run the models on. 'cuda' or 'cpu' or GPU index (default is 'cuda').

    Returns:
    - List[Tuple]: A list of tuples containing (entity_id, entity_name, similarity_score).
    """

    from transformers import pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    # Load models if not provided
    # if sbert_model is None:
    #     sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    # if ner_pipeline is None:
    #     if device == 'cuda':
    #         device_index = 0
    #     elif device == 'cpu':
    #         device_index = -1
    #     elif isinstance(device, int):
    #         device_index = device
    #     else:
    #         device_index = 0
    #     ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple", device=device_index)

    # Prepare data
    entity_names = list(entity_id_to_name.values())
    entity_ids = list(entity_id_to_name.keys())

    # Extract topic entities using NER
    # entities = ner_pipeline(query_sentence)

    topic_entities = []
    current_entity = ''
    current_label = ''

    # for entity in entities:
    #     word = entity['word']
    #     label = entity['entity_group']

    #     if label in ['ORG', 'LOC', 'PER', 'MISC']:  # 根据需要调整实体类型
    #         if word.startswith('##'):
    #             word = word[2:]
    #             current_entity += word
    #         else:
    #             if current_entity != '':
    #                 topic_entities.append(current_entity)
    #                 current_entity = ''
    #             current_entity = word
    #     else:
    #         if current_entity != '':
    #             topic_entities.append(current_entity)
    #             current_entity = ''

    # if current_entity != '':
    #     topic_entities.append(current_entity)
    # for i in topic_entities:
    #     for j in entity_names:
    #         if i in j:
    #             topic_entities.remove(i)
    #             break

    # parts = re.split(r'\s-\s(?!\d)', query_sentence)
    
    # # 去除每个部分前后的空白字符
    # parts = [part.strip().strip('"#/') for part in parts]
    
    # # 选择位于奇数位置的元素
    # topic_entities = [parts[i] for i in range(len(parts)) if i % 2 == 0]

    parts = re.split(r'\s-\s(?![^()]*\))', query_sentence)
    
    # 清理每个部分，移除引号和前后空白
    topic_entities = [re.sub(r'[“”"]', '', part).strip() for part in parts]

    for i in topic_entities:
        if i in topic_exsiting:
            topic_entities.remove(i)
    print("Extracted topic entities:", topic_entities)
    if len(topic_entities) == 0:
        return [],[]
    

    # 将实体名称和主题实体合并为语料库
    corpus = entity_names + topic_entities

    # 创建 TF-IDF 矢量化器
    vectorizer = TfidfVectorizer()

    # 拟合并转换语料库
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # 分割 TF-IDF 矩阵为实体和主题实体的向量
    entity_tfidf = tfidf_matrix[:len(entity_names)]
    topic_tfidf = tfidf_matrix[len(entity_names):]

    # 计算主题实体向量的平均值
    topic_vector = topic_tfidf.mean(axis=0)

    # **修改这里，转换为 np.array**
    topic_vector = topic_vector.A1  # 或者使用 .toarray().ravel()

    # 计算实体向量与主题向量之间的余弦相似度
    similarities = cosine_similarity(entity_tfidf, topic_vector.reshape(1, -1))

    # 将相似度数组展平
    similarities = similarities.flatten()

    # 获取具有最高相似度分数的 top_k 索引
    top_k_indices = similarities.argsort()[-top_k:][::-1]

    # 汇总结果
    top_entities = []
    for idx in top_k_indices:
        entity_id = entity_ids[idx]
        entity_name = entity_names[idx]
        similarity_score = similarities[idx]
        if similarity_score > 0.8:
            if entity_name not in topic_exsiting:
                top_entities.append((entity_id, entity_name, similarity_score))

    return top_entities, topic_entities
    # # Encode entity names
    # entity_embeddings = sbert_model.encode(entity_names, batch_size=64, show_progress_bar=False)

    # # Encode topic entities
    # if len(topic_entities) > 0:
    #     topic_entity_embeddings = sbert_model.encode(topic_entities)
    #     topic_entity_embedding = np.mean(topic_entity_embeddings, axis=0)
    # else:
    #     print("No topic entities extracted. Using the entire query sentence for encoding.")
    #     topic_entity_embedding = sbert_model.encode([query_sentence])[0]

    # # Compute similarities
    # similarities = cosine_similarity([topic_entity_embedding], entity_embeddings)[0]

    # # Get top_k similar entities
    # top_k_indices = similarities.argsort()[-top_k:][::-1]

    # # Compile results
    # top_entities = []
    # for idx in top_k_indices:
    #     entity_id = entity_ids[idx]
    #     entity_name = entity_names[idx]
    #     similarity_score = similarities[idx]
    #     top_entities.append((entity_id, entity_name, similarity_score))

    # return top_entities

import sys
if __name__ == '__main__':
    if len(sys.argv) < 7:
        print("Error Usage: python PoG_multi.py <Dataset_name> <sum/unsum> <beam_search:1, 12, 13, 123> <PoG/PoGE> <gpt3/gpt4> <max_deepth 1/2/3>")
        print("Note:")
        print("<Dataset_name>: webqsp, cwq, grailqa, webquestions, simpleqa")
        print("<sum/unsum>: sum for using paths summary, unsum for not using summary")
        print("<beam_search:1, 12, 13, 123>: 1 for only using fuzzy selection, 12 for using step1+BranchReduced, 13 for using step1+preciese selection, 123 for using all steps")
        print("<PoG/PoGE>: PoG for using only all relation, PoGE for using radom one relation")
        print("<gpt3/gpt4>: gpt3 for using gpt3, gpt4 for using gpt4")
        print("<max_deepth 1/2/3>: 1 for using only 1 hop, 2 for using 2 hop, 3 for using 3 hop")
        sys.exit()
    file_name = sys.argv[1]

    # db_path = f'{file_name}_subgraph.db'
    subgraph_db = f'subgraph/{file_name}_main_Subgraphs.db'
    NL_subgraph_db = f'subgraph/{file_name}_main_nl_Subgraphs.db'
    path_db = f'subgraph/{file_name}_path_db.db'

    answer_add = ""
    if sys.argv[2] == "sum":
        using_summary = True
        answer_add = "_sum"
    else:
        using_summary = False
        answer_add = "_unsum"
    using_beam_step1_2 = False
    using_beam_step1_3 = False
    using_beam_step1_only = False
    if_using_all_r = False
    if sys.argv[3] == "1":
        using_beam_step1_only = True
        answer_add += "_BS1"
        print("************* active using Fuzzy selection Only ***********")

    elif sys.argv[3] == "12":
        using_beam_step1_2 = True
        answer_add += "_BS12"
        print("*************active using Fuzzy and BranchR selection***********")

    elif sys.argv[3] == "13":
        using_beam_step1_3 = True
        answer_add += "_BS13"
        print("*************active using Fuzzy and Path selection***********")

    elif sys.argv[3] == "123":
        answer_add += "_BS123_20"
        print("*************active using ALL Beam Search***********")
    non_allr = answer_add

    if sys.argv[4] == "PoG":
        if_using_all_r = True
        answer_add += "_allr"
        print("*************active using ALL related edges***********")
    else:
        print("*************active using ONE related edges***********")
    if sys.argv[5] == "gpt4":
        LLM_model = "gpt4"
        changemode("gpt4")
    else:
        LLM_model = "gpt3"
        changemode("gpt3")
    
    change_depth(int(sys.argv[6]))
        
    # answer_db = f'subgraph/{file_name}_answer_{Global_depth}_gpt_3.db'
    # datas, question_string = prepare_dataset(file_name)
    datas, question_string,Q_id = prepare_dataset(file_name)
    if "gpt4" in LLM_model:
        version = "4"
    else:
        version = "3"
    # initialize_db(db_path)
    initialize_large_database(subgraph_db)
    initialize_large_database(path_db)
    initialize_large_database(NL_subgraph_db)
    # initialize_large_database(answer_db)

    # for data in tqdm(datas[5+28+6+3+54:1000]):
    # for Global_depth in [2,3,4,5,1]:
    for Global_depth in [Global_depth]:
        print("Global_depth:", Global_depth)
        answer_db = f'answer/{file_name}_answer_{Global_depth}_gpt_{version}{answer_add}.db'
        print(f'subgraph/{file_name}_answer_{Global_depth}_gpt_{version}{answer_add}.db')
        answer_gpt3_db = f'subgraph/{file_name}_answer_{Global_depth}_gpt_3{answer_add}.db'
        if "allr" in answer_add:
            previouse_db_one_r = f'subgraph/{file_name}_answer_{Global_depth}_gpt_{version}{non_allr}.db'
            initialize_large_database(previouse_db_one_r)
        
        print(f'subgraph/{file_name}_answer_{Global_depth}_gpt_3{answer_add}.db')
        initialize_large_database(answer_db)
        initialize_large_database(answer_gpt3_db)
        for data in tqdm(datas[0:200]):
        # for data in tqdm(datas[0:200]):
        # for data in tqdm(datas[200:400]):
        # for data in tqdm(datas[0:100]):
        # for data in tqdm(datas[0:2]):
        # for data in tqdm(datas[382:400]):

            depth, path, graph_storage, NL_formatted_paths, NL_subgraph = None, None, None, None, None
            question = data[question_string]
            topic_entity = data['topic_entity']
            question_id = data[Q_id] 
            question_real_answer = check_answerlist(file_name, question_string, question, datas,data)

            # if len(topic_entity) > 2:
            #     continue
            # if question_id == 'WebQTest-1348_67d901eb5314fe0563f6e2602b72f478':
            #     continue
            if len(topic_entity) < 2:
                continue
            else:
                inital_num()
            answer = load_from_large_db(answer_db, question_id)
            if answer:
                print("qurstion_id:", question)
                print("answer is found in the database")
                if not using_beam_step1_only:
                    continue
   

        # continue  
            print("\n Question:", question)
            # print("Machine_question:", data['machine_question'])
            print("answer:", question_real_answer)
            print("topic_entity:", topic_entity)



            prompt_split = split_question_prompt + "\nQ:\n Question: " + question + "\n"
            prompt_split += "Main Topic Entities: \n" + str(data['topic_entity']) + "\n" +"A:\n"
            split_answer = run_LLM(prompt_split, LLM_model)[2:]
            Num_run_LLM += 1
            print(f"split_answer: {split_answer}")
            predict_length, thinking_cot_line = extract_path_length_from_text(split_answer)
            split_question = extract_split_questions(split_answer)
            print("predict CoT length:", predict_length)
            
            print(topic_entity.values())
            print("list(topic_entity):", list(topic_entity))
            sorted_topic_entity_name = reorder_entities(thinking_cot_line, list(topic_entity.values()))
            sorted_topic_entity_id = []
            for name in sorted_topic_entity_name:
                for id, entity in topic_entity.items():
                    if entity == name:
                        sorted_topic_entity_id.append(id)
                        break
            print("sorted_topic_entity_id:", sorted_topic_entity_id)
            print("sorted_topic_entity_id_name:", sorted_topic_entity_name)


            
            
            subgraph_dict = load_from_large_db(subgraph_db, question_id)
            sub_cap = False
            if subgraph_dict:
                print("Data found in the database.")
                print(subgraph_dict.keys())
                
                graph = subgraph_dict['subgraph']
                all_entities = subgraph_dict['all_entities']
                depth = subgraph_dict['hop']
                outter_entity = subgraph_dict['outter_entity']
                if "sub_cap" in subgraph_dict:
                    sub_cap = subgraph_dict['sub_cap']

            else:
                print("Data not found in the database. Exploring the graph...")
                start = time.time()
                sub_cap, graph, depth,all_entities, outter_entity, dict, if_inside = explore_graph_from_entities_by_hop_neighbor_1(list(topic_entity), Global_depth, question_real_answer)

                total_id_to_name_dict = dict
                NL_name_set = set(dict.values())
                
                delete_data_by_question_id(NL_subgraph_db, question_id)
                NL_subgraph= {
                    "total_id_to_name_dict": total_id_to_name_dict,
                    "NL_name_set": NL_name_set
                }
                save_to_large_db(NL_subgraph_db, question_id, NL_subgraph)
                NL_subgraph = None

                subgraph_dict = {
                    "question": question,
                    "machine_question": data[question_string],
                    "question_id": question_id,
                    "results": question_real_answer,
                    "topic_entity": topic_entity,
                    "hop": depth,
                    "subgraph": graph,
                    "all_entities": all_entities,
                    "outter_entity": outter_entity,
                    "sub_cap": sub_cap
                }
                delete_data_by_question_id(subgraph_db, question_id)
                save_to_large_db(subgraph_db, question_id, subgraph_dict)

            if sub_cap == True:

                in_sub = False
                NL_subgraph = load_from_large_db(NL_subgraph_db, question_id)
                if NL_subgraph:
                    total_id_to_name_dict = NL_subgraph['total_id_to_name_dict']
                    NL_name_set = NL_subgraph['NL_name_set']

                    # if question_real_answer in NL_name_set:  
                    #     in_sub = True
                    #     print("Answer in subgraph")
                else:
                    print("NL_subgraph is not found")
                    continue
                if_inside = False
                

                
                entity_id_to_name = total_id_to_name_dict




                print('''
                    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    +++++++++++++++++++++++ Part1: Topic entity path +++++++++++++++++++++++++++
                    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++                  
                    ''')
                temp_hop = 1
                ori_graph = graph
                ori_total_id_to_name_dict = total_id_to_name_dict
                final_entity_path = []
                Summary_main_entity_path = []
                final_COT_path = []
                Summary_COT_path = []
                final_expand_path = []
                Summary_expand_path = []
                previouse_all_entity_path = []
                result = ""
                result_f = ""
                
                flag_start_change = False


                
                while temp_hop <= Global_depth:
                    itersection_name = set()
                    if len(final_entity_path)==0:
                        if temp_hop < min(math.ceil(predict_length/2),Global_depth):
                            temp_hop += 1
                            print("subgraph not reached the predict length, temp_hop+1")
                            continue

                    intersection = bfs_with_intersection_only(ori_graph, sorted_topic_entity_id, temp_hop)
                    end = time.time()
                    if not intersection:
                        temp_hop += 1
                        print("topic entity not connected, temp_hop+1")
                        continue
                    # itersection_name = set()
                    intersection_new = set()
                    for i in intersection:
                        itersection_name.add(ori_total_id_to_name_dict[i])
                        intersection_new.add(i)
                        check = True
                        for real_name in question_real_answer:
                            if real_name not in itersection_name:
                                check = False
                                break
                        if check:
                            break

                    print("intersection",len(intersection_new))



                    
                    if len(intersection_new) > 100000:
                        if len(final_entity_path)>0:
                            break
                        else: 
                            temp_hop -= 1
                            print("intersection is too large, temp_hop-1")
                            intersection_new = bfs_with_intersection_only(ori_graph, sorted_topic_entity_id, temp_hop)
                    
                    if len(intersection_new) == 0:
                        break
                    
                    # if data["answer"] in itersection_name:
                    #     print("answer in the intersection")

                    if flag_start_change == True and len(graph.keys()) == len(ori_graph.keys()):
                        print("no change need in the graph")
                    else:
                        reduced_graph, reduced_name_dict = create_subgraph_through_intersections(ori_graph, list(topic_entity), intersection_new, ori_total_id_to_name_dict,temp_hop)
                        entity_id_to_name = reduced_name_dict
                        total_id_to_name_dict = reduced_name_dict
                        graph = reduced_graph
                        flag_start_change = True
                    print("graph nodes number before:", len(ori_graph.keys()))
                    print("graph nodes number after:", len(graph.keys()))


                        
                    start = time.time()
                    
                    all_paths = []


                    top_entities, key_words = find_top_similar_entities(entity_id_to_name, thinking_cot_line, top_k=3, topic_exsiting=list(topic_entity.values()))

                    if len(top_entities) > 0:
                        print("\nTop entities most similar to the topic entities:\n")

                        for entity_id, entity_name, similarity_score in top_entities:
                            if entity_id in topic_entity.keys():
                                continue
                            print(f"Entity ID: {entity_id}, Entity Name: {entity_name}, Similarity: {similarity_score:.4f}")
                            if similarity_score < 0.95:
                                continue
                            implemented_entity_list =  reorder_entities(thinking_cot_line, list(topic_entity.values())+[entity_name])
                            Exp_list = get_name_to_id(entity_name, total_id_to_name_dict)
                            for id in Exp_list:
                                sorted_smain_entity_id = []
                                for temp_name in implemented_entity_list:
                                    found = False
                                    for te_id, te_entity in topic_entity.items():
                                        if te_entity == temp_name:
                                            found = True
                                            sorted_smain_entity_id.append(te_id)
                                            break
                                    if found == False:
                                        sorted_smain_entity_id.append(id)
                                    
                                print("sorted_CoT_entity_id:", sorted_smain_entity_id)
                                all_paths += find_all_paths_bibfs_itersection_limit(graph, sorted_smain_entity_id, temp_hop,if_using_all_r)
                    
                    if len(all_paths) == 0:
                        all_paths += find_all_paths_bibfs_itersection(graph, sorted_topic_entity_id, temp_hop,if_using_all_r)

                    print("all paths number:", len(all_paths))
                    # nl_path=format_paths_to_natural_language_id_with_name([all_paths[0]],entity_id_to_name)
                    # print(nl_path)
                    end = time.time()
                    print("Find path Time:", end - start)
                    # exit()
                
                #############################################################
                # LLM prunning for the main entity_path
                #############################################################
                    print(f'''
                        +++++++++++++++++++++++  obtained from  path  in Global_depth {temp_hop} +++++++++++++++++++++++
                        ''')
                    
                    if len(all_paths) > 3:
                        pruned_path = final_entity_path + format_paths_to_natural_language_id_with_name(all_paths,entity_id_to_name) 
                        previouse_all_entity_path = pruned_path

                        print("number of all_paths:", len(all_paths))
                        final_entity_path = Beam_search(question, split_answer,thinking_cot_line, data, entity_id_to_name, pruned_path,[],2)
                    else:   
                        final_entity_path = format_paths_to_natural_language_id_with_name(all_paths,entity_id_to_name)
                        previouse_all_entity_path = final_entity_path

                    
                    for i in range(0, len(final_entity_path)):
                        print("final entity path",i,": ", final_entity_path[i])

                    print('''
                    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    +++++++++++++++++++++        Check the answer         ++++++++++++++++
                    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    ''')
                    if using_summary:
                        result = check_n_explor_v4(question, data, split_answer, final_entity_path,[],Summary_COT_w_splitQ_prompt)
                                
                        Summary_main_entity_path = extract_cots_as_strings(result)
                        # break


                        # print("Summary_main_entity_path:", Summary_main_entity_path)
                        for i in range(0, len(Summary_main_entity_path)):
                            print("Summary_main_entity_path",i,": ", Summary_main_entity_path[i])
                        for i in range(0, len(final_entity_path)):
                            print("final entity path",i,": ", final_entity_path[i])

                    result = check_n_explor(question,split_answer, data, Summary_main_entity_path + final_entity_path, [],answer_n_explore_prompt)
                    
                    print("Result:", result)
                    print("the real answer is:", question_real_answer)
                    if "{Yes}" in result:
                        print("Yes in the answer")
                        final_path_toal =final_entity_path + Summary_main_entity_path + final_COT_path + Summary_COT_path + final_expand_path + Summary_expand_path
                        result = check_n_explor(question,split_answer, data, final_path_toal, [], answer_generated_direct)+result                        
                        LLM_call = display_LLM_calls()
                        error_message = display_error_status()
                        total_reasonning_token_input = display_input_token_length()
                        final_path_toal =final_entity_path + Summary_main_entity_path + final_COT_path + Summary_COT_path + final_expand_path + Summary_expand_path
                        answer_dict = {
                            "LLM_answer": result,
                            "real_answer": question_real_answer,
                            "question": question,
                            "split_answer": split_answer,
                            "final_entity_path": final_path_toal,
                            "LLM_call": LLM_call,
                            "main_path":1,
                            "cot":0,
                            "gpt":0,
                            "error_message": error_message,
                            "total_reasonning_token_input": total_reasonning_token_input
                        }
                        delete_data_by_question_id(answer_db, question_id)
                        save_to_large_db(answer_db, question_id, answer_dict)
                        
                        
                        break
                    temp_hop += 1
                # exit()
                
                
                if "{Yes}" in result:
                    continue
                # graph = ori_graph
                # total_id_to_name_dict = ori_total_id_to_name_dict

                
                explored_path_entity = []

                candidate_list = []

                print('''
                    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    +++++++++++++++++++++     start LLM supplement part      +++++++++++++++++++++++
                    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    ''')

                CoT_entity, CoT_list = CoT_entity_expand_inchain(question, split_answer, data, Summary_main_entity_path + final_entity_path)

                similary_threshold = 0.8
                # if len(itersection_name) == 0:
                #     itersection_name = NL_name_set
                high_sub_entities = calculate_cosine_similarity(total_id_to_name_dict.values(), CoT_entity, similary_threshold)
                print("high_sub_entities:", high_sub_entities)
                result = ''

                CoT_high_entity = entity_need_explore(topic_entity, Summary_main_entity_path + final_entity_path, high_sub_entities)[:Global_depth]
                
                final_COT_path = []
                total_revise_COT_path = []
                if len(CoT_high_entity) > 0:
                    CoT_high_entity = reorder_entities(",".join(CoT_entity), CoT_high_entity)

                print("exploring CoT predict: ", CoT_high_entity)
                for name in CoT_high_entity:
                    print(f'''
                    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                            CoT entity exploretion for entity: {name}
                    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++''')
                    COT_part_path = []
                    check_previouse_path = []
                    print("length of previouse path:", len(previouse_all_entity_path))
                    for pre_path in previouse_all_entity_path:
                        if name in pre_path:
                            check_previouse_path.append(pre_path)


                    if len(check_previouse_path) > 0:
                        print("checked_previouse_path:", len(check_previouse_path))
                        COT_part_path.extend(check_previouse_path)
                        Exp_list = []
                    else:

                        Exp_list = get_name_to_id(name, total_id_to_name_dict)

                    CoT_indicter = ""
                    for i in CoT_list:
                        if name in i:
                            CoT_indicter = i
                            break
                    sorted_CoT_entity_name = reorder_entities(CoT_indicter, list(topic_entity.values())+[name])
                    print("coT_indicter:", CoT_indicter)
                    print("sorted_CoT_entity_name:", sorted_CoT_entity_name)
                    
                    for id in Exp_list:
                        sorted_CoT_entity_id = []
                        for temp_name in sorted_CoT_entity_name:
                            found = False
                            for te_id, te_entity in topic_entity.items():
                                if te_entity == temp_name:
                                    found = True
                                    sorted_CoT_entity_id.append(te_id)
                                    break
                            if found == False:
                                sorted_CoT_entity_id.append(id)
                            
                        print("sorted_CoT_entity_id:", sorted_CoT_entity_id)
                        total = []
                        # CoT_hop = math.ceil(predict_length/2) - 1
                        CoT_hop = 1
                        while len(total)==0 and CoT_hop <= Global_depth:
                            print("CoT_hop:", CoT_hop)
                    
                            # intersection_new_COT = bfs_with_intersection_only(graph, sorted_CoT_entity_id, CoT_hop)
                            # if not intersection_new_COT:
                            #     CoT_hop += 1
                            #     continue    
                            # CoT_new_graph,_ = create_subgraph_through_intersections(graph, sorted_CoT_entity_id, intersection_new_COT, total_id_to_name_dict,CoT_hop)
                            # print("graph nodes number before:", len(graph.keys()))
                            # print("graph nodes number after:", len(CoT_new_graph.keys()))
                            start = time.time()

                            total = []
                            for i in range(1, len(sorted_CoT_entity_id)-1):
                                new = find_all_paths_bibfs_itersection(graph, [sorted_CoT_entity_id[i-1],sorted_CoT_entity_id[i]], CoT_hop, if_using_all_r)
                                
                                if len(new) == 0:
                                    total = []
                                    break
                                new = Beam_search_step1(CoT_indicter, new, total_id_to_name_dict, 10)
                                
                                if len(total) == 0:
                                    total = new
                                else:
                                    total = concatenate_paths_with_unlinked(total, new)
                           
                            # new = find_all_paths_bibfs_itersection_limit(CoT_new_graph, sorted_CoT_entity_id, CoT_hop, if_using_all_r)
                            print("time:", time.time()-start)
                            CoT_hop += 1
                        COT_part_path.extend(total)
                        # COT_part_path += new

                        
                    
                    COT_part_path = Beam_search_step1(CoT_indicter, COT_part_path, total_id_to_name_dict, 50)
                    final_COT_path.extend(COT_part_path)



                print("number of total_cot_path(before beam search):", len(final_COT_path))
                # print("final_COT_path:", final_COT_path)
                final_COT_path = Beam_search(question, split_answer, thinking_cot_line, data, total_id_to_name_dict, final_COT_path, [], 2)
                print("number of total_cot_path (after beam search):", len(final_COT_path))

                print('''
                ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                +++++++++++++++++++++        Check the answer for COT        ++++++++++++++++
                ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                ''')
                        
                if using_summary:
                    result = check_n_explor_v4(question, data, split_answer, final_entity_path+ final_COT_path,[],Summary_COT_w_splitQ_prompt)
                    
                    # print("Result:", result)
                    Summary_COT_path = extract_cots_as_strings(result)
                    for i in Summary_COT_path:
                        print("Summary CoT:", i)
                    for i in final_COT_path:
                        print("Final CoT:", i)



                result = check_n_explor(question,split_answer, data, Summary_COT_path+final_COT_path, [],answer_n_explore_prompt)
                print("Check answer by KG paths:", result)


                if "Yes" in result:
                    LLM_call = display_LLM_calls()
                    final_path_toal =final_entity_path + Summary_main_entity_path + final_COT_path + Summary_COT_path + final_expand_path + Summary_expand_path
                    result = check_n_explor(question,split_answer, data, final_path_toal, [], answer_generated_direct) + result
                    final_path_toal =final_entity_path + Summary_main_entity_path + final_COT_path + Summary_COT_path + final_expand_path + Summary_expand_path
                    error_message = display_error_status()
                    total_reasonning_token_input = display_input_token_length()
                    answer_dict = {
                        "LLM_answer": result,
                        "real_answer": question_real_answer,
                        "question": question,
                        "split_answer": split_answer,
                        "final_entity_path": final_path_toal,
                        "LLM_call": LLM_call,
                        "main_path":0,
                        "cot":1,
                        "error_message": error_message,
                        "gpt":0,
                        "total_reasonning_token_input": total_reasonning_token_input
                    }
                    delete_data_by_question_id(answer_db, question_id)
                    save_to_large_db(answer_db, question_id, answer_dict)
                    continue
                

                final_expand_path = Beam_search(question, split_answer,thinking_cot_line+ "\n".join(CoT_list), data, total_id_to_name_dict, Summary_main_entity_path+final_entity_path+Summary_COT_path+final_COT_path,[],2)
                print("""
                        ++++++++++++++++++++++++++++++++++++++++++
                        Part3: start to explore the answer
                            ++++++++++++++++++++++++++++++++++++++++++
                        """)

                Total_GPT = []
                result_v = True
                explored_path_entity = []
                current_explored_depth = 1

                while current_explored_depth <= Global_depth:
                    candidate_list1 = []
                    candidate_id_list1 = []
                    main_path_entities = extract_entities_from_strings(final_expand_path+Total_GPT)
                    print("main_path_entities:", main_path_entities)
                
                    for id,name in main_path_entities.items():
                        input_temp = f"{id}: {name}"
                        if id not in total_id_to_name_dict.keys():
                            continue
                        if id not in topic_entity.keys():
                            if input_temp not in explored_path_entity:
                                candidate_list1.append(input_temp)
                                candidate_id_list1.append(id)

                    if len(candidate_list1)>0:
                        new_nl_related_paths1 = []
                        for id in candidate_id_list1:
                            name = total_id_to_name_dict[id]
                            input_temp = f"{id}: {name}"
                            explored_path_entity.append(input_temp)
                            print(f'''
                            ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                            ++++++++++++   GPT 1-hop path Beam search for: {input_temp}   ++++++++++++++
                            ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                            ''')
                            # graph, all_entities, exlored_entities, _, total_id_to_name_dict = explore_graph_from_one_topic_entities([id], graph, total_id_to_name_dict, exlored_entities,all_entities)

                            new_nl_related_paths1 += find_1_hop_relations_and_entities(id, graph, total_id_to_name_dict,if_using_all_r)
                        current_explored_depth += 1

                        
                        while len(new_nl_related_paths1) > 3:
                            if using_beam_step1_only: 
                                # new_nl_related_paths1 = concatenate_paths_with_unlinked(Total_GPT, new_nl_related_paths1)

                                new_nl_related_paths1 = Beam_search_step1(thinking_cot_line, new_nl_related_paths1, total_id_to_name_dict, 3)
                            else:
                                new_nl_related_paths1 = beam_path_expand_select(question, split_answer,data, new_nl_related_paths1, final_expand_path, total_id_to_name_dict,thinking_cot_line)
                            # print("involved_path:", involved_path)
                            # prunned_path_number = beam_path_expand_select(question, split_answer,data, new_nl_related_paths1, final_expand_path)
                            # new_nl_related_paths1 = [new_nl_related_paths1[i-1] for i in prunned_path_number]
                        if len(Total_GPT) > 0:
                            Total_GPT = concatenate_paths_with_unlinked(Total_GPT, new_nl_related_paths1)
                        else:
                            Total_GPT += new_nl_related_paths1

                        while len(Total_GPT) > 10:
                            # print("involved_path:", involved_path)
                            if using_beam_step1_only:
                                Total_GPT = Beam_search_step1(thinking_cot_line, Total_GPT, total_id_to_name_dict, 10)
                            else:
                                Total_GPT = beam_path_expand_select(question, split_answer,data, Total_GPT, final_expand_path, total_id_to_name_dict,thinking_cot_line)
                            
                            
                            # prunned_path_number = beam_path_expand_select(question, split_answer,data, Total_GPT, final_expand_path)
                            # Total_GPT = [Total_GPT[i-1] for i in prunned_path_number]
                        print(f'''
                            ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                            ++++++++++++          Check the answer                 ++++++++++++++
                            ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                            ''')

                        print(id)
                        if using_summary:
                            result = check_n_explor_v4(question, data, split_answer, final_expand_path+Total_GPT,[],Summary_COT_w_splitQ_prompt)
                                    
                            Summary_expand_path = extract_cots_as_strings(result)
                            print("Summary_expand_path:", Summary_expand_path)

                            for i in Summary_expand_path:
                                print("Summary_expand_path:", i)
                            for i in Total_GPT:
                                print("Total_GPT:", i)


                            result = check_n_explor(question,split_answer, data, Summary_expand_path, [], answer_n_explore_prompt)
                        else:
                            result = check_n_explor(question,split_answer, data, final_expand_path+Total_GPT, [], answer_n_explore_prompt)

                        # result = check_n_explor(question, data, New_total_path, [],answer_n_explore_prompt)

                        # result = check_n_explor(question, data, New_total_path, [],answer_n_explore_prompt)
                        print("Result:", result)

                        print("the real answer is:", question_real_answer)

                        # Total_GPT += new_nl_related_paths1
                        if "Yes" in result:
                            final_path_toal =final_entity_path + Summary_main_entity_path + final_COT_path + Summary_COT_path + final_expand_path + Summary_expand_path
                            result = check_n_explor(question,split_answer, data, final_path_toal, [], answer_generated_direct)+result
                            error_message = display_error_status()
                            LLM_call = display_LLM_calls()
                            total_reasonning_token_input = display_input_token_length()
                            final_path_toal =final_entity_path + Summary_main_entity_path + final_COT_path + Summary_COT_path + final_expand_path + Summary_expand_path
                            answer_dict = {
                                "LLM_answer": result,
                                "real_answer": question_real_answer,
                                "question": question,
                                "split_answer": split_answer,
                                "final_entity_path": final_path_toal,
                                "LLM_call": LLM_call,
                                "main_path":0,
                                "cot":0,
                                "error_message": error_message,
                                "gpt":1,
                                "total_reasonning_token_input": total_reasonning_token_input
                            }
                            delete_data_by_question_id(answer_db, question_id)
                            save_to_large_db(answer_db, question_id, answer_dict)
                            
                            
                            break
                    else:
                        break
                if "Yes" not in result:
                    final_path_toal =final_entity_path + Summary_main_entity_path + final_COT_path + Summary_COT_path + final_expand_path + Summary_expand_path
                    if using_summary:
                        result = check_n_explor_v4(question, data, split_answer, final_path_toal,[],Summary_COT_w_splitQ_prompt)
                                
                        Summary_expand_path = extract_cots_as_strings(result)
                        print("Summary_expand_path:", Summary_expand_path)

                        result = check_n_explor(question,split_answer, data, Summary_expand_path, [], answer_generated_direct)
                    else:
                        result = check_n_explor(question,split_answer, data, final_path_toal, [], answer_generated_direct)
                    
                    print("Result:", result)

                    print("the real answer is:", question_real_answer)

                    # Total_GPT += new_nl_related_paths1
                    # if "Yes" in result_f:
                    LLM_call = display_LLM_calls()
                    error_message = display_error_status()
                    total_reasonning_token_input = display_input_token_length()
                    final_path_toal =final_entity_path + Summary_main_entity_path + final_COT_path + Summary_COT_path + final_expand_path + Summary_expand_path
                    answer_dict = {
                        "LLM_answer": result,
                        "real_answer": question_real_answer,
                        "question": question,
                        "split_answer": split_answer,
                        "final_entity_path": final_path_toal,
                        "LLM_call": LLM_call,
                        "main_path":0,
                        "cot":0,
                        "gpt":1,
                        "error_message": error_message,
                        "total_reasonning_token_input": total_reasonning_token_input
                    }
                    delete_data_by_question_id(answer_db, question_id)
                    save_to_large_db(answer_db, question_id, answer_dict)
