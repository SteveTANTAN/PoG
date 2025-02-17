from tqdm import tqdm
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

LLM_model = "gpt3"




            
def check_in_path(paths, answer_list):
    for i in paths:
        for answer in answer_list:
            if answer in i:
                return True
    return False


def re_check_answer(data,answer_dict,answer_list,question_id, answer_db):
    question = answer_dict['question']
    split_answer = answer_dict['split_answer']
    final_path_toal = answer_dict['final_entity_path']

    reasult = check_n_explor(question,split_answer, data, final_path_toal, [],answer_generated_direct)
    an_dict={}
    an_dict["LLM_answer"] = reasult
    print("Check answer by KG paths:", reasult)

    if check_answer(an_dict, answer_list):
        answer_dict["LLM_answer"] = reasult

        delete_data_by_question_id(answer_db, question_id)
        save_to_large_db(answer_db, question_id, answer_dict)
    # continue
    
if __name__ == '__main__':
    import sys
    import os
    if len(sys.argv) < 7:
        print("Error Usage: python check_answer.py <Dataset_name> <sum/unsum> <beam_search:1, 12, 13, 123> <PoG/PoGE> <gpt3/gpt4> <max_deepth 1/2/3>")
        print("Note:")
        print("<Dataset_name>: webqsp, cwq, grailqa, webquestions, simpleqa")
        print("<sum/unsum>: sum for using paths summary, unsum for not using summary")
        print("<beam_search:1, 12, 13, 123>: 1 for only using fuzzy selection, 12 for using step1+BranchReduced, 13 for using step1+preciese selection, 123 for using all steps")
        print("<PoG/PoGE>: PoG for using only all relation, PoGE for using radom one relation")
        print("<gpt3/gpt4>: gpt3 for using gpt3, gpt4 for using gpt4")
        print("<max_deepth 1/2/3>: 1 for using only 1 hop, 2 for using 2 hop, 3 for using 3 hop")
        sys.exit()
    if len(sys.argv[1]) >0:
        file_name = sys.argv[1]
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

        if sys.argv[4] == "allr":
            if_using_all_r = True
            answer_add += "_allr"
            print("*************active using ALL related edges***********")
        else:
            print("*************active using ONE related edges***********")
    if len(sys.argv[5]) >0:
        
        if "4" in sys.argv[5]:
            LLM_model = "gpt4"
            version = 4
        else:
            LLM_model = "gpt3"
            version = 3
        # LLM_model = sys.argv[2]
        print("LLM_model:", LLM_model)
    recheck = False
    countall_ = False
    Global_depth_1 = int(sys.argv[6])

    # if sys.argv[1] == "gpt3":
    # file_name = "cwq_multi"
    # file_name = "cwq"
    # file_name = "grailqa"
    # file_name = "webqsp"
    # file_name = "webquestions"
    # file_name = "simpleqa"
    error_summary = {}

    datas, question_string,Q_id = prepare_dataset(file_name)

    # for data in tqdm(datas[20+4+13+2+4+10+10+11+22+8+2+29+69+22+9+68+11:1000]):


    # for Global_depth in [1,2,3,4,5]:
    for Global_depth in [Global_depth_1]:
        print("Global_depth:", Global_depth)

        answer_db = f'answer/{file_name}_answer_{Global_depth}_gpt_{version}{answer_add}.db'
        print("answer_db:", answer_db)

        # answer_db = f'subgraph/{file_name}_answer_{Global_depth}_gpt_3{answer_add}.db'
        


        initialize_large_database(answer_db)
        
        obtained_answer = 0
        Num_run_LLM = 0
        total_question = 0
        generated_answer_by_KG = 0  
        total_main_path = 0
        total_CoT = 0
        total_gpt = 0
        error_reasoning = 0

        total_llm_token = 0
        total_f1 = 0
        total_LLM_only = 0


        total_time = 0
        total_memory = 0

        TTT_a = 0
        for data in tqdm(datas[0:]):

        # for data in tqdm(datas[23:25 ]):
        # for data in tqdm(datas[382:400]):
            depth, path, graph_storage, NL_formatted_paths, NL_subgraph = None, None, None, None, None
            question = data[question_string]
            topic_entity = data['topic_entity']
            question_id = data[Q_id] 
            TTT_a += 1
            answer = load_from_large_db(answer_db, question_id)

            if answer:
                # print("Data found in the database.")

                answer_dict = {
                    "LLM_answer": answer['LLM_answer'],
                    "real_answer": answer['real_answer'],
                    "question": answer['question'],
                    "split_answer": answer['split_answer'],
                    "final_entity_path": answer['final_entity_path'],
                    "LLM_call": answer['LLM_call'],
                    "main_path": answer['main_path'] if 'main_path' in answer.keys() else 0,
                    "cot": answer['cot'] if 'cot' in answer.keys() else 0,
                    "gpt": answer['gpt'] if 'gpt' in answer.keys() else 0,
                    "total_reasonning_token_input": answer['total_reasonning_token_input'] if 'total_reasonning_token_input' in answer.keys() else 0,
                    "error_message": answer['error_message'] if 'error_message' in answer.keys() else "",
                    "run_time": answer['run_time'] if 'run_time' in answer.keys() else 0,
                    "memory": answer['memory'] if 'memory' in answer.keys() else 0
                }

                answer_list = check_answerlist(file_name, question_string, answer['question'], datas,data)

                total_question += 1
                
                obtain = check_answer(answer_dict, answer_list)
                if obtain:
                    obtained_answer += 1
                    total_f1 += calculate_f1_score(answer_dict, answer_list)
                    if check_in_path(answer_dict['final_entity_path'], answer_list):
                        generated_answer_by_KG += 1
                    if len(answer_dict['final_entity_path']) == 0:
                        total_LLM_only += 1


                    

                    # prompt_split = split_question_prompt + "\nQ:\n Question: " + question + "\n"
                    # prompt_split += "Main Topic Entities: \n" + str(data['topic_entity']) + "\n" +"A:\n"
                    # split_answer = run_LLM(prompt_split, LLM_model)[2:]
                    # Num_run_LLM += 1
                    # print(f"split_answer: {split_answer}")
                    # print("question_string:", answer_dict['question'])
                    # print("answer:", answer_list)
                    # print("topic_entity:", topic_entity)

                    # for i in answer_dict['final_entity_path']:
                    #     print("path :",i)
                    # print ("LLM answer:", answer_dict['LLM_answer'])
                else:
                    if check_in_path(answer_dict['final_entity_path'], answer_list):
                        if countall_:
                            obtained_answer += 1
                        error_reasoning += 1
                        if recheck:
                            re_check_answer(data,answer_dict,answer_list,question_id, answer_db)
                        # re_check_answer(data,answer,answer_list,question_id, answer_db)
                    # print(answer_dict.keys())
                    # exit()
                    if "error_message" in answer_dict.keys():
                        if check_in_path(answer_dict['final_entity_path'], answer_list):
                            if "reasoning error" not in error_summary.keys():
                                error_summary["reasoning error"] = 1
                            else:
                                error_summary["reasoning error"] += 1
                        elif answer_dict['error_message'] not in error_summary.keys():
                            error_summary[answer_dict['error_message']] = 1
                        else:
                            error_summary[answer_dict['error_message']] += 1


                Num_run_LLM += answer_dict['LLM_call']
                if obtain:
                    total_main_path += answer_dict['main_path']
                    total_CoT += answer_dict['cot']
                    total_gpt += answer_dict['gpt']
                total_llm_token += answer_dict['total_reasonning_token_input']
                total_time += answer_dict['run_time']
                total_memory += answer_dict['memory']
            # else:
                # print("Data not found in the database. Exploring the graph...")
                # print("question_id:", question_id)
                # print("question:", question)
                # print("topic_entity:", topic_entity)


        print("TTT_a:", TTT_a)
        
        print("Global_depth:", Global_depth)
        print("total_question:", total_question)
        print("obtained_answer:", obtained_answer)
        print("accuracy in % (2 decimal):", round(obtained_answer/total_question*100,2))
        print("Num_run_LLM:", Num_run_LLM)
        print("generated_answer_by_KG:", generated_answer_by_KG)

        print("answed by Main Paths exploration in % (2 decimal):", round(total_main_path/total_question*100,2))
        print("answed by LLMs Supplyment paths in % (2 decimal):", round(total_CoT/total_question*100,2))
        print("answed by Node expand exploration in % (2 decimal):", round(total_gpt/total_question*100,2))
        print("error_reasoning:", error_reasoning)
        print("average_LLM_call:", Num_run_LLM/total_question)
        print("average token input:", total_llm_token/200)
        print("total_LLM_only:", total_LLM_only)

        print("KG generated answer:", round((obtained_answer-total_CoT)/obtained_answer*100,2))
        print("KG+LLM answer:", round((total_CoT)/obtained_answer*100,2))
        print("LLM answer:", round((obtained_answer-generated_answer_by_KG)/obtained_answer*100,2))
        print(total_main_path+total_CoT+total_gpt)
        print("error_summary:", error_summary)




    # exit()
 