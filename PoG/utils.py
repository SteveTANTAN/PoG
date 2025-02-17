from freebase_func import *
from openai import OpenAI
import json
import re
import time
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import util
from sentence_transformers import SentenceTransformer

import google.generativeai as genai
import os
import openai
import re
import time
import tiktoken  # 使用OpenAI的tiktoken工具来计算token数量

def count_tokens(text, model="gpt-3.5-turbo"):
    """使用tiktoken计算token数量"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def run_LLM(prompt, model,temperature=0.4):
    result = ''
    if "google" in model:
        genai.configure(api_key="your_api_keys")

        # model = genai.GenerativeModel('gemini-1.5-flash')
        model = genai.GenerativeModel("gemini-1.5-flash")
        system_message = "You are an AI assistant that helps people find information."

        chat = model.start_chat(
            history=[
                {"role": "user", "parts": system_message},
            ]
        )

        try_time = 0
        while try_time<3:
            try:
                response = chat.send_message(prompt)
                print("Google response: ")
                return (response.text)
                break
            except Exception as e:
                error_message = str(e)
                print(f"Google error: {error_message}")
                print("Retrying in 2 seconds...")
                try_time += 1
                time.sleep(40)
                    

    # openai_api_base = "http://localhost:8000/v1"
    elif "gpt" in model:
        openai_api_key = "your_api_keys"
        if model == "gpt4":
            # model = "gpt-4-0613"
            model = "gpt-4-turbo"
        else:
            model = "gpt-3.5-turbo-0125"
        # model = "gpt-3.5-turbo-0125"
        # model = "gpt-3.5-turbo"
        # model = "gpt-4-turbo"
        # model = "gpt-4o"
        client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=openai_api_key,
            # base_url=openai_api_base,
        )
    else:
        openai_api_base = "http://localhost:8000/v1"
        openai_api_key = "EMPTY"
        client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
    system_message = "You are an AI assistant that helps people find information."
    messages = [{"role": "system", "content": system_message}]
    message_prompt = {"role": "user", "content": prompt}
    messages.append(message_prompt)
    try_time = 0
    while try_time<3:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=512,
                frequency_penalty= 0,
                presence_penalty=0
            )
            result = response.choices[0].message.content
            break
        except Exception as e:
            error_message = str(e)
            print(f"OpenAI error: {error_message}")
            print("Retrying in 2 seconds...")
            try_time += 1
            time.sleep(2)

    print(f"{model} response: ")

        # print("end openai")

    return result

def prepare_dataset(dataset_name):
    if dataset_name == 'cwq':
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
        ID = 'ID'

    elif dataset_name == 'cwq_multi':
        with open('../data/cwq_multi.json',encoding='utf-8') as f:
            datas = json.load(f)
        ID = 'ID'
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
        ID = "QuestionId"
        # answer = ""
    elif dataset_name == 'webqsp_multi':
        with open('../data/webqsp_multi.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'grailqa':
        with open('../data/grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
        ID = 'qid'
        
    elif dataset_name == 'simpleqa':
        with open('../data/SimpleQA.json',encoding='utf-8') as f:
            datas = json.load(f)    
        question_string = 'question'
        ID = 'question'
    elif dataset_name == 'qald':
        with open('../data/qald_10-en.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'   
    elif dataset_name == 'webquestions':
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        ID = 'question'
        question_string = 'question'
    elif dataset_name == 'trex':
        with open('../data/T-REX.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'zeroshotre':
        with open('../data/Zero_Shot_RE.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'creak':
        with open('../data/creak.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'sentence'
    else:
        print("dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}.")
        exit(-1)
    return datas, question_string, ID
import time
import concurrent.futures
import time


# Example usage (outside of function definitions):
# results = explore_graph_from_entities(["entity1", "entity2"])
def Multi_relation_search(entity_id, head=True):
    # Fetch head relations
    if (head == True):
        sparql_relations_extract_head = sparql_head_relations % (format(entity_id))
        head_relations = execurte_sparql(sparql_relations_extract_head)
        relations = replace_relation_prefix(head_relations)
    else:
    # Fetch tail relations
        sparql_relations_extract_tail = sparql_tail_relations % (format(entity_id))
        tail_relations = execurte_sparql(sparql_relations_extract_tail)
        relations = replace_relation_prefix(tail_relations)


    # # Prune unnecessary relations
    # if args.remove_unnecessary_rel:
    #     head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
    #     tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
    
    return relations


def Multi_entity_search(entity, relation, head=True):
    if head:
        sparql_query = sparql_tail_entities_extract % (format(entity), format(relation))
    else:
        # sparql_query = sparql_head_entities_extract % (format(entity), format(relation))
        sparql_query = sparql_head_entities_extract % (format(relation), format(entity))

    entities = execurte_sparql(sparql_query)  # ensure this function is correctly spelled as `execute_sparql`
    entity_ids = replace_entities_prefix(entities)
    new_entity = [entity for entity in entity_ids if entity.startswith("m.")]
    return new_entity



def bfs_expand_one_hop(entity, graph_storage, is_head):
    """Perform a single hop expansion for a given entity."""
    # start = time.time()
    # print(f"Expanding {entity} in {'head' if is_head else 'tail'} direction...")
    relations = Multi_relation_search(entity, is_head)
    # end = time.time()
    # print(f"Time taken to fetch {entity} in {'head' if is_head else 'tail'}  relations: {end - start:.2f} seconds")
    new_entities = set()
    if (len(relations) > 0):
        for relation in relations:
            # start2 = time.time()
            connected_entities = Multi_entity_search(entity, relation, is_head)
            # end2_2 = time.time()
            # print(f"Time taken to fetch entities: {end2_2 - start2:.2f} seconds")
            if len(connected_entities) > 0:
                if is_head:
                    if graph_storage.get((entity, relation)) is None:
                        graph_storage[(entity, relation)] = connected_entities
                    else:
                        for connected_entity in connected_entities:
                            if connected_entity not in graph_storage[(entity, relation)]:
                                graph_storage[(entity, relation)].append(connected_entity)
                        # graph_storage[(entity, relation)].extend(connected_entities)
                else:
                    for connected_entity in connected_entities:
                        if graph_storage.get((connected_entity, relation)) is None:
                            graph_storage[(connected_entity, relation)] = [entity]
                        else:
                            for entity in connected_entities:
                                if entity not in graph_storage[(connected_entity, relation)]:

                                     graph_storage[(connected_entity, relation)].append(entity)
                # graph_storage[(entity, relation)] = connected_entities
                new_entities.update(connected_entities)
    #         end2_3 = time.time()
    #         # print(f"Time taken to update graph storage: {end2_3 - end2_2:.2f} seconds")
    #     end2 = time.time()
    # print(f"Time taken to fetch {entity} in {'head' if is_head else 'tail'} entities: {end2 - end:.2f} seconds")

    return new_entities

from concurrent.futures import ThreadPoolExecutor


def replace_prefix1(data):
    if data is None:
        print("Warning: No data available to process in replace_prefix1.")
        return []
    # Function to process results and replace prefixes or format data
    return [{key: value['value'].replace("http://rdf.freebase.com/ns/", "") for key, value in result.items()} for result in data]


def search_relations_and_entities(entity_id, head=True):
    if head:
        sparql_query = sparql_head_entities_and_relations % entity_id
    else:
        sparql_query = sparql_tail_entities_and_relations % entity_id
    results = execute_sparql(sparql_query)
    return replace_prefix1(results)


import concurrent.futures
from threading import Lock
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
def search_relations_and_entities_combined(entity_id):
    sparql_query = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?relation ?connectedEntity ?direction
    WHERE {
        {
            ns:%s ?relation ?connectedEntity .
            BIND("tail" AS ?direction)
        }
        UNION
        {
            ?connectedEntity ?relation ns:%s .
            BIND("head" AS ?direction)
        }
    }
    """ % (entity_id, entity_id)
    results = execute_sparql(sparql_query)
    return replace_prefix1(results)
# Ensure you have proper implementations for search_relations_and_entities_combined and are_entities_connected
def search_relations_and_entities_combined_1(entity_id):
    sparql_query = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?relation ?connectedEntity ?connectedEntityName ?direction
    WHERE {
        {
            ns:%s ?relation ?connectedEntity .
            OPTIONAL {
                ?connectedEntity ns:type.object.name ?name .
                FILTER(lang(?name) = 'en')
            }
            BIND(COALESCE(?name, "Unnamed Entity") AS ?connectedEntityName)
            BIND("tail" AS ?direction)
        }
        UNION
        {
            ?connectedEntity ?relation ns:%s .
            OPTIONAL {
                ?connectedEntity ns:type.object.name ?name .
                FILTER(lang(?name) = 'en')
            }
            BIND(COALESCE(?name, "Unnamed Entity") AS ?connectedEntityName)
            BIND("head" AS ?direction)
        }
    }
    """ % (entity_id, entity_id)
    results = execute_sparql(sparql_query)
    return replace_prefix1(results)

def explore_graph_from_entities_by_hop_neighbor_1(entity_ids, max_depth=5,answer_name=[]):
    current_entities = set(entity_ids)
    all_entities = set(entity_ids)
    found_answer = False
    entity_names = {entity: id2entity_name_or_type(entity) for entity in entity_ids}  # 默认所有初始实体名称为"unnamedentity"
    graph = {entity: {} for entity in all_entities}  # 初始化图
    storage_lock = Lock()  # 创建线程安全锁
    answer_name_set = set(answer_name)
    empty_set = set()
    if len(entity_ids) == 1:
        connect = True
    else:
        connect = False
    hopnumber = 5

    for depth in range(1, max_depth + 1):
        print(f"Exploring entities at depth {depth}...")
        start = time.time()
        new_entities = set()

        with ThreadPoolExecutor(max_workers=80) as executor:
            futures = {executor.submit(search_relations_and_entities_combined_1, entity): entity for entity in current_entities}
            for future in as_completed(futures):
                results = future.result()
                entity = futures[future]
                for result in results:
                    relation, connected_entity, connected_name, direction = result['relation'], result['connectedEntity'], result['connectedEntityName'], result['direction']

                    if connected_entity.startswith("m."):

                        if connected_name in answer_name_set:
                            empty_set.add(connected_entity)
                            if len(empty_set) == len(answer_name_set):
                                found_answer = True
                        with storage_lock:
                            # 更新或添加实体名称
                            entity_names[connected_entity] = connected_name
                            # 确保图中包含相关实体和关系
                            if entity not in graph:
                                graph[entity] = {}
                            if connected_entity not in graph:
                                graph[connected_entity] = {}
                            if connected_entity not in graph[entity]:
                                graph[entity][connected_entity] = {'forward': set(), 'backward': set()}
                            if entity not in graph[connected_entity]:
                                graph[connected_entity][entity] = {'forward': set(), 'backward': set()}
                            # 更新关系
                            if direction == "tail":
                                graph[entity][connected_entity]['forward'].add(relation)
                                graph[connected_entity][entity]['backward'].add(relation)
                            else:  # direction is "head"
                                graph[entity][connected_entity]['backward'].add(relation)
                                graph[connected_entity][entity]['forward'].add(relation)
                        new_entities.add(connected_entity)


        new_entities.difference_update(all_entities)
        all_entities.update(new_entities)
        current_entities = new_entities
        end = time.time()
        print(f"Time taken to explore depth {depth}: {end - start:.2f} seconds")
        if connect == False:
            connect = are_entities_connected(graph, entity_ids, all_entities)
            if connect:
                print(f"All entities are connected within {depth} hops.")
                hopnumber = depth

        if found_answer and connect:
            if depth == hopnumber:
                return (True, graph, hopnumber, all_entities, current_entities, entity_names, True)
            return (True, graph, hopnumber,all_entities, current_entities, entity_names, False)

    print("Entities are not fully connected or answer entity not found within the maximum allowed hops.")
    return (False, graph, hopnumber,all_entities, current_entities, entity_names, False)
# Ensure you have proper implementations for search_relations_and_entities_combined and are_entities_connected



def bfs_expand_one_hop2(entity, graph_storage, is_head, executor):
    relations = Multi_relation_search(entity, is_head)
    new_entities = set()
    if relations:
        future_to_relation = {executor.submit(Multi_entity_search, entity, relation, is_head): relation for relation in relations}
        results = {}
        for future in concurrent.futures.as_completed(future_to_relation):
            relation = future_to_relation[future]
            try:
                results[relation] = future.result()
            except Exception as e:
                print(f"Error processing {relation}: {e}")
                continue

        for relation, connected_entities in results.items():
            if connected_entities:
                # with threading.Lock():  # 确保线程安全
                if is_head:
                    graph_storage.setdefault((entity, relation), set()).update(connected_entities)
                else:
                    for connected_entity in connected_entities:
                        graph_storage.setdefault((connected_entity, relation), set()).add(entity)
                new_entities.update(connected_entities)
    return new_entities

def explore_graph_from_entities2(total_entities, max_depth=5):
    graph_storage = {}
    current_entities = set(total_entities)
    all_entities = set(total_entities)
    def process_entity(entity):
        # Both head and tail expansion for a single entity
        new_head_entities = bfs_expand_one_hop2(entity, graph_storage, True, executor)
        new_tail_entities = bfs_expand_one_hop2(entity, graph_storage, False, executor)

        return new_head_entities | new_tail_entities  # Union of sets 

    with ThreadPoolExecutor(max_workers=150) as executor:
        for depth in range(1, max_depth + 1):
            print(f"Exploring entities at depth {depth}...")
            future_to_entity = {executor.submit(process_entity, entity): entity for entity in current_entities}
            next_entities = set()
            for future in concurrent.futures.as_completed(future_to_entity):
                next_entities.update(future.result())

            new_current_entities = next_entities - all_entities
            all_entities.update(next_entities)
            current_entities = new_current_entities

            print(f"Checking connectivity at depth {depth}...")
            if are_entities_connected(graph_storage, total_entities):
                print(f"All entities are connected within {depth} hops.")
                return (True, graph_storage, all_entities, depth)

    print("Entities are not fully connected within the maximum allowed hops.")
    return (False, graph_storage, all_entities, max_depth)



# Global ThreadPoolExecutor

def bfs_expand_one_hop3(entity, graph_storage, is_head):
    executor1 = concurrent.futures.ThreadPoolExecutor(max_workers = 80)

    """Perform a single hop expansion for a given entity."""
    relations = Multi_relation_search(entity, is_head)
    new_entities = set()
    if relations:
        # Perform entity searches in parallel
        future_to_relation = {executor1.submit(Multi_entity_search, entity, relation, is_head): relation for relation in relations}
        results = {}
        for future in concurrent.futures.as_completed(future_to_relation):
            relation = future_to_relation[future]
            results[relation] = future.result()

        # Update graph_storage and new_entities
        for relation, connected_entities in results.items():
            if connected_entities:
                if is_head:
                    if graph_storage.get((entity, relation)) is None:
                        graph_storage[(entity, relation)] = set(connected_entities)
                    else:
                        graph_storage[(entity, relation)].update(connected_entities)
                else:
                    for connected_entity in connected_entities:
                        if graph_storage.get((connected_entity, relation)) is None:
                            graph_storage[(connected_entity, relation)] = {entity}
                        else:
                            graph_storage[(connected_entity, relation)].add(entity)
                new_entities.update(connected_entities)
    return new_entities

def explore_graph_from_entities3(total_entities, max_depth=5):
    graph_storage = {}
    current_entities = set(total_entities)  # Start with the initial set of entities
    all_entities = set(total_entities)      # To track all discovered entities

    for depth in range(1, max_depth + 1):
        print(f"Exploring entities at depth {depth}...")
        next_entities = set()
        
        for entity in current_entities:
            new_head_entities = bfs_expand_one_hop3(entity, graph_storage, True)
            new_tail_entities = bfs_expand_one_hop3(entity, graph_storage, False)
            
        next_entities.update(new_head_entities)
        next_entities.update(new_tail_entities)

        # Calculate new current_entities before updating all_entities
        new_current_entities = next_entities - all_entities

        # Update the set of all entities
        all_entities.update(next_entities)

        # Update current_entities to only include newly discovered entities
        current_entities = new_current_entities

        print("Checking connectivity at depth {depth}...")
        if are_entities_connected(graph_storage, total_entities):
            print(f"All entities are connected within {depth} hops.")
            # all_paths = find_all_paths(graph_storage, total_entities, all_entities)
            # for path in all_paths:
            #     print("Path:", " -> ".join(path))
            return (True, graph_storage, all_entities, depth)

    print("Entities are not fully connected within the maximum allowed hops.")
    return (False, graph_storage, all_entities, depth)



from collections import deque
# from collections import deque

def are_entities_connected(graph, total_entities, all_entities):
    """
    Check if starting from the first entity in total_entities, all other entities in total_entities can be visited.
    graph: Dictionary with entity as key and another dictionary {connected_entity: {'forward': set(), 'backward': set()}} as value.
    total_entities: Set of initial entities to check connectivity from.
    """
    if not total_entities:
        return True  # If no entities are provided, they are trivially connected.

    total_entities_set = set(total_entities)
    if len(total_entities_set) == 1:
        return True  # Only one entity, trivially connected to itself.

    start_entity = next(iter(total_entities_set))  # Start BFS from any entity in the set
    visited = set()
    queue = deque([start_entity])

    while queue:
        current = queue.popleft()
        if current not in visited:
            visited.add(current)
            # Early termination check
            if total_entities_set.issubset(visited):
                return True

            # Add connected entities to the queue
            for connected_entity, relations in graph[current].items():
                if connected_entity not in visited:
                    queue.append(connected_entity)

    # Final check in case not all entities are connected
    return False



# def are_entities_connected1(graph_storage, total_entities):
    """
    Check if starting from the first entity in total_entities, all other entities in total_entities can be visited.
    graph_storage: Dictionary storing connections (head, relation) -> [connected_entities]
    total_entities: List or Set of initial entities to check connectivity from.
    """
    if not total_entities:
        return True  # If no entities are provided, they are trivially connected.

    total_entities_set = set(total_entities)

    if len(total_entities_set) == 1:
        return True  # Only one entity, trivially connected to itself.

    start_entity = next(iter(total_entities_set)) 

    visited = set()
    queue = deque([start_entity])  # Using deque for efficient pops from the front

    while queue:
        current = queue.popleft()  # O(1) time complexity
        if current in visited:
            continue
        visited.add(current)

        if total_entities_set.issubset(visited):
            return True

        # Process each connection where the current entity is involved
        for (head, relation), tails in graph_storage.items():
            if head == current:
                for tail in tails:
                    if tail not in visited:
                        queue.append(tail)
            elif current in tails and head not in visited:
                queue.append(head)

    return False

import concurrent.futures
import time

# Global ThreadPoolExecutor


def bfs_expand_one_hop1(entity, graph_storage, is_head):
    """Perform a single hop expansion for a given entity."""
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=100)
    relations = Multi_relation_search(entity, is_head)
    new_entities = set()
    if relations:
        # Perform entity searches in parallel
        future_to_relation = {executor.submit(Multi_entity_search, entity, relation, is_head): relation for relation in relations}
        results = {}
        for future in concurrent.futures.as_completed(future_to_relation):
            relation = future_to_relation[future]
            results[relation] = future.result()

        # Update graph_storage and new_entities
        for relation, connected_entities in results.items():
            if connected_entities:
                if is_head:
                    if graph_storage.get((entity, relation)) is None:
                        graph_storage[(entity, relation)] = set(connected_entities)
                    else:
                        graph_storage[(entity, relation)].update(connected_entities)
                else:
                    for connected_entity in connected_entities:
                        if graph_storage.get((connected_entity, relation)) is None:
                            graph_storage[(connected_entity, relation)] = {entity}
                        else:
                            graph_storage[(connected_entity, relation)].add(entity)
                new_entities.update(connected_entities)
    return new_entities




def initialize_graph(graph_storage, all_entities):
    graph = {entity: {} for entity in all_entities}
    for (head, relation), tails in graph_storage.items():
        for tail in tails:
            if tail not in graph[head]:
                graph[head][tail] = {}
            if 'forward' not in graph[head][tail]:
                graph[head][tail]['forward'] = set()
            graph[head][tail]['forward'].add(relation)

            # 存储反向关系
            if tail not in graph:
                graph[tail] = {}
            if head not in graph[tail]:
                graph[tail][head] = {}
            if 'backward' not in graph[tail][head]:
                graph[tail][head]['backward'] = set()
            graph[tail][head]['backward'].add(relation)
    return graph


from functools import lru_cache




from collections import deque


from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import itertools

def process_node(start_paths, goal_paths, node):
    local_paths = []
    for f_path in start_paths[node]:
        for b_path in goal_paths[node]:
            # 确保 f_path 和 b_path 始终为列表
            f_path = f_path if isinstance(f_path, list) else [f_path]
            b_path = b_path if isinstance(b_path, list) else [b_path]
            # local_paths.append(tuple(f_path))
            # local_paths.append(tuple(b_path))
            try:
                if len(b_path) > 1:
                    combined_path = f_path + b_path[::-1][1:]  # 确保 b_path 长度大于1
                else:
                    combined_path = f_path  # 如果 b_path 只有一个元素，只取 f_path
                if len(combined_path)>1:
                    local_paths.append(tuple(combined_path))
            except TypeError as e:
                print(f"TypeError combining paths: {e}")
                print(f"f_path: {f_path}, b_path: {b_path}")  # 输出问题数据

    return local_paths


def node_expand_with_paths(graph, start, hop):
    queue = deque([(start, [start])])  # 存储节点和到该节点的路径
    visited = {start: [start]}  # 记录到达每个节点的所有路径

    while queue:
        current_node, current_path = queue.popleft()
        if current_node not in graph:
            print(f"Skipping non-existent node {current_node}")
            continue
        current_layer = len(current_path) - 1
        if current_layer < hop:  # 只扩展到给定的层数
            for neighbor in graph[current_node]:
                if neighbor in current_path:
                    continue  # 跳过已经在路径中的节点，防止回环
                new_path = current_path + [neighbor]
                if neighbor not in visited:
                    visited[neighbor] = []
                    queue.append((neighbor, new_path))
                visited[neighbor].append(new_path)  # 记录到此节点的每条路径

    return visited
def bfs_with_intersection_only(graph, entity_list, hop):
    # 使用多线程并行执行node_expand
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(node_expand_with_paths, graph, entity, hop): entity for entity in entity_list}
        paths_dict = {entity: future.result() for entity, future in zip(entity_list, as_completed(futures))}
    
    # 计算所有实体的路径交集
    intersection = set.intersection(*(set(paths.keys()) for paths in paths_dict.values()))
    return intersection



def create_subgraph_through_intersection3s(graph, entity_list, hop):
    from collections import defaultdict
    import copy

    # Initialize a function to safely add nodes and relationships to the subgraph
    def safe_add_edge(subgraph, src, dst, relation, direction):
        if src not in subgraph:
            subgraph[src] = {}
        if dst not in subgraph[src]:
            subgraph[src][dst] = {'forward': set(), 'backward': set()}
        subgraph[src][dst][direction].add(relation)

    # Use ThreadPoolExecutor to handle node expansion
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(node_expand_with_paths, graph, entity, hop): entity for entity in entity_list}
        paths_dict = {entity: future.result() for entity, future in zip(entity_list, as_completed(futures))}

    subgraph = {}

    # 计算所有实体的路径交集
    intersection = set.intersection(*(set(paths.keys()) for paths in paths_dict.values()))
    print("Find all the intersection nodes")

    # Iterate through each entity's paths to build the subgraph
    for node in intersection:
        for paths in paths_dict.values():
            for path in paths[node]:
                path = path if isinstance(path, list) else [path]
                if len(path) > 1:
                    for i in range(0, len(path) - 1):
                        src, dst = path[i], path[i + 1]
                        if src in graph and dst in graph[src]:
                            for direction in ['forward', 'backward']:
                                for relation in graph[src][dst][direction]:
                                    safe_add_edge(subgraph, src, dst, relation, direction)
                                for relation in graph[dst][src][direction]:
                                    safe_add_edge(subgraph, dst, src, relation, direction)
                        else:
                            print(path)
                            print(f"Missing edge or node in the graph from {src} to {dst}")

    return subgraph


def create_subgraph_through_intersections(graph, entity_list, intersection, total_id_to_name_dict, hop):
    # 使用多线程并行执行 node_expand_with_paths


    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(node_expand_with_paths, graph, entity, hop): entity for entity in entity_list}
        paths_dict = {entity: future.result() for entity, future in zip(entity_list, as_completed(futures))}

    complete_subgraph = {}
    reduce_entity_names = {}

    for paths in paths_dict.values():
        for end_node, all_paths in paths.items():
            if end_node in intersection:
                # 将所有通过交集节点的路径加入到子图中
                for path in all_paths:
                    path = path if isinstance(path, list) else [path]
                    if len(path) > 1:
                        try:
                            for i in range(len(path) - 1):
                                head, tail = path[i], path[i + 1]
                                if head not in complete_subgraph:
                                    complete_subgraph[head] = {}
                                if tail not in complete_subgraph[head]:
                                    complete_subgraph[head][tail] = graph[head][tail].copy()
                                if tail not in complete_subgraph:
                                    complete_subgraph[tail] = {}
                                if head not in complete_subgraph[tail]:
                                    complete_subgraph[tail][head] = graph[tail][head].copy()
                        except KeyError as e:
                            print(f"An error occurred when processing edge from {head} to {tail}: {e}")
    for en_id in complete_subgraph.keys():
        reduce_entity_names[en_id] = total_id_to_name_dict.get(en_id, "Unnamed Entity")

    return complete_subgraph,reduce_entity_names



def find_all_paths_bibfs_itersection(graph, total_entities, hop, if_using_all_r):
    all_paths = []
    # entity_list = sorted(total_entities, key=lambda x: len(graph.get(x, {})))

    raw_paths = bfs_with_intersection(graph, total_entities, hop)
    if if_using_all_r:
        for path in raw_paths:
            all_paths.extend(add_relations_to_path_with_all_R(graph, path))

        # all_paths = all_paths.extend(add_relations_to_path_with_all_R(graph, path) for path in raw_paths)
    else:
        all_paths = [add_relations_to_path1(graph, path) for path in raw_paths]

    return merge_paths_by_relations(all_paths)
    # return all_paths

def find_all_paths_bibfs_itersection_limit(graph, total_entities, hop, if_using_all_r):
    all_paths = []
    # entity_list = sorted(total_entities, key=lambda x: len(graph.get(x, {})))

    raw_paths = bfs_with_intersection_inter(graph, total_entities, hop)

    # raw_paths = bfs_with_intersection_testv1(graph, total_entities, hop)
    if if_using_all_r:
        for path in raw_paths:
            all_paths.extend(add_relations_to_path_with_all_R(graph, path))
    else:
        all_paths = [add_relations_to_path1(graph, path) for path in raw_paths]

    return merge_paths_by_relations(all_paths)
    # return all_paths

import difflib

def find_best_matching_substring(entity, cot_line):
    len_entity = len(entity)
    len_cot = len(cot_line)

    # Consider substrings within reasonable lengths
    min_len = max(1, len_entity // 2)
    max_len = min(len_cot, len_entity * 2)

    best_score = 0
    best_start = -1

    for length in range(min_len, max_len + 1):
        for start in range(len_cot - length + 1):
            substring = cot_line[start:start + length]
            score = difflib.SequenceMatcher(None, entity, substring).ratio()
            if score > best_score:
                best_score = score
                best_start = start

    return best_score, best_start

def reorder_entities(cot_line, topic_entity_dict):
    entity_positions = []

    for entity in topic_entity_dict:
        score, position = find_best_matching_substring(entity, cot_line)
        # Assign a high position if no match is found
        if position != -1:
            entity_positions.append((position, entity))
        else:
            entity_positions.append((float('inf'), entity))

    # Sort entities based on their positions in cot_line
    entity_positions.sort()
    sorted_entities = [entity for position, entity in entity_positions]
    return sorted_entities

def bfs_with_intersection_inter(graph, entity_list, hop):
    # 使用多线程并行执行node_expand
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(node_expand_with_paths, graph, entity, hop): entity for entity in entity_list}
        paths_dict = {entity: future.result() for entity, future in zip(entity_list, as_completed(futures))}
    paths = set()
    
    # 计算所有实体的路径交集
    if len(entity_list) == 1:
        local_paths = []
        for nei in paths_dict[entity_list[0]].values():
            for f_path in nei:
                # 确保 f_path 和 b_path 始终为列表
                f_path = f_path if isinstance(f_path, list) else [f_path]
                try:

                    paths.add(tuple(f_path))
                except TypeError as e:
                    print(f"TypeError combining paths: {e}")
        print("Only one entity, return all paths")
        # print("local_paths", len(local_paths))
        # print("local_paths", len(paths))
        # print("local_paths", local_paths)
        # paths.update(local_paths)
        return list(paths)
    
    intersection = set.intersection(*(set(paths.keys()) for paths in paths_dict.values()))
    print("Find all the intersection nodes")
    if not intersection:
        return []

    combination_path_dict = {}
    with ThreadPoolExecutor() as executor:
        for i in range(1, len(entity_list)):
            futures = []
            start_entity_paths = paths_dict[entity_list[i - 1]]
            target_entity_paths = paths_dict[entity_list[i]]
            # intersection = set(start_entity_paths.keys()) & set(target_entity_paths.keys())
            if not intersection:
                return []
            for node in intersection:
                futures.append(executor.submit(process_node, start_entity_paths, target_entity_paths, node))
            
            # Collect results for this entity pair
            combination_paths = []
            for future in as_completed(futures):
                combination_paths.extend(future.result())
            combination_path_dict[(entity_list[i - 1], entity_list[i])] = combination_paths

    # Combine all paths
    total_paths = combine_all_paths(combination_path_dict, entity_list)
    print(entity_list)
    return total_paths

def bfs_with_intersection(graph, entity_list, hop):
    # Perform node expansion in parallel
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(node_expand_with_paths, graph, entity, hop): entity for entity in entity_list}
        paths_dict = {entity: future.result() for entity, future in zip(entity_list, as_completed(futures))}

    if len(entity_list) == 1:
        paths = set()
        for nei in paths_dict[entity_list[0]].values():
            for f_path in nei:
                f_path = f_path if isinstance(f_path, list) else [f_path]
                if len(f_path) > 1:
                    paths.add(tuple(f_path))
        print("Only one entity, return all paths")
        return list(paths)
    
    combination_path_dict = {}
    total_paths = None
    
    for i in range(len(entity_list) - 1):
        start_entity = entity_list[i]
        target_entity = entity_list[i+1]
        start_entity_paths = paths_dict[start_entity]
        target_entity_paths = paths_dict[target_entity]
        intersection = set(start_entity_paths.keys()) & set(target_entity_paths.keys())
        if not intersection:
            return []
        temp_paths = []
        for node in intersection:
            temp_paths.extend(process_node(start_entity_paths, target_entity_paths, node))
        if total_paths is None:
            total_paths = temp_paths
        else:
            combined_paths = []
            for path1 in total_paths:
                for path2 in temp_paths:
                    if path1[-1] == path2[0]:
                        combined_paths.append(path1 + path2[1:])
            total_paths = combined_paths
            if not total_paths:
                return []

    return total_paths


def combine_all_paths(combination_path_dict, entity_list):
    # Start with the paths between the first pair
    
    total_paths = combination_path_dict.get((entity_list[0], entity_list[1]), [])
    for i in range(2, len(entity_list)):
        next_paths = combination_path_dict.get((entity_list[i - 1], entity_list[i]), [])
        combined_paths = []
        for path1 in total_paths:
            for path2 in next_paths:
                # Ensure the paths can be connected
                if path1[-1] == path2[0]:
                    # Avoid duplicate nodes
                    combined_path = path1 + path2[1:]
                    combined_paths.append(combined_path)
        if not combined_paths:
            return []
        total_paths = combined_paths
    return total_paths


def add_relations_to_path1(graph, path):
    """Add relation information to a completed path."""
    full_path = []
    for i in range(len(path) - 1):
        node = path[i]
        next_node = path[i + 1]
        relations_dict = graph[node][next_node]
        relation_strings = []
        for direction, relations in relations_dict.items():
            direction_symbol = " ->" if direction == 'forward' else " <-"
            if isinstance(relations, set):
                relations = list(relations) 
            for relation in relations:
                relation_strings.append(f"{direction_symbol} {relation} {direction_symbol}")
        relation_strings.sort()
        top1 = relation_strings[0]
        relation_string = "{" + (top1) + "}"

        # relation_string = "{" + ", ".join(relation_strings) + "}"
        full_path.append(node)
        full_path.append(relation_string)
    full_path.append(path[-1])
    return full_path

def add_relations_to_path_with_all_R(graph, path):
    """Add all relation information to a completed path, generating different paths accordingly."""
    import itertools

    # Build a list of possible relations between each pair of nodes
    relations_list = []
    for i in range(len(path) - 1):
        node = path[i]
        next_node = path[i + 1]
        relations_dict = graph[node][next_node]
        relation_strings = []
        for direction, relations in relations_dict.items():
            direction_symbol = " ->" if direction == 'forward' else " <-"
            if isinstance(relations, set):
                relations = list(relations)
            for relation in relations:
                relation_strings.append(f"{direction_symbol} {relation} {direction_symbol}")
        relations_list.append(relation_strings)

    # Generate all combinations of relations
    relation_combinations = list(itertools.product(*relations_list))

    # For each combination, build the full path
    paths = []
    for combination in relation_combinations:
        full_path = []
        for i in range(len(path) - 1):
            node = path[i]
            relation_string = "{" + combination[i] + "}"
            full_path.append(node)
            full_path.append(relation_string)
        full_path.append(path[-1])
        paths.append(full_path)
    return paths


def task(graph, entity_list, hop):
    if start in graph and end in graph:
        raw_paths = bfs_with_intersection(graph, entity_list, hop)
        return [add_relations_to_path1(graph, path) for path in raw_paths]
    return []



def expand_node(node, path, graph):
    """扩展给定节点，返回所有可能的下一步"""
    expansions = []
    for next_node, relations_dict in graph[node].items():
        if next_node not in path:  # 防止环路
            new_path = path + [next_node]
            expansions.append((next_node, new_path))
    return expansions



def create_relation_strings(relations_dict, reverse=False):
    relation_strings = []
    for direction, relations in relations_dict.items():
        direction_symbol = " ->" if direction == 'forward' else " <-"
        if reverse:
            direction_symbol = direction_symbol[::-1]  # Reverse the arrow directions
        for relation in set(relations):
            relation_strings.append(f"{direction_symbol} {relation} {direction_symbol}")
    relation_strings.sort()
    return "{" + ", ".join(relation_strings) + "}"

def merge_paths(graph, path_from_start, path_from_goal, is_direct_meet):
    # Ensure that the last element of path_from_start and the first element of path_from_goal are nodes
    last_node_start = path_from_start[-2] if len(path_from_start) > 1 else path_from_start[0]
    first_node_goal = path_from_goal[-2] if len(path_from_goal) > 1 else path_from_goal[0]
    
    if is_direct_meet:
        return path_from_start[:-1] + path_from_goal[::-1]

    try:
        middle_relation = create_relation_strings(graph[last_node_start][first_node_goal], reverse=True)
        return path_from_start + [middle_relation] + path_from_goal[::-1][1:]
    except KeyError as e:
        print(f"KeyError encountered: {e}")
        return []  # Return an empty list or handle the error as appropriate for your application



def merge_paths_by_relations(paths):
    from collections import defaultdict
    import itertools

    # Organize paths by their relation sequences
    paths_by_relations = defaultdict(list)
    for path in paths:
        relations = tuple(path[i] for i in range(1, len(path), 2))
        paths_by_relations[relations].append(path)

    # Merge paths with the same relation sequences
    merged_paths = []
    for relations, paths in paths_by_relations.items():
        # This will hold the final merged path
        merged_path = []
        # We know all paths have the same length and relations, so we iterate through entities
        for i in range(0, len(paths[0]), 2):  # Only iterate over entity indices
            # Gather all entities at this position across all paths
            entities = {path[i] for path in paths}
            if len(entities) > 1:
                merged_entity = "{" + ", ".join(sorted(set(entities))) + "}"
            else:
                merged_entity = list(entities)[0]  # Just take the single entity
            merged_path.append(merged_entity)
            if i < len(paths[0]) - 1:  # Add the relation if it's not the last element
                merged_path.append(paths[0][i+1])

        merged_paths.append(merged_path)

    return merged_paths



def merge_paths_custom_format(paths, intersection_nodes):
    from collections import defaultdict

    # First, separate paths by intersection node presence and collect segments after the first intersection node
    segments_by_intersection = defaultdict(list)
    for path in paths:
        for idx, node in enumerate(path):
            if node in intersection_nodes:
                before = tuple(path[:idx+1])  # Segment up to and including intersection node
                after = tuple(path[idx+1:])  # Everything after intersection node
                segments_by_intersection[before].append(after)
                break

    # Prepare to merge and format the paths
    merged_paths = []
    for before, afters in segments_by_intersection.items():
        merged_path = list(before)  # Start with the segment before the intersection node, excluding it
        # Handle multiple different segments after the intersection
        for after in afters:
            if after:  # Ensure there is a segment to process
                merged_path += ['{'+'AND}', before[-1]] + list(after)  # Include 'AND', the intersection node, and the segment

        # Remove the first 'AND' for proper formatting
        if merged_path[0] == '{'+'AND}':
            merged_path = merged_path[2:]

        merged_paths.append(merged_path)

    return merged_paths

def merge_and_format_paths(paths, intersection_nodes):
    from collections import defaultdict

    # First, merge by relations
    paths_by_relations = defaultdict(list)
    for path in paths:
        relations = tuple(path[i] for i in range(1, len(path), 2))
        paths_by_relations[relations].append(path)

    # Initial merge based on relation sequences
    preliminary_merged_paths = []
    for relations, paths in paths_by_relations.items():
        merged_path = []
        for i in range(0, len(paths[0]), 2):  # Iterate over entity indices
            entities = {path[i] for path in paths}
            merged_entity = "{" + ", ".join(sorted(entities)) + "}" if len(entities) > 1 else list(entities)[0]
            merged_path.append(merged_entity)
            if i < len(paths[0]) - 1:
                merged_path.append(paths[0][i+1])
        preliminary_merged_paths.append(merged_path)

    # Then, merge by intersection and format
    segments_by_intersection = defaultdict(list)
    for path in preliminary_merged_paths:
        for idx, node in enumerate(path):
            if node.strip('{}').split(', ')[0] in intersection_nodes:  # Adjusting for merged entities
                before = tuple(path[:idx+1])
                after = tuple(path[idx+1:])
                segments_by_intersection[before].append(after)
                break

    # Final merging and formatting
    final_merged_paths = []
    for before, afters in segments_by_intersection.items():
        merged_path = list(before)  # Exclude the intersection node initially
        for after in afters:
            if after:  # Ensure there is something to process
                merged_path += ['{'+'AND}', before[-1]] + list(after)
        if merged_path[0] == '{'+'AND}':
            merged_path = merged_path[2:]
        final_merged_paths.append(merged_path)

    return final_merged_paths




def merge_and_format_paths_segmented(paths, intersection_nodes, main_entities):
    from collections import defaultdict

    # First, merge by relations
    paths_by_relations = defaultdict(list)
    for path in paths:
        relations = tuple(path[i] for i in range(1, len(path), 2))
        paths_by_relations[relations].append(path)

    # Initial merge based on relation sequences
    preliminary_merged_paths = []
    for relations, paths in paths_by_relations.items():
        merged_path = []
        for i in range(0, len(paths[0]), 2):  # Iterate over entity indices
            entities = {path[i] for path in paths}
            merged_entity = "{" + ", ".join(sorted(entities)) + "}" if len(entities) > 1 else list(entities)[0]
            merged_path.append(merged_entity)
            if i < len(paths[0]) - 1:
                merged_path.append(paths[0][i+1])
        preliminary_merged_paths.append(merged_path)

    # Then, merge by intersection and format
    segments_by_intersection = defaultdict(list)
    for path in preliminary_merged_paths:
        for idx, node in enumerate(path):
            if node.strip('{}').split(', ')[0] in intersection_nodes:  # Adjusting for merged entities
                before = tuple(path[:idx+1])
                after = tuple(path[idx+1:])
                segments_by_intersection[before].append(after)
                break

    # Final merging and formatting
    final_merged_paths = []
    for before, afters in segments_by_intersection.items():
        merged_path = list(before)  # Exclude the intersection node initially
        initial_entities = set(before[::2])  # Capture initial entities in the path before the intersection
        for after in afters:
            # Append only main entities not in the initial segment
            filtered_entities = [ent for ent in main_entities if ent not in initial_entities and ent in after]
            if after:  # Ensure there is something to process
                merged_path += ['{'+'AND}', before[-1]]  # Add 'AND' and the intersection node
                for ent in filtered_entities:
                    # Append each entity exactly once along with its associated relations if any
                    ent_idx = after.index(ent)
                    if ent_idx < len(after) - 1:  # Ensure there is a relation to follow
                        merged_path += [ent, after[ent_idx + 1]]
        final_merged_paths.append(merged_path)

    return final_merged_paths

from collections import deque
from concurrent.futures import ThreadPoolExecutor


from collections import deque







def extract_first_ten_words(text):
    # 将字符串按空格分割成单词列表
    words = text.split()
    
    # 提取前10个单词
    first_ten_words = words[:10]
    
    # 将单词列表重新组合成字符串
    return ' '.join(first_ten_words)
def format_paths_to_natural_language_id_with_name(paths, entity_id_to_name, version =1):
    natural_language_paths = []
    print("version", version)

    for path in paths:
        formatted_path = []
        # print(type(path))    

        for i, element in enumerate(path):
            if i % 2 == 0:  # Assuming even indices are entities and odd are relations
                try:
                    # print(element)
                    # print(type(element))    
                    if element.startswith('{'):

                        entities = element.strip('{}').split(', ')
                        formatted_entities = []
                        for e in entities[:20]:
                            if version == 2:
                                # entity_name = entity_id_to_name[element] if element in entity_id_to_name else id2entity_name_or_type(element)

                                entity_name = entity_id_to_name.get(e.strip(), id2entity_name_or_type(e.strip()))
                            else:
                                entity_name = entity_id_to_name.get(e.strip())
                            formatted_entities.append(e.strip() + ": " + extract_first_ten_words(entity_name))
                        # Limiting to first 5 unique entities if more than 5 are present
                        formatted_entities = list(set(formatted_entities))
                        formatted_path.append("{" + ", ".join(formatted_entities) + "}")
                    else:
                        # Single entity handling
                        # entity_name = entity_id_to_name.get(element, id2entity_name_or_type(element))
                        # print("element 2")
                        if version == 2:
                            # entity_name = entity_id_to_name[element] if element in entity_id_to_name else id2entity_name_or_type(element)

                            entity_name = entity_id_to_name.get(element, id2entity_name_or_type(element))
                        else:
                            entity_name = entity_id_to_name.get(element)
                        formatted_path.append("{" + element + ": " + extract_first_ten_words(entity_name) + "}")
                except:
                    print(type(element))
                    print(path)
                    print(f"KeyError encountered for element {element}")
                    exit()
            else:
                # Adding relation as is
                formatted_path.append(element)
        # Creating a readable natural language path
        natural_language = " - ".join(formatted_path)
        natural_language_paths.append(natural_language)

    return natural_language_paths


def merge_paths_by_relations_remove_usless(paths):
    from collections import defaultdict
    import itertools

    # Organize paths by their relation sequences
    paths_by_relations = defaultdict(list)
    for path in paths:
        relations = tuple(path[i] for i in range(1, len(path), 2))
        paths_by_relations[relations].append(path)

    # Merge paths with the same relation sequences
    merged_paths = []
    for relations, paths in paths_by_relations.items():
        # This will hold the final merged path
        merged_path = []
        # We know all paths have the same length and relations, so we iterate through entities
        for i in range(0, len(paths[0]), 2):  # Only iterate over entity indices
            # Gather all entities at this position across all paths
            entities = {path[i] for path in paths}
            if len(entities) > 1:

                merged_entity = "{" + ", ".join(sorted(set(entities))) + "}"

            else:
                merged_entity = list(entities)[0]  # Just take the single entity
            merged_path.append(merged_entity)
            if i < len(paths[0]) - 1:  # Add the relation if it's not the last element
                merged_path.append(paths[0][i+1])

        merged_paths.append(merged_path)

    return merged_paths



def find_1_hop_relations_and_entities(entity, graph,entity_id_to_name, ifusing_all_R):
    # results = search_relations_and_entities_combined(entity)

    all_path = []
    for r_entity in graph[entity]:
        # continue
        if ifusing_all_R:


            path = add_relations_to_path_with_all_R(graph, [entity, r_entity])   
            all_path.extend(path)

        else:
            path = add_relations_to_path1(graph, [entity, r_entity])
            all_path.append(path)

    merge_path = merge_paths_by_relations_remove_usless(all_path)
    new_nl_related_paths = format_paths_to_natural_language_id_with_name(merge_path,entity_id_to_name)

    # if id2entity_name_or_type(entity) == "Unnamed Entity":
    #     format_paths_to_natural_language_new_parall(merge_path)
    # new_nl_related_paths1, entity_id_to_name = format_paths_to_natural_language_new_parall_remove_nonname(merge_path, entity_id_to_name)


    return new_nl_related_paths





def explore_graph_from_one_topic_entities(current_entities, graph, entity_names, exlored_entities,all_entities):
   
    # all_entities = set(entity_ids)


    storage_lock = Lock()  # 创建线程安全锁



    print(f"Exploring entities ...")
    start = time.time()
    new_entities = set()

    with ThreadPoolExecutor(max_workers=80) as executor:
        futures = {executor.submit(search_relations_and_entities_combined_1, entity): entity for entity in current_entities}
        exlored_entities.update(current_entities)
        for future in as_completed(futures):
            results = future.result()
            entity = futures[future]
            for result in results:
                relation, connected_entity, connected_name, direction = result['relation'], result['connectedEntity'], result['connectedEntityName'], result['direction']
                if connected_entity.startswith("m."):
                    if connected_entity in exlored_entities:
                        continue
                    with storage_lock:
                        # 更新或添加实体名称
                        entity_names[connected_entity] = connected_name
                        # 确保图中包含相关实体和关系
                        if entity not in graph:
                            graph[entity] = {}
                        if connected_entity not in graph:
                            graph[connected_entity] = {}
                        if connected_entity not in graph[entity]:
                            graph[entity][connected_entity] = {'forward': set(), 'backward': set()}
                        if entity not in graph[connected_entity]:
                            graph[connected_entity][entity] = {'forward': set(), 'backward': set()}
                        # 更新关系
                        if direction == "tail":
                            graph[entity][connected_entity]['forward'].add(relation)
                            graph[connected_entity][entity]['backward'].add(relation)
                        else:  # direction is "head"
                            graph[entity][connected_entity]['backward'].add(relation)
                            graph[connected_entity][entity]['forward'].add(relation)
                    new_entities.add(connected_entity)


        new_entities.difference_update(exlored_entities)
        all_entities.update(new_entities)
        current_entities = new_entities
        # print ((all_entities))
        # print((exlored_entities))
        # print ((current_entities))

    # print("Entities are not fully connected or answer entity not found within the maximum allowed hops.")
    return (graph, all_entities, exlored_entities, current_entities, entity_names)
# Ensure you have proper implementations for search_relations_and_entities_combined and are_entities_connected
def extract_brace_contents(path):
    """
    提取路径中所有大括号内的内容。
    """
    return re.findall(r'\{([^}]+)\}', path)

def concatenate_paths_with_unlinked(list1, list2):
    """
    连接两个路径列表中的路径，如果 list1 中路径的最后一个大括号内容
    与 list2 中路径的第一个大括号内容相同，则将它们连接起来。
    
    同时，将未被连接的路径单独输出。
    
    返回一个包含所有连接后路径和未连接路径的列表。
    """
    # 创建一个字典，键为 list2 中路径的第一个大括号内容，值为路径本身
    list2_dict = {}
    for path2 in list2:
        braces2 = extract_brace_contents(path2)
        if braces2:
            first_brace2 = braces2[0]
            if first_brace2 not in list2_dict:
                list2_dict[first_brace2] = []
            list2_dict[first_brace2].append(path2)
    
    concatenated_paths = []
    linked_list1 = set()
    linked_list2 = set()
    
    # 遍历 list1，检查每个路径的最后一个大括号内容是否在 list2_dict 中
    for idx1, path1 in enumerate(list1):
        braces1 = extract_brace_contents(path1)
        if braces1:
            last_brace1 = braces1[-1]
            if last_brace1 in list2_dict:
                for path2 in list2_dict[last_brace1]:
                    braces2 = extract_brace_contents(path2)
                    if braces2:
                        # 移除 path2 的第一个大括号内容，以避免重复
                        concatenated_braces = braces1 + braces2[1:]
                        # 重新构建连接后的路径
                        concatenated_path = " - ".join(f'{{{brace}}}' for brace in concatenated_braces)
                        concatenated_paths.append(concatenated_path)
                        # 记录已连接的路径
                        linked_list1.add(idx1)
                        linked_list2.add(list2.index(path2))
    
    # 收集未被连接的路径
    unlinked_paths = []
    
    # 未连接的 list1 路径
    for idx1, path1 in enumerate(list1):
        if idx1 not in linked_list1:
            unlinked_paths.append(path1)
    
    # 未连接的 list2 路径
    for idx2, path2 in enumerate(list2):
        if idx2 not in linked_list2:
            unlinked_paths.append(path2)
    
    # 合并连接后的路径和未连接的路径
    result = concatenated_paths + unlinked_paths
    
    return result



def check_answerlist(dataset_name, question_string, ori_question, ground_truth_datas, origin_data):
    answer_list= []
    # origin_data = [j for j in ground_truth_datas if j[question_string] == ori_question]
    if dataset_name == 'cwq':
        answer_list.append(origin_data["answer"])

    elif dataset_name == 'webqsp':
        answers = origin_data["Parses"]
        for answer in answers:
            for name in answer['Answers']:
                if name['EntityName'] == None:
                    answer_list.append(name['AnswerArgument'])
                else:
                    answer_list.append(name['EntityName'])

    elif dataset_name == 'grailqa':
        answers = origin_data["answer"]
        for answer in answers:
            if "entity_name" in answer:
                answer_list.append(answer['entity_name'])
            else:
                answer_list.append(answer['answer_argument'])

    elif dataset_name == 'simpleqa':
        answers = origin_data["answer"]
        answer_list.append(answers)

        
    elif dataset_name == 'webquestions':
        answer_list = origin_data["answers"]


    return list(set(answer_list))

def check_answer(answer, answer_list):
    if not answer or not answer["LLM_answer"]:
    # if "LLM_answer" not in answer:
        return False
    # if not answer["LLM_answer"]:
        # return False
    lower_answer = answer["LLM_answer"].strip().replace(" ","").lower()
    getanswer = clean_results(lower_answer)
    for answer_name in answer_list:
        lower_answer_name = answer_name.strip().replace(" ","").lower()
        if lower_answer_name in lower_answer:
            # print("answer is found in the LLM answer")
            return True

    if len(getanswer) > 0:
        for getanswer_e in getanswer:
            for answer_name in answer_list:
                lower_answer_name = answer_name.strip().replace(" ","").lower()
                if getanswer_e in lower_answer_name:
                    # print("answer is found in the LLM answer")
                    return True
    return False


def calculate_f1_score(answer, answer_list):
    if not answer or not answer["LLM_answer"]:
    # if "LLM_answer" not in answer:
        return False
    # if not answer["LLM_answer"]:
        # return False
    lower_answer = answer["LLM_answer"].strip().replace(" ","").lower()
    prediction = clean_results(lower_answer)
    ground_truth = [answer_name.strip().replace(" ","").lower() for answer_name in answer_list]

    interseaction_number = 0
    for gt in ground_truth:
        if gt in lower_answer:
            interseaction_number += 1
            continue
        for pred in prediction:
            
            if pred in gt:
                interseaction_number += 1
                break
            elif gt in pred:
                interseaction_number += 1
                break

    # if len(prediction) == 0 and interseaction_number == 0:
    #     inter = 0
    #     for gt in ground_truth:
    #         if gt in lower_answer:
    #             inter += 1
    #     interseaction_number = inter
    #     prediction = inter
    #     # return 0
    if len(prediction) == 0:
        # preception = interseaction_number/len(prediction)
        if len(prediction) == 0 and interseaction_number >= 0:
            preception = interseaction_number/len(ground_truth)
        # else:
        # preception = interseaction_number/len(prediction)

            # prediction = interseaction_number
    else:
        preception = interseaction_number/len(prediction)
    recall = interseaction_number/len(ground_truth)
    if preception + recall == 0:
        return 0
    f1 = 2 * preception * recall / (preception + recall)
    return f1

def clean_results(string):
    # top_list_str = ""
    # match = re.search(r'swer:\s*\{([^}]+)\}', string)
    # if not match:
    #     # match = re.search(r'list:\s*\{([^}]+)\}', text)
    #     return ""
    
    # top_list_str = match.group(1)
    # return top_list_str
   
# Adjust the function to handle a single string input

    # Split the input string by 'answer:' to isolate each section
    # print("++++++++++++++++++++")
    # print(string)
    if "answer:{" not in string:
        return []
    # else:   
        # print("+++==========++++++++++")
    sections = string.split('answer:')
    
    all_answers = []
    
    for section in sections[1:]:  # Skip the first part since it doesn't contain an answer
        # Extract the part between curly braces
        # print(section)
        # print("++++++++++++++++++++")
        # print(string)
        #get string after "answer:" in one line from section
        string = section.split("\n")[0]
        replace_string = string.replace("{",",").replace("}",",")
        # answers = section.split('{')[1].split('}')[0]
        # Split by comma and strip any spaces
        answers_list = [answer.strip() for answer in replace_string.split(',')]
        # Add to the overall list
        all_answers.extend(answers_list)
    
    # Remove duplicates and return the final list
    all_answers = list(set(all_answers))
    all_answers = [x for x in all_answers if x != ""]
    return list(set(all_answers))



def check_refuse(string):
    refuse_words = ["however", "sorry"]
    return any(word in string.lower() for word in refuse_words)


def exact_match(response, answers):
    clean_result = response.strip().replace(" ","").lower()
    for answer in answers:
        clean_answer = answer.strip().replace(" ","").lower()
        if clean_result == clean_answer or clean_result in clean_answer or clean_answer in clean_result:
            return True
    return False
