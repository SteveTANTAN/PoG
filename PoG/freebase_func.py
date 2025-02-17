from SPARQLWrapper import SPARQLWrapper, JSON, XML
# SPARQLPATH = "http://localhost:8890/sparql"  # depend on your own internal address and port, shown in Freebase folder's readme.md
SPARQLPATH = "http://localhost:8890/sparql"  # depend on your own internal address and port, shown in Freebase folder's readme.md


sparql_head_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ns:%s ?relation ?x .\n}"""
sparql_tail_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ?x ?relation ns:%s .\n}"""
sparql_tail_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\nns:%s ns:%s ?tailEntity .\n}""" 
sparql_head_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\n?tailEntity ns:%s ns:%s  .\n}"""
sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity ns:type.object.name ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n}"""


sparql_tail_entities_and_relations = """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?relation ?tailEntity
WHERE {
    ns:%s ?relation ?tailEntity .
}
"""

sparql_head_entities_and_relations = """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?relation ?headEntity
WHERE {
    ?headEntity ?relation ns:%s .
}
"""


def check_end_word(s):
    words = [" ID", " code", " number", "instance of", "website", "URL", "inception", "image", " rate", " count"]
    return any(s.endswith(word) for word in words)

def abandon_rels(relation):
    if relation == "type.object.type" or relation == "type.object.name" or relation.startswith("common.") or relation.startswith("freebase.") or "sameAs" in relation:
        return True

def format(entity_id):
    return entity_id

def format1(entity_id):
    if "http://" in entity_id:
        return f"<{entity_id}>"
    else:
        return f"ns:{entity_id}"
    return entity_id

import time
from urllib.error import HTTPError

def execurte_sparql(sparql_txt):
    # Assuming SPARQLPATH is a variable that holds your SPARQL endpoint URL
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_txt)
    sparql.setReturnFormat(JSON)
    
    attempts = 0
    while attempts < 3:  # Set the number of retries
        try:
            results = sparql.query().convert()
            return results["results"]["bindings"]
        except Exception as e:
            print("404 Error encountered. Retrying after 2 seconds...")
            print(e)
            time.sleep(2)  # Sleep for 2 seconds before retrying
            attempts += 1  

    print("Failed to execute after multiple attempts.")
    return None

def execute_sparql(sparql_txt):
    # Assuming SPARQLPATH is a variable that holds your SPARQL endpoint URL
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_txt)
    sparql.setReturnFormat(JSON)
    
    attempts = 0
    while attempts < 3:  # Set the number of retries
        try:
            results = sparql.query().convert()
            return results["results"]["bindings"]
        except Exception as e:
            print("404 Error encountered. Retrying after 2 seconds...")
            print(e)

            time.sleep(2)  # Sleep for 2 seconds before retrying
            attempts += 1  

    print("Failed to execute after multiple attempts.")
    return None

def replace_relation_prefix(relations):
    if relations is None:
        return []  
    return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/","") for relation in relations]

def replace_entities_prefix(entities):
    if entities is None:
        return []  
    return [entity['tailEntity']['value'].replace("http://rdf.freebase.com/ns/","") for entity in entities]



from functools import lru_cache
import re
@lru_cache(maxsize=1024)
def id2entity_name_or_type(entity_id):
    # sparql_id = "YOUR_SPARQL_QUERY_HERE"
    init_id = entity_id
    entity_id = sparql_id % (format(entity_id), format(entity_id))
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(entity_id)
    sparql.setReturnFormat(JSON)
    # results = sparql.query().convert()
    results = []
    attempts = 0
    while attempts < 3:  # Set the number of retries
        try:
            results = sparql.query().convert()
            break
            # return results["results"]["bindings"]
        except Exception as e:
            print("404 Error encountered. Retrying after 2 seconds...")
            print(e)
            time.sleep(2)  # Sleep for 2 seconds before retrying
            attempts += 1  

    if attempts == 3:
        print("Failed to execute after multiple attempts.")

    if len(results["results"]["bindings"]) == 0:
        return "Unnamed Entity"
    else:
        # First, filter to find results with 'xml:lang': 'en'
        english_results = [result['tailEntity']['value'] for result in results["results"]["bindings"] if result['tailEntity'].get('xml:lang') == 'en']
        if english_results:
            return english_results[0]  # Return the first English result

        # If no English results, find entries that match English letters or numbers
        alphanumeric_results = [result['tailEntity']['value'] for result in results["results"]["bindings"]
                                if re.match("^[a-zA-Z0-9 ]+$", result['tailEntity']['value'])]
        if alphanumeric_results:
            return alphanumeric_results[0]  # Return the first alphanumeric result

        return "Unnamed Entity"

