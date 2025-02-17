split_question_prompt = """
You will receive a multi-hop question, which is composed of several interconnected queries, along with a list of topic entities that serve as the main keywords for the question. Please split the multi-hop question into parts for clarity and depth.

please give me the Thinking CoT which contains chain relations of "ALL" the topic entities and predicted answer type, and serval split questions.

Each entity in the topic list is already included in the knowledge graph. Your task is to consider how to obtain the answer by using all the topic entities and split the multi-hop question into multiple simpler questions using one Topic Entity once. Each split question should explore the relationship between one of the topic entities and the others entities or the answer. Your goal is to determine how to derive the final answer by systematically addressing each split question.

note: the number of split questions should be equal to the number of topic entities. since each split question is work for one topic entity.




For example:
Q: 
Question: What country bordering France contains an airport that serves Nijmegen?
also writen as: what countries share borders with france and is the location contains and airport that server Nijmegen?
Main Topic Entities: {m.05g2b: Nijmegen, m.0f8l9c: France}
A:
Thinking Cot:  "Nijmegen" - service by - airport - owned by - answer(country) - border with - "France"
split_question1: What country contains an airport that serves "Nijmegen"?
split_question2: What country bordering "France"?

please provide answer(A) section only in the same format as the example above.

The question is:

"""


answer_n_explore_prompt = """
Given a main question, an uncertain LLM-generated thinking Cot that consider all the entities, a few split questions that you can use stepply and finally obtain the final answer, and the associated retrieved knowledge graph path, {set of entities (with id start with "m.")} -> {set of relationships} -> {set of entities(with id start with "m.")}, 

Your task is to determine if this knowledge graph path is sufficient to answer the given main question or with your pretrained knowledge. 

If it's sufficient, you need to respose {Yes}, and provide the answer to the main question. If the answer is obtained from the given knowledge path, it should be the entity name from the path. Otherwise, you need to respose {No}, then explain the reason. 

for example:
Q:What educational institution has a football sports team named Northern Colorado Bears is in Greeley, Colorado? 
Thinking CoT:  "Northern Colorado Bears football" - has team - answer(educational institution) - located in - "Greeley"
split_question1: What educational institution has a football team named "Northern Colorado Bears football"?
split_question2: What educational institution is located in "Greeley"? 
Topic entity paths: 
path 1 :  {m.0lqj7vt: Northern Colorado Bears football} -> { -> sports.school_sports_team.school  ->,  <- education.educational_institution.sports_teams  <-} -> {m.01_k7f: University of Northern Colorado} -> { -> location.location.containedby  ->,  <- location.location.contains  <-} -> {m.0rczx: Greeley}
A: 
response: {Yes}. 
answer: {University of Northern Colorado}
reason: Given {m.0lqj7vt: Northern Colorado Bears football} -> { -> sports.school_sports_team.school  ->,  <- education.educational_institution.sports_teams  <-} -> {m.01_k7f: University of Northern Colorado}, {University of Northern Colorado} is answer of the split question1 , 
and given {m.01_k7f: University of Northern Colorado} -> { -> location.location.containedby  ->,  <- location.location.contains  <-} -> {m.0rczx: Greeley}, the location of the {University of Northern Colorado} is {Greeley}, 
therefore, the provided knowledge graph path is sufficient to answer the overall question, and the answer is {University of Northern Colorado}.

Q: Where did the "Country Nation World Tour" concert artist go to college?
Thinking CoT:  Country Nation World Tour - performed by - Brad Paisley - graduated from - answer(educational institution) - with - Bachelor's degree
split_question1: Who performed the "Country Nation World Tour"?
split_question2: What college did Brad Paisley graduate from? 
Topic entity path: 
Path:1 {m.010qhfmm: Country Nation World Tour} -> { -> music.concert_tour.artist  ->,  <- music.artist.concert_tours  <-} -> {m.03gr7w: Brad Paisley} -> { -> people.person.profession  ->,  <- people.profession.people_with_this_profession  <-} -> {m.02hrh1q: Actor, m.09jwl: Musician} -> { <- base.yupgrade.user.topics  <-} -> {m.07y51vk: Unnamed Entity, m.06wfq8f: Good Morning America, m.06j0_km: RainnWilson} -> { -> base.yupgrade.user.topics  ->} -> {m.019v9k: Bachelor's degree}
Path:2 {m.010qhfmm: Country Nation World Tour} -> { -> music.concert_tour.album_or_release_supporting  ->,  <- music.album.supporting_tours  <-} -> {m.010xc046: Moonshine in the Trunk, m.0r4tvsy: Wheelhouse} -> { -> music.album.artist  ->,  <- music.artist.album  <-} -> {m.03gr7w: Brad Paisley} -> { -> people.person.education  ->,  <- education.education.student  <-} -> {m.0h3d7qj: Unnamed Entity} -> { -> education.education.degree  ->,  <- education.educational_degree.people_with_this_degree  <-} -> {m.019v9k: Bachelor's degree}
Path:3 {m.010qhfmm: Country Nation World Tour} -> { -> music.concert_tour.artist  ->,  <- music.artist.concert_tours  <-} -> {m.03gr7w: Brad Paisley} -> { -> people.person.education  ->,  <- education.education.student  <-} -> {m.0h3d7qj: Unnamed Entity} -> { -> education.education.degree  ->,  <- education.educational_degree.people_with_this_degree  <-} -> {m.019v9k: Bachelor's degree}
A: 
response: {No}.
Reason: From the Topic entity path, the m.03gr7w: Brad Paisley from the candidate list in path 2&3 is answer of the split question1, and the m.0h3d7qj: Unnamed Entity in the path 2&3 is related the college information which granted a bachlor degree, and the specific name of the school is not shown in the supplymentary edge section. therefore, the provided knowledge graph path is not sufficient to answer the overall question.


Q: When did the team with Crazy crab as their mascot win the world series?
Thinking CoT: "Crazy Crab" - mascot of - team - won - World Series 
split_question1: What team has "Crazy Crab" as their mascot?
split_question2: When did the team with "Crazy Crab" as their mascot win the World Series?
Topic entity paths:  
path 1 :  {m.02q_hzh: Crazy Crab} - { -> sports.mascot.team  ->} - {m.0713r: San Francisco Giants} - { <- sports.sports_championship_event.runner_up  <-} - {m.06jwmt: 1987 National League Championship Series, m.04tfr4: 1962 World Series, m.0468cj: 2002 World Series, m.04j747: 1989 World Series, m.0dqwx9: 1971 National League Championship Series}
path 2 :  {m.02q_hzh: Crazy Crab} - { -> sports.mascot.team  ->} - {m.0713r: San Francisco Giants} - { -> sports.sports_team.championships  ->} - {m.09gnk2r: 2010 World Series, m.0ds8qct: 2012 World Series, m.0117q3yz: 2014 World Series}
A:
response: {Yes}.
answer: {2010 World Series, 2012 World Series, 2014 World Series}
reason: From the given path {m.02q_hzh: Crazy Crab} - { -> sports.mascot.team  ->} - {m.0713r: San Francisco Giants}, {San Francisco Giants} is answer of the split question1, 
and from {m.0713r: San Francisco Giants} - { -> sports.sports_team.championships  ->} - {m.09gnk2r: 2010 World Series, m.0ds8qct: 2012 World Series, m.0117q3yz: 2014 World Series},  the World Series won by the {San Francisco Giants} are {2010, 2012, 2014}, 
therefore, the provided knowledge graph path is sufficient to answer the overall question, and the answer is {2010 World Series, 2012 World Series, 2014 World Series}.

Q: who did tom hanks play in apollo 13?
Thinking CoT: "Tom Hanks" - played - character - in - "Apollo 13" 
split_question1: Who did Tom Hanks play in "Apollo 13"?
split_question2: Which character did Tom Hanks portray in the movie "Apollo 13"?
Topic entity paths:  
path 1:  {m.0bxtg: Tom Hanks} - { -> film.actor.film  ->} - {m.0jtp74: Unnamed Entity} - { -> film.performance.film  ->} - {m.011yd2: Apollo 13}
path 2:  {m.0bxtg: Tom Hanks} - { -> award.award_winner.awards_won  ->} - {m.0b79lv8: Unnamed Entity, m.0b79zf0: Unnamed Entity, m.0mzydgm: Unnamed Entity} - { -> award.award_honor.honored_for  ->} - {m.011yd2: Apollo 13}
A:
response: {No}.
reason: The character that Tom Hanks played in "Apollo 13" is represented as {Unnamed Entity}, but the name is not given. therefore, we need additional information to answer the question.

The question is:
"""


answer_generated_direct = """
Given a main question, an uncertain LLM-generated thinking Cot that consider all the entities, a few split questions that you can use stepply and finally obtain the final answer, and the associated retrieved knowledge graph path, {set of entities (with id start with "m.")} -> {set of relationships} -> {set of entities(with id start with "m.")}, 

Your task is to generated the answer based on the given knowledge graph path and your own knowledge.

please give me the answer section only in the same format as the example below:

for example:
Q:What educational institution has a football sports team named Northern Colorado Bears is in Greeley, Colorado? 
Thinking CoT:  "Northern Colorado Bears football" - has team - answer(educational institution) - located in - "Greeley"
split_question1: What educational institution has a football team named "Northern Colorado Bears football"?
split_question2: What educational institution is located in "Greeley"? 
Topic entity paths: 
path 1 :  {m.0lqj7vt: Northern Colorado Bears football} -> { -> sports.school_sports_team.school  ->,  <- education.educational_institution.sports_teams  <-} -> {m.01_k7f: University of Northern Colorado} -> { -> location.location.containedby  ->,  <- location.location.contains  <-} -> {m.0rczx: Greeley}
A: 
answer: {University of Northern Colorado}
reason: Given {m.0lqj7vt: Northern Colorado Bears football} -> { -> sports.school_sports_team.school  ->,  <- education.educational_institution.sports_teams  <-} -> {m.01_k7f: University of Northern Colorado}, {University of Northern Colorado} is answer of the split question1 , 
and given {m.01_k7f: University of Northern Colorado} -> { -> location.location.containedby  ->,  <- location.location.contains  <-} -> {m.0rczx: Greeley}, the location of the {University of Northern Colorado} is {Greeley}, 
therefore, the provided knowledge graph path is sufficient to answer the overall question, and the answer is {University of Northern Colorado}.

The question is:
"""



split_answer = """
Given a main question, and the associated retrieved knowledge graph path, {set of entities (with id start with "m.")} -> {set of relationships} -> {set of entities(with id start with "m.")}, 

Your task is to determine if this knowledge graph path is sufficient to answer the given question or with your pretrained knowledge. 

If it's sufficient, you need to respose {Yes}, and provide the answer and reason. If the answer is obtained from the given knowledge path, it should be the entity name from the path. Otherwise, you need to respose {No}, then explain the reason. 

for example:
Q: What educational institution has a football team named "Northern Colorado Bears football"?
Topic entity paths: 
path 1:  {m.0lqj7vt: Northern Colorado Bears football} -> { -> sports.school_sports_team.school  ->,  <- education.educational_institution.sports_teams  <-} -> {m.01_k7f: University of Northern Colorado} -> { -> location.location.containedby  ->,  <- location.location.contains  <-} -> {m.0rczx: Greeley}
A: 
response: {Yes}. 
answer: {University of Northern Colorado}
reason: from {m.0lqj7vt: Northern Colorado Bears football} -> { -> sports.school_sports_team.school  ->,  <- education.educational_institution.sports_teams  <-} -> {m.01_k7f: University of Northern Colorado}, the {University of Northern Colorado} is the answer of the question, and the location of the {University of Northern Colorado} is {Greeley}, therefore, the provided knowledge graph path is sufficient to answer the overall question.

Q: When did the team with "Crazy Crab" as their mascot win the World Series?
Topic entity paths:  
path 1 :  {m.02q_hzh: Crazy Crab} - { -> sports.mascot.team  ->} - {m.0713r: San Francisco Giants} - { <- sports.sports_championship_event.runner_up  <-} - {m.06jwmt: 1987 National League Championship Series, m.04tfr4: 1962 World Series, m.0468cj: 2002 World Series, m.04j747: 1989 World Series, m.0dqwx9: 1971 National League Championship Series}
path 2 :  {m.02q_hzh: Crazy Crab} - { -> sports.mascot.team  ->} - {m.0713r: San Francisco Giants} - { -> sports.sports_team.championships  ->} - {m.09gnk2r: 2010 World Series, m.0ds8qct: 2012 World Series, m.0117q3yz: 2014 World Series}
A:
response: {Yes}.
answer: {2010 World Series, 2012 World Series, 2014 World Series}
reason: from {m.02q_hzh: Crazy Crab} - { -> sports.mascot.team  ->} - {m.0713r: San Francisco Giants} - { -> sports.sports_team.championships  ->} - {m.09gnk2r: 2010 World Series, m.0ds8qct: 2012 World Series, m.0117q3yz: 2014 World Series}, the World Series won by the {San Francisco Giants} are {2010, 2012, 2014}.


Q: What college did Brad Paisley graduate from? 
Topic entity path: 
Topic Path:1 {m.010qhfmm: Country Nation World Tour} -> { -> music.concert_tour.artist  ->,  <- music.artist.concert_tours  <-} -> {m.03gr7w: Brad Paisley} -> { -> people.person.profession  ->,  <- people.profession.people_with_this_profession  <-} -> {m.02hrh1q: Actor, m.09jwl: Musician} -> { <- base.yupgrade.user.topics  <-} -> {m.07y51vk: Unnamed Entity, m.06wfq8f: Good Morning America, m.06j0_km: RainnWilson} -> { -> base.yupgrade.user.topics  ->} -> {m.019v9k: Bachelor's degree}
Topic Path:2 {m.010qhfmm: Country Nation World Tour} -> { -> music.concert_tour.album_or_release_supporting  ->,  <- music.album.supporting_tours  <-} -> {m.010xc046: Moonshine in the Trunk, m.0r4tvsy: Wheelhouse} -> { -> music.album.artist  ->,  <- music.artist.album  <-} -> {m.03gr7w: Brad Paisley} -> { -> people.person.education  ->,  <- education.education.student  <-} -> {m.0h3d7qj: Unnamed Entity} -> { -> education.education.degree  ->,  <- education.educational_degree.people_with_this_degree  <-} -> {m.019v9k: Bachelor's degree}
Topic Path:3 {m.010qhfmm: Country Nation World Tour} -> { -> music.concert_tour.artist  ->,  <- music.artist.concert_tours  <-} -> {m.03gr7w: Brad Paisley} -> { -> people.person.education  ->,  <- education.education.student  <-} -> {m.0h3d7qj: Unnamed Entity} -> { -> education.education.degree  ->,  <- education.educational_degree.people_with_this_degree  <-} -> {m.019v9k: Bachelor's degree}
supplymentary edge:
A: 
response: {No}.
Reason: From the Topic entity path, the m.0h3d7qj: Unnamed Entity in the path 2&3 is related the college information which granted a bachlor degree, and the specific name of the school is not shown in the supplymentary edge section. therefore, the provided knowledge graph path is not sufficient to answer the overall question.



The question is:
"""



explor_prompt_zero_shot = """
Given a question and the associated retrieved knowledge graph path 
{set of entities (with id start with "m.")} -> {set of relationships} -> {set of entities(with id start with "m.")}, 
your task is to identify which entity is required to further exploration to answer the question.

please give me a summary Exploration Required Entities in the format in below in each question beginning

For example:
Q: question
A: {entity1, entity2, entity3,...}

The question is:

"""


GPT_prompt = """
Given a question and the associated retrieved entities list, 
Please score and give me the top three entities can be highly to be the answer of the question.

please give me a summary final answer like "top_entity":["a","b","c"] in each question end
"""


main_path_select_prompt = """
Given a question and the associated retrieved entities lists, 
Please score and give me the top three lists that can be highly to be the answer to the question.

please give me a summary final answer exactly same to the format below in each question beginning

Answer: top_list: {path 1, path 2, path3}
Explanation: The top three lists are path 1, path 2, and path 3....

"""


explore_prompt_v2 = """
Given a question, the associated retrieved knowledge graph path and candidate entities list
please score and give me the top entities from the candiate list that can be highly to be the answer of the question, in the decreasing order.
please give me a summary final answer in the format in below in each question beginning.
Warnning: you have to select the entities form the candidate list, not the entities in the path.

please give me a summary final answer in the format in below in each question beginning

A: top_entity: {Entity 1, Entity 2, Entity3}
Explanation: The top three Entities are Entity 1, Entity 2, and Entity 3....

for example:

Q: 
Question: Where did the "Country Nation World Tour" concert artist go to college?
The question also writen as: where did the artist had a concert tour named Country Nation World Tour graduate from college
candidate list: [m.010qhfmm: Country Nation World Tour, m.03gr7w: Brad Paisley, m.0h3d7qj: Unnamed Entity, m.06j0_km: RainnWilson, m.06wfq8f: Good Morning America, m.07v6r3f: Unnamed Entity, m.019v9k: Bachelor's degree]
entity path: 
Path:1 {m.010qhfmm: Country Nation World Tour} -> { -> music.concert_tour.artist  ->,  <- music.artist.concert_tours  <-} -> {m.03gr7w: Brad Paisley} -> { -> people.person.nationality  ->} -> {m.09c7w0: United States of America} -> { <- base.yupgrade.user.topics  <-} -> {m.06j0_km: RainnWilson, m.06wfq8f: Good Morning America, m.07v6r3f: Unnamed Entity} -> { -> base.yupgrade.user.topics  ->} -> {m.019v9k: Bachelor's degree}
Path:2 {m.010qhfmm: Country Nation World Tour} -> { -> music.concert_tour.artist  ->,  <- music.artist.concert_tours  <-} -> {m.03gr7w: Brad Paisley} -> { -> people.person.profession  ->,  <- people.profession.people_with_this_profession  <-} -> {m.02hrh1q: Actor, m.09jwl: Musician} -> { <- base.yupgrade.user.topics  <-} -> {m.06j0_km: RainnWilson, m.06wfq8f: Good Morning America, m.07y51vk: Unnamed Entity} -> { -> base.yupgrade.user.topics  ->} -> {m.019v9k: Bachelor's degree}
Path:3 {m.010qhfmm: Country Nation World Tour} -> { -> music.concert_tour.album_or_release_supporting  ->,  <- music.album.supporting_tours  <-} -> {m.0r4tvsy: Wheelhouse, m.010xc046: Moonshine in the Trunk} -> { -> music.album.artist  ->,  <- music.artist.album  <-} -> {m.03gr7w: Brad Paisley} -> { -> people.person.education  ->,  <- education.education.student  <-} -> {m.0h3d7qj: Unnamed Entity} -> { -> education.education.degree  ->,  <- education.educational_degree.people_with_this_degree  <-} -> {m.019v9k: Bachelor's degree}
A:
top_entity: {m.0h3d7qj: Unnamed Entity, m.03gr7w: Brad Paisley, m.019v9k: Bachelor's degree}
Explanation: m.03gr7w: Brad Paisley - Strong connection through multiple paths directly linking to the artist's education. m.019v9k: Bachelor's degree - Directly mentioned in the education path and linked to the artist. m.0h3d7qj: Unnamed Entity - Mentioned in the educational context in Path 3, likely representing an educational institution or a specific aspect of education.

The question is:
"""

explored_path_select_prompt = """
Given a main question, a LLM-generated thinking Cot that consider all the entities, a few split questions that you can use stepply and finally obtain the final answer, and the associated retrieved knowledge graph path, {set of entities (with id start with "m.")} -> {set of relationships} -> {set of entities(with id start with "m.")}, 

Please score and give me the top three lists from the candidate set can be highly to be the answer of the question.
please answer in the same format, such as, top_list:{Candidate Edge 1, Candidate Edge 3, Candidate Edge4}.

For example:
Q: question
exsiting path: {set of entities (with id start with "m.")} -> {set of relationships} -> {set of entities(with id start with "m.")}
Candidate Edge 1:  {set of entities (with id start with "m.")} -> {set of relationships} -> {set of entities(with id start with "m.")}  
Candidate Edge 2:  {set of entities (with id start with "m.")} -> {set of relationships} -> {set of entities(with id start with "m.")}  
Candidate Edge 3:  {set of entities (with id start with "m.")} -> {set of relationships} -> {set of entities(with id start with "m.")}  
Candidate Edge 4:  {set of entities (with id start with "m.")} -> {set of relationships} -> {set of entities(with id start with "m.")}  
A: top_list:{Candidate Edge 1, Candidate Edge 3, Candidate Edge4}

the question is:
"""



"""

Explanation: The top three lists are Candidate Edge 1, Candidate Edge 2, and Candidate Edge 3....

"""

path_select_to_remove_prompt = """
Given a question and the associated retrieved entities lists, 
Please score and give me the top lists that can be highly unlikely related to the answer of the question.

please give me a summary final answer in the format in below in each question beginning

Answer: top_list: {path 1, path 2, path3}
Explanation: The top three lists are path 1, path 2, and path 3....

"""

'''



'''

explore_prompt_v31 = """
Given a question, a candidate entities list, the associated accuracy retrieved knowledge path, and unsure LLM-generated thought chains, 
The LLM-generated thought chains are the possible answers to the question based on the retrieved knowledge path but not sure about their correctness.
Your task is to explore the candidate entities list and select new entities can lead to answer, by consider the retrieved knowledge path and LLM-generated thought chains.

please score and give me the top three entities from the candidate list, in the decreasing order.
please give me a summary final answer in the format in below in each question beginning.
Warning: you have to select the entities from the candidate list, not the entities in the path.

please give me a summary final answer in the format below in each question beginning
top_entity: {Entity 1, Entity 2, Entity3}
Explanation: The top three entities are Entity 1, Entity 2, and Entity 3....

For example:
Q: 
Question: Where did the "Country Nation World Tour" concert artist go to college?
The question also writen as: where did the artist had a concert tour named Country Nation World Tour graduate from college
Candidate List:  ['m.010qhfmm: Country Nation World Tour', 'm.03gr7w: Brad Paisley', 'm.09c7w0: United States of America', 'm.06wfq8f: Good Morning America', "m.019v9k: Bachelor's degree", 'm.09jwl: Musician', 'm.0r4tvsy: Wheelhouse', 'm.0h3d7qj: Unnamed Entity', 'Music artist', 'Brad Paisley Christmas', 'Belmont University']
entity path: 
Path:1 {m.010qhfmm: Country Nation World Tour} -> { -> music.concert_tour.artist  ->,  <- music.artist.concert_tours  <-} -> {m.03gr7w: Brad Paisley} -> { -> people.person.nationality  ->} -> {m.09c7w0: United States of America} -> { <- base.yupgrade.user.topics  <-} -> {m.06j0_km: RainnWilson, m.06wfq8f: Good Morning America, m.07v6r3f: Unnamed Entity} -> { -> base.yupgrade.user.topics  ->} -> {m.019v9k: Bachelor's degree}
Path:2 {m.010qhfmm: Country Nation World Tour} -> { -> music.concert_tour.artist  ->,  <- music.artist.concert_tours  <-} -> {m.03gr7w: Brad Paisley} -> { -> people.person.profession  ->,  <- people.profession.people_with_this_profession  <-} -> {m.02hrh1q: Actor, m.09jwl: Musician} -> { <- base.yupgrade.user.topics  <-} -> {m.06j0_km: RainnWilson, m.06wfq8f: Good Morning America, m.07y51vk: Unnamed Entity} -> { -> base.yupgrade.user.topics  ->} -> {m.019v9k: Bachelor's degree}
Path:3 {m.010qhfmm: Country Nation World Tour} -> { -> music.concert_tour.album_or_release_supporting  ->,  <- music.album.supporting_tours  <-} -> {m.0r4tvsy: Wheelhouse, m.010xc046: Moonshine in the Trunk} -> { -> music.album.artist  ->,  <- music.artist.album  <-} -> {m.03gr7w: Brad Paisley} -> { -> people.person.education  ->,  <- education.education.student  <-} -> {m.0h3d7qj: Unnamed Entity} -> { -> education.education.degree  ->,  <- education.educational_degree.people_with_this_degree  <-} -> {m.019v9k: Bachelor's degree}
LLM_generated answer: 
CoT1: {Country Nation World Tour} -> { -> music.concert_tour.artist ->, <- music.artist.concert_tours <-} -> {Brad Paisley} -> { -> people.person.education ->, <- education.education.student <-} -> {Belmont University} -> { -> education.education.degree ->, <- education.educational_degree.people_with_this_degree <-} -> {Bachelor's degree}
CoT2: {Country Nation World Tour} -> { -> music.concert_tour.artist ->, <- music.artist.concert_tours <-} -> {Brad Paisley} -> { -> people.person.education ->, <- education.education.student <-} -> {Belmont University} -> { -> education.education.degree ->, <- education.educational_degree.people_with_this_degree <-} -> {Bachelor's degree}
CoT3: {Country Nation World Tour} -> { -> music.concert_tour.album_or_release_supporting ->, <- music.album.supporting_tours <-} -> {Wheelhouse, Moonshine in the Trunk} -> { -> music.album.artist ->, <- music.artist.album <-} -> {Brad Paisley} -> { -> people.person.education ->, <- education.education.student <-} -> {Belmont University} -> { -> education.education.degree ->, <- education.educational_degree.people_with_this_degree <-} -> {Bachelor's degree}
A: 
top_entity: {Belmont University, m.0h3d7qj: Unnamed Entity, Brad Paisley}
Explanation: The top three entities,  Belmont University, and m.0h3d7qj: Unnamed Entity, and Brad Paisley, directly relate to answering where the artist from the "Country Nation World Tour" went to college. Brad Paisley is the central figure connecting to Belmont University where he earned his Bachelor's degree, reflecting his educational path directly relevant to the query.

The question is:
"""


explore_prompt_v3 = """
Given a question, a candidate entities list, the associated accuracy retrieved knowledge path, and unsure LLM-generated thought chains, 
The LLM-generated thought chains are the possible answers to the question based on the retrieved knowledge path but not sure about their correctness.
Your task is to explore the candidate entities list and select new entities can lead to answer, by consider the retrieved knowledge path and LLM-generated thought chains.

please score and give me the top three entities from the candidate list, in the decreasing order.
Warning: you have to select the entities from the candidate list, not the entities in the path.

For example:
Q: 
Question: What country bordering France contains an airport that serves Nijmegen?
also writen as: what countries share borders with france and is the location contains and airport that server Nijmegen?

Candidate List:  ['m.05g2b: Nijmegen', 'm.02llzg: Central European Time Zone', 'm.0f8l9c: France', 'm.06fz_: Rhine', ':France', 'Netherlands', 'Luxembourg', 'Location(s)', 'Belgium (remix)', 'Germany', 'France 2', 'France 3', 'Live in Nijmegen', 'Le France', 'Central Time Zone', 'The Netherlands', 'Belgium', 'Location', 'La France', 'France 7', 'Est, Netherlands']
entity path: 
path 1:  {m.05g2b: Nijmegen} -> { -> location.location.time_zones  ->,  <- time.time_zone.locations_in_this_time_zone  <-} -> {m.02llzg: Central European Time Zone} -> { -> time.time_zone.locations_in_this_time_zone  ->,  <- location.location.time_zones  <-} -> {m.0f8l9c: France}
path 2:  {m.05g2b: Nijmegen} -> { <- geography.river.cities  <-} -> {m.06fz_: Rhine} -> { -> geography.river.basin_countries  ->,  -> location.location.partially_containedby  ->,  <- location.location.partially_contains  <-} -> {m.0f8l9c: France}
LLM_generated answer: 
CoT1: {Nijmegen} -> { -> location.location.time_zones ->} -> {Central European Time Zone} -> { -> time.time_zone.locations_in_this_time_zone ->} -> {France} -> { -> location.location.contains ->} -> {Belgium} -> { -> aviation.airport_serves_city ->} -> {Nijmegen}
CoT2: {Nijmegen} -> { -> location.location.time_zones ->} -> {Central European Time Zone} -> { -> time.time_zone.locations_in_this_time_zone ->} -> {France} -> { -> location.location.contains ->} -> {Luxembourg} -> { -> aviation.airport_serves_city ->} -> {Nijmegen}
CoT3: {Nijmegen} -> { <- geography.river.cities <-} -> {Rhine} -> { -> geography.river.basin_countries ->} -> {Germany} -> { -> location.location.contains ->} -> {Netherlands} -> { -> aviation.airport_serves_city ->} -> {Nijmegen}
A: 
top_entity: {Germany, France, Netherlands}
Explanation: The top three entities, France, Netherlands, and Germany, directly relate to the country bordering France that contains an airport serving Nijmegen. France borders the Netherlands, which includes the airport serving Nijmegen, making these countries key in the context of the question.

please give me a summary final answer in the format below in each question beginning
top_entity: {Entity 1, Entity 2, Entity3}
Explanation: The top three entities are Entity 1, Entity 2, and Entity 3....

The question is:
"""



explor_with_COT_prompt = """
Given a question, the associated accuracy retrieved knowledge graph path
{set of entities (with id start with "m.")} -> {set of relationships} -> {set of entities(with id start with "m.")}, 


your task is first to summarize the provided Chains and to identify if the provided knowledge graph path with your knowledge is sufficient to answer the question or not. 

If not, please select which entity is required to further explore to answer the question and revise and generate a new thought chain that uses the newly explored entity can lead to an answer.

please only give me an answer section in the same format below in each question beginning

If Yes:
A: {Yes}
answer: {your answer}
new CoT: entity1 - relationship - entity2 - relationship - entity3
Explanation: The provided knowledge graph paths meet the CoT's thinking and are sufficient to answer the question.

If No:
A: {No}
Exploration Required: {entity1}
New CoT: entity1 - relationship - entity2 - relationship - entity3
Explanation: The provided knowledge graph paths do not meet the CoT's thinking and are not sufficient to answer the question. Further exploration is needed specifically in the entity {entity1} to find the answer.

For example:
Q: 
Question: What country bordering France contains an airport that serves Nijmegen?
also writen as: what countries share borders with france and is the location contains and airport that server Nijmegen?
Main question Path:
path 1:  {m.05g2b: Nijmegen} -> { -> location.location.time_zones  ->,  <- time.time_zone.locations_in_this_time_zone  <-} -> {m.02llzg: Central European Time Zone} -> { -> time.time_zone.locations_in_this_time_zone  ->,  <- location.location.time_zones  <-} -> {m.0f8l9c: France}
path 2:  {m.05g2b: Nijmegen} -> { <- geography.river.cities  <-} -> {m.06fz_: Rhine} -> { -> geography.river.basin_countries  ->,  -> location.location.partially_containedby  ->,  <- location.location.partially_contains  <-} -> {m.0f8l9c: France}
Related_path: 
path 0 :  {m.0345h: Germany} -> { <- film.film.country  <-} -> {m.0ngfc2h: Hannah Arendt, m.0q4137w: Layla Fourie, m.0gxs2pq: A meditation on love, life, death and the human voice, m.04nm0ly: Baksy, m.0_zb111: Judgment In Hungary} -> { -> film.film.country  ->} -> {m.0f8l9c: France}
path 1 :  {m.0345h: Germany} -> { <- film.film.country  <-} -> {m.02qxnt9: César and Rosalie, m.04qbhwn: Dragon Hunters, m.04nl_vc: Khamsa, m.07kcx7c: Bab El-Oued City, m.0fjzwp: Gabrielle} -> { -> film.film.country  ->,  -> media_common.netflix_title.netflix_genres  ->,  <- media_common.netflix_genre.titles  <-} -> {m.0f8l9c: France}
path 2 :  {m.0345h: Germany} -> { -> biology.breed_origin.breeds_originating_here  ->,  <- biology.animal_breed.place_of_origin  <-} -> {m.01p2dp: Affenpinscher, m.0m1ct: Wirehaired Pointing Griffon, m.05sfy0: Löwchen} -> { -> biology.animal_breed.place_of_origin  ->,  <- biology.breed_origin.breeds_originating_here  <-} -> {m.0f8l9c: France}

path 3 :  {m.05g2b: Nijmegen} -> { -> location.location.nearby_airports  ->,  <- aviation.airport.serves  <-} -> {m.06cm5d: Weeze Airport} -> { -> location.location.containedby  ->,  <- location.location.contains  <-} -> {m.0345h: Germany}
path 4 :  {m.012x_5p_: Germany} -> { -> music.release.region  ->} -> {m.02j71: Earth} -> { <- music.release.region  <-} -> {m.0sprr4w: So Serene, m.0g801ww: A State of Trance 2009, m.0126m_r6: Alexandria, m.03xsqd1: Bad Dreams / Omissions, m.049ljb7: Myam James, Part 1} -> { -> music.release.region  ->} -> {m.059j2: Netherlands} -> { -> location.country.second_level_divisions  ->,  -> location.location.contains  ->,  <- location.administrative_division.second_level_division_of  <-,  <- location.location.containedby  <-} -> {m.05g2b: Nijmegen}
path 5 :  {m.012x_5p_: Germany} -> { -> music.release.region  ->} -> {m.02j71: Earth} -> { -> base.aareas.schema.administrative_area.administrative_children  ->,  -> base.locations.planets.countries_within  ->,  <- base.aareas.schema.administrative_area.administrative_parent  <-,  <- base.locations.countries.planet  <-} -> {m.0f8l9c: France} -> { -> location.location.partially_contains  ->,  <- geography.river.basin_countries  <-,  <- location.location.partially_containedby  <-} -> {m.06fz_: Rhine} -> { -> geography.river.cities  ->} -> {m.05g2b: Nijmegen}
LLM_generated CoT:
{Germany} - {neighboring} - {France} - {houses} - {airport serving} - {Nijmegen}

A: {Yes}
answer: {Germany}
new CoT: Germany - neighboring - France - houses - airport serving - Nijmegen
Explanation: The provided knowledge graph paths meet the CoT's thinking and are sufficient to answer the question. Germany borders France and contains Weeze Airport, which serves Nijmegen, satisfying the criteria of the question.

The question is:

"""



revise_COT_prompt = """

Given a question, the associated accuracy knowledge based thought chain (CoT) based on retrieved knowledge paths, and unsure LLM-generated thought chain (CoT), 
The LLM-generated thought chains are the possible guideline to the question but not correct. and you can consider the knowledge based thought chain (CoT) are correct.

Your task is to fix the exsiting and generate new LLM-generated thought based on knowledge based CoTs as new guideline related to the answer. 
Please using the new guideline, and select new entities to explore the answer from the given knowledge based thought chain (CoT).

please give me a summary final answer in the format below in each question beginning
top_entity: {Entity 1, Entity 2, Entity3}
CoT1:{set of entities (with id start with "m.")} -> {relationship} -> {set of entities(with id start with "m.")}, 
reason: {your reason}

If the given LLM_generated CoT unlikly generated answer, please answer in the format below in each question beginning
answer: {give up}
reason: {your reason}

please only give me an answer section in the same format below in each question beginning

for example:

Q: 
 Question: Where did the "Country Nation World Tour" concert artist go to college?
The question also writen as: where did the artist had a concert tour named Country Nation World Tour graduate from college

knowledge based CoTs: 
KB_CoT1: {m.010qhfmm: Country Nation World Tour} -> { <- music.concert_tour.artist <-} -> {m.03gr7w: Brad Paisley} -> { -> people.person.education ->} -> {m.0h3d7qj: Unnamed Entity} -> { -> education.education.degree ->} -> {m.019v9k: Bachelor's degree}
KB_CoT2: {m.010qhfmm: Country Nation World Tour} -> {<- music.artist.concert_tours <-} -> {m.03gr7w: Brad Paisley} -> { -> people.person.education ->} -> {m.0h3d7qj: Unnamed Entity} -> { -> education.education.degree ->} -> {m.019v9k: Bachelor's degree}
LLM_generated CoT:
CoT1: {Country Nation World Tour} -> { -> music.concert_tour.artist ->} -> {Brad Paisley} -> { -> people.person.education ->} -> {Vanderbilt University}

A:
Top_entity: {m.03gr7w: Brad Paisley, m.0h3d7qj: Unnamed Entity, m.019v9k: Bachelor's degree}
CoT1: {m.010qhfmm: Country Nation World Tour} -> {<- music.concert_tour.artist <-} -> {m.03gr7w: Brad Paisley} -> { -> people.person.education ->} -> {m.0h3d7qj: Unnamed Entity} -> { -> education.education.degree ->} -> {m.019v9k: Bachelor's degree}
Reason: The knowledge-based CoTs (KB_CoT1 and KB_CoT2) indicate a clear pathway from the "Country Nation World Tour" to the artist Brad Paisley, and then to his educational background, culminating in a Bachelor's degree. The LLM-generated CoT mistakenly identifies Vanderbilt University as the college attended by Brad Paisley, which is not supported by the knowledge graph data. Instead, the corrected chain of thought follows the artist from the tour to the actual institution and degree, which are specified but not named in the knowledge graph, suggesting the need for further specification of the 'Unnamed Entity'.
"""





Summary_COT_prompt = """

Given a question, the associated accuracy retrieved knowledge paths from the Related_path section, and main topic entities
Your task is to summarize the provided knowledge triple in Related_path section and generate a chain of thoughts by the knowledge triple related to the main topic entities of question, which will used for generating the answer further.
you have to make sure you summarize correctly by use the provided knowledge triple, you can only use the entity with id from the given path and you can not skip in steps.

please only give me an answer section in the same format below in each question beginning

for example:

Q: 
Question: What country bordering France contains an airport that serves Nijmegen?
also writen as: what countries share borders with france and is the location contains and airport that server Nijmegen?
Main Topic Entities: {m.05g2b: Nijmegen, m.0f8l9c: France}
Related_path: 
path 1 :  {m.05g2b: Nijmegen} -> { -> location.location.nearby_airports  ->,  <- aviation.airport.serves  <-} -> {m.06cm5d: Weeze Airport} -> { -> location.location.containedby  ->,  <- location.location.contains  <-} -> {m.0345h: Germany}
path 2 :  {m.012x_5p_: Germany} -> { -> music.release.region  ->} -> {m.02j71: Earth} -> { <- music.release.region  <-} -> {m.0sprr4w: So Serene, m.0g801ww: A State of Trance 2009, m.0126m_r6: Alexandria, m.03xsqd1: Bad Dreams / Omissions, m.049ljb7: Myam James, Part 1} -> { -> music.release.region  ->} -> {m.059j2: Netherlands} -> { -> location.country.second_level_divisions  ->,  -> location.location.contains  ->,  <- location.administrative_division.second_level_division_of  <-,  <- location.location.containedby  <-} -> {m.05g2b: Nijmegen}
path 3 :  {m.012x_5p_: Germany} -> { -> music.release.region  ->} -> {m.02j71: Earth} -> { -> base.aareas.schema.administrative_area.administrative_children  ->,  -> base.locations.planets.countries_within  ->,  <- base.aareas.schema.administrative_area.administrative_parent  <-,  <- base.locations.countries.planet  <-} -> {m.0f8l9c: France} -> { -> location.location.partially_contains  ->,  <- geography.river.basin_countries  <-,  <- location.location.partially_containedby  <-} -> {m.06fz_: Rhine} -> { -> geography.river.cities  ->} -> {m.05g2b: Nijmegen}

A:
CoT1: {m.05g2b: Nijmegen} -> {<- airport serves to <-} -> {m.06cm5d: Weeze Airport} -> { -> containedby ->} -> {m.0345h: Germany} -> { -> location borders ->} -> {m.0f8l9c: France}
reason: {m.05g2b: Nijmegen} -> {<- aviation.airport.serves <-} -> {m.06cm5d: Weeze Airport} found in path 1, {m.06cm5d: Weeze Airport} -> { -> location.location.containedby ->} -> {m.0345h: Germany} found in path 1, {m.0345h: Germany} -> { -> location.location.borders ->} -> {m.0f8l9c: France} found in path 3, which can be summarized as Nijmegen's airport, Weeze Airport, is contained by Germany, which borders France.

The question is:
"""




Summary_COT_w_splitQ_prompt = """
Given a main question, an uncertain LLM-generated thinking Cot that consider all the entities, a few split questions that you can use stepply and finally obtain the final answer, the associated accuracy retrieved knowledge paths from the Related_path section, and main topic entities
Your task is to summarize the provided knowledge triple in Related_path section and generate a chain of thoughts by the knowledge triple related to the main topic entities of question, which will used for generating the answer for the main question and split question further.
you have to make sure you summarize correctly by use the provided knowledge triple, you can only use the entity with id from the given path and you can not skip in steps.

please only give me an answer section in the same format below in each question beginning

for example:

Q: 
Question: What country bordering France contains an airport that serves Nijmegen?
also writen as: what countries share borders with france and is the location contains and airport that server Nijmegen?
Main Topic Entities: {m.05g2b: Nijmegen, m.0f8l9c: France}
Thinking Cot:  "Nijmegen" - service by - airport - owned by - answer(country) - border with - "France"
split_question1: What country contains an airport that serves "Nijmegen"?
split_question2: What country bordering "France"?
Related_path: 
path 1 :  {m.05g2b: Nijmegen} -> { -> location.location.nearby_airports  ->,  <- aviation.airport.serves  <-} -> {m.06cm5d: Weeze Airport} -> { -> location.location.containedby  ->,  <- location.location.contains  <-} -> {m.0345h: Germany}
path 2 :  {m.012x_5p_: Germany} -> { -> music.release.region  ->} -> {m.02j71: Earth} -> { <- music.release.region  <-} -> {m.0sprr4w: So Serene, m.0g801ww: A State of Trance 2009, m.0126m_r6: Alexandria, m.03xsqd1: Bad Dreams / Omissions, m.049ljb7: Myam James, Part 1} -> { -> music.release.region  ->} -> {m.059j2: Netherlands} -> { -> location.country.second_level_divisions  ->,  -> location.location.contains  ->,  <- location.administrative_division.second_level_division_of  <-,  <- location.location.containedby  <-} -> {m.05g2b: Nijmegen}
path 3 :  {m.012x_5p_: Germany} -> { -> music.release.region  ->} -> {m.02j71: Earth} -> { -> base.aareas.schema.administrative_area.administrative_children  ->,  -> base.locations.planets.countries_within  ->,  <- base.aareas.schema.administrative_area.administrative_parent  <-,  <- base.locations.countries.planet  <-} -> {m.0f8l9c: France} -> { -> location.location.partially_contains  ->,  <- geography.river.basin_countries  <-,  <- location.location.partially_containedby  <-} -> {m.06fz_: Rhine} -> { -> geography.river.cities  ->} -> {m.05g2b: Nijmegen}
A:
CoT1: {m.05g2b: Nijmegen} -> {<- airport serves to <-} -> {m.06cm5d: Weeze Airport} -> { -> containedby ->} -> {m.0345h: Germany} -> { -> location borders ->} -> {m.0f8l9c: France}
reason: {m.05g2b: Nijmegen} -> {<- aviation.airport.serves <-} -> {m.06cm5d: Weeze Airport} found in path 1, {m.06cm5d: Weeze Airport} -> { -> location.location.containedby ->} -> {m.0345h: Germany} found in path 1, {m.0345h: Germany} -> { -> location.location.borders ->} -> {m.0f8l9c: France} found in path 3, which can be summarized as Nijmegen's airport, Weeze Airport, is contained by Germany, which borders France.

Question: When did the team with Crazy crab as their mascot win the world series?
Thinking CoT: "Crazy Crab" - mascot of - team - won - World Series (2 steps)
split_question1: What team has "Crazy Crab" as their mascot?
split_question2: When did the team with "Crazy Crab" as their mascot win the World Series?
Related_path: 
path 1 :  {m.02q_hzh: Crazy Crab} - { -> sports.mascot.team  ->} - {m.0713r: San Francisco Giants} - { -> baseball.baseball_team.team_stats  ->} - {m.05n6d4c: Unnamed Entity, m.05n6bn9: Unnamed Entity, m.05n6d9g: Unnamed Entity, m.05n6fgn: Unnamed Entity, m.05n6b1q: Unnamed Entity, m.05n6b62: Unnamed Entity, m.05n6chh: Unnamed Entity, m.05n69l5: Unnamed Entity, m.05n6fpw: Unnamed Entity, m.05n6fy2: Unnamed Entity, m.05n69rf: Unnamed Entity, m.05n6f7n: Unnamed Entity, m.05n6fby: Unnamed Entity, m.05n6cr0: Unnamed Entity, m.0dl3646: Unnamed Entity, m.05n69sr: Unnamed Entity, m.05n6dq4: Unnamed Entity, m.05n6cb0: Unnamed Entity, m.05n6bdf: Unnamed Entity, m.05n6c4s: Unnamed Entity}
path 2 :  {m.02q_hzh: Crazy Crab} - { -> sports.mascot.team  ->} - {m.0713r: San Francisco Giants} - { <- sports.sports_championship_event.runner_up  <-} - {m.06jwmt: 1987 National League Championship Series, m.04tfr4: 1962 World Series, m.0468cj: 2002 World Series, m.04j747: 1989 World Series, m.0dqwx9: 1971 National League Championship Series}
path 3 :  {m.02q_hzh: Crazy Crab} - { -> sports.mascot.team  ->} - {m.0713r: San Francisco Giants} - { -> sports.sports_team.championships  ->} - {m.09gnk2r: 2010 World Series, m.0ds8qct: 2012 World Series, m.0117q3yz: 2014 World Series}
A:
CoT1: {m.02q_hzh: Crazy Crab} - { -> sports.mascot.team  ->} - {m.0713r: San Francisco Giants} - { -> sports.sports_team.championships  ->} - {m.09gnk2r: 2010 World Series, m.0ds8qct: 2012 World Series, m.0117q3yz: 2014 World Series}
reason: The San Francisco Giants, represented by the mascot "Crazy Crab," won the World Series in 2010, 2012, and 2014, as indicated by the knowledge graph data.


The question is:
"""

cot_prompt_in_chain = """
Given a main question, an uncertain LLM-generated thinking Cot that consider all the entities, a few split questions that you can use stepply and finally obtain the final answer, the associated accuracy retrieved knowledge paths from the Related_path section, and main topic entities
Please provied three predict result and three possible Chains of Thought that can lead to the predict result in the same formate below by the given knowledge path and your own knowledge.
If the answer unclear, please give predicted answers to replace the entity in chains based on your knowledge in the same format as the examples.


Q: What country bordering France contains an airport that serves Nijmegen?
also writen as: what countries share borders with france and is the location contains and airport that server Nijmegen?
Main Topic Entities: {m.05g2b: Nijmegen, m.0f8l9c: France}
Thinking Cot:  "Nijmegen" - service by - airport - owned by - answer(country) - border with - "France"
split_question1: What country contains an airport that serves "Nijmegen"?
split_question2: What country bordering "France"?

path 0: {m.05g2b: Nijmegen} - { -> location.location.nearby_airports  ->} - {m.06cm5d: Weeze Airport} - { -> location.location.containedby  ->} - {m.0345h: Germany} - { -> location.location.adjoin_s  ->} - {m.02wrxl5: Unnamed Entity} - { -> location.adjoining_relationship.adjoins  ->} - {m.0f8l9c: France}
path 1: {m.05g2b: Nijmegen} - { -> location.location.containedby  ->} - {m.049nq: Kingdom of the Netherlands} - { -> location.location.containedby  ->} - {m.02j9z: Europe} - { -> location.location.contains  ->} - {m.0852h: Western Europe} - { -> location.location.contains  ->} - {m.0f8l9c: France}
path 2: {m.05g2b: Nijmegen} - { -> location.location.containedby  ->} - {m.049nq: Kingdom of the Netherlands} - { -> location.country.languages_spoken  ->} - {m.02bv9: Dutch Language} - { -> language.human_language.region  ->} - {m.02j9z: Europe} - { -> location.location.contains  ->} - {m.0f8l9c: France}
A:
Predicted: {Germany, Belgium, Luxembourg}
CoT1: {m.05g2b: Nijmegen} - { -> location.location.nearby_airports -> } - {m.06cm5d: Weeze Airport} - { -> location.location.containedby -> } - {m.0345h: Germany}  - { -> location.location.adjoins -> } - {m.0f8l9c: France}
CoT2: {m.05g2b: Nijmegen} - { -> location.location.nearby_airports -> } - {Brussels Airport} - {-> location.location.containedby ->} - {Belgium} - { -> location.location.adjoins -> } - {m.0f8l9c: France}
CoT3: {m.05g2b: Nijmegen} - { -> location.location.nearby_airports -> } - {Luxembourg airport} - { -> location.location.containedby -> } - {Luxembourg} - { -> location.location.adjoins -> } - {m.0f8l9c: France}



The question is:
"""



cot_prompt_in_chain4 = """

Instructions:

Given the following:
- Main Question: A question that needs to be answered.
- Initial Chain of Thought (CoT): An initial reasoning path generated by an LLM that considers all relevant entities.
- Split Questions: Sub-questions that help break down the problem step by step to reach the final answer.
- Associated Knowledge Paths: Accurate knowledge retrieval paths from the Related Paths section.
- Main Topic Entities: Key entities involved in the question.

Your Tasks:
1. Provide Three Predicted Results: Based on the information provided and your own knowledge, predict three possible answers to the main question.
2. Provide Three Possible Chains of Thought (CoTs): For each predicted result, create a reasoning path that leads to that result. Use the given Topic Entities, the given knowledge paths and your own knowledge. Present these CoTs in the same format as the answer section in examples.
Note: If the answer is unclear or certain entities are missing, use your best judgment to predict possible answers and adjust the entities in the chains accordingly. Maintain the format demonstrated in the examples.

Example Start:
Q: 
Question: What country bordering France contains an airport that serves Nijmegen?
also writen as: what countries share borders with france and is the location contains and airport that server Nijmegen?
Main Topic Entities: {m.05g2b: Nijmegen, m.0f8l9c: France}
Thinking Cot:  "Nijmegen" - service by - airport - owned by - answer(country) - border with - "France"
split_question1: What country contains an airport that serves "Nijmegen"?
split_question2: What country bordering "France"?
path 0: {m.05g2b: Nijmegen} - { -> location.location.nearby_airports  ->} - {m.06cm5d: Weeze Airport} - { -> location.location.containedby  ->} - {m.0345h: Germany} - { -> location.location.adjoin_s  ->} - {m.02wrxl5: Unnamed Entity} - { -> location.adjoining_relationship.adjoins  ->} - {m.0f8l9c: France}
path 1: {m.05g2b: Nijmegen} - { -> location.location.containedby  ->} - {m.049nq: Kingdom of the Netherlands} - { -> location.location.containedby  ->} - {m.02j9z: Europe} - { -> location.location.contains  ->} - {m.0852h: Western Europe} - { -> location.location.contains  ->} - {m.0f8l9c: France}
path 2: {m.05g2b: Nijmegen} - { -> location.location.containedby  ->} - {m.049nq: Kingdom of the Netherlands} - { -> location.country.languages_spoken  ->} - {m.02bv9: Dutch Language} - { -> language.human_language.region  ->} - {m.02j9z: Europe} - { -> location.location.contains  ->} - {m.0f8l9c: France}
A:
answer:
Predicted: {Germany, Belgium, Luxembourg}
CoT1: {m.05g2b: Nijmegen} - { -> location.location.nearby_airports -> } - {m.06cm5d: Weeze Airport} - { -> location.location.containedby -> } - {m.0345h: Germany}  - { -> location.location.adjoins -> } - {m.0f8l9c: France}
CoT2: {m.05g2b: Nijmegen} - { -> location.location.nearby_airports -> } - {Brussels Airport} - {-> location.location.containedby ->} - {Belgium} - { -> location.location.adjoins -> } - {m.0f8l9c: France}
CoT3: {m.05g2b: Nijmegen} - { -> location.location.nearby_airports -> } - {Luxembourg airport} - { -> location.location.containedby -> } - {Luxembourg} - { -> location.location.adjoins -> } - {m.0f8l9c: France}
Example End


The question is:

"""