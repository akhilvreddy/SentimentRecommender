import random
import pinecone    
from ModelTraining import my_function


pinecone.init(      
	api_key='f58d07a8-35ae-46f2-8cec-df247cb8020b',      
	environment='gcp-starter'      
)      
index = pinecone.Index('connotation-database')

print("Connection successful")

active_indexes = pinecone.list_indexes()
print(active_indexes)

print(pinecone.describe_index("connotation-database"))

testData = [{"id": "This is just a very nice day.", "vector": [2, 7]}]

index.upsert(testData)

result = my_function("[2]. Test sentence.")

query_vector = [0.1, 0.2, 0.3]  # Replace with the actual query vector you want 
# query_vector = [-1*myfunction(setnece)[1], myfunction(sentecne)[2]]

# call the "distance neighbor"
# finalAnswer = query_vector[0]

# throw question -> does this meet your expectations? 
answer = random.choice([True, False])

if True: 
    exit

# start reinforcement learning 

# throw more questions -> is it more positive or more negative -> choose percentages -> shift in database


results = index.query(queries=[query_vector])


# # DISCONNECT (do not run with the while)
# pinecone.deinit()