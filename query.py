from check import retrieval_chain
my_query = "which state has bottom most per capita income"
response = retrieval_chain.run(my_query)
print(response)  