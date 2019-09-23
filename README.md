# NetworkScienceA1
The assignment 1 repository for Network Science course.

Q1.py is the python code for question 1

Q3.py is the python code for question 3 to generate AB model networks

ABbonus.py is the python code for the bonus modified AB model

# How to run it

Put all network edgelist.txt files into a subfolder named "networks"

For Q1.py

$ python Q1.py --file ****.edgelist.txt --directed 0

directed = 0 means the network is undirected, directed = 1 means the network is directed

For Q3.py

$ python Q3.py --file "name" --size x --edges y

"name" is the filename for the generated network, a edgelist.txt file will be generated. use size x and edges y to specify the number of nodes and edges

Same for ABbonus.py

$ python ABbonus.py --file "name" --size x --edges y

# Dependencies

scipy

matplotlib

pandas 
