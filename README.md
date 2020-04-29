# Group 3: Unified Framework for Graph Representation Learning

## Team
* Ben Mosbah, Nahla
* Debbichi, Firas
* Oumarou, Oumarou
* Banin Panyin, Stephen
* Ulyanov, Alexander


## Project Structure

* `lib` - the folder with the implementations of algorithms
* `input` - the folder containing the edge lists for input graphs
* `output` - the output produced by the program
    * `output/embeddings` - graph embedding files
* `main.py` - a script that generates embeddings for 2 graphs using each algorithm
* `report` - the folder containig the final lab report

## Generating Embeddings

To accomodate for the differences in Python versions between graph representation learning libraries, and to ensure proper sharing of requirements, and execution of the program we designed our program to run on the University's Linux System. It should, however, run on any Unix system.

### To connect to the university computer
1. If not yet created, generate SSH keys for your computer:
	* In Terminal, run: ```ssh-keygen```
2. Ensure you are connected through the university's VPN
3. Using terminal, connect to the university computer using SSH:
	* ```ssh your_fim_username@normandy.fim.uni-passau.de```
	* This example uses `normandy` computer, but other computers can be used. Ex: `babylon5`
4. Accept the login fingerprint:
	* Type: ```yes```

### To run the program
1. Clone the project
	* ```git clone https://git.fim.uni-passau.de/ulyanov/group3-unified-framework-for-graph-representation-learning.git```
	* If you are receiving a CA-Certificate error, run: ```git config --global http.sslVerify false```
	* Then, retry to clone the repo again
2. Create virtual environment with Python 2.7:
	* Ensure you are using Python version 2.7: ```python -V``` (If not, please install Python 2.7)
	* Ensure that you have virtual environment module installed: ```python -m pip install virtualenv```
	* Creating a new  virtual environment (Linux): ```python -m virtualenv venv```
	* Activate the new virtual environment: ```. venv/bin/activate```
3. Install the required packages:
	* ```pip install -r requirements.txt```
4. Run the code to generate embeddings:
	* ```python main.py```
5. Look for generated embeddings in ```output/embeddings``` folder

The expected screen output of the program:
```
(venv) $ python main.py
-------------------------------------------------
Generating embeddings for input/cora.edgelist
	generating node2vec ..  Done.	Saved in: output/embeddings/cora.node2vec
	generating Walklets ..  Done.	Saved in: output/embeddings/cora.walklets
	generating struc2vec ..  Done.	Saved in: output/embeddings/cora.struct2vec
	generating HARP ..  Done.	Saved in: output/embeddings/cora.harp
Completed.
-------------------------------------------------
Generating embeddings for input/citeseer.edgelist
	generating node2vec ..  Done.	Saved in: output/embeddings/citeseer.node2vec
	generating Walklets ..  Done.	Saved in: output/embeddings/citeseer.walklets
	generating struc2vec ..  Done.	Saved in: output/embeddings/citeseer.struct2vec
	generating HARP ..  Done.	Saved in: output/embeddings/citeseer.harp
Completed.
```
