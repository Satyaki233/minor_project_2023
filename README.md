# Minor Project Rcciit
# Image Recognition using entropy 

## Setup and Execution

1. Create a virtual enviornment with following command  
`python -m venv <venv>` 
venv is your virtual enviornment name
2. Activate your virtual enviornment
3. install packages using requirements.txt  
`pip install -r requirements.txt`  
**You can aslo run this as a spyder project**
4. Start running python scripts  
5. Create following folders:  
    * images/  
    * database/  

6. Extract database on database folder  
7. Run `face_detection.py` script  
it will execute fuction named `detect_face_from_video(path:str,load_file:bool=True)`  
write the path to your database in path argument and if you want to save progress then keep load_file=True  
8. Run `image_entropy.py` script  
execute `calculate_entropy()` function  
9. Run the `project.ipynb` jupyter notebook  
You can see the result here !






