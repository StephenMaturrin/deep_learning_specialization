import pandas as pd
import io
import requests
url="https://github.com/andersy005/deep-learning-specialization-coursera/blob/master/01-Neural-Networks-and-Deep-Learning/week2/Programming-Assignments/datasets/test_catvnoncat.h5"
s=requests.get(url).content
c=pd.read_csv(io.StringIO(s.decode('utf-8')))