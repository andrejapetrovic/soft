#!/usr/bin/python
import sys
import apptest

res = []
n = 0
with open('proj-lvl3-data/res.txt') as file:	
    data = file.read()
    lines = data.split('\n')
    for id, line in enumerate(lines):
        if(id>0):
            cols = line.split('\t')
            if(cols[0] == ''):
                continue
            cols[1] = cols[1].replace('\r', '')
            res.append(float(cols[1]))
            n += 1

correct = 0
student = []
student_results = apptest.retSums()
print(student_results)

diff = 0
for index, res_col in enumerate(res):
    diff += abs(res_col - student_results[index])
percentage = 100 - abs(diff/sum(res))*100

print("RA158 Andreja Petrovic")
print(student)
print('Procenat tacnosti:\t'+str(percentage))
print('Ukupno:\t'+str(n))
