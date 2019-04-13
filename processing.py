import os

from string import ascii_lowercase as letters
import numpy as np
from PIL import Image

def get_links(info,index,num_classes,cur_class, num=None):
   for i in os.listdir(os.curdir):
       if(i == "prediction"):
               if(not cur_class+num in info.keys()):
                     info[cur_class+num] = [[0 for j in range(num_classes)]]
               info[cur_class+num][0][index] = os.getcwd()+"/" + i
       elif(i == "signal"):
               if(not cur_class+num in info.keys()):
                     info[cur_class+num] = [[0 for j in range(num_classes)]]
               info[cur_class+num].append(os.getcwd()+"/" + i)
       elif(i == "00" or i == "01" or i=="02" or i=="03" or i=="04"):
               temp = os.getcwd()
               os.chdir(i)
               get_links(info,index,num_classes,cur_class, num=i)
               os.chdir(temp)
       elif(not os.path.isfile(i)):
               temp = os.getcwd()
               os.chdir(i)
               get_links(info,index,num_classes,cur_class,num)
               os.chdir(temp)
               
dictionary = {}

def generate_links(info, classes=None,curclass = None):
        for i in os.listdir(os.curdir):
                if(i=="v1-synthesized"):
                        temp = os.getcwd()
                        os.chdir(i)
                        classes = os.listdir(os.curdir)
                        if('classes.txt' in classes):
                           classes.remove('classes.txt')
                        #classes.append('membrane_caax_63x')
                        classes.sort()

                        
                        F = open("classes.txt","a")
                        F.write(str(classes) + "\n")
                        F.close()
                        
                        for j in classes:
                                temp2 = os.getcwd()
                                os.chdir(j)
                                generate_links(info, classes, j)
                                os.chdir(temp2)
                        os.chdir(temp)
                elif(classes and i in classes):
                        temp = os.getcwd()
                        os.chdir(i)
                        get_links(info,classes.index(i),len(classes),curclass)
                        os.chdir(temp)
                elif(not os.path.isfile(i)):
                        temp = os.getcwd()
                        os.chdir(i)
                        generate_links(info,classes,curclass)
                        os.chdir(temp)



temp = os.getcwd()
os.chdir('images')
generate_links(dictionary)
os.chdir(temp)

def generate_names(n=64,prefix="x"):
    counter = 0
    result = []
    for i in letters:
        for j in letters:
            for k in letters:
                if(counter == n):
                    return result
                result.append(prefix + i + j + k)
                counter+=1

def generate_array(path, names):
    return np.array( [ np.array(Image.open(path + "/" + i + ".png"))/65535.0 for i in names])

def write_array(dictionary):
   if( os.path.isdir("data") ):
      return
   os.mkdir('data')
   for i in range(14):
      os.mkdir('data/' + str(i))
   counter = 0
   info = open("info.txt","w")
   
   for key in sorted(dictionary.keys()):
      pairs = dictionary[key]
      arrays = []
      print(counter)
      for i in range(len(pairs[0])):
         names = generate_names(len(os.listdir(pairs[0][i])))
         arrays.append(generate_array(pairs[0][i], names))
      print(pairs[1])
      arrays.append(generate_array(pairs[1],names))
      # assert correctness
      assert len(arrays)==14
      for i in range(1,len(arrays)):
         assert arrays[i].shape == arrays[i-1].shape
   
      for d in range(32,arrays[0].shape[0]+1,32):
         for h in range(64,arrays[0].shape[1]+1,64):
            for w in range(64, arrays[0].shape[2]+1,64):
               for i in range(len(arrays)):
                  array = arrays[i][d-32:d,h-64:h,w-64:w]
                  assert array.shape == (32,64,64)
                  np.save('data/'+str(i) + '/' + str(counter), array)
                  info = open('data/'+str(i) + '/info.txt','a')
                  if(i == 13):
                     path = pairs[1]
                  else:
                     path = pairs[0][i]
                  info.write(str(counter) + "\t" + str((d-32,d)) + "\t" + str((h-64,h)) + "\t"  + str((w-64,w)) + "\t" + path + "\n")
                  info.close()
               counter = counter + 1
      for d in range(32,arrays[0].shape[0]+1,32):
         for h in range(64,arrays[0].shape[1]+1,64):
            for w in range(64, arrays[0].shape[2]+1,64):
               for i in range(len(arrays)):
                  shape = arrays[i].shape
                  array = arrays[i][shape[0]-d:shape[0]-d+32,shape[1]-h:shape[1]-h+64,shape[2]-w:shape[2]-w+64]
                  assert array.shape == (32,64,64)
                  np.save('data/'+str(i) + '/' + str(counter), array)
                  info = open('data/'+str(i) + '/info.txt','a')
                  if(i == 13):
                     path = pairs[1]
                  else:
                     path = pairs[0][i]
                  info.write(str(counter) + "\t" + str((shape[0]-d,shape[0]-d+32)) + "\t" + str((shape[1]-h,shape[1]-h+64)) + "\t"  + str((shape[2]-w,shape[2]-w+64)) + "\t" + path + "\n")
                  info.close()
               counter = counter + 1
               
write_array(dictionary)
         
#indices = []
#validation_failed = False

"""
for key in sorted(dictionary.keys()):
        #if(key == 'alpha_tubulin03' or key == 'fibrillarin03' or key == 'lamin_b100'):
        #   continue
        pairs = dictionary[key]
        s = ""
        index = 0
        for i in pairs[0]:
                #if(index>=len(indices)):
                #        indices.append(i.split('\\')[11])
                #else:
                #        validation_failed = i.split('\\')[11] != indices[index]
                s+= i + ","
                index+=1
        s=s[:-1]
        s=s+"\t"+pairs[1]
        print(s)
"""

#print indices
#print validation_failed


"""print os.getcwd()
os.chdir("images")
os.chdir('AICS')
os.chdir('BF-FM')
os.chdir('v1-synthesized')
os.chdir('dna')
os.chdir('evaluate')
os.chdir('dna')
os.chdir('03')
print os.getcwd()
print os.listdir(os.getcwd())"""
