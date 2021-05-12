#!/usr/bin/env python
# coding: utf-8

# # Hopfield Network

# Developer: Thiago Fellipe Ortiz de Camargo

# ## Standard Imports

# In[3]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

# ## Hopfield

# In[361]:
class hopfield(object):
    
    def __init__(self, patterns, noise_percentage, pattern_n_row, pattern_n_column, ib, epochs):
        self.patterns = patterns
        self.noise    = 1-noise_percentage
        self.nrow     = pattern_n_row
        self.ncol     = pattern_n_column
        self.fmn      = len(patterns)
        self.dim      = len(self.patterns[0])
        self.ib       = ib
        self.epc      = epochs
        self.scape    = False
        
    def noise_attribution(self, patt):
        self.pattern = patt
        self.randM   = np.random.rand(self.nrow,self.ncol)
        self.auxA    = self.noise > self.randM
        self.auxB    = self.noise < self.randM
        self.randM[self.auxA] =  1
        self.randM[self.auxB] = -1
        self.new_patter       = self.pattern.reshape(self.nrow,self.ncol)*self.randM
        return self.new_patter.reshape(self.dim,1)
    
    def weights(self):
        self.auxW = 0
        
        for patt in self.patterns:
            self.auxW += patt*patt.reshape(self.dim,1)
            
        self.W = ((1/self.dim)*self.auxW)-((self.fmn/self.dim)*np.zeros((self.dim,self.dim)))
        
    
    def run(self):
        
        self.outputs    = pd.DataFrame()
        self.noised_img = pd.DataFrame()
        for patt, i in zip(self.patterns,range(self.fmn)):
            self.weights()
            self.v_current  = self.noise_attribution(patt)
            self.noised_img = pd.concat((self.noised_img, pd.DataFrame(self.v_current).T))
            self.it = 0
            self.scape = False

            while(self.scape == False):
                self.v_past    = self.v_current
                self.u         = np.dot(self.W,self.v_past)+self.ib
                self.v_current = np.sign(np.tanh(self.u))

                if pd.DataFrame(self.v_current).equals(pd.DataFrame(self.v_past)):
                    self.scape = True

                if(self.it >= self.epc):
                    self.scape = True

                self.it += 1
                
            self.outputs = pd.concat((self.outputs,pd.DataFrame(self.v_current).T))
    
    


# ## Examples

# In[332]:

arq = open('/home/otluiz/Workspace/Datum/numeros.dat', 'r')
numeros = arq.readlines()

for index in range(len(numeros)):
    numeros[index] = numeros[index].rstrip('\n')      # remove o '\n'
    numeros[index] = numeros[index].replace(' ', ',') # coloca ',' no ' '

N0 = np.array((-1,-1,+1,-1,-1,-1,+1,+1,-1,-1,-1,-1,+1,-1,-1,-1,-1,+1,-1,-1,-1,-1,+1,-1,-1,-1,-1,+1,-1,-1,-1,+1,+1,+1,-1))
N1 = np.array((-1,+1,+1,+1,-1,-1,-1,-1,-1,+1,-1,-1,-1,-1,+1,-1,-1,-1,+1,-1,-1,-1,+1,-1,-1,-1,+1,-1,-1,-1,-1,+1,+1,+1,+1))
N2 = np.array((-1,+1,+1,+1,-1,+1,-1,-1,-1,+1,-1,-1,-1,-1,+1,-1,-1,+1,+1,-1,-1,-1,-1,-1,+1,+1,-1,-1,-1,+1,-1,+1,+1,+1,-1))
N3 = np.array((-1,-1,-1,-1,+1,-1,-1,-1,+1,+1,-1,-1,+1,-1,+1,-1,+1,-1,-1,+1,-1,+1,+1,+1,+1,-1,-1,-1,-1,+1,-1,-1,-1,-1,+1))
N4 = np.array((+1,+1,+1,+1,+1,+1,-1,-1,-1,-1,+1,-1,-1,-1,-1,+1,+1,+1,+1,-1,-1,-1,-1,-1,+1,-1,-1,-1,-1,+1,+1,+1,+1,+1,-1))
N5 = np.array((-1,+1,+1,+1,+1,+1,-1,-1,-1,-1,+1,-1,-1,-1,-1,+1,+1,+1,+1,-1,+1,-1,-1,-1,+1,+1,-1,-1,-1,+1,-1,+1,+1,+1,-1))
N6 = np.array((+1,+1,+1,+1,+1,-1,-1,-1,-1,+1,-1,-1,-1,+1,-1,-1,-1,-1,+1,-1,-1,-1,+1,-1,-1,-1,-1,+1,-1,-1,-1,+1,-1,-1,-1))
N7 = np.array((-1,+1,+1,+1,-1,+1,-1,-1,-1,+1,+1,-1,-1,-1,+1,-1,+1,+1,+1,-1,+1,-1,-1,-1,+1,+1,-1,-1,-1,+1,-1,+1,+1,+1,-1))
N8 = np.array((-1,+1,+1,+1,-1,+1,-1,-1,-1,+1,+1,-1,-1,-1,+1,-1,+1,+1,+1,+1,-1,-1,-1,-1,+1,-1,-1,-1,-1,+1,+1,+1,+1,+1,-1))
N9 = np.array((-1,+1,+1,+1,-1,+1,-1,-1,-1,+1,+1,-1,-1,-1,+1,+1,-1,-1,-1,+1,+1,-1,-1,-1,+1,+1,-1,-1,-1,+1,-1,+1,+1,+1,-1))

N = np.array((N0,N1,N2,N3,N4,N5,N6,N7,N8,N9))


# In[333]:

percentage = 0.50 ## percentual para ruído

hp = hopfield(patterns=N, noise_percentage=percentage, 
              pattern_n_row=7, pattern_n_column=5, ib=0, epochs=1000)
hp.run()


# In[334]:
#----------- Informação para metrica das imagens -------------------

per_med = [] ## lista para guardar a acuracia
#per_med.append("Ruído: ")
per_med.append(percentage) ##adiciona o primeiro elemento à lista

dif = lambda x, y: x - y ## diferença entre matrizes X e Y

def acuracia(x, n = 0):
    for i in range(0, len(x)):
      if (x[i] == 0): 
        n += 1
    return n/len(x)

#-------------- Impressão arquivos ----------------------------------
def write_csv(palavras, caminho): 
    row_list = [["Ruído", "1","2","3","4","5","6","7","8","9","0"],
                palavras]
    with open(caminho, "w", newline = '', encoding = 'utf-8') as caminho:
        writer = csv.writer(caminho, quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
        writer.writerows(row_list)


fig, axs = plt.subplots(nrows=6, ncols=5, figsize=(8, 15))

# ------- N0 -------
axs[0][0].set_title('Pattern 0')
axs[0][0].imshow(N0.reshape(7,5))

axs[1][0].set_title('Noised 0')
axs[1][0].imshow(hp.noised_img.iloc[0,:].values.reshape(7,5))

axs[2][0].set_title('HP Pattern 0')
axs[2][0].imshow(hp.outputs.iloc[0,:].values.reshape(7,5))

A0 = N0
B0 = hp.outputs.iloc[0,:].values
per_med.append(acuracia(dif(A0,B0)))
#print(per_med)

# ------- N1 -------
axs[0][1].set_title('Pattern 1')
axs[0][1].imshow(N1.reshape(7,5))

axs[1][1].set_title('Noised 1')
axs[1][1].imshow(hp.noised_img.iloc[1,:].values.reshape(7,5))

axs[2][1].set_title('HP Pattern 1')
axs[2][1].imshow(hp.outputs.iloc[1,:].values.reshape(7,5))

A1 = N1
B1 = hp.outputs.iloc[1,:].values
per_med.append(acuracia(dif(A1,B1)))
#print(per_med)

# ------- N2 -------
axs[0][2].set_title('Pattern 2')
axs[0][2].imshow(N2.reshape(7,5))

axs[1][2].set_title('Noised 2')
axs[1][2].imshow(hp.noised_img.iloc[2,:].values.reshape(7,5))

axs[2][2].set_title('HP Pattern 2')
axs[2][2].imshow(hp.outputs.iloc[2,:].values.reshape(7,5))

A2 = N2
B2 = hp.outputs.iloc[2,:].values
per_med.append(acuracia(dif(A2,B2)))
#print(per_med)

# ------- N3 -------
axs[0][3].set_title('Pattern 3')
axs[0][3].imshow(N3.reshape(7,5))

axs[1][3].set_title('Noised 3')
axs[1][3].imshow(hp.noised_img.iloc[3,:].values.reshape(7,5))

axs[2][3].set_title('HP Pattern 3')
axs[2][3].imshow(hp.outputs.iloc[3,:].values.reshape(7,5))

A3 = N3
B3 = hp.outputs.iloc[3,:].values
per_med.append(acuracia(dif(A3,B3)))
#print(per_med)

# ------- N4 -------
axs[0][4].set_title('Pattern 4')
axs[0][4].imshow(N4.reshape(7,5))

axs[1][4].set_title('Noised 4')
axs[1][4].imshow(hp.noised_img.iloc[4,:].values.reshape(7,5))

axs[2][4].set_title('HP Pattern 4')
axs[2][4].imshow(hp.outputs.iloc[4,:].values.reshape(7,5))

A4 = N4
B4 = hp.outputs.iloc[4,:].values
per_med.append(acuracia(dif(A4,B4)))
#print(per_med)

# ------- N5 -------
axs[3][0].set_title('Pattern 5')
axs[3][0].imshow(N5.reshape(7,5))

axs[4][0].set_title('Noised 5')
axs[4][0].imshow(hp.noised_img.iloc[5,:].values.reshape(7,5))

axs[5][0].set_title('HP Pattern 5')
axs[5][0].imshow(hp.outputs.iloc[5,:].values.reshape(7,5))

A5 = N5
B5 = hp.outputs.iloc[5,:].values
per_med.append(acuracia(dif(A5,B5)))
#print(per_med)

# ------- N6 -------
axs[3][1].set_title('Pattern 6')
axs[3][1].imshow(N6.reshape(7,5))

axs[4][1].set_title('Noised 6')
axs[4][1].imshow(hp.noised_img.iloc[6,:].values.reshape(7,5))

axs[5][1].set_title('HP Pattern 6')
axs[5][1].imshow(hp.outputs.iloc[6,:].values.reshape(7,5))

A6 = N6
B6 = hp.outputs.iloc[6,:].values
per_med.append(acuracia(dif(A6,B6)))
#print(per_med)

# ------- N7 -------
axs[3][2].set_title('Pattern 7')
axs[3][2].imshow(N7.reshape(7,5))

axs[4][2].set_title('Noised 7')
axs[4][2].imshow(hp.noised_img.iloc[7,:].values.reshape(7,5))

axs[5][2].set_title('HP Pattern 7')
axs[5][2].imshow(hp.outputs.iloc[7,:].values.reshape(7,5))

A7 = N7
B7 = hp.outputs.iloc[7,:].values
per_med.append(acuracia(dif(A7,B7)))
#print(per_med)

# ------- N8 -------
axs[3][3].set_title('Pattern 8')
axs[3][3].imshow(N8.reshape(7,5))

axs[4][3].set_title('Noised 8')
axs[4][3].imshow(hp.noised_img.iloc[8,:].values.reshape(7,5))

axs[5][3].set_title('HP Pattern 8')
axs[5][3].imshow(hp.outputs.iloc[8,:].values.reshape(7,5))

A8 = N8
B8 = hp.outputs.iloc[8,:].values
per_med.append(acuracia(dif(A8,B8)))
#print(per_med)

# ------- N9 -------
axs[3][4].set_title('Pattern 9')
axs[3][4].imshow(N9.reshape(7,5))

axs[4][4].set_title('Noised 9')
axs[4][4].imshow(hp.noised_img.iloc[9,:].values.reshape(7,5))

axs[5][4].set_title('HP Pattern 9')
axs[5][4].imshow(hp.outputs.iloc[9,:].values.reshape(7,5))

A9 = N9
B9 = hp.outputs.iloc[9,:].values
per_med.append(acuracia(dif(A9,B9)))
#print(per_med)

path = "/home/otluiz/Doutorado/RedesNeurais/Relatorio/Lista-3/numeros/"
plt.savefig(path + "numeros_050.png", bbox_inches = 'tight')

print(per_med)

plt.show()
    
write_csv(per_med, path + "numeros_050.csv")