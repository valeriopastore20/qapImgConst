import gym
import numpy as np
import os
import sys
import math
from gym import spaces

class QapImgConstEnv(gym.Env):

    def __init__(self):
        #genera e legge i frequent item sets
        path = os.getenv("HOME")+"/fisFolder/fisFile"+str(sys.argv[1])+".txt"
        self.matrix_fq = self.readFisFile(path)
        self.matrix_fq = self.matrix_fq/np.max(self.matrix_fq)
        self.num_prod = len(self.matrix_fq)
        self.num_loc = self.num_prod
        #inizializza il dizionario delle azioni (In questo modo possiamo avere un action space discreto)
        self.dict = {}
        k=0
        for a in range(self.num_prod):
            for b in range(a,self.num_prod):
                self.dict.update({k : [a,b]})
                k+=1
        # Inizializza la matrice dei prodotti
        path = os.getenv("HOME")+"/prodLocFolder/prodLocFile"+str(self.num_prod)+".txt"
        self.matrix_pl_original = self.get_location_matrix(path,self.num_prod)
        self.matrix_pl = self.matrix_pl_original.copy()
        #inizializza matrice delle distanze tra locazioni (e' quadrata simmetrica e sulla diagonale c'e' la distanza con l'uscita)
        step = int(math.sqrt(self.num_prod))
        self.matrix_dist = np.zeros((self.num_loc,self.num_loc), dtype=int)
        k = 1
        i = 0
        while i < self.num_loc:
            for j in range(step):
                self.matrix_dist[i,i] = j+k
                i+=1
            k+=1
        for i in range(self.num_loc):
            for j in range(i+1,self.num_loc):
                x = abs(int(i/step) - int(j/step))
                y = abs(int(i%step) - int(j%step))
                self.matrix_dist[i,j] = self.matrix_dist[j,i] = x + y
        self.matrix_dist = self.matrix_dist/np.max(self.matrix_dist)

        matrix_dp = np.dot(np.dot(self.matrix_pl,self.matrix_dist),np.transpose(self.matrix_pl))
        self.matrix_wd = matrix_dp*self.matrix_fq
        self.initial_sum = np.sum(self.matrix_wd)
        self.mff_sum = self.compute_mff_sum(matrix_dp)
        self.done = False
        self.final_sum = 1000

        self.action_space = spaces.Discrete(len(self.dict))
        self.observation_space = spaces.Box(low=0, high=255,shape=(self.num_prod, self.num_prod, 1), dtype=np.uint8)


    # metodo che viene invocato alla fine di ogni game e resetta l'environment
    def reset(self):
        self.matrix_pl = self.matrix_pl_original.copy()
        matrix_dp = np.dot(np.dot(self.matrix_pl,self.matrix_dist),np.transpose(self.matrix_pl))
        self.matrix_wd = matrix_dp*self.matrix_fq
        self.current_sum = np.sum(self.matrix_wd)
        self.count = 0
        return np.reshape(self.matrix_wd,(self.num_prod,self.num_prod,1))

    # metodo per effettuare il rendere dell'environment
    def render(self):
        #np.set_printoptions(threshold=3000)
        print("R E N D E R")
        #print(self.matrix_wd)
        print("INITIAL SUM: {0:.2f}".format(self.initial_sum))
        print("CURRENT SUM: {0:.2f}".format(self.current_sum))
        print("CURRENT IMPROVEMENT: {0:.2f}%".format((self.initial_sum-self.current_sum)/self.initial_sum*100))
        print("MFF SUM: {0:.2f}".format(self.mff_sum))
        print("MFF IMPROVEMENT: {0:.2f}%".format((self.initial_sum-self.mff_sum)/self.initial_sum*100))
        print("R E N D E R")

    def step(self,actionKey):
        self.done = False #e' qui perche' la callback viene chiamata dopo il reset, quindi nn saremmo in grado di
                        # vedere i risultati dopo l'ultima azione
        #converte il valore dell'action nella corrispondente azione
        action = self.dict[actionKey]
        # effettua lo swap sulla matrice di prodotto e ricalcola la matrice finale
        self.matrix_pl[[action[0], action[1]]] = self.matrix_pl[[action[1], action[0]]]
        matrix_dp = np.dot(np.dot(self.matrix_pl,self.matrix_dist),np.transpose(self.matrix_pl))
        self.matrix_wd = matrix_dp*self.matrix_fq
        self.matrix_wd *= (255/self.matrix_wd.max())
        self.matrix_wd = self.matrix_wd.astype(int)
        #calcola il reward come differenza tra la somma allo sato precedente e la somma allo stato corrente
        sum = np.sum(self.matrix_wd/10000)
        reward = (self.current_sum - sum)
        self.current_sum = sum
        self.count+=1
        if(self.count == self.num_prod+10):
            self.done = True
            self.final_sum = sum
        return np.reshape(self.matrix_wd,(self.num_prod,self.num_prod,1)), reward.item(), self.done, {}


# UTILITY METHODS

    # calcola la somma della matrice disposta come mff
    def compute_mff_sum(self,matrix):
        diag = np.diag(matrix)
        diag.setflags(write=1)
        min_ind = np.argmin(diag,0)
        matrix_mff = self.matrix_pl[min_ind]
        diag[min_ind] = 9000
        for i in range(1,self.num_prod):
            min_ind = np.argmin(diag,0)
            if diag[min_ind] == 9000:
                break
            matrix_mff = np.vstack((matrix_mff,self.matrix_pl[min_ind]))
            diag[min_ind] = 9000
        matrix_dp = np.dot(np.dot(matrix_mff,self.matrix_dist),np.transpose(matrix_mff))
        matrix_wd = matrix_dp*self.matrix_fq
        mff_sum = np.sum(matrix_wd/10000)
        return mff_sum

    # legge il file contenete la disposizione iniziale dei prodotti e crea la relativa matrice
    def get_location_matrix(self,path,num_prod):
        prod_loc_matrix = np.zeros((num_prod,num_prod), int)
        file = open(path,"r")
        line = file.readline()
        for i in range(num_prod):
            k = [int(s) for s in line.split() if s.isdigit()]
            p1 = k[0]
            p2 = k[1]
            prod_loc_matrix[p1,p2] = 1
            line = file.readline()
        file.close()
        return prod_loc_matrix

    # Metodo che legge il file contenente i frequent item sets e produce la relativa matrice
    def readFisFile(self,path):
        file = open(path,"r")
        line = file.readline()
        num_prod = [int(s) for s in line.split() if s.isdigit()][0]
        fis = np.zeros((num_prod,num_prod),int) # 1000 max product
        line = file.readline()
        k = [int(s) for s in line.split() if s.isdigit()]
        while len(k) == 2:
            prod = k[0]
            freq = k[1]
            fis[prod,prod] = freq
            line = file.readline()
            k = [int(s) for s in line.split() if s.isdigit()]
        while line:
            p1 = k[0]
            p2 = k[1]
            freq = k[2]
            fis[p1,p2] = freq
            fis[p2,p1] = freq
            fis[p1,p1] -= freq
            fis[p2,p2] -= freq
            line = file.readline()
            k = [int(s) for s in line.split() if s.isdigit()]
        file.close()
        return fis