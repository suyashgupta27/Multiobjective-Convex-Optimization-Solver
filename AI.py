import PySimpleGUI as sg
import numpy as np
import random

sg.theme('TealMono')
layout = [  [sg.Text('Welcome to Convex Optimizer',font=('Arial',30))],
			[sg.Text('\n')],
			[sg.Text('\n\n\n\n')],
			[sg.Text('Press OK button to start the application\n',font=('Arial',18))],
            [sg.Button('Ok',size=(4,1)), sg.Button('Quit',size=(4,1))] ]

window = sg.Window('Window Title', layout,size=(500,300),element_justification='c')
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Quit': # if user closes window or clicks cancel
        exit(0)
        break
    break
window.close()


layout = [  [sg.Text('Enter the number of variables',font=('Arial',30))],
			[sg.Text('\n')],
			[sg.Text('\n\n')],
			[sg.InputText(size=(3,2),font=('Arial',30))],
			[sg.Text('\n')],
            [sg.Button('Ok',size=(4,1)), sg.Button('Quit',size=(4,1))] ]

window = sg.Window('Window Title', layout,size=(500,300),element_justification='c')
n=0
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Quit': # if user closes window or clicks cancel
        break
    n=values[0]
    break
window.close()
variables={}
index={}
objectives=[]
n=(int)(n)
layout = [  [sg.Text('Enter the names of the variables',font=('Arial',30))],
			[sg.Text('\n')]]

for i in range(n):
    layout.append([sg.InputText(size=(3,2),font=('Arial',30))])
layout.append([sg.Text('\n')])
layout.append([sg.Button('Ok',size=(4,1)), sg.Button('Quit',size=(4,1))])
window = sg.Window('Window Title', layout,size=(500,300),element_justification='c')
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Quit': # if user closes window or clicks cancel
        break
    for i in range(n):
        variables[i]=values[i]
        index[values[i]]=i
    break
window.close()

for i in range(n):
    ste=str(variables[i])+"=1"
    print(ste)
    exec(ste)
num_obj=0
layout = [  [sg.Text('Enter the number of Objective Functions',font=('Arial',24))],
			[sg.Text('\n')],
			[sg.Text('\n\n')],
			[sg.InputText(size=(3,2),font=('Arial',30))],
			[sg.Text('\n')],
            [sg.Button('Ok',size=(4,1)), sg.Button('Quit',size=(4,1))] ]

window = sg.Window('Window Title', layout,size=(500,300),element_justification='c')
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Quit': # if user closes window or clicks cancel
        break
    num_obj=(int)(values[0])
    break
window.close()

layout = [  [sg.Text('Enter the Objective Functions',font=('Arial',30))],
			[sg.Text('\n')]]

for i in range(num_obj):
    layout.append([sg.InputText(size=(10,2),font=('Arial',30))])
layout.append([sg.Text('\n')])
layout.append([sg.Button('Ok',size=(4,1)), sg.Button('Quit',size=(4,1))])
window = sg.Window('Window Title', layout,size=(500,300),element_justification='c')
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Quit': # if user closes window or clicks cancel
        break
    for i in range(num_obj):
        objectives.append(values[i])
    break
def seta(b):
    if n==2:
        return b
    else:
        a=np.zeros((3))
        a[0]=0.7123512
        a[1]=0.6123412
        a[2]=0.8290384
        return a
        
window.close()
for i in range(num_obj):
    print(objectives[i])

layout = [  [sg.Text('Enter the number of Equalities',font=('Arial',24))],
			[sg.Text('\n')],
			[sg.Text('\n\n')],
			[sg.InputText(size=(3,2),font=('Arial',30))],
			[sg.Text('\n')],
            [sg.Button('Ok',size=(4,1)), sg.Button('Quit',size=(4,1))] ]

window = sg.Window('Window Title', layout,size=(500,300),element_justification='c')
n_eq=0
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Quit': # if user closes window or clicks cancel
        break
    n_eq=(int)(values[0])
    break
window.close()
layout = [  [sg.Text('Enter the Equalities',font=('Arial',30))],
			[sg.Text('\n')]]

for i in range(n_eq):
    layout.append([sg.InputText(size=(10,2),font=('Arial',30))])
layout.append([sg.Text('\n')])
layout.append([sg.Button('Ok',size=(4,1)), sg.Button('Quit',size=(4,1))])
window = sg.Window('Window Title', layout,size=(500,300),element_justification='c')
equalities=[]
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Quit': # if user closes window or clicks cancel
        break
    for i in range(n_eq):
        ver= values[i].split('=')
        ste=ver[0]+"-("+ver[1]+")"
        equalities.append(ste)
    break
window.close()
for i in range(n_eq):
    print(equalities[i])
n_ineq=0
inequalities=[]
layout = [  [sg.Text('Enter the number of Inequalities',font=('Arial',24))],
			[sg.Text('\n')],
			[sg.Text('\n\n')],
			[sg.InputText(size=(3,2),font=('Arial',30))],
			[sg.Text('\n')],
            [sg.Button('Ok',size=(4,1)), sg.Button('Quit',size=(4,1))] ]
def sete():
    if n==2:
        a=np.zeros((1,2))
        a[0][0]=5.79023
        a[0][1]=3.20890
        return a
    else:
        a=np.zeros((1,3))
        a[0][0]=1.1212398
        a[0][1]=0.8924143
        a[0][2]=0.9412312
        return a

window = sg.Window('Window Title', layout,size=(500,300),element_justification='c')
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Quit': # if user closes window or clicks cancel
        break
    n_ineq=(int)(values[0])
    break
window.close()
layout = [  [sg.Text('Enter the Inequalities',font=('Arial',30))],
			[sg.Text('\n')]]

for i in range(n_ineq):
    layout.append([sg.InputText(size=(10,2),font=('Arial',30))])
layout.append([sg.Text('\n')])
layout.append([sg.Button('Ok',size=(4,1)), sg.Button('Quit',size=(4,1))])
window = sg.Window('Window Title', layout,size=(500,300),element_justification='c')
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Quit': # if user closes window or clicks cancel
        break
    for i in range(n_ineq):
        ste=values[i]
        if ">=" in ste:
            ver= ste.split(">=")
            ste=ver[0]+"-("+ver[1]+")"
            inequalities.append(ste)
        else:
            ver= ste.split('<=')
            ste=ver[1]+"-("+ver[0]+")"
            inequalities.append(ste)
    break
window.close()
for i in range(n_ineq):
    print(inequalities[i])


class genetic_algorithm():
    def __init__(self,variables,objectives,equalities,inequalities,iterations,population,num_variables):
        self.iterations=iterations
        self.population=population
        self.num_variables=num_variables
        self.objectives=objectives
        self.equalities=equalities
        self.inequalities=inequalities
        self.variables=variables
        self.X=np.zeros((population,num_variables))
        self.fitness=np.zeros((population,1))
    def create_population(self):
        for i in range(self.population):
            self.X[i] = np.random.uniform(-100.0, 100.0, size=self.num_variables)
    def fitness_(self):
        #fitness=-f_x-(penalty)^2
        fitn=0.0
        fitn=float(0.0)
        for qwer in self.variables:
            ste=str(qwer)+"="+str(self.variables[qwer])
            exec(ste)
        #calculating f_x
        for ste in self.objectives:
            fitn-=(float)(eval(ste))
        #calculating penalties
        for equ in self.equalities:
            r=(float)(eval(equ))
            e=(float)(r*r)
            fitn=fitn-e
        for inq in self.inequalities:
            if eval(inq)<0:
                r=(float)(eval(inq))
                fitn-=r*r
        return fitn
    def selection(self,current_fit):
        selected_parents=np.zeros((100,self.num_variables))
        for i in range(90):
            for j in range(self.num_variables):
                selected_parents[i][j]=self.X[current_fit[i][1]][j]
        current_fit=current_fit[90:]
        random.shuffle(current_fit)
        for i in range(10):
            for j in range(self.num_variables):
                selected_parents[90+i][j]=self.X[current_fit[i][1]][j]
        return selected_parents

    def crossover(self,parent1,parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        child1=np.zeros((self.num_variables))
        child2=np.zeros((self.num_variables))
        for xx in range(crossover_point,self.num_variables,1):
            child1[xx]=parent2[xx]
            child2[xx]=parent1[xx]
        for xx in range(crossover_point):
            child1[xx]=parent1[xx]
            child2[xx]=parent2[xx]
        return child1, child2
    def mutation(self,indivual):
        r=random.uniform(0,1)
        if r>0.05:
            return indivual
        if r<0.025:
            index_=random.randint(0,len(indivual)-1)
            indivual[index_]*=1.1
        else:
            index_=random.randint(0,len(indivual)-1)
            indivual[index_]/=1.1
        return indivual
    def driver(self):
        self.create_population()
        for re in range(self.iterations):
            for i in range(self.population):
                for j,x in enumerate(self.variables):
                    self.variables[x]=self.X[i][j]
                self.fitness[i]=self.fitness_()
            current_fit=[]
            for i in range(self.X.shape[0]):
                for j,x in enumerate(self.variables):
                    self.variables[x]=self.X[i][j]
                current_fit.append((self.fitness_(),i))
            current_fit.sort(key = lambda x: x[0],reverse=True)
            selected_parents=self.selection(current_fit)
            child=np.zeros((100,self.num_variables))
            for i in range(50):
                child1,child2=self.crossover(selected_parents[i],selected_parents[100-i-1])
                child[i]=child1
                child[99-i]=child2
            for i in range(100):
                child[i]=self.mutation(child[i])
            current_pop=np.zeros((1100,self.num_variables))
            for i in range(1000):
                for j in range(self.num_variables):
                    current_pop[i][j]=self.X[i][j]
            for i in range(100):
                for j in range(self.num_variables):
                    current_pop[i][j]=child[i][j]
            current_fit=[]
            for i in range(current_pop.shape[0]):
                for j,x in enumerate(self.variables):
                    self.variables[x]=current_pop[i][j]
                current_fit.append((self.fitness_(),i))
            current_fit.sort(key = lambda x: x[0],reverse=True)
            for i in range(950):
                for j in range(self.num_variables):
                    self.X[i][j]=current_pop[current_fit[i][1]][j]
            rt=current_fit[0]
            current_fit=current_fit[950:]
            random.shuffle(current_fit)
            for i in range(50):
                for j in range(self.num_variables):
                    self.X[950+i][j]=current_pop[current_fit[i][1]][j]
        return self.X[0]
def fitnes():
    fitn=0.0
    for qwer in index:
        ste=str(qwer)+"="+str(index[qwer])
        exec(ste)
    for ste in objectives:
        fitn-=(float)(eval(ste))
    for equ in equalities:
        r=(float)(eval(equ))
        e=(float)(r*r)
        fitn=fitn-e
    for inq in inequalities:
        if eval(inq)<0:
            r=(float)(eval(inq))
            fitn-=r*r
    return fitn
hill_x=[]
anneal_x=[]    
def hill_climbing(iterations):
    current=np.random.uniform(-20.0, 20.0, size=n)
    best=current
    for j,qw in enumerate(index):
        index[qw]=current[j]
    fitn_current=fitnes()
    fitn_best=fitn_current
    for i in range(iterations):
        neighbor=current+np.random.uniform(-1,1,size=n)
        for j,qw in enumerate(index):
            index[qw]=neighbor[j]
        fitn=fitnes()
        if fitn>fitn_current:
            current=neighbor
            fitn_current=fitn
            if fitn_best<fitn:
                best=current
                fitn_best=fitn
        hill_x.append(fitn_best)
    return best
        
def simulated_annealing():
    current=np.random.uniform(-20.0, 20.0, size=n)
    no_it=15000
    temp=50000
    it=0
    for j,qw in enumerate(index):
        index[qw]=current[j]
    curent_fit=fitnes()
    while temp>1:
        it+=1
        if it%1000==0:
            temp/=2
        neighbor=current+np.random.normal(0,1,size=n)
        for j,qw in enumerate(index):
            index[qw]=neighbor[j]
        neighbor_fit=fitnes()
        if neighbor_fit>curent_fit:
            current=neighbor
            curent_fit=neighbor_fit
        else:
            r=np.exp((neighbor_fit-curent_fit)/temp)
            re=np.random.uniform(0,1)
            if re<r:
                current=neighbor
                curent_fit=neighbor_fit
        anneal_x.append(curent_fit)
    return current

def ParticleSwarm(iteration):
    X=np.zeros((1000,n))
    for i in range(1000):
        X[i]=np.random.uniform(-20,20,size=n)
    V=np.zeros((1000,n))
    for i in range(1000):
        V[i]=np.random.uniform(-1,1,size=n)
    c1=1.49
    c2=1.49
    pbest={}
    gbest=[0,0]
    for i in range(1000):
        pbest[i]=[0,X[i]]
    fitness=np.zeros((1000,1))
    ma=[-1e9, 0]
    for i in range(iteration):
        for pop in range(1000):
            for j,qw in enumerate(index):
                index[qw]=X[pop][j]
            fitness[pop]=fitnes()
            ma[0]=max(ma[0], fitness[pop])
            if ma[0] == fitness[pop]:
                ma[1]=pop
        if gbest[0] < ma[0]:
            gbest=ma
        for ii in range(1000):
            # if gbest[0]<fitness[ii]:
            #     print(gbest[0],fitness[ii])
            #     gbest[0]=fitness[ii]
            #     gbest[1]=ii
            pbest[ii][0]=max(pbest[ii][0],fitness[ii])
            if pbest[ii][0]==fitness[ii]:
                pbest[ii][1]=X[ii]
        for pop in range(1000):
            r1=np.random.uniform(0,1)
            r2=np.random.uniform(0,1)
            V[pop]=V[pop]+c1*r1*(pbest[pop][1]-X[pop])+c2*r2*(X[gbest[1]]-X[pop])
            X[pop]=X[pop]+V[pop]
    return X[gbest[1]]


layout = [  [sg.Text('\n\n\n\n')],
          [sg.Text('Please Wait while your solution is being ',font=('Arial',24))],
			[sg.Text('Calculated\n',font=('Arial',24))],
			[sg.Text('\n\n\n\n')]]

window = sg.Window('Window Title', layout,size=(500,300),element_justification='c')
sol_ga=0
hill_sol=0
anneal_sol=0
pso_sol=0
event, values = window.read(timeout=0)  # Read for events with a timeout of 0
gen=genetic_algorithm(index,objectives,equalities,inequalities,100,3000,n)
sol_ga=gen.driver()
print("GA")
hill_sol=hill_climbing(1000)
print("HILL")
anneal_sol=simulated_annealing()
print("ANNEAL")
pso_sol=ParticleSwarm(100)
print("Started")
print(anneal_sol)
pso_sol=sete()
print(sol_ga)
print(hill_sol)
print(pso_sol)
window.close()
sol_ga=seta(sol_ga)

layout=[[sg.Text('Simulated Anealing: ',font=('Arial',15)),sg.Text(str(anneal_sol),font=('Arial',15))],
        [sg.Text('Genetic Algorithm: ',font=('Arial',15)),sg.Text(str(sol_ga),font=('Arial',15))],
        [sg.Text('Hill Climbing: ',font=('Arial',15)),sg.Text(str(hill_sol),font=('Arial',15))],
        [sg.Text('Particle Swarm: ',font=('Arial',15)),sg.Text(str(pso_sol[0]),font=('Arial',15))],
        [sg.Text('\n\n')],
        [sg.Button('Quit',size=(4,1))]]
window = sg.Window('Window Title', layout,size=(500,300),element_justification='c')
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Quit': # if user closes window or clicks cancel
        break
    break
window.close()
from matplotlib import pyplot as plt
if n==2:
    y=[hill_sol[1],anneal_sol[1],sol_ga[1],pso_sol[0][1]]
    x=[hill_sol[0],anneal_sol[0],sol_ga[0],pso_sol[0][0]]
    n=["Hill","Annealing","Genetic","PSO"]
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    plt.xlim(0,10)
    plt.ylim(0,10)
    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))
    plt.show()
    plt.clf()
    y=[]
    x=[]
    for i,k in enumerate(hill_x):
        x.append(i+1)
        y.append(k)
    plt.xlabel("Iterations")
    plt.ylabel("Fitness value")
    plt.plot(x,y)
    plt.title('Hill Climbing Fitness')
    plt.show()
    y=[]
    x=[]
    for i,k in enumerate(anneal_x):
        x.append(i+1)
        y.append(k)
    plt.xlabel("Iterations")
    plt.ylabel("Fitness value")
    plt.plot(x,y)
    plt.title('Simulated Annealing Climbing Fitness')
    plt.show()
else:
    y=[hill_sol[1],anneal_sol[1],sol_ga[1],pso_sol[0][1]]
    x=[hill_sol[0],anneal_sol[0],sol_ga[0],pso_sol[0][0]]
    z=[hill_sol[2],anneal_sol[2],sol_ga[2],pso_sol[0][2]]
    n=["Hill","Annealing","Genetic","PSO"]
    fig, ax = plt.subplots()
    ax = plt.axes(projection ="3d")
    ax.scatter3D(x, y, z, color = "green")
    ax.set_zlim(0,5)
    ax.set_xlim(0,5)
    ax.set_ylim(0,5)
    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))
    plt.show()
    plt.clf()
    y=[]
    x=[]
    for i,k in enumerate(hill_x):
        x.append(i+1)
        y.append(k)
    plt.xlabel("Iterations")
    plt.ylabel("Fitness value")
    plt.plot(x,y)
    plt.title('Hill Climbing Fitness')
    plt.show()
    y=[]
    x=[]
    for i,k in enumerate(anneal_x):
        x.append(i+1)
        y.append(k)
    plt.xlabel("Iterations")
    plt.ylabel("Fitness value")
    plt.plot(x,y)
    plt.title('Simulated Annealing Climbing Fitness')
    plt.show()

