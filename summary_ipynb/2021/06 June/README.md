# Summary of 2021 June

### 6/03
Run serveral experiment on real machine to get some data and figure for the papaer. They are saved at 'model', 
'montreal' dir.

### 6/05
Found that classification for "unbalanced" dataset is not working with noisy simulation. This is weired since previous 
experements shows that it is perfectly fine

### 6/23
Going through Git logs to find answer

### 6/24
Found that the problem was 'QuantumCircuit.uc' bug. I think the problem is that uc is not good at decomposing matrix to 
single qubit gates. Thus, I used sequence of ucry and ucry to avoid decomposition. 

### 6/25
Conclude that qubit mapping method has little impact on the performance.

{yi - i1 - i0 - xi - a - xj - j0 - j1 - yj} v.s. {(i0, i1) = yi - xi - a - xj - yj = (j0, j1)} 

Also, It seems that when k=1 & unbalanced dataset, performance of STC < QASVM. I now let C=None, k=1. The results are 
saved at './naive(montreal)/' dir.