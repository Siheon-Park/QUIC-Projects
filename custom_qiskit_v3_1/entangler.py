from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector

class Entangler(QuantumCircuit):
    """ ref: https://arxiv.org/abs/1905.10876 """
    def __init__(self, qr:QuantumRegister, id:int, layer:int, param_name:str='theta'):
        super().__init__(qr, name=f'Entangler{id}')
        n = len(qr)
        self.id = id
        self.layer = layer

        # circuit id 0: uniform weight
        if id==0:
            k=0
            self.theta = None
            self.h(qr)

        # circuit id 1
        if id==1:
            k = 2*n
            self.theta = ParameterVector(param_name, layer*k)
            for l in range(layer):
                self._rx_column(qr, self.theta[l*k:l*k+n])
                self._rz_column(qr, self.theta[l*k+n:l*k+2*n])
        
        # circuit id 2
        elif id==2:
            k = 2*n
            self.theta = ParameterVector(param_name, layer*k)
            for l in range(layer):
                self._rx_column(qr, self.theta[l*k:l*k+n])
                self._rz_column(qr, self.theta[l*k+n:l*k+2*n])
                for i in range(n-1):
                    self.cnot(qr[n-1-i], qr[n-2-i])
        
        # circuit id 3
        elif id==3:
            k = 3*n-1
            self.theta = ParameterVector(param_name, layer*k)
            t=0
            for l in range(layer):
                self._rx_column(qr, self.theta[t:t+n]); t+=n
                self._rz_column(qr, self.theta[t:t+n]); t+=n
                for i in range(n-1):
                    self.crz(self.theta[t], qr[n-1-i], qr[n-2-i]); t+=1

        #circuit id 4
        elif id==4:
            k = 3*n-1
            self.theta = ParameterVector(param_name, layer*k)
            t=0
            for l in range(layer):
                self._rx_column(qr, self.theta[t:t+n]); t+=n
                self._rz_column(qr, self.theta[t:t+n]); t+=n
                for i in range(n-1):
                    self.crx(self.theta[t], qr[n-1-i], qr[n-2-i]); t+=1

        # circuit id 5
        elif id==5:
            k = 2*n+2*n+n*(n-1)
            self.theta = ParameterVector(param_name, layer*k)
            t=0
            for l in range(layer):
                self._rx_column(qr, self.theta[t:t+n]); t+=n
                self._rz_column(qr, self.theta[t:t+n]); t+=n
                for i in range(n):
                    for j in range(n):
                        if i!=j:
                            self.crz(self.theta[t], qr[n-1-i], qr[n-1-j]); t+=1
                self._rx_column(qr, self.theta[t:t+n]); t+=n
                self._rz_column(qr, self.theta[t:t+n]); t+=n             

        # circuit id 5
        elif id==6:
            k = 2*n+2*n+n*(n-1)
            self.theta = ParameterVector(param_name, layer*k)
            t=0
            for l in range(layer):
                self._rx_column(qr, self.theta[t:t+n]); t+=n
                self._rz_column(qr, self.theta[t:t+n]); t+=n
                for i in range(n):
                    for j in range(n):
                        if i!=j:
                            self.crx(self.theta[t], qr[n-1-i], qr[n-1-j]); t+=1
                self._rx_column(qr, self.theta[t:t+n]); t+=n
                self._rz_column(qr, self.theta[t:t+n]); t+=n 

        # circuit id 7
        elif id==7:
            k = 2*n+2*n+n-1
            self.theta = ParameterVector(param_name, layer*k)
            t = 0
            for l in range(layer):
                self._rx_column(qr, self.theta[t:t+n]); t+=n
                self._rz_column(qr, self.theta[t:t+n]); t+=n
                for i in range(n-1):
                    if i%2==0:
                        self.crx(self.theta[t], qr[i+1], qr[i]); t+=1
                self._rx_column(qr, self.theta[t:t+n]); t+=n
                self._rz_column(qr, self.theta[t:t+n]); t+=n
                for i in range(n-1):
                    if i%2==1:
                        self.crx(self.theta[t], qr[i+1], qr[i]); t+=1

        # circuit id 8
        elif id==8:
            k = 2*n+2*n+n-1
            self.theta = ParameterVector(param_name, layer*k)
            t = 0
            for l in range(layer):
                self._rx_column(qr, self.theta[t:t+n]); t+=n
                self._rz_column(qr, self.theta[t:t+n]); t+=n
                for i in range(n-1):
                    if i%2==0:
                        self.crz(self.theta[t], qr[i+1], qr[i]); t+=1
                self._rx_column(qr, self.theta[t:t+n]); t+=n
                self._rz_column(qr, self.theta[t:t+n]); t+=n
                for i in range(n-1):
                    if i%2==1:
                        self.crz(self.theta[t], qr[i+1], qr[i]); t+=1

        # circuit id 9
        elif id==9:
            k = n
            self.theta = ParameterVector(param_name, layer*k)
            t=0
            for l in range(layer):
                self.h(qr)
                for i in range(n-1):
                    self.cz(qr[i+1], qr[i])
                self._rx_column(qr, self.theta[t:t+n]); t+=n

        # circuit id 10
        elif id==10:
            k = n
            self.theta = ParameterVector(param_name, n+layer*k)
            t=0
            self._ry_column(qr, self.theta[t:t+n]); t+=n
            for l in range(layer):
                for i in range(n):
                    self.cz(qr[(n-1-i)%n], qr[(n-2-i)%n])
                self._ry_column(qr, self.theta[t:t+n]); t+=n

        # circuit id 11
        elif id==11:
            k = 4*n-4
            self.theta = ParameterVector(param_name, layer*k)
            t=0
            for l in range(layer):
                self._ry_column(qr, self.theta[t:t+n]); t+=n
                self._rz_column(qr, self.theta[t:t+n]); t+=n
                for i in range(n-1):
                    if i%2==0:
                        self.cx(qr[i+1], qr[i])
                self._ry_column(qr[1:n-1], self.theta[t:t+n-2]); t=t+n-2
                self._rz_column(qr[1:n-1], self.theta[t:t+n-2]); t=t+n-2
                for i in range(n-1):
                    if i%2==1:
                        self.cx(qr[i+1], qr[i])

        # circuit id 12
        elif id==12:
            k = 4*n-4
            self.theta = ParameterVector(param_name, layer*k)
            t=0
            for l in range(layer):
                self._ry_column(qr, self.theta[t:t+n]); t+=n
                self._rz_column(qr, self.theta[t:t+n]); t+=n
                for i in range(n-1):
                    if i%2==0:
                        self.cz(qr[i+1], qr[i])
                self._ry_column(qr[1:n-1], self.theta[t:t+n-2]); t=t+n-2
                self._rz_column(qr[1:n-1], self.theta[t:t+n-2]); t=t+n-2
                for i in range(n-1):
                    if i%2==1:
                        self.cz(qr[i+1], qr[i])

        # circuit id 13
        elif id==13:
            k=4*n
            self.theta = ParameterVector(param_name, layer*k)
            t=0
            for l in range(layer):
                self._ry_column(qr, self.theta[t:t+n]); t+=n
                for i in range(n):
                    self.crz(self.theta[t], qr[(n-1-i)%n], qr[(n-i)%n]); t+=1
                self._ry_column(qr, self.theta[t:t+n]); t+=n
                for i in range(n):
                    self.crz(self.theta[t], qr[(n-1+i)%n], qr[(n-2+i)%n]); t+=1

        # circuit id 14
        elif id==14:
            k=4*n
            self.theta = ParameterVector(param_name, layer*k)
            t=0
            for l in range(layer):
                self._ry_column(qr, self.theta[t:t+n]); t+=n
                for i in range(n):
                    self.crx(self.theta[t], qr[(n-1-i)%n], qr[(n-i)%n]); t+=1
                self._ry_column(qr, self.theta[t:t+n]); t+=n
                for i in range(n):
                    self.crx(self.theta[t], qr[(n-1+i)%n], qr[(n-2+i)%n]); t+=1

        # circuit id 15
        elif id==15:
            k=2*n
            self.theta = ParameterVector(param_name, layer*k)
            t=0
            for l in range(layer):
                self._ry_column(qr, self.theta[t:t+n]); t+=n
                for i in range(n):
                    self.cz(qr[(n-1-i)%n], qr[(n-i)%n])
                self._ry_column(qr, self.theta[t:t+n]); t+=n
                for i in range(n):
                    self.cz(qr[(n-1+i)%n], qr[(n-2+i)%n])

        # circuit id 16
        elif id==16:
            k = 3*n-1
            self.theta = ParameterVector(param_name, layer*k)
            t=0
            for l in range(layer):
                self._ry_column(qr, self.theta[t:t+n]); t+=n
                self._rz_column(qr, self.theta[t:t+n]); t+=n
                for i in range(n-1):
                    if i%2==0:
                        self.crz(self.theta[t], qr[i+1], qr[i]); t+=1
                for i in range(n-1):
                    if i%2==1:
                        self.crz(self.theta[t], qr[i+1], qr[i]); t+=1      
        # circuit id 17
        elif id==17:
            k = 3*n-1
            self.theta = ParameterVector(param_name, layer*k)
            t=0
            for l in range(layer):
                self._ry_column(qr, self.theta[t:t+n]); t+=n
                self._rz_column(qr, self.theta[t:t+n]); t+=n
                for i in range(n-1):
                    if i%2==0:
                        self.crx(self.theta[t], qr[i+1], qr[i]); t+=1
                for i in range(n-1):
                    if i%2==1:
                        self.crx(self.theta[t], qr[i+1], qr[i]); t+=1    

        # circuit id 18
        elif id==18:
            k=3*n
            self.theta = ParameterVector(param_name, layer*k)
            t=0
            for l in range(layer):
                self._ry_column(qr, self.theta[t:t+n]); t+=n
                self._rz_column(qr, self.theta[t:t+n]); t+=n
                for i in range(n):
                    self.crz(self.theta[t], qr[(n-1-i)%n], qr[(n-i)%n]); t+=1

        # circuit id 19
        elif id==19:
            k=3*n
            self.theta = ParameterVector(param_name, layer*k)
            t=0
            for l in range(layer):
                self._ry_column(qr, self.theta[t:t+n]); t+=n
                self._rz_column(qr, self.theta[t:t+n]); t+=n
                for i in range(n):
                    self.crx(self.theta[t], qr[(n-1-i)%n], qr[(n-i)%n]); t+=1
         

    def _rx_column(self, qr:QuantumRegister, theta:ParameterVector):
        n = len(qr)
        [self.rx(theta[i], qr[i]) for i in range(n)]

    def _rz_column(self, qr:QuantumRegister, theta:ParameterVector):
        n = len(qr)
        [self.rz(theta[i], qr[i]) for i in range(n)]

    def _ry_column(self, qr:QuantumRegister, theta:ParameterVector):
        n = len(qr)
        [self.ry(theta[i], qr[i]) for i in range(n)]



if __name__=='__main__':
    qr = QuantumRegister(3)
    Entangler(qr, 1, 4).draw('mpl')