import sympy as sp
import numpy as np
'''
最先写的模拟模型运行的代码
'''

def compute_derivative(func):
    """
    计算传入函数的导数并求出其逆函数，返回一个包含所有逆函数解的列表。
    """
    # 定义符号变量 x 和 y
    x, y = sp.symbols('x y')
    # 生成符号表达式
    expr = func(x)
    # 求导函数
    derivative = sp.diff(expr, x)

    # 求导函数的逆函数（列表形式）
    inverse = sp.solve(sp.Eq(y, derivative), x)

    return inverse


def evaluate_inverse(func, y_value, index=0):
    """
    使用逆函数计算特定 y 值对应的 x 值。默认使用第一个解（index=0）。
    """
    inverse_func_list = compute_derivative(func)

    # 确保索引合法
    if index >= len(inverse_func_list):
        raise IndexError("所选解的索引超出范围")

    # 将 y 替换为具体的值
    x_value = inverse_func_list[index].subs('y', y_value)

    return x_value

def A_function(A_bar, alpha, X):
    """
    参数:
    A_bar (float): 向量
    alpha (float): 向量
    X (numpy.ndarray): 3D 矩阵 (a, b, c)

    返回:
    numpy.ndarray: b维度上所有的值求和+1 结果的alpha次方,再乘以A_bar
    """
    # 按 b 维度求和
    m = np.sum(X, axis=(0, 2)) + 1  # 求和并加 1

    # 计算 m 的 alpha 次方
    q = m ** alpha

    # 计算 result
    result= [q[i] * A_bar[i] for i in range(len(q))]    

    return result

def Mu_function(Mu_bar, beta, X):
    """
    参数:
    Mu_bar (float): 向量
    beta (float): 向量
    X (numpy.ndarray): 3D 矩阵 (a, b, c)

    返回:
    numpy.ndarray: b维度上所有的值求和+1 结果的beta次方,再乘以Mu_bar
    """
    # 按 b 维度求和
    m = np.sum(X, axis=(1, 2)) + 1  # 求和并加 1

    # 计算 m 的 beta 次方
    q = m ** beta

    # 计算 result     
    result= [q[i] * Mu_bar[i] for i in range(len(q))]  

    return result  

def travel_time(T,X,E,K):
    """
    参数:
    T: 3维矩阵
    X: 3维矩阵
    E: 3维矩阵
    K: 3维矩阵
    返回:
    3维矩阵
    """
    T=np.array(T)
    X=np.array(X)
    E=np.array(E)
    K=np.array(K)
    d1,d2,d3=X.shape
    result = np.zeros((d1, d2, d3))
    for i in range(len(result)):
        for j in range(len(result[0])):
            for k in range(len(result[0][0])):                    
                EX=sum([E[i][j][o]*X[i][j][o] for o in range(len(result[0][0]))])-E[i][j][k]*X[i][j][k]
                result[i][j][k]=T[i][j][k]*((1+X[i][j][k]+EX))**K[i][j][k]                   
    return result

def get_Y(T,Mu,A,lam,gamma):
    T= np.array(T)
    d1,d2,d3=T.shape  
    result = np.zeros((d1, d2, d3))
    for i in range(d1):
        for j in range(d2):
            for k in range(d3):
                result[i][j][k]=Mu[i]+A[j]-T[i][j][k]+lam[i]-gamma[j]
    return result

def get_X(Y,func):
    Y= np.array(Y)
    d1,d2,d3=Y.shape  
    result = np.zeros((d1, d2, d3))
    for i in range(d1):
        for j in range(d2):
            for k in range(d3):
                temp=evaluate_inverse(func,Y[i][j][k])
                print(temp)
                result[i][j][k]= 1 if temp >= 1 else (0 if temp <= 0 else temp)

    return result

def update_lam_gamma(L,Mu,Y):
    numerator=L-Mu
    temp=-np.exp(1) ** Y 
    Hessian= np.sum(temp , axis=2)
    Hessian_diagonal=np.diagonal(Hessian)
    denominator=Hessian_diagonal
    result=numerator/denominator
    return result

def check_sub_Convergence(New,Old,threshold):
    temp=np.max(np.abs(New-Old))
    return True if threshold >= temp else False

def check_Convergence(T_next,T_bar,threshold_T,A_next,A_bar,threshold_A,mu_next,mu_bar,threshold_mu,lam_next,lam_bar,threshold_lam,gamma_next,gamma_bar,threshold_gamma):
    if not check_sub_Convergence(T_next,T_bar,threshold_T):
        print('T is not convergence.')
        return False
    if not check_sub_Convergence(A_next,A_bar,threshold_A):
        print('A is not convergence.')
        return False
    if not check_sub_Convergence(mu_next,mu_bar,threshold_mu):
        print('mu is not convergence.')
        return False
    if not check_sub_Convergence(lam_next,lam_bar,threshold_lam):
        print('lambda is not convergence.')
        return False
    if not check_sub_Convergence(gamma_next,gamma_bar,threshold_gamma):
        print('lambda is not convergence.')
        return False
    return True




def Iteration(counter,mu_bar,A_bar,T_bar,lam_bar,gamma_bar,kappa,zeta,func,beta,alpha,learn_rate,threshold_T,threshold_A,threshold_mu,threshold_lam,threshold_gamma,L_lam,L_gamma):
    mu_bar = mu_bar
    A_bar = A_bar
    T_bar = T_bar
    lam_bar =lam_bar
    gamma_bar =gamma_bar
    kappa = kappa
    zeta = zeta
    func=func
    beta=beta
    alpha=alpha
    learn_rate=learn_rate
    threshold_T=threshold_T
    threshold_A=threshold_A
    threshold_mu=threshold_mu
    threshold_lam=threshold_lam
    threshold_gamma=threshold_gamma
    L_lam=L_lam
    L_gamma=L_gamma
    for _ in range(counter):
        Y=get_Y(T_bar,mu_bar,A_bar,lam_bar,gamma_bar)
        X=get_X(Y,func)
        #update T
        T_i= travel_time(T_bar,X,zeta,kappa)
        T_next=T_bar+learn_rate*(T_i-T_bar)
        
        #update mu
        mu_i=Mu_function(mu_bar,beta,X)
        mu_next=mu_bar+learn_rate*(mu_i-mu_bar)
        #update A
        A_i=A_function(A_bar,alpha,X)
        A_next=A_bar+learn_rate*(A_i-A_bar)
        #update lam
        lam_next=lam_bar+learn_rate*update_lam_gamma(L_lam,mu_bar,Y)
        #update gamma
        gamma_next=gamma_bar+learn_rate*update_lam_gamma(L_gamma,mu_bar,Y)

        # check if it hit the Convergence

        check_point=check_Convergence(T_next,T_bar,threshold_T,A_next,A_bar,threshold_A,mu_next,mu_bar,threshold_mu,lam_next,lam_bar,threshold_lam,gamma_next,gamma_bar,threshold_gamma)
        
        if check_point:
            print("nice, hit the Convergence")
            return T_next,A_next,mu_next,lam_next,gamma_next,X
        
        
        T_bar=T_next
        mu_bar=mu_next
        A_bar=A_next
        lam_bar=lam_next
        gamma_bar=gamma_next
    print("Did not get Convergence,return the latest values:")
    return T_next,A_next,mu_next,lam_next,gamma_next,X

def check_condition(T_bar,kappa,zeta):
    T_bar= np.array(T_bar)
    kappa= np.array(kappa)
    zeta= np.array(zeta)
    if T_bar.shape==kappa.shape and zeta.shape==T_bar.shape:
        pass
    else:
        print('T_bar,kappa,zeta,have to be same size.')
        return False
    if len(T_bar.shape)==3:
        pass
    else:
        print('T_bar,kappa,zeta,have to be 3 dimention.')
        return False
    d1,d2,d3=T_bar.shape  
    for i in range(d1):
        for j in range(d2):            
            temp1=T_bar[i][j][0]*kappa[i][j][0]*zeta[i][j][1]
            temp2=T_bar[i][j][1]*kappa[i][j][1]*zeta[i][j][0]
            if temp1 !=temp2:
                print(f'T is not symmetric.T_bar{[i]}{[j]}={temp1},T_bar{[i]}{[j]}={temp2}')
                return False
    return True
def check_existence(alpha,beta,mu_bar,A_bar,T_bar,kappa,zeta,L_lam,L_gamma,X):
    temp_vector1=alpha*A_bar*(L_gamma+1)**(alpha-1)
    temp_vector1=beta*mu_bar*(L_lam+1)**(beta-1)
    kappa= np.array(kappa)
    d1,d2,d3=kappa.shape  
    temp_metrix = np.zeros((d1, d2, d3))
    for i in range(d1):
        for j in range(d2):
            temp_metrix[i][j][0]=kappa[i][j][0]*T_bar[i][j][0]*(1+X[i][j][0]+zeta[i][j][1]*X[i][j][1])**(kappa[i][j][0]-1)
            temp_metrix[i][j][1]=kappa[i][j][1]*T_bar[i][j][1]*(1+X[i][j][1]+zeta[i][j][0]*X[i][j][0])**(kappa[i][j][1]-1)

    for i in range(d1):
        for j in range(d2):
            temp_metrix[i][j][0]=temp_vector1[0]+temp_vector1[0]-temp_metrix[i][j][0]
            temp_metrix[i][j][1]=temp_vector1[0]+temp_vector1[0]-temp_metrix[i][j][1]
    if np.min(temp_metrix) <=0:
        return True
    else:
        print('not existence!')
        return False
    




            



           


def main():    
    mu_bar = np.array([1, 6])
    A_bar = np.array([10, 9])
    T_bar = np.array([[[10, 2], [10, 40]], [[20, 80], [15, 60]]])
    lam_bar =np.array([1, 2])
    gamma_bar =np.array([2, 2])
    kappa = np.array([[[2, 1],[2,1]],[[2,1],[2,1]]])
    zeta = np.array([[[2,0.2],[0.2,0.1]],[[1,0.5],[4,2]]])    
    func=lambda x: (1 + x) * sp.log(1 + x) - x    
    beta=3
    alpha=3
    learn_rate=0.1
    threshold_T=0.1
    threshold_A=0.1
    threshold_mu=0.1
    threshold_lam=0.1
    threshold_gamma=0.1
    L_lam=np.array([1, 2])
    L_gamma=np.array([1, 2])
    counter=500
    check_condition1=check_condition(T_bar,kappa,zeta)
    if not check_condition1:
        print("faled on check_condition")
        exit()
    Y=get_Y(T_bar,mu_bar,A_bar,lam_bar,gamma_bar)
    X=get_X(Y,func)    
    check_condition2=check_existence(alpha,beta,mu_bar,A_bar,T_bar,kappa,zeta,L_lam,L_gamma,X)
    if not check_condition2:
        print("faled on check_existence")
        exit()
    
    result=Iteration(counter,mu_bar,A_bar,T_bar,lam_bar,gamma_bar,kappa,zeta,func,beta,alpha,learn_rate,threshold_T,threshold_A,threshold_mu,threshold_lam,threshold_gamma,L_lam,L_gamma)
    
    print(result)

if __name__ == "__main__":
    main()
