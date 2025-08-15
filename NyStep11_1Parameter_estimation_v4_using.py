"""
parameter estimation(whole NY network)
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count
import math
from sklearn.preprocessing import MinMaxScaler
def compute_demand(bar_d, eta, c_pv, c_pt,l):
    """
    Compute travel demand.
    """
    return max((bar_d - eta * (c_pv+c_pt+(1+l[0])*math.log(1+l[0]) - l[0] + (1+l[1])*math.log(1+l[1]) - l[1])),0)
def get_t(bar_t,kappa1,l):
   
    return kappa1*l[0]+bar_t[0]

def creat_parameter_space(nu_vehicles_D=None,nu_transit_D=None,kappa1_D=None,eta_D=None):
    if nu_vehicles_D:
        nu_vehicles_range = np.round(np.arange(nu_vehicles_D[0],nu_vehicles_D[1], nu_vehicles_D[2]),4)
    if nu_transit_D:
        nu_transit_range = np.round(np.arange(nu_transit_D[0], nu_transit_D[1], nu_transit_D[2]),4)
    if kappa1_D:
        kappa1_range = np.round(np.arange(kappa1_D[0], kappa1_D[1], kappa1_D[2]),4)
    parameter_space = []
    if eta_D:
        eta_range = np.round(np.arange(eta_D[0], eta_D[1], eta_D[2]), 3)
        i = 1
        for nu_vehicles in nu_vehicles_range:
            for nu_transit in nu_transit_range:
                for kappa1 in kappa1_range: 
                    for eta in eta_range:
                        parameter_space.append([i, nu_vehicles, nu_transit, kappa1,eta])
                        i+=1
    else:
        i = 1
        for nu_vehicles in nu_vehicles_range:
            for nu_transit in nu_transit_range:
                for kappa1 in kappa1_range: 
                    parameter_space.append([i, nu_vehicles, nu_transit, kappa1])
                    i+=1
    print(f"length of parameter space is:{len(parameter_space)}")         
    return parameter_space

def compute_l(t, lambda_, nu_vehicles, nu_transit, bar_c,d):
    """
    Compute commuting flows while ensuring constraints.
    """
    c_pv = nu_vehicles * float(t[0]) + bar_c[0]
    c_pt = nu_transit * float(t[1]) + bar_c[1]    
    c_total = max(1e-6, c_pv + c_pt)
    c_pv /= c_total
    c_pt /= c_total
    temp1=(-lambda_-c_pv)
    temp2=(-lambda_-c_pt)
    l_pv = max(0, np.exp(temp1) - 1)
    l_pt = max(0, np.exp(temp2) - 1)
 
    return np.array([l_pv, l_pt]), c_pv, c_pt  # Return cost values as well


def iterate_equilibrium(bar_t, bar_c, bar_d, nu_vehicles, nu_transit,kappa1,
                        max_iter=1000, tol=0.0003, verbose=True):
    """
    Iteratively solve for equilibrium commuting flows and travel times.
    """
    t = bar_t.copy()
    lambda_ = -1
    d = bar_d  # Initialize d before loop
    step1=0.05
    step2=0.05
    prev_loss1=0
    prev_loss2=0
    Lambda_convergence=False
    T_convergence=False
    for iteration in range(max_iter):        
        l, c_pv, c_pt = compute_l(t, lambda_, nu_vehicles, nu_transit, bar_c,d) 
        if l[0]==float('inf'):
            l[0]=d
            l[1]=0
            break
        if l[1]==float('inf'):
            l[1]=d
            l[0]=0
            break
        
        
        # Compute the gradient for lambda
        Denominator=l[1]+l[0]+2        
        gradient_lambda = (d - l[0] - l[1])/ Denominator     

        Denominator=kappa1*nu_vehicles*(1+l[0])+1   
        temp=get_t(t,kappa1,l) 
        gradient_t0=(t[0]-temp)/Denominator
        
        # Update lambda and t
        lambda_new = lambda_-step1 * gradient_lambda
        t_new=np.array((t[0]-step2 * gradient_t0,t[1]))
        loss1=abs(lambda_new-lambda_)
        loss2=abs(t_new[0]-t[0])
        if loss1 < tol:            
            Lambda_convergence=True           
        else:
            if loss1>prev_loss1:
                step1 *= 0.85
                prev_loss1=loss1
            else:  # 误差下降
                step1 *= 1.05  
                prev_loss1=loss1
            lambda_=lambda_new
        if loss2<tol:
            T_convergence=True
            
        else:
            if loss2>prev_loss2:
                step2 *= 0.85
                prev_loss2=loss2
            else:  # 误差下降
                step2 *= 1.05  
                prev_loss2=loss2        
            t=t_new
        
        if lambda_<=-20*bar_d or lambda_>=20*bar_d:
            print(f"exit lambda: {lambda_},D : {d}, l:{l}")
            break
        if T_convergence and Lambda_convergence:
            convergence=True
            break
    if not convergence:
        print(f"this one did not get convergence: d:{d},nu_vehicles:{nu_vehicles},nu_transit:{nu_transit}")
    return lambda_, l, d, t  # Return d as well


def process_theta(theta, df):
    nu_vehicles, nu_transit, kappa1 = theta[1:]
    l_star_list = []
    if theta[0]==400 or theta[0]==500:
        print(f"processing {theta}")
    for row in df.itertuples(index=False, name=None):
        bar_d = row[5]
        if bar_d==0:
            break
        bar_c = np.array([row[8], row[9]])  # Convert to NumPy array
        bar_t = np.array([row[3], row[4]])  # Convert to NumPy array        
        total_flow= int(row[6]+row[7])
        lambda_, l, d, t = iterate_equilibrium(
            bar_t, bar_c, bar_d, nu_vehicles, nu_transit, kappa1, 
             max_iter=1000, tol=0.0003, verbose=True
        )
        
        row_sums = np.sum(l)
        if row_sums > 0:
            l *= total_flow / row_sums
        l = l.tolist()
        l.insert(0, theta[0])
        l = np.round(l, 3)
        l_star_list.append(l)
    
    return l_star_list

def solve_optimal_flow_parallel(parameter_space, network_links_data, L_flow_star='L_Flow_star'):   
    num_processes = cpu_count() -1   
    with Pool(num_processes) as pool:
        results = pool.starmap(process_theta, [(theta, network_links_data) for theta in parameter_space])
    
    l_star_list = [item for sublist in results for item in sublist]  # Flatten the list
    results_df = pd.DataFrame(l_star_list)
    save_data=False
    if save_data:
        n_name = f"{L_flow_star}.pkl"
        with open(n_name, 'wb') as f:
            pickle.dump(results_df, f)
        
        output_file = f"{L_flow_star}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}.")
    
    return results_df



def process_row(row, nu_vehicles, nu_transit, kappa1):
    """ 计算单行数据的 equilibrium 结果 """
    bar_d = row[5]
    if bar_d==0:
        return (00000, [0,0], 0)     
    bar_c = np.array([row[8], row[9]])  # Convert to NumPy array
    bar_t = np.array([row[3], row[4]])  # Convert to NumPy array
    total_flow=int(row[6]+row[7])

    lambda_, l, d,t = iterate_equilibrium(bar_t, bar_c, bar_d, nu_vehicles, nu_transit, kappa1, max_iter=100000, tol=0.0003, verbose=False)
    
    row_sums = np.sum(l)
    if row_sums > 0:
        l *= total_flow / row_sums
    
    l = np.round(l.tolist(), 3)  
    return (lambda_, l, total_flow)

def solve_equilibrium_parallel(nu_vehicles, nu_transit, kappa1, network_links_data, solved_equilibrium="solved_equilibrium"):
    """ 使用多进程计算 equilibrium 并保存结果 """
    
    # 设置进程数（一般用 CPU 核心数）
    num_workers = max(1, cpu_count() - 1)
    
    # 准备数据
    data_list = [(row, nu_vehicles, nu_transit, kappa1) for row in network_links_data.itertuples(index=False, name=None)]
    
    # 使用多进程计算
    with Pool(num_workers) as pool:
        results = pool.starmap(process_row, data_list)

    # 转换为 DataFrame
    result_df = pd.DataFrame(results, columns=['lambda_', 'l', 'd'])
    
    # 添加噪声数据

    result_df['l_1'] = result_df['l'].apply(lambda lst: [x + np.random.poisson(lam=3, size=1)[0] for x in lst])
    result_df['l_1'] = result_df['l_1'].apply(lambda lst: [round(float(x), 0) for x in lst])
    
    # 保存结果
    result_df.to_csv(f"{solved_equilibrium}.csv", index=False)
    with open(f"{solved_equilibrium}.pkl", 'wb') as f:
        pickle.dump(result_df, f)
    
    return result_df    


def get_Q(L_flow_star="L_flow_star",network_links_data="matrix.pkl",parameter_space=None):
    if isinstance(L_flow_star,str):
        A=pd.read_pickle(f"{L_flow_star}.pkl")
    else:
        A=pd.DataFrame(L_flow_star)
    A.columns = A.columns.astype(str)
    if  isinstance(network_links_data, str):
        B=pd.read_pickle(network_links_data)
    else:
        B=pd.DataFrame(network_links_data)
    selected_columns = B[['VE_travelor', 'PT_travelor']]   
    # print(selected_columns.head(2))
    repeat_times = len(A) // len(selected_columns) + 1
    B_repeated = pd.DataFrame(
        np.tile(selected_columns.values, (repeat_times, 1)), 
        columns=selected_columns.columns
    ).iloc[:len(A)]
    D= pd.concat([A, B_repeated], axis=1)
    D["Q_value"] = np.sqrt((D["1"] - D["VE_travelor"])**2 + (D["2"] - D["PT_travelor"])**2)
    mean_distance = D.groupby("0")["Q_value"].mean()
    mean_distance_df = pd.DataFrame(mean_distance).reset_index()
    mean_distance_df.columns = ['theta_index', 'Q_value']
    min_row = mean_distance_df.loc[mean_distance_df['Q_value'].idxmin()]
    # print(min_row)
    pioint=int(min_row['theta_index'])
    Q_value=min_row['Q_value']   
    # print(pioint)                       
    print(f"theta: {pioint}, nu_vehicles: {parameter_space[pioint][1]}, nu_transit: {parameter_space[pioint][2]}, kappa1:{parameter_space[pioint][3]}, Q_value:{Q_value}")
    return parameter_space[pioint][1],parameter_space[pioint][2],parameter_space[pioint][3],Q_value               
                
   
    
def NRMSE(L_flow_star="L_flow_star",network_links_data="matrix.pkl"):
    A=pd.read_pickle(f"{L_flow_star}.pkl")
    A.columns = A.columns.astype(str)
    B=pd.read_pickle(network_links_data)
    
    selected_columns = B[['VE_travelor', 'PT_travelor']]
    D= pd.concat([A, selected_columns], axis=1)
    # print(D.head(2))
    max=selected_columns['VE_travelor'].max()
    min=selected_columns['VE_travelor'].min()
    D[["l_ve", "l_pt"]] = D["l"].apply(pd.Series)
    D["distance"] = np.sqrt((D["l_ve"]- D["VE_travelor"])**2 + (D["l_pt"] - D["PT_travelor"])**2)
    mean_distance = D["distance"].mean()/(max-min)
    return mean_distance
    
def Draw_graph(solved_equilibrium,network_links_data):
    if isinstance(solved_equilibrium,str):
        A=pd.read_pickle(solved_equilibrium)
    else:
        A=solved_equilibrium
    A.columns = A.columns.astype(str)
    if isinstance(network_links_data,str):
        B=pd.read_pickle(network_links_data)
    else:
        B=network_links_data   
    selected_columns = B[['VE_travelor', 'PT_travelor']]
    
    A=A.reset_index(drop=True)
    selected_columns=selected_columns.reset_index(drop=True)
    D= pd.concat([A, selected_columns], axis=1)
    D[["l_ve", "l_pt"]] = D["l"].apply(pd.Series)

    Regularization_VE=D['VE_travelor'].max()-D['VE_travelor'].min()
    D['VE_travelor']=D['VE_travelor']/Regularization_VE
    Regularization_PT=D['PT_travelor'].max()-D['PT_travelor'].min()
    D['PT_travelor']=D['PT_travelor']/Regularization_PT
    Regularization_l_ve=D['l_ve'].max()-D['l_ve'].min()
    D['l_ve']=D['l_ve']/Regularization_l_ve
    Regularization_l_pt=D['l_pt'].max()-D['l_pt'].min()
    D['l_pt']=D['l_pt']/Regularization_l_pt
    # a=D["l_ve"]-D["VE_travelor"]
    # print(a)
    # for index, row in D.iterrows():
    #     print(row)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 第一张图：l_ve vs VE_travelor
    axes[0].scatter(D["l_ve"], D["VE_travelor"], color="blue", label="Data Points")
    axes[0].plot([0, max(D["l_ve"].max(), D["VE_travelor"].max())], 
                [0, max(D["l_ve"].max(), D["VE_travelor"].max())], 
                'r-', label="45-degree line")  # 45度对角线
    axes[0].set_xlabel("Predicted VE")
    axes[0].set_ylabel("Observed VE")
    axes[0].set_title("Scatter Plot: Predicted VE vs Observed VE")
    axes[0].legend()
    # axes[0].grid(True)

    # 第二张图：l_pt vs PT_travelor
    axes[1].scatter(D["l_pt"], D["PT_travelor"], color="green", label="Data Points")
    axes[1].plot([0, max(D["l_pt"].max(), D["PT_travelor"].max())], 
                [0, max(D["l_pt"].max(), D["PT_travelor"].max())], 
                'r-', label="45-degree line")  # 45度对角线
    axes[1].set_xlabel("Predicted PT")
    axes[1].set_ylabel("Observed PT")
    axes[1].set_title("Scatter Plot: Predicted PT vs Observed PT")
    axes[1].legend()
    # axes[1].grid(True)

    # 调整子图间距
    plt.tight_layout()
    plt.savefig("graph~~~~~~~~.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

def all_in_one(dataset,parameter_space): 
    network_links_data=pd.read_pickle(dataset)
    train_data, test_data = train_test_split(network_links_data, test_size=0.3, random_state=42) 
    L_flow_star=solve_optimal_flow_parallel(parameter_space,train_data,"L_flow_star_group")
    L_flow_star="L_flow_star_group"    
    nu_vehicles,nu_transit,kappa1,Q_value  = get_Q(L_flow_star,train_data,parameter_space)
    solved_equilibrium=solve_equilibrium_parallel(nu_vehicles,nu_transit,kappa1,test_data,solved_equilibrium="solved_equilibrium")
    Draw_graph(solved_equilibrium,test_data)

def bootstrap(dataset,parameter_space):
    network_links_data=pd.read_pickle(dataset)
    train_data=network_links_data.sample(n=int(len(network_links_data)*0.7), replace=True)
    L_flow_star=solve_optimal_flow_parallel(parameter_space,train_data)
    result  = get_Q(L_flow_star,train_data,parameter_space)
    return result
def run():
    for i in [1,2,3,4,5]:
        dataset=f"normalized_training_data_group{i}.pkl"
        nu_vehicles_D=[0.2495,0.3557,0.01]
        nu_transit_D=[0.0856,0.5504,0.05]
        if i==1:
            kappa1_D=[0.0497,0.1082,0.01] #group1
        elif i==2:
            kappa1_D=[0.1082,0.1667,0.01] #group2
        elif i==3:
            kappa1_D=[0.1667,0.2252,0.01] #group3
        elif i==4:
            kappa1_D=[0.2252,0.2837,0.01] #group4
        elif i==5:
            kappa1_D=[0.2837,0.3422,0.01] #group5


        parameter_space=creat_parameter_space(nu_vehicles_D,nu_transit_D,kappa1_D)
        # print(parameter_space[0])
        # all_in_one(dataset,parameter_space)
        results=[]
        for i in range(50):
            result=bootstrap(dataset,parameter_space)
            results.append(result)
            print(result)
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"group{i}_50_times_estimation.csv",index=False)
        results_df.to_pickle(f"group{i}_50_times_estimation.pkl")

if __name__== "__main__":
    run()






