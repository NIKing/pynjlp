import numpy as np

def bfgs(f, grad_f, x0, tol=7.89e-30, max_iter= 100):
    """
    BFGS optimization algorithm

    Parameters:
        - f: Objective function to minimize
        - grad_f: Gradient of the objective function
        - x0: Initial guess for the minimum
        - tol: Tolerance for convergence    收敛容忍度
        - max_iter: Maximum number of iterations

    Retruns:
        - x: Estimated location of the minimum
        - f_val: Value of the obejctive function at the minimum
        - n_iter: Number of iterations performed
    """
    
    x = x0
    n = len(x0)
    H = np.eye(n) # Initial Hessian approximation as the identity matrix, 它是一个单位矩阵，类似数字 1
    #print('H0:')
    #print(H)
    #print('')

    for n_iter in range(max_iter):
        # Step 1: 获取向量的梯度变化
        # 梯度计算(x的导数, 一阶导数)，即 计算每个特征的变化率和方向
        grad = grad_f(x)
        
        print('grad:', grad)
        #print(np.linalg.norm(grad))
        #print('')
        if np.linalg.norm(grad) < tol:
            break
        
        # Step 2: 获取梯度在Hessian矩阵中的变化
        # search direction 方向
        # -1: 在神经网络中，我们总是希望能够最小化损失函数，因此往往我们需要沿着负梯度的方向更新参数，沿着负梯度方向更新参数，函数值会减小的最快。
        # 通过向量点积计算，获取两个向量的相似度（注意，点乘获取到的是标量，但是在这里H是二维的矩阵，而grad是一维的向量，因此点乘就发生了降维，得到不是标量而是向量）
        d = np.dot(H, grad) * (-1) # eg: [-2. -4]
        print('d:', d)
        #print(np.dot(grad, d))
        #print('')

        # Step 3: 获取梯度合适的下降步长
        # line search (simple backtracking) 步长
        # 寻找让目标函数下降最多的步长, 可以认为是学习率
        # x + alpha * d 表示线性计算，f(x + alpha * d) 表示移动到新位置后的损失值
        # alpha * np.dot(grad, d), 
        #alpha = 1.0
        #while f(x + alpha * d) > f(x) + 1e-4 * alpha * np.dot(grad, d):
        #    alpha *= 0.5
        
        # 固定步长
        #alpha = 0.3

        # 随着迭代逐步增大
        alpha = 0.1 * (n_iter + 1)

        # 先小，中间大，后面小
        #alpha = 0.5 * (1 - n_iter / max_iter)
        
        # Step 4: 计算下降的距离，并叠加之前的位置形成新的位置
        # update step
        s = alpha * d   
        x_new = x + s   
        
        # Step 5: 计算新位置的梯度变化，并计算两个位置的梯度变化量
        grad_new = grad_f(x_new)    
        y = grad_new - grad         
        
        # Step 6: 更新 Hessian 矩阵，不过这个计算与公式不符???????
        # BFGS update
        rho = 1.0 / (np.dot(y, s) + 1e-10)
        I = np.eye(n)
        #H = np.dot((I - rho * np.outer(s, y)), np.dot(H, (I - rho * np.outer(y,s))) + rho * np.outer(s,s))

        print('H:', H)
        
        # Step 7: 更新位置
        x = x_new
        print(f'迭代{n_iter}: 新的位置={x_new}, 损失值={f(x)}, 方向={d}, 步长={alpha}')
        print('')

    #print(f'H{n_iter}=')
    #print(H)

    return x, f(x), n_iter


def f(x):
    """
    目标函数，二次函数 = (x_0 - 1)^2 + (x_1 - 2)^2，需要说的是 (1, 2) 是目标值的意思。
    目标函数通过平方范数公式计算当前特征向量与目标向量的距离, 返回的时候距离也可以认为是损失值。
    -param x 特征向量    target function : 
    """
    return (x[0] - target_x[0]) ** 2 + (x[1] - target_x[1]) ** 2

def grad_f(x):
    """
    梯度函数, 指使了目标函数在当前点的最陡上升方向(向上的方向)
    -param x 特征向量
    是根据目标函数来定义的，具体来说，梯度表示目标函数在某一点的导数向量，表示了函数在该点的变化率和方向。

    方向，在数学意义上，是函数增大最快的方向。
    变化率，是指向量的大小（或模长），表示该方向上函数值的变化有多大。

    计算梯度就是计算目标函数中每个变量的偏导数。
    """
    return np.array([2 * (x[0] - target_x[0]), 2 * (x[1] - target_x[1])])


# bfgs 算法
# 利用迭代更新的方式来近似一个Hessian矩阵，来逼近真实的Hessian矩阵。以达到函数能在某个区域的快速找到区域内函数的最小值。
# 特别注意，bfgs是准牛顿法，它并不直接计算二阶导数（虽然，Hessian矩阵是二阶导数的矩阵），而是通过迭代过程中一阶导数信息来更新Hessian矩阵的近似

# f(x) 可以看作是模型的决策函数，以优化角度考虑，它也是损失函数。当然，对于条件概率模型而言，损失函数是另外的函数，比如经验风险函数或结构化风险函数。
# 不能把 f(x) 当做模型的决策函数, 它就是损失函数，常见的目标函数是CrossEntepyLoss(交叉损失值), 这样的话，x0 就是模型的预测值。而目标函数中的最优解(1, 2)，则可以认为是真实标签。 -- 2024年8月17日

# x0 看作是输入的特征值，需要特别注意的是，f(x) 和 grad_f(x) 的设计中，它们的公式每项都与x0的两个特征值相关联，若修改x0，则同时也要修改这两个函数。
x0 = np.array([0.0, 0.0])
target_x = np.array([10, 20])

x_min, f_min, n_iter = bfgs(f, grad_f, x0)

print('')
print('到达的位置:', x_min)
print('距离目标位置还需要多少:', f_min)
print('迭代次数:', n_iter)
