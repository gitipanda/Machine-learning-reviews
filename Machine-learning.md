# Machine learning

- What is machine learning?
    
    Data → Feature extraction, dimension reduction →  build models (maybe assume parameters) or has some laws, optimization to solve → Result → Evaluation
    
    Machine → “Model”, learning → “Use historical data / samples, to have ability to diagnose or predict”
    
- Feature sacaling
    - Min-max nomalization:
        
        $\tilde{x}=\frac{x-min(x_{sample})}{max(x_{sample})-min(x_{sample})}$, then  $\tilde{x}\in(0,1).$
        
    - Mean normalization:
        
        $\tilde{x}=\frac{x-avg(x_{sample})}{max(x_{sample})-min(x_{sample})}$, then $\tilde{x}\in(-1,1).$
        
    - Z-score nomalization:
        
        $\tilde{x}=\frac{x-avg(x_{sample})}{\sigma(x_{sample})}$, with $\sigma(x_{sample})$ be the standard deviation of the sampled data.
        
- Metric
    
    $\|\cdot\|_2$, hamming distance, shared key words...
    
- Missing data
    
    Delete the whole record, Use mean/median to substitute the value, Use regression and fit the value, Nearest Neighbor or common chosen to fit the value, EM (random guess the value $x'$, step 1: estimate the parameters $\theta$, by using $x'$; step 2: known $\theta$, to estimate best $x'$). 
    
- KL divergence, entropy, cross entropy, mutual information
    
    KL divergence is a statistical distance (but not metric) between two distributions : $p(x)$ and $q(x)$. KL divergence is also called information gain or relative entropy, which describes how much information achieved if using P instead of using Q.
    
    $$
    D_{KL}(P||Q)=\mathbb{E}_P(\log\frac{P}{Q})
    $$
    
    Entropy describes how much uncertainty is in a statistical system. It is an average-level measure as follows:
    
    $H(P)=-\mathbb{E}_P(\log P)$
    
    If the true distribution is $P$, but the coding scheme used as $Q$, then cross entropy describes how much bits needed to identify the fact.
    
    $H(P,Q)=H(P)+D_{KL}(P||Q)=-\mathbb{E}_P(\log Q)$
    
    The mutual information measures the mutual dependence of two random variable, which is defined as follows:
    
    $I(X,Y)=D_{KL}(P_{X,Y}||P_X \otimes P_Y)$
    

### Choose hyperparameters and evaluation

- Decide hyperparameter
    - How to choose $\lambda$?
        
        use validation set or cross-validation to choose a good $\lambda.$
        
- Evaluation - validation / cross validation

### Bias and variance tradeoff

- Interpretation
    
    $$
    \int_{data}\int_x\int_y \left(h_{data}(x)-y\right)^2\mathbb{P}(data)p(x,y)dxdyd(data)
    $$
    
    $$
    =\mathbb{E}_{data}(\int_x\int_y (h_{data}(x)-\mathbb{E}(h_{data}(x)))^2 dxdy)+\mathbb{E}_{data}(\int_x\int_y(\mathbb{E}_{data}(h_{data}(x))-y)^2dxdy)+\text{Noise},
    $$
    
    the first term is the variance due to limited data and the second term is the bias of the model due to poorness of $h(\cdot)$.
    

### Regression model, solver, optimization and evaluation

- Linear regression model
    
    **Model:**
    
    If $y\in \mathbb{R}$, and $\bm{x}\in\mathbb{R}$:
    
    $$
    y=w_0+wx.
    $$
    
    if $\bm{y}\in\mathbb{R}^m$, and $\bm{x}\in\mathbb{R}^n$:
    
    $$
    y=b+\bm{w}^T\bm{x}.
    $$
    
    Noisy model: assume $e$~$\mathcal{N}(\bm{0},\sigma^2)$, and 
    
    $$
    y=b+\bm{w}^T\bm{x}+e
    $$
    
    **Solution:**
    
    Minimize RSS(residual sum of squares): 
    
    $$
    b,\bm{w}=\arg\min \|\bm{y}_{data}-\bm{X}_{data}\bm{w}\|^2.
    $$
    
    The minimum is reached when
    
    $$
    \frac{\partial RSS(\bm{w},b)}{\partial \bm{w}}=0\text{, and } \frac{\partial RSS(\bm{w},b)}{\partial b}=0.
    $$
    
    That is,
    
    $$
    \bm{w}=(\bm{X}_{data}^T\bm{X}_{data})^{-1}\bm{X}_{data}\bm{y}_{data}.
    $$
    
    **Probabilistic interpretation** for the noisy model, maximize log-likelihood:
    
    $$
    \bm{w},b = \arg\max \log \mathbb{P}(\bm{y}_{data}|\bm{x}_{data}),
    $$
    
    which is equivalent to minimize RSS(residual sum of squares). Moreover, $\sigma^*=\sqrt{\frac{1}{m}\sum \|\bm{y}_i-\bm{w}^T\bm{x}_i-b\|^2}$ reveals the noise level. And, one can do statistical test to show whether this model is reasonable.
    
    Assume a noisy model: $y=\bm{w}^T\bm{x}+e$, with $e$ ~ $\mathcal{N}(0,1)$, then we have 
    
    $$
    \hat{\bm{w}}^{LMS}=\hat{\bm{w}}^{MLE},
    $$
    
    LMS standing for Least mean squares and MLE standing for maximize likelihood estimator.
    
    **Large-scale computing solution:**
    
    As calculating inverse of a high-dimensional matrix cost heavy computation power, one can use SGD to reduce computational cost. Critical computational cost happens when calculate $(X_{data}^TX_{data})^{-1}$, methods include Gaussian-Jordan elimination, Newton’s method, blockwise inversion, etc. However, batch GD (gradient descent) and Stochastic Gradient Descent (SGD).
    
    - Batch GD:
    
    $$
    \bm{w}_{k+1}=\bm{w}_k-\alpha g_k,
    $$
    
    $\text{with }g_k=\nabla RSS(\bm{w}_k)=\bm{X}_{data}^T(\bm{X}_{data}\bm{w}_k-\bm{y}_{data}).$ Computational cost per iteration: $\mathcal{O}(mN).$
    
    - SGD:
    
    Use the gradient of a single sample instead of all samples. Similar to batch GD, but with
    
    $$
    g_k=(\bm{x}_k^T\bm{w}_k-y_k)\bm{x}_k.
    $$
    
    This is use one random sample to approximate the gradient, rather than all samples. Computational cost per iteration: $\mathcal{O}(N).$
    
    **Batch size**: tradeoff between computational cost and accuracy.
    
    Try different stepsize on validation dataset, and choose the one with best tradeoff between speed and stability.
    
- Non-linear regression
    
    Assume $y=\bm{w}^T\phi(\bm{x})+b$, assume
    
    $\Phi = \begin{pmatrix}\Phi(\bm{x}_1)^T\\\Phi(\bm{x}_2)^T\\...\\\Phi(\bm{x}_m)^T\end{pmatrix}$, we have $\bm{w}^{LMS}=(\Phi^T\Phi)^{-1}\Phi^T\bm{y}_{data}$.
    

### Optimization

- **Advanced iterative scheme**
    
    机器之心： [https://www.jiqizhixin.com/graph/technologies/7eab38a3-23ec-494c-a677-415b6f85e6c5](https://www.jiqizhixin.com/graph/technologies/7eab38a3-23ec-494c-a677-415b6f85e6c5)
    
    1951 SGD: 
    
    Robbins, H., & Monro, S. (1951). A stochastic approximation method.
    
    [https://www.jstor.org/stable/pdf/2236626.pdf?casa_token=eixnlEZn6esAAAAA:Q-yfoI6TW6OLEdAPa66hdKNEZ5EfoJFwsBSd7jN2v3sJ9Rf-edelufy7PoWbDNV4gEQ89sWZQh0IDouCcpyweYxnCZW7MMLUv1vWn-yL3tyomE5ni_ii](https://www.jstor.org/stable/pdf/2236626.pdf?casa_token=eixnlEZn6esAAAAA:Q-yfoI6TW6OLEdAPa66hdKNEZ5EfoJFwsBSd7jN2v3sJ9Rf-edelufy7PoWbDNV4gEQ89sWZQh0IDouCcpyweYxnCZW7MMLUv1vWn-yL3tyomE5ni_ii)
    
    2011 Adagrad
    
    Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12(Jul), 2121-2159.
    
    [https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    
    2012 AdaDelta
    
    Zeiler, M. D. (2012). ADADELTA: an adaptive learning rate method. 
    
    [https://arxiv.org/abs/1212.5701](https://arxiv.org/abs/1212.5701)
    
    2014 Adam
    
    Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. 
    
    [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
    
    Review paper: see [https://arxiv.org/pdf/1609.04747.pdf](https://arxiv.org/pdf/1609.04747.pdf)
    
    comment: very practical and quick review of lines of work on adaptive iterative schemes in optimization. 
    
    **Momentum**
    
    **AdaGrad (adaptive gradient algorithm)**
    
    Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. *Journal of machine learning research*, *12*(7).
    
    link: [https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    
    **Adam**
    
    **AdaDelta**
    
- **Randomness**

### Regularization

- Ridge regression
    
    There are reasons that make $\bm{X}_{data}^T\bm{X}_{data}$ not invertible or rank-deficient, eg. not enough samples, redundant features. If we look at the generalized inverse of $\bm{X}_{data}^T\bm{X}_{data}$, some component would contribute $+-\infty$, making $\hat{\bm{w}}$ to be singular. To overcome the singular property of the inverse, one can use $\|\bm{w}\|_2^2$ as regularization to avoid the singular cases.
    
    $$
    \min_{\bm{w}} \|\bm{y}_{data}-\bm{X}_{data}\bm{w}\|_2^2+\lambda\|\bm{w}\|_2^2.
    $$
    
    The redundancy of features will results in minimizer of least squares to be not a singular point, but a “line” or ridge instead. Therefore, we need the regularization term $\|\bm{w}\|_2^2$ to avoid the consequence of rank deficient of $\bm{X}_{data}^T\bm{X}_{data}$ and the riege property of the argmin.
    
    **Probabilistic interpretation**:
    
    Assume the model is: $y=\bm{w}^T\bm{x}+e$, with $\bm{w}\sim \mathcal{N}(\bm{0}^N,\sigma^2\bm{I}_N)$ and $e\sim\mathcal{N}(0,1)$. Therefore, the MAP solution is 
    
    $\hat{\bm{w}}^{MAP}=\arg\max \mathbb{P}(\bm{y}_{data}|\bm{w})\cdot \mathbb{P}(\bm{w})=\arg\max \sum \frac{1}{\sigma_0^2}(\bm{w}^T\bm{x_i}-y_i)^2+\frac{1}{\sigma^2}\|\bm{w}\|_2^2.$ 
    
- LASSO
    
    LASSO stands for least absolute shrinkage and selection operator. If the coefficient is sparse, then one can use LASSO to do feature selection. The $\|\cdot\|_1$ helps make the coefficient to be sparse.
    
    $$
    \min_{\bm{w}}\frac{1}{2m}\|\bm{y}_{data}-\bm{X}_{data}\bm{w}\|_2^2+\lambda \|\bm{w}\|_1
    $$
    
    or 
    
    $$
    \min_{\bm{w}}\frac{1}{2m}\|\bm{y}_{data}-\bm{X}_{data}\bm{w}\|_2^2\text{, subject to }\|\bm{w}\|_1\leq \eta.
    $$
    
- Elastic net
    
    $$
    \min_{\bm{w}}\frac{1}{2m}\|\bm{y}_{data}-\bm{X}_{data}\bm{w}\|_2^2+\lambda_1 \|\bm{w}\|_1+\frac{\lambda_2}{2}\|\bm{w}\|_2^2
    $$
    
- Total variation
    
    Fit the videlity term and then regularize the function to have sparse gradient with $\|\cdot\|_1$ norm.
    
- Nuclear norm
    
    Fit the videlity term and then regularize the matrix to be low-rank with $\|\cdot\|_*$ norm.
    

### Classification

Linear classifier

The idea is to search for a best hyperplane, and use the hyperplane to cut off between different classes. The criteria for “best” can be different.

- LDA(Linear discriminant analysis)
    
    Assume $p(\bm{x}|y=0)$ and $p(\bm{x}|y=1)$ are both normal distribution, $\mathcal{N}(\bm{\mu_1},\bm{\Sigma_1})$ and $\mathcal{N}(\bm{\mu_1},\bm{\Sigma_1})$. By considering the log-likelihood ratio, we predict the label to be “1” if the following ratio is greater than a threshold $T$.
    
    $$
    (\bm{x}-\bm{\mu_1})^T\bm{\Sigma_1}^{-1}(\bm{x}-\bm{\mu_1})+\ln |\bm{\Sigma_1}|-(\bm{x}-\bm{\mu_0})^T\bm{\Sigma_0}^{-1}(\bm{x}-\bm{\mu_0})-\ln |\bm{\Sigma_0}|>T.
    $$
    
    If we assume $\bm{\Sigma_1}=\bm{\Sigma_2}$, the solution is: if $\bm{x}^T \bm{\Sigma}^{-1}(\bm{\mu_2}-\bm{\mu_1})>c$, with $c=(\bm{\mu_2}-\bm{\mu_1})\bm{\Sigma}^{-1}(\bm{\mu_2}-\bm{\mu_1})$.
    
    **Fisher linear discriminat**
    
    The idea is to find a hyperplane to discriminate class-1 and class-2. Assume the hyperplane is $\bm{w}^T\bm{x}=c$, so the first job it to solve what $\bm{w}$ is and then let $c$ to be the level between two sample mean.
    
    To find a good $\bm{w}$, one can maximize $\max_{\bm{w}} \frac{\bm{w}^T\bm{S_{bg}}\bm{w}}{\bm{w}^T\bm{S_{wg}}\bm{w}}$, where the between group scatter matrix is $\bm{S}_{bg}=(\bm{\mu_1}-\bm{\mu_2})(\bm{\mu_1}-\bm{\mu_2})^T$ and the within group scatter matrix is $\bm{S}_{wg}=\sum_{i}(\bm{x_i}^{j}-\bm{\mu_j})(\bm{x_i}^{j}-\bm{\mu_j})^T$, where $j\in\{1,2\}$. The solution is $\bm{w}=\bm{S}_{wg}^{-1}(\bm{\mu_2}-\bm{\mu_1})$.
    
- Naive Bayes
    
    Eg. To identify “Spam” from email. Use the bag of words to get feature, and then calculate the critical ratio for $\frac{p(x|Spam)p(Spam)}{p(x|Spam^c)p(Spam^c)}$
    
    $p(x|Spam)=p(word_1|Spam)^{k_1}p(word_2|Spam)^{k_2}...p(word_n|Spam)^{k_n}$, and similar for the divisor part.
    
    This is based on the assumption that any of two words are independent, as $p(word_i,word_j|x)=p(word_i|x)p(word_j|x)$.
    
    What parameters should be learnt from training data? $p(x=Spam)$, $p(word_i|Spam)$ and similar for $Spam^c$.
    
    Problem: if some word never shows up, then $p(word_i|Spam)=0$, this would lead to log-likelihood to be $-\infty$.
    
    Solution: 1. delete the $word_i$ from the features. 2. Use Laplacian smoothing - “pseudo count”, use $count(word_i)+\alpha$ to substitute $count(word_i)$, with $\alpha>0$, for all $i$’s. Pros: help with the singular case. Cons: introduce bias, but asymptotic unbias.
    
- SVM
    
    Cons: does not require to use whole dataset as in Naive Bayes and Logistic regression. Instead, it only requires little memory and less computational cost, which means a lot in cases under high dimensional setting. And, SVM is less sensitive to outliers.
    
    SVM is a kind of linear classifier, the criteria to define the “best” hyperplane to make cut off between classes is related to maximizing the margins. The distance from point $\bm{a}$ to the hyperplane $\bm{w}^T\bm{x}+\bm{b}=0$ can be calculated as:
    
    $$
    dist(\bm{a},\mathcal{S})=\frac{1}{\|\bm{w}\|_2}\cdot (\bm{w}^T\bm{a}+\bm{b})
    $$
    
    We first change the class label of class “0” and class “1” to be “-1” and “1”. Then, we solve the optimization problem as follows to maximize the margin of different classes:
    
    $\max_{\bm{w},\bm{b}}\left(\min_{i\in[m]} \frac{y_i(\bm{w}^T\bm{x_i}+b)}{\|\bm{w}\|_2}\right)$
    
    Since for any $k\neq 0$, we have $(\bm{w},b)$ is equivalent to $(k\bm{w},kb)$. 
    
    Then, **if the training set is seperable**, we can fix $\min_{i\in[m]} y_i(\bm{w}^T\bm{x_i}+b)=1$, and then to maximize $\frac{1}{\|\bm{w}\|_2}$. Therefore, $(*)$ is equivalent to 
    
    $$
    \begin{align*}&\max_{\bm{w},b}\quad\frac{1}{\|\bm{w}\|_2}\\&s.t.\quad\quad y_i(\bm{w}^T\bm{x_i}+b)\geq 1\text{, for }\forall i\in[m]\end{align*}
    $$
    
    For **non-seperable training set**, we should add **slack variable** to handle the case, which is as follows
    
    $$
    \begin{align*}&\min_{\bm{w},b}\quad\frac{1}{2}\|\bm{w}\|_2^2+\lambda\sum_{i}\xi_i\\&s.t.\quad \quad y_i(\bm{w}^T\bm{x_i}+b)\geq 1-\xi_i\text{, for }\forall i\in[m]\quad\quad\quad\quad\quad\quad\quad\quad\quad (*)\\&\quad\quad\quad \quad \xi_i\geq 0\text{, for }\forall i\in[m]\end{align*}
    $$
    
    Meanings for support vectors: (1)$\xi_i=0$ (2)$\xi_i\in(0,1]$(3) $\xi_i>1$
    
    The Lagrangian of the problem $(*)$ is 
    
    $L(\bm{w},b,\{\xi_i\},\mu_i,\theta_i)=\frac{1}{2}\|\bm{w}\|_2^2+\lambda\sum_i\xi_i+\sum_i\mu_i(1-\xi_i-y_i(\bm{w}^T\bm{x_i}+b))-\sum_i\theta_i\xi_i$
    
    under constraints $\mu_i\geq 0$, $\theta_i\geq 0$. 
    
    (把不等式全部写成$\leq$, 然后引入dual非负的变量加入原来要最小化的objective function得到Lagrangian)
    
    Primal variables: $\bm{w}, b, \xi_i$’s
    
    Dual vairables: $\mu_i, \theta_i$’s 
    
    对Lagrangian分别对primal 变量求导并且使其为0, i.e. $\frac{\partial L}{\partial \bm{w}}=0$, $\frac{\partial L}{\partial b}=0$, $\frac{\partial L}{\partial \xi_i}=0$. It becomes the following system:
    
    $$
    \begin{align*}\max_{\mu_i's} \quad&-\frac{1}{2}\sum_i\sum_j\mu_i\mu_jy_iy_j\bm{x_i}^T\bm{x_j}+\sum_i\mu_i \\s.t. \quad&0\leq \mu_i\leq C \\&\sum_i \mu_iy_i=0\end{align*}
    $$
    
    For this problem, the optimization is not related to the dimension of $\bm{x_i}$’s and only related to $K(i,j)=\bm{x_i}^T\bm{x_j}$. Another good property is that many $\mu_i=0$, and it is non-trivial only when $\bm{x_i}$ is support vector. Further, as $\bm{w}=\sum_i\mu_iy_i\bm{x_i}$, so the solution of the parameter $\bm{w}$ only depends on a small number of data points who are support vectors. By KKT complementatry and slackness condition:
    
    $\mu_i(1-\xi_i-y_i(\bm{w}^T\bm{x_i}+b))=0\text{, for }\forall i$
    
    Therefore, we have if ${\mu_i}>0$ then $1-\xi_i=y_i(\bm{w}^T\bm{x_i}+b)$. Further, if $\xi_i=0$, then $\bm{x_i}$ is the support vector. If $\xi_i>0$, then $\bm{x_i}$ is an outlier. 
    
    Label prediction: $sgn(\bm{w}^T\bm{x}+b)$
    
    Kernel SVM: change $K(i,j)=\phi(\bm{x_i})^T\phi(\bm{x_j})$, sometimes one can use $K(i,j)=e^{-\gamma\|\bm{x_i}-\bm{x_j}\|_2^2}$
    
    Label prediction: $sgn(\bm{w}^T\phi(\bm{x})+b)$, since in the dual problem, we only need to know $K(\cdot,\cdot)$, so the prediction can be simplified to be $sgn(\sum_i\mu_iy_i K(\bm{x_i},\bm{x})+b)$. **You don’t need to know exactly $\phi(\cdot)$ is, but** $K(\cdot,\cdot)$!
    
- Logistic regression
    
    A linear model $a=\bm{w}^Tx$ along with a sigmoid function $f(a)=\frac{1}{1+e^{-a}}$ are lifting vector $\bm{x}$ to a integer in $(0,1)$. Benifit from good properties of  the sigmoid function that (1) probabilistic interpretation (2) monotonically increasing helps develop classification rules. In conclusion, the model can be described as:
    
    $Prob(y_x=1)=\frac{1}{1+e^{-(\bm{w}^T\bm{x}+b)}}$
    
    and then we have to determine the parameters $(\bm{w},b)$. The parameters can be determinate by maximizing the log-likelihood or cross entropy error as follows:
    
    $Prob(\mathcal{D}|\bm{w},b)=\Pi_{i\in[m]}  \hat{Prob(y_{x_i}=1)}^{y_i}\hat{Prob(y_{x_i}=0)}^{1-y_i}$   
    
    Here, if $y_{x_i}=1$, then the corresponding proportion of $x_i$ is $h^1\cdot (1-h)^0=h$, with $h=\hat{Prob(y_{x_i}=1)}$, we should maximize the $h$.
    
    Here, if $y_{x_i}=0$, then the corresponding proportion of $x_i$ is $h^0\cdot (1-h)^1=1-h$, with $h=\hat{Prob(y_{x_i}=1)}$, we should maximize the $1-h$.
    
    Therefore, the corresponding log-likelihood is:
    
    $log(Prob(\mathcal{D}|\bm{w},b))=\sum_{i\in[m]}  y_i\log\hat{Prob(y_{x_i}=1)}+(1-y_i)\log(1-\hat{Prob(y_{x_i}=1))}$ 
    
    Define the **cross-entropy error** is:
    
    $\mathcal{E}(\bm{w},b)=-\sum_{i\in[m]}  y_i\cdot  \log f(\bm{w},b,x_i)+(1-y_i)\log(1-f(\bm{w},b,x_i))$ 
    
    Here, $f(\bm{w},b,x_i)=\hat{Prob(y_{x_i}=1)}$.
    
    Calculate $\nabla_{\bm{w}} \mathcal{E}(\bm{w})=\sum_{i\in[m]}\left(\frac{1}{1+e^{-\bm{w}^T\bm{x_i}}}-y_i\right)\bm{x_i}$, then we can use Gradient descent or SGD to solve the parameters $\bm{w}$. Here, we rewrite $\bm{x}=(\bm{x},1)$ and $\bm{w}=(\bm{w},b)$. (No closed form solution, so one turn to iterative solver.)
    
    Multi-class with cross-entropy error function:
    
    $\mathcal{E}(\bm{w_1},\bm{w_2},...,\bm{w_k})=-\sum_{i\in[m]}\sum_{j\in[k]} y_{i,j}\log\left(\frac{e^{\bm{w_j}^T\bm{x_i}}}{\sum_{j'\in[k]}e^{\bm{w_{j'}}^T\bm{x_i}}}\right)$
    
    The form of regularized logistic regression (labeling by “-1” and “1”) can be also approximated by a hinge loss function, which makes the loss function similar to that of SVM.
    
    $\min_{\bm{w},b}\sum_i \max(0,1-y_i(\bm{w}^T\bm{x_i}+b))+\frac{\lambda}{2}\|\bm{w}\|_2^2$
    
    This is so-called geometric formulation of hinge loss. 
    
- Perceptron

Non-linear classifier

- kNN (k-neareast neighbor)
    
    The definition for k-nearest neighbor of any given data point $\bm{x}$ is
    
    $knn(\bm{x})=\{nn_1(\bm{x}),nn_2(\bm{x}),...,nn_k(\bm{x})\}$
    
    with $nn_1(\bm{x})=argmin_{i\in[m]}dist(\bm{x},\bm{x_i})$, $nn_2(\bm{x})=argmin_{i\in[m]\setminus \{nn_1(\bm{x})\}}dist(\bm{x},\bm{x_i})$,$nn_2(\bm{x})=argmin_{i\in[m]\setminus \{nn_1(\bm{x}),nn_2(\bm{x})\}}dist(\bm{x},\bm{x_i})$, ...
    
    As $k$ increases, the boundary becomes more smooth. However, the method is nonlinear.
    
    Weakness: “carry” data every time around, and it is also difficult to choose $k$ and the distance. The model is also sensitive to unbalanced data and if the data has many irrelavant features (this can be treated by “feature selection”). 
    
- Decision trees
    
    Entropy
    
    For random varibale $a$, which has different categories, $a_1, a_2,..., a_K$, then the entropy of such variable is:
    
    $H(a)=-\sum_{k\in[K]}Prob(a=a_k)\log(Prob(a=a_k))$
    
    Conditional entropy
    
    If we have two random variables $a, b$. Then, the conditional entropy of $a|b$ is:
    
    $H(a|b)=\sum_{l\in[L]}Prob(b=b_l)\cdot H(a|b=b_l)$
    
    Information gain
    
    The information gain is $I(b;a)=H(a)-H(a|b)$.
    
    - Choose which feature to split: split a tree based on information gains.
    - Choose the threshold value: if one outcome per category based on information gains.
    - The values for the leaves.
    
    What will cause overfit:
    
    - Too much depth. Maxdepth is a hyperparameter, or alternatively, create a very deep tree and prune it later.
    - Training data size is too small.
    - Including irrelevant attributes.
    
    How to prevent overfit:
    
    - Stop when data split is not statistically significant. (based on information gain)
    - Acquire enough training data.
    - Remove irrelevant attributes. (Manually and logically)
    - Grow a full tree and then prune.
    
    Property of decision tree:
    
    - Nonlinear boundaries. Nonlinear regression/classification.
    - High variance due to dependence on the training data.
    - Heuristic training technique: finding the optimzal partition is NP hard.
    - Can has good interpretation.
    - Computationally efficient
    - Can have both numerically and categorical features.
    - Compact representation, non-parametric
    
    Pruning a tree: to minimize the cost complexity by greedily collapse the node in the full tree, as follows:
    
    $C_{\lambda}=\sum_{i=1}^{|T|}error_i(T)+\lambda|T|$
    
    The index $i=1,2,...,|T|$ means the indexes for the leaf nodes. This lead to increasing the error rate the least.
    
- Random forest
- Boosting
- Multilayer perceptron
- CNN

### Clustering

- k-means
    
    Step 0: Initialize $k$ points as estimator for k-cluster centers (”凝结核”)
    
    Step 1: Assign labels to each data point according to the label of nearest cluster center.
    
    Step 2: Re-estimate the k-cluster centers.
    
    Step 3: Stop if the clusters are stable. Otherwise, re-do from step 1.
    
    The mathematical formulation of the above is to minimize the following objective function:
    
    $Loss=\sum_{k\in[K]}\sum_{i\in[m]}r_{k,i}\cdot \|\bm{x_i}-\bm{\mu_k}\|_2^2$, with $r_{k,i}=1$ if $\bm{x_i}$ belongs to cluster $k$, otherwise $r_{k,i}=0$. $\bm{\mu_k}$ is the center of cluster $k$.
    
    How to use k-means ++ to reduce running time?
    
    As the traditional k-means need $\mathcal{O}(mkd)$ calculation, one can choose “better” initialization to reduce running time.
    
    In step 0, allocate less probability to some data points as new cluster center if it is close to already-chosen cluster centers. e.g. $Prob(\bm{\mu_j}=\bm{x_{i_0}})\propto\|\bm{x_{i_0}}-\bm{\mu_i}\|_2^2$ .
    
    How to choose $k$? Draw the relationship of “within cluster error” vs. “k”. Choose $k$ when no much significant “within cluster error” can be reduce if change $k$ to $k+1$.
    
    k-means is an alternating approach, where the objective function decreases at each step. This lead to convergence to local minimum.
    
- Gaussian mixture model
    
    GMM(Gaussian mixture model) is an unsupervised learning approach, which assume the data to be combination of several Gaussian distributions.
    
    $p(x)=\sum_{k\in[K]}w_k\cdot\mathcal{N}(\bm{x}|\bm{\mu_k},\bm{\Sigma_k})$, with $w_k\geq0$ and $\sum_{k\in[K]}w_k=1$.
    
    How to learn the parameters $(\bm{\mu_k},\bm{\Sigma_k},w_k)\text{, for all }k\in[K]$. 
    
    Assume the hidden variable: $z_i\in[K]$ indicates the cluster of $\bm{x_i}$, and $r_{i,k}\in[0,1]$ indicates the probability that $\bm{x}_i$ belongs to cluster $k\in[K]$. The log-likelihood becomes:
    
    $\sum_{k\in[K]}\sum_{i\in[m]}r_{i,k}\left(\log w_k+\log(\bm{x_i}|\bm{\mu_k},\bm{\Sigma_k})\right)$.
    
    If we know the $r_{i,k}$’s along with data $\bm{x_i}$’s, then we can have MLE for parameter $(w_k,\bm{\mu_k},\bm{\Sigma_k})$ as follows:
    
    $w_k=\frac{\sum_i r_{i,k}}{\sum_k\sum_i r_{i,k}}$, $\bm{\mu_k}=\frac{\sum_i r_{i,k}\bm{x_i}}{\sum_i r_{i,k}}$, $\bm{\Sigma_k}=\frac{1}{\sum_i r_{i,k}}\sum_i r_{i,k}(\bm{x_i}-\bm{\mu_k})(\bm{x_i}-\bm{\mu_k})^T$.
    
    If we have the parameter $(w_k,\bm{\mu_k},\bm{\Sigma_k})$, then we can have guess for $r_{i,k}=Prob(\bm{x_i}\text{ belongs to cluster }k)=\frac{p(\bm{x_i}|z_i=k)p(z_i=k)}{p(\bm{x_i})}=\frac{p(\bm{x_i}|z_i=k)p(z_i=k)}{\sum_{k'}p(\bm{x_i}|z_i=k')p(z_i=k')}$.
    
    Therefore, the solution for GMM is an example of EM (expectation maximization) algorithm, as follows:
    
    Step 0: Initialization for parameters $(w_k,\bm{\mu_k},\bm{\Sigma_k})$.
    
    Step 1: Using Bayes rule to calculate
    
    Step 2: Re-estimate the k-cluster centers.
    
    Step 3: Stop if the clusters are stable. Otherwise, re-do from step 1.
    
    From the above process, we can know that GMM is a “soft” k-means and k-means is like a “hard” GMM.
    
    k-mean: more straightforward, less accurate
    
    GMM: more accurate with more information ($\bm{\mu_k}, \bm{\Sigma_k}$)’s, but cost more computation 
    
- DBSCAN

### Embedding

- PCA
    
    PCA assumptions (linearity, orthogonality) are not always appropriate, one can also turn to PCA extensions, such as: manifold learning, kernal PCA, ICA
    
    Assume data $\bm{X}\in\mathbb{R}^{m\times d}$, here we extract principal component of such data, which (1) has $k$ components (2) each components are linear combination of the original features, (3) components are unit and orthogonal to each other. We assume such $C=XP$, with $C\in\mathbb{R}^{n\times k}$ and $P\in\mathbb{R}^{d\times k}$. Therefore, the covariance matrix has relationship as $\bm{\Sigma}_C=P^T\Sigma_{X^TX}P$. Do eigen decomposition for $\Sigma_{X^TX}$, and choose P letting $\Sigma_C$ be the best rank $k$-approximation of $\Sigma_{X^TX}$. Meaning, one should choose $P$ to be the first $k$-eigenvectors of $\Sigma_{X^TX}$.
    
- robust PCA
    
    The robust PCA problem assumes a SPSD matrix $A=L+S$ to be a low-rank SPSD matrix $L$ plus some sparse noise $S$, which is a robust version of traditional PCA.
    
- Graphical models

### Online learning

### Parallel computing