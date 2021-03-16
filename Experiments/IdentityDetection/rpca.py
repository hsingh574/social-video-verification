#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def thres(x, mu):
    y = np.maximum(x - mu, 0)
    y = y + np.minimum(x + mu, 0)
    return y

def solve_proj2(m, U, lambda1, lambda2):
    """
    solve the problem:
    min_{v, s} 0.5*|m-Uv-s|_2^2 + 0.5*lambda1*|v|^2 + lambda2*|s|_1

    solve the projection by APG
    Parameters
    ----------
    m: nx1 numpy array, vector
    U: nxp numpy array, matrix
    lambda1, lambda2: tuning parameters

    Returns:
    ----------
    v: px1 numpy array, vector
    s: nx1 numpy array, vector
    """
    # intialization
    n, p = U.shape
    v = np.zeros(p)
    s = np.zeros(n)
    I = np.identity(p)
    converged = False
    maxIter = np.inf
    k = 0
    # alternatively update
    UUt = np.linalg.inv(U.transpose().dot(U) + lambda1*I).dot(U.transpose())
    while not converged:
        k += 1
        vtemp = v
        # v = (U'*U + lambda1*I)\(U'*(m-s))
        v = UUt.dot(m - s)
        stemp = s
        s = thres(m - U.dot(v), lambda2)
        stopc = max(np.linalg.norm(v - vtemp), np.linalg.norm(s - stemp))/n
        if stopc < 1e-6 or k > maxIter:
            converged = True

    return v, s



def pcp(M, lam=np.nan, mu=np.nan, factor=1, tol=10**(-7), maxit=1000):
    m, n = M.shape
    unobserved = np.isnan(M)
    M[unobserved] = 0
    S = np.zeros((m,n))
    L = np.zeros((m,n))
    Lambda = np.zeros((m,n)) # the dual variable

    # parameter setting
    if np.isnan(mu):
        mu = 0.25/np.abs(M).mean()
    if np.isnan(lam):
        lam = 1/np.sqrt(max(m,n)) * float(factor)

    # main
    for niter in range(maxit):
        normLS = np.linalg.norm(np.concatenate((S,L), axis=1), 'fro')
        # dS, dL record the change of S and L, only used for stopping criterion

        X = Lambda / mu + M
        # L - subproblem
        # L = argmin_L ||L||_* + <Lambda, M-L-S> + (mu/2) * ||M-L-S||.^2
        # L has closed form solution (singular value thresholding)
        Y = X - S;
        dL = L;
        U, sigmas, V = np.linalg.svd(Y, full_matrices=False);
        rank = (sigmas > 1/mu).sum()
        Sigma = np.diag(sigmas[0:rank] - 1/mu)
        L = np.dot(np.dot(U[:,0:rank], Sigma), V[0:rank,:])
        dL = L - dL

        # S - subproblem
        # S = argmin_S  lam*||S||_1 + <Lambda, M-L-S> + (mu/2) * ||M-L-S||.^2
        # Define element wise softshinkage operator as
        #     softshrink(z; gamma) = sign(z).* max(abs(z)-gamma, 0);
        # S has closed form solution: S=softshrink(Lambda/mu + M - L; lam/mu)
        Y = X - L
        dS = S
        S = thres(Y, lam/mu) # softshinkage operator
        dS = S - dS

        # Update Lambda (dual variable)
        Z = M - S - L
        Z[unobserved] = 0
        Lambda = Lambda + mu * Z;

        # stopping criterion
        RelChg = np.linalg.norm(np.concatenate((dS, dL), axis=1), 'fro') / (normLS + 1)
        if RelChg < tol:
            break

    return L, S, niter, rank



def stoc_rpca(M, burnin,  lambda1=np.nan, lambda2=np.nan):
    m, n = M.shape
    # calculate pcp on burnin samples and find rank r
    Lhat, Shat, niter, r = pcp(M[:,:burnin])
    Uhat, sigmas_hat, Vhat = np.linalg.svd(Lhat)
    if np.isnan(lambda1):
        lambda1 = 1.0/np.sqrt(m)/np.mean(sigmas_hat[:r])
    if np.isnan(lambda2):
        lambda2 = 1.0/np.sqrt(m)

    # initialization
    U = np.random.rand(m, r)
#    Uhat, sigmas_hat, Vhat = np.linalg.svd(Lhat)
#    U = Uhat[:,:r].dot(np.sqrt(np.diag(sigmas_hat[:r])))
    A = np.zeros((r, r))
    B = np.zeros((m, r))
    for i in range(burnin, n):
        mi = M[:, i]
        vi, si = solve_proj2(mi, U, lambda1, lambda2)
        Shat = np.hstack((Shat, si.reshape(m,1)))
        A = A + np.outer(vi, vi)
        B = B + np.outer(mi - si, vi)
        U = update_col(U, A, B, lambda1)
        Lhat = np.hstack((Lhat, U.dot(vi).reshape(m,1)))
        #Lhat = np.hstack((Lhat, (mi - si).reshape(m,1)))

    return Lhat, Shat, r, U

def update_col(U, A, B, lambda1):
    m, r = U.shape
    A = A + lambda1*np.identity(r)
    for j in range(r):
        bj = B[:,j]
        uj = U[:,j]
        aj = A[:,j]
        temp = (bj - U.dot(aj))/A[j,j] + uj
        U[:,j] = temp/max(np.linalg.norm(temp), 1)

    return U
