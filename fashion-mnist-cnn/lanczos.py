import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

import torch.nn.functional as F

import gc



# Convert the gradients graph to vector
def to_vector(grads):

    J = torch.cat([g.contiguous().view(-1) for g in grads])

    return(J)

def set_rvec_to(model, vec):

    if vec.dim == 2:
        vec = vec.view(-1)
    prev_ind = 0
    rvec = []
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))

        if param.dim() ==1:
            rvec.append(vec[prev_ind:prev_ind + flat_size].view(param.unsqueeze(0).size()))
        else:
            rvec.append(vec[prev_ind:prev_ind + flat_size].view(param.size()))


        prev_ind += flat_size

    return rvec




####################################################################
#  Compute Non-centralized Covariance Mt sum(g_t*g_t^T)/N-vector Procudct
####################################################################

def eval_Mt_vec_prod(vec, model, dataloader, device,args):
    model.eval() # in evaluation mode, the parameters won't change
#    model.zero_grad() # clears grad for every parameter in the net


    num_params = sum(p.numel() for p in model.parameters())

    Mt = torch.zeros(num_params,1,device=device)

    for i,(data,labels) in enumerate(dataloader):
        if args.arch =='relunet' and args.data == 'mnist':
            inputs = data.reshape(-1, 28*28).to(device)
        elif args.arch =='relunet' and args.data =='cifar10':
            inputs = data.reshape(-1, 32*32*3).to(device)
        else:
            inputs = data.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(inputs)
        #model.zero_grad()
        loss = F.nll_loss(outputs, labels)

        # first order gradient
        grads = torch.autograd.grad(loss, inputs=(model.parameters()))# tuple
        gt = to_vector(grads)

        #print(torch.sum(gt.view(-1,1)*vec).size())

        #grad_v = torch.mm(gt.view(-1,1).t(),vec)
        #Mt= Mt+gt.mul(grad_v).view(-1,1)
        #Mt+= gt.mul(torch.sum(gt.view(-1,1)*vec)).view(-1,1)
        Mt+= gt.mul(torch.mm(gt.view(-1,1).t(),vec)).view(-1,1)


        del gt
        del grads


    Mt=Mt/(i+1)

    return Mt



def Mt_Rop(model,inputs,labels,device,vec):


    #print(inputs.size())
    l = int(len(model.layers)/2)
    batch = inputs.size()[0]
    # Initialization at layer l = 0 correspond to the input layer
    a_l = [inputs] # batch x len(inputs) [5,1,2]
    z_l = [inputs] # [5,1,2]

    # forward propagation
    for i in range(l): # 0,1
        w = model.layers[i*2].weight  # no. (l+1) x no. (l)
        b = model.layers[i*2].bias # no. (1+1)


        z= torch.matmul(a_l[i],w.t())+b  # z_j^{l+1}
        a= model.layers[i*2+1](z)


        a_l.append(a)
        z_l.append(z)



    # backward propagation

    # for the final layer, the error term is dL/dz_i = p_i-y_i
    # https://deepnotes.io/softmax-crossentropy
    #zsum = torch.exp(z).sum(1).unsqueeze(1).expand(-1,z.size()[1])

    p = torch.exp(a_l[-1])


    p1 = p.clone()
    p1[torch.arange(inputs.size()[0]),labels]-=1
    delta = [p1]




    for i in range(l-1,0,-1): #[1,0]

        w = model.layers[(i*2)].weight  # no. (l+1) x no. (l)
        b = model.layers[(i*2)].bias # no. (1+1)
        hp = torch.zeros(z_l[i].size(),device=device)
        hp[z_l[i]>0]=1
        delta.append(hp*(torch.mm(delta[-1],w)))

    delta.reverse()


    g = []
    for i in range(l):
        dL_dw = torch.matmul(delta[i].unsqueeze(1).transpose(1,2),a_l[i].unsqueeze(1))
        dL_db = delta[i]

        g.append(dL_dw.view(batch,-1))
        g.append(dL_db)


    g = torch.cat(g,1)
    #print(g.size())

    g = g.unsqueeze(1)

    Mt =torch.matmul(g.transpose(1,2),(torch.matmul(g,vec))).mean(0)

    del g,dL_dw,dL_db,delta,hp,b,w,p1,z_l,a_l,z,a
    gc.collect()

    #return(torch.matmul(g.transpose(1,2),(torch.matmul(g,vec))).mean(0))

    return Mt




def eval_Mt_vec_prod_with_rop(args,vec, model, dataloader, device):
    """
    Evaluate product of the Hessian of the loss function with a direction vector "vec".
    The product result is saved in the grad of net.
    Args:
        vec: a list of tensor with the same dimensions as "params".
        params: the parameter list of the net (ignoring biases and BN parameters).
        net: model with trained parameters.
        criterion: loss function.
        dataloader: dataloader for the dataset.
        use_cuda: use GPU.
    """
    model.eval() # in evaluation mode, the parameters won't change


    for _ in model.parameters():
        _.requires_grad_(False)

    num_params = sum(p.numel() for p in model.parameters())
    Mt = torch.zeros(num_params,1,device=device)


    for i,(images,labels) in enumerate(dataloader):
        # Move tensors to the configured device

        if args.arch =='relunet' and args.data =='mnist':
            inputs = images.reshape(-1, 28*28).to(device)
        elif args.arch =='relunet' and args.data =='cifar10':
            inputs = images.reshape(-1, 32*32*3).to(device)
        else:
            inputs = images.to(device)


        #inputs = images.to(device)
        #images = images.to(device)
        labels = labels.to(device)

        Mt = Mt+ Mt_Rop(model,inputs,labels,device,vec)



        del inputs,labels
        gc.collect()

    for _ in model.parameters():
            _.requires_grad_(True)


    return Mt/(i+1)










def lanczos_tridiag(args,model,train_loader,device, M,sig2 = 0,isHessian = True):
    """
    Inputs:
        Linear operator H in R^{pxp}
        Number of iterations M.
    Outputs:
        Top M eigenvelues/eigenvectors of H?
    """

    # Create storage for q_mat, alpha,and beta
    # q_mat - orthogonal matrix of decomp
    # alpha - main diagonal of T
    # beta -  off diagonal of T
    num_params = sum(p.numel() for p in model.parameters())
    t_mat = torch.zeros(M, M, device=device)
    q_mat = torch.zeros(num_params,M,device=device)# pxM


    # Initilization
    q_prev = torch.zeros(num_params,1,device=device)
    q = torch.randn(num_params,1,device=device)
    q = q / torch.norm(q, 2) # q_1: norm=1
    r = q
    beta = 1
    beta_prev = 0

    for m in range(M):

        q = r/beta

        if m >0:
            # reothogonoliztion: w=w-VmVm^T w
            V = torch.mm(q_mat[:,:m].t(),q)
            q = q-torch.mm(q_mat[:,:m],V)

        q_mat[:,m].copy_(q.view(-1))

        if isHessian:
            #print('Compute Hessian')
            v = eval_hess_vec_prod(q, model, train_loader, device,args)-sig2*q
        else:
            #print('Compute Non-central Covariance Mt')
            v = eval_Mt_vec_prod(q, model, train_loader, device,args)-sig2*q


        alpha = torch.mm(q.t(),v)

        t_mat[m, m].copy_(alpha.squeeze())


        r = v-alpha*q-beta_prev*q_prev


        beta = torch.norm(r,2)


        if m+1 <M:
            t_mat[m, m+1].copy_(beta.squeeze())
            t_mat[m+1, m].copy_(beta.squeeze())



        # Update curr to prev
        beta_prev = beta
        q_prev = q
        #print('finishing {}-subspace'.format(m))
        del v,alpha
        if m>0:
            del V
        gc.collect()

    # We're done!
    return t_mat,q_mat#eigenvector with p element


def lanczos_tridiag_with_rop(args,model,dataloader,device,M,sig2 = 0,isHessian = True):
    """
    Inputs:
        Linear operator H in R^{pxp}
        Number of iterations M.
        sig2: the value used to shift the absolute values of our eigenvalues so that the lanczos can pick up the desired eigenvalue-eigenvector pairs
    Outputs:
        Top M eigenvelues/eigenvectors of H?
    """
    #torch.cuda.manual_seed(2019)

    # Create storage for q_mat, alpha,and beta
    # q_mat - orthogonal matrix of decomp
    # alpha - main diagonal of T
    # beta -  off diagonal of T
    num_params = sum(p.numel() for p in model.parameters())
    t_mat = torch.zeros(M, M, device=device)
    q_mat = torch.zeros(num_params,M,device=device)# pxM

    # Initilization
    q_prev = torch.zeros(num_params,1,device=device)
    q = torch.randn(num_params,1,device=device)
    q = q / torch.norm(q, 2) # q_1: norm=1
    r = q
    beta = 1
    beta_prev = 0

    for m in range(M):

        q = r/beta

        if m >0:
            # reothogonoliztion: w=w-VmVm^T w
            V = torch.mm(q_mat[:,:m].t(),q)
            q = q-torch.mm(q_mat[:,:m],V)

        q_mat[:,m].copy_(q.view(-1))




        if isHessian:
            #print('Compute Hessian')
            v = eval_hess_vec_prod_with_rop(args,q, model, dataloader, device)-sig2*q
            #print(v.sum())
        else:
            #print('Compute Non-central Covariance Mt')
            v = eval_Mt_vec_prod_with_rop(args,q, model, dataloader, device)-sig2*q


        alpha = torch.mm(q.t(),v)

        t_mat[m, m].copy_(alpha.squeeze())


        r = v-alpha*q-beta_prev*q_prev


        beta = torch.norm(r,2)


        if m+1 <M:
            t_mat[m, m+1].copy_(beta.squeeze())
            t_mat[m+1, m].copy_(beta.squeeze())



        # Update curr to prev
        beta_prev = beta
        q_prev = q
        #print('finishing {}-subspace'.format(m))

        #print('finish lanczos iteration {}'.format(m+1))
        del v,alpha
        if m>0:
            del V
        gc.collect()

    # We're done!
    return (t_mat,q_mat)#eigenvector with p element


#### Lanczos to compute residual Hp = Hf-Mt

def lanczos_tridiag_with_rop_residual(args,model,dataloader,device,M):
    """
    Inputs:
        Linear operator H in R^{pxp}
        Number of iterations M.
    Outputs:
        Top M eigenvelues/eigenvectors of H?
    """
    #torch.cuda.manual_seed(2019)

    # Create storage for q_mat, alpha,and beta
    # q_mat - orthogonal matrix of decomp
    # alpha - main diagonal of T
    # beta -  off diagonal of T
    num_params = sum(p.numel() for p in model.parameters())
    t_mat = torch.zeros(M, M, device=device)
    q_mat = torch.zeros(num_params,M,device=device)# pxM

    # Initilization
    q_prev = torch.zeros(num_params,1,device=device)
    q = torch.randn(num_params,1,device=device)
    q = q / torch.norm(q, 2) # q_1: norm=1
    r = q
    beta = 1
    beta_prev = 0

    for m in range(M):

        q = r/beta

        if m >0:
            # reothogonoliztion: w=w-VmVm^T w
            V = torch.mm(q_mat[:,:m].t(),q)
            q = q-torch.mm(q_mat[:,:m],V)

        q_mat[:,m].copy_(q.view(-1))




        v = eval_Mt_vec_prod_with_rop(args,q, model, dataloader, device)-eval_hess_vec_prod_with_rop(args,q, model, dataloader, device)


        alpha = torch.mm(q.t(),v)

        t_mat[m, m].copy_(alpha.squeeze())


        r = v-alpha*q-beta_prev*q_prev


        beta = torch.norm(r,2)


        if m+1 <M:
            t_mat[m, m+1].copy_(beta.squeeze())
            t_mat[m+1, m].copy_(beta.squeeze())



        # Update curr to prev
        beta_prev = beta
        q_prev = q
        #print('finishing {}-subspace'.format(m))

        print('finish lanczos iteration {}'.format(m+1))
        del v,alpha
        if m>0:
            del V
        gc.collect()

    # We're done!
    return (t_mat,q_mat)#eigenvector with p element
