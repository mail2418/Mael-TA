import torch

def ssl_loss_v2(gx, x, mask, S, D, device) :
    Lm = 0
    Lum = 0
    # loss = 0
    lamb=0.5
    s0,s1,s2 = x.shape
    ss0,ss1,ss2 = gx.shape
    
    # if s1 < ss1:
    #     gx=gx[:,:s1,:]
    # elif s1 > ss2:
    minS1 = min(s1,ss1)

    x_new = x.clone()
    gx_new = gx.clone()
    mask_new = mask.clone()

    x_new=x_new[:,:minS1,:].to(device)
    gx_new=gx_new[:,:minS1,:].to(device)
    mask_new=mask_new[:,:minS1,:].to(device)

    # m_one = torch.ones(s0,s1,s2).cuda()
    m_one = torch.ones(s0,minS1,s2).to(device)

    # for i in range(0,s1):
    #     tm = mask[i] * (torch.norm(gx[:,i,:]-x[:,i,:]) ** 2)
    #     tum = (1-mask[i]) * (torch.norm(gx[:,i,:]-x[:,i,:]) ** 2)
    #     Lm = Lm + tm
    #     Lum = Lum + tum
    # print(gx.shape)
    # print(x.shape)
    # print(mask.shape)

    tm = (mask_new * ((gx_new-x_new)** 2)).flatten().sum()
    tum = ((m_one-mask_new) * ((gx_new-x_new) ** 2)).flatten().sum()

    Lm = tm / mask_new.flatten().sum()
    Lum = tum / (m_one-mask_new).flatten().sum()

    return (lamb*Lm)+((1-lamb)*Lum)