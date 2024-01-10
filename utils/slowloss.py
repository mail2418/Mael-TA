import torch

def ssl_loss_v2(gx, x, mask, S, D):
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
    x=x[:,:minS1,:]
    gx=gx[:,:minS1,:]
    mask=mask[:,:minS1,:]

    # m_one = torch.ones(s0,s1,s2).cuda()
    m_one = torch.ones(s0,minS1,s2)

    # for i in range(0,s1):
    #     tm = mask[i] * (torch.norm(gx[:,i,:]-x[:,i,:]) ** 2)
    #     tum = (1-mask[i]) * (torch.norm(gx[:,i,:]-x[:,i,:]) ** 2)
    #     Lm = Lm + tm
    #     Lum = Lum + tum
    # print(gx.shape)
    # print(x.shape)
    # print(mask.shape)


    tm = (mask * ((gx-x) ** 2)).flatten().sum()
    tum = ((m_one-mask) * ((gx-x) ** 2)).flatten().sum()

    Lm = tm / mask.flatten().sum()
    Lum = tum / (m_one-mask).flatten().sum()

    return (lamb*Lm)+((1-lamb)*Lum)