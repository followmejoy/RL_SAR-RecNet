import torch
from torch.nn.functional import pad
from torch.fft import fft, ifft, fftshift


def ftx(s):
    if s.dim() == 2:
        return fftshift(fft(fftshift(s, 0), dim=0), 0)
    elif s.dim() == 5:
        return fftshift(fft(fftshift(s, 3), dim=3), 3)
    else:
        return fftshift(fft(fftshift(s, 2), dim=2), 2)


def iftx(fs):
    if fs.dim() == 2:
        return fftshift(ifft(fftshift(fs, 0), dim=0), 0)
    elif fs.dim() == 5:
        return fftshift(ifft(fftshift(fs, 3), dim=3), 3)
    else:
        return fftshift(ifft(fftshift(fs, 2), dim=2), 2)


def fty(s):
    if s.dim() == 2:
        return fftshift(fft(fftshift(s, 1), dim=1), 1)
    elif s.dim() == 5:
        return fftshift(fft(fftshift(s, 4), dim=4), 4)
    else:
        return fftshift(fft(fftshift(s, 3), dim=3), 3)


def ifty(fs):
    if fs.dim() == 2:
        return fftshift(ifft(fftshift(fs, 1), dim=1), 1)
    elif fs.dim() == 5:
        return fftshift(ifft(fftshift(fs, 4), dim=4), 4)
    else:
        return fftshift(ifft(fftshift(fs, 3), dim=3), 3)


# thetas为loadmat('thetas.mat')的结果，包含Theta1~Thetas3三个变量，需根据实际的y与x的尺寸来生成
def CSA_echo(imag, thetas):
    Theta1 = thetas[0]
    Theta2 = thetas[1]
    Theta3 = thetas[2]
    if imag.dim() == 5:
        Theta1 = Theta1.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        Theta2 = Theta2.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        Theta3 = Theta3.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    if imag.dim() == 4:
        Theta1 = Theta1.unsqueeze(0).unsqueeze(0)
        Theta2 = Theta2.unsqueeze(0).unsqueeze(0)
        Theta3 = Theta3.unsqueeze(0).unsqueeze(0)

    # imag = pad(imag, (imag.size()[-1]*3//2, imag.size()[-1]*3//2, \
    #                   imag.size()[-2]*3//2, imag.size()[-2]*3//2))
    S1 = ftx(imag) * torch.conj(Theta3)
    S2 = fty(S1) * torch.conj(Theta2)
    S3 = ifty(S2) * torch.conj(Theta1)
    return iftx(S3)


def CSA_imag(echo, thetas):
    Theta1, Theta2, Theta3 = thetas
    if echo.dim() == 5:
        Theta1 = Theta1.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        Theta2 = Theta2.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        Theta3 = Theta3.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    if echo.dim() == 4:
        Theta1 = Theta1.unsqueeze(0).unsqueeze(0)
        Theta2 = Theta2.unsqueeze(0).unsqueeze(0)
        Theta3 = Theta3.unsqueeze(0).unsqueeze(0)
    S1 = ftx(echo) * Theta1
    S2 = fty(S1) * Theta2
    S3 = ifty(S2) * Theta3
    return iftx(S3)  # [..., echo.size()[-2]*3//8 : echo.size()[-2]*5//8, \
    #      echo.size()[-1]*3//8 : echo.size()[-1]*5//8]

def CSA_rangecompression(echo, thetas):
    Theta1, Theta2, Theta3 = thetas
    if echo.dim() == 5:
        Theta1 = Theta1.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    if echo.dim() == 4:
        Theta1 = Theta1.unsqueeze(0).unsqueeze(0)
    S1 = iftx(ftx(echo) * Theta1)

    return S1   #size:[batch_size, 128, 128] image = s(448:575,448:575);