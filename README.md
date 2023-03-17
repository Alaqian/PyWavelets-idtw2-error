# PyWavelets idtw2 error

This was made in response to this post on the PyTorch forums: https://discuss.pytorch.org/t/why-differnce-coeffs-size-wavelets-transform/129636

More information on this thread: https://github.com/PyWavelets/pywt/issues/54

You've got your answer right there:
`coeffs_d_2[1][1].size`:4489
`coeffs_u_4.size`:4624

This happens when you end up with non-integer dimensions when downsampling, and pywt.dwt2 rounds it up.
for example, you have an image and `data.shape = (250, 400)`.

`cA1, (cH1, cV1, cD1) = pywt.dwt2(data, 'haar')` << `cA1.shape = (125, 200)`

`cA2, (cH2, cV2, cD2) = pywt.dwt2(CA1, 'haar')` << `cA2.shape = (63, 100)`

`ca1 =  pywt.dwt2((cA2, (cH2, cV2, cD2)), 'haar')` >> `ca1.shape = (126, 200)`

`data2=  pywt.dwt2((ca1, (cH1, cV1, cD1)), 'haar')` >> `ca1.shape = (126, 200)`, '`cA1.shape = cH1.shape = cv1.shape = cD1.shape = (125, 200)`'

pywt.idwt2 does not like different dimensions of coefficients. I am not familiar with 'db3' but I tried your code with an image of shape `(760, 680)` and printed .shape of the coefficients:

Link to my notebook: https://github.com/Alaqian/PyWavelets-idtw2-error/blob/main/pywt-idwt2-error.ipynb

**Code:**
```
import pywt.data

import cv2

import numpy as np

import matplotlib.pyplot as plt

original = cv2.imread('gigachad.jpg',cv2.IMREAD_GRAYSCALE)
print(original.shape)

coeffs_d_1 = pywt.dwt2(original,'db3')
print(f"coeffs_d_1[1][1].shape = {coeffs_d_1[1][1].shape}")
coeffs_d_2 = pywt.dwt2(coeffs_d_1[0],'db3')
print(f"coeffs_d_2[1][1].shape = {coeffs_d_2[1][1].shape}")
coeffs_d_3 = pywt.dwt2(coeffs_d_2[0],'db3')
print(f"coeffs_d_3[1][1].shape = {coeffs_d_3[1][1].shape}")
coeffs_d_4 = pywt.dwt2(coeffs_d_3[0],'db3')
print(f"coeffs_d_4[1][1].shape = {coeffs_d_4[1][1].shape}")
coeffs_d_5 = pywt.dwt2(coeffs_d_4[0],'db3')
print(f"coeffs_d_5[1][1].shape = {coeffs_d_5[1][1].shape}")
coeffs_d_6 = pywt.dwt2(coeffs_d_5[0],'db3')
print(f"coeffs_d_6[1][1].shape = {coeffs_d_6[1][1].shape}")

coeffs_u_1 = pywt.idwt2((np.empty(coeffs_d_6[0].shape),coeffs_d_6[1]),'db3')
print(f"coeffs_u_1.shape = {coeffs_u_1.shape}")
coeffs_u_2 = pywt.idwt2((coeffs_u_1,coeffs_d_5[1]),'db3')
print(f"coeffs_u_2.shape = {coeffs_u_2.shape}")
coeffs_u_3 = pywt.idwt2((coeffs_u_2,coeffs_d_4[1]),'db3')
print(f"coeffs_u_3.shape = {coeffs_u_3.shape}")
```

**Output:**
> original.shape = (760, 680)
>
> coeffs_d_1[1][1].shape = (382, 342)
>
> coeffs_d_2[1][1].shape = (193, 173)
>
> coeffs_d_3[1][1].shape = (99, 89)
>
> coeffs_d_4[1][1].shape = (52, 47)
>
> coeffs_d_5[1][1].shape = (28, 26)
>
> coeffs_d_6[1][1].shape = (16, 15)
>
> coeffs_u_1.shape = (28, 26)
>
> coeffs_u_2.shape = (52, 48)
>
> ---------------------------------------------------------------------------
>
> ValueError                                Traceback (most recent call last)
>
> Cell In[6], line 29
>
>      27 coeffs_u_2 = pywt.idwt2((coeffs_u_1,coeffs_d_5[1]),'db3')
>
>      28 print(f"coeffs_u_2.shape = {coeffs_u_2.shape}")
>
>
> ---> 29 coeffs_u_3 = pywt.idwt2((coeffs_u_2,coeffs_d_4[1]),'db3')
>
>      30 print(f"coeffs_u_3.shape = {coeffs_u_3.shape}")
>
>      31 coeffs_u_4 = pywt.idwt2((coeffs_u_3,coeffs_d_3[1]),'db3')
>
> 
>
> File ~\miniconda3\envs\ImV\lib\site-packages\pywt\_multidim.py:118, in idwt2(coeffs, wavelet, mode, axes)
>
>     115     raise ValueError("Expected 2 axes")
>
>     117 coeffs = {'aa': LL, 'da': HL, 'ad': LH, 'dd': HH}
>
> --> 118 return idwtn(coeffs, wavelet, mode, axes)
>
> 
>
> File ~\miniconda3\envs\ImV\lib\site-packages\pywt\_multidim.py:280, in idwtn(coeffs, wavelet, mode, axes)
>
>     277     raise ValueError("`coeffs` must contain at least one non-null wavelet "
>
>     278                      "band")
>
>     279 if any(s != coeff_shape for s in coeff_shapes):
>
> --> 280     raise ValueError("`coeffs` must all be of equal size (or None)")
>
>     282 if axes is None:
>
>     283     axes = range(ndim_transform)
>
> 
>
> ValueError: `coeffs` must all be of equal size (or None)

So `coeffs_d_4[1][1].shape = (52, 47)`  and `coeffs_u_2.shape = (52, 48)` which idwt2 does not know how to deal with.
One solution is to check the dimensions of the coefficients and truncate the first row using `coeffs_u_2[1:, :]` but I am not a fan of this solution and the reconstruction results are poorer. The solution that worked for me using `haar` was cropping the original image so that it is a multiple of `2 ** d`, where d is the level of decomposition. In your case, `d = 6`, so your dimensions should be a multiple of `2**6 = 64` if you are using 'haar', but `db3` works differently.
