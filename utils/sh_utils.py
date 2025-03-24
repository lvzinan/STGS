#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import torch
import numpy as np
import math
from math import sqrt
import time

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

kSqrt02_01  = sqrt( 2.0 /  1.0)
kSqrt01_02  = sqrt( 1.0 /  2.0)
kSqrt03_02  = sqrt( 3.0 /  2.0)
kSqrt01_03  = sqrt( 1.0 /  3.0)
kSqrt02_03  = sqrt( 2.0 /  3.0)
kSqrt04_03  = sqrt( 4.0 /  3.0)
kSqrt01_04  = sqrt( 1.0 /  4.0)
kSqrt03_04  = sqrt( 3.0 /  4.0)
kSqrt05_04  = sqrt( 5.0 /  4.0)
kSqrt01_05  = sqrt( 1.0 /  5.0)
kSqrt02_05  = sqrt( 2.0 /  5.0)
kSqrt03_05  = sqrt( 3.0 /  5.0)
kSqrt04_05  = sqrt( 4.0 /  5.0)
kSqrt06_05  = sqrt( 6.0 /  5.0)
kSqrt08_05  = sqrt( 8.0 /  5.0)
kSqrt09_05  = sqrt( 9.0 /  5.0)
kSqrt01_06  = sqrt( 1.0 /  6.0)
kSqrt05_06  = sqrt( 5.0 /  6.0)
kSqrt07_06  = sqrt( 7.0 /  6.0)
kSqrt02_07  = sqrt(02.0 /  7.0)
kSqrt06_07  = sqrt( 6.0 /  7.0)
kSqrt10_07  = sqrt(10.0 /  7.0)
kSqrt12_07  = sqrt(12.0 /  7.0)
kSqrt15_07  = sqrt(15.0 /  7.0)
kSqrt16_07  = sqrt(16.0 /  7.0)
kSqrt01_08  = sqrt( 1.0 /  8.0)
kSqrt03_08  = sqrt( 3.0 /  8.0)
kSqrt05_08  = sqrt( 5.0 /  8.0)
kSqrt07_08  = sqrt( 7.0 /  8.0)
kSqrt09_08  = sqrt( 9.0 /  8.0)
kSqrt05_09  = sqrt( 5.0 /  9.0)
kSqrt08_09  = sqrt( 8.0 /  9.0)
kSqrt01_10  = sqrt( 1.0 / 10.0)
kSqrt03_10  = sqrt( 3.0 / 10.0)
kSqrt07_10  = sqrt( 7.0 / 10.0)
kSqrt09_10  = sqrt( 9.0 / 10.0)
kSqrt01_12  = sqrt( 1.0 / 12.0)
kSqrt07_12  = sqrt( 7.0 / 12.0)
kSqrt11_12  = sqrt(11.0 / 12.0)
kSqrt01_14  = sqrt( 1.0 / 14.0)
kSqrt03_14  = sqrt( 3.0 / 14.0)
kSqrt15_14  = sqrt(15.0 / 14.0)
kSqrt04_15  = sqrt( 4.0 / 15.0)
kSqrt07_15  = sqrt( 7.0 / 10.0)
kSqrt14_15  = sqrt(14.0 / 15.0)
kSqrt16_15  = sqrt(16.0 / 15.0)
kSqrt01_16  = sqrt( 1.0 / 16.0)
kSqrt03_16  = sqrt( 3.0 / 16.0)
kSqrt07_16  = sqrt( 7.0 / 16.0)
kSqrt15_16  = sqrt(15.0 / 16.0)
kSqrt01_18  = sqrt( 1.0 / 18.0)
kSqrt01_24  = sqrt( 1.0 / 24.0)
kSqrt03_25  = sqrt( 3.0 / 25.0)
kSqrt09_25  = sqrt( 9.0 / 25.0)
kSqrt14_25  = sqrt(14.0 / 25.0)
kSqrt16_25  = sqrt(16.0 / 25.0)
kSqrt18_25  = sqrt(18.0 / 25.0)
kSqrt21_25  = sqrt(21.0 / 25.0)
kSqrt24_25  = sqrt(24.0 / 25.0)
kSqrt03_28  = sqrt( 3.0 / 28.0)
kSqrt05_28  = sqrt( 5.0 / 28.0)
kSqrt01_30  = sqrt( 1.0 / 30.0)
kSqrt01_32  = sqrt( 1.0 / 32.0)
kSqrt03_32  = sqrt( 3.0 / 32.0)
kSqrt15_32  = sqrt(15.0 / 32.0)
kSqrt21_32  = sqrt(21.0 / 32.0)
kSqrt11_36  = sqrt(11.0 / 36.0)
kSqrt35_36  = sqrt(35.0 / 36.0)
kSqrt01_50  = sqrt( 1.0 / 50.0)
kSqrt03_50  = sqrt( 3.0 / 50.0)
kSqrt21_50  = sqrt(21.0 / 50.0)
kSqrt15_56  = sqrt(15.0 / 56.0)
kSqrt01_60  = sqrt( 1.0 / 60.0)
kSqrt01_112 = sqrt( 1.0 / 112.0)
kSqrt03_112 = sqrt( 3.0 / 112.0)
kSqrt15_112 = sqrt(15.0 / 112.0)

def dp(n, a, b):
    # 确保输入是张量
    a = a[:,:,:n]  # 取前 n 个元素
    b = b[:,:,:n]  # 取前 n 个元素

    result = (a * b).sum(dim=2)  # 计算点积并求和
    return result

def RotateSH(R, n, sh):

    sh_rotated = torch.zeros([sh.shape[0],sh.shape[1], 16]).cuda()

    sh_rotated[:,:, 0] = sh[:,:, 0].clone()

    if (n < 1):
        return

    R1 = torch.tensor([
        [R[1, 1], R[2, 1], R[0, 1]],
        [R[1, 2], R[2, 2], R[0, 2]],
        [R[1, 0], R[2, 0], R[0, 0]]
    ]).cuda()
    sh1 = R1.unsqueeze(0).repeat(sh.shape[1], 1, 1).unsqueeze(0).repeat(sh.shape[0], 1, 1, 1).cuda()

    sh_rotated[:,:, 1] = dp(3, sh.clone(), sh1[:,:, 0].clone())
    sh_rotated[:,:, 2] = dp(3, sh.clone(), sh1[:,:, 1].clone())
    sh_rotated[:,:, 3] = dp(3, sh.clone(), sh1[:,:, 2].clone())

    if (n < 2):
        return sh_rotated


    sh2 = torch.zeros((sh.shape[0],sh.shape[1], 5, 5)).cuda()
    sh122=sh1[:,:, 2, 2]
    sh100=sh1[:,:, 0, 0]
    sh120=sh1[:,:, 2, 0]
    sh102=sh1[:,:, 0, 2]
    sh121=sh1[:,:, 2, 1]
    sh112=sh1[:,:, 1, 2]
    sh101=sh1[:,:, 0, 1]
    sh111=sh1[:,:, 1, 1]
    sh110=sh1[:,:, 1, 0]

    sh2[:,:, 0,0] = kSqrt01_04 * (
                (sh122 * sh100 + sh120 * sh102) + (sh102 * sh120 + sh100 * sh122))
    sh2[:,:, 0,1] = (sh121 * sh100 + sh101 * sh120)
    sh2[:,:, 0,2] = kSqrt03_04 * (sh121 * sh101 + sh101 * sh121)
    sh2[:,:, 0,3] = (sh121 * sh102 + sh101 * sh122)
    sh2[:,:, 0,4] = kSqrt01_04 * (
                (sh122 * sh102 - sh120 * sh100) + (sh102 * sh122 - sh100 * sh120))

    sh_rotated[:,:, 4] = dp(5, sh.clone(), sh2[:,:, 0].clone())

    sh2[:,:, 1,0] = kSqrt01_04 * (
                (sh112 * sh100 + sh110 * sh102) + (sh102 * sh110 + sh100 * sh112))
    sh2[:,:, 1,1] = sh111 * sh100 + sh101 * sh110
    sh2[:,:, 1,2] = kSqrt03_04 * (sh111 * sh101 + sh101 * sh111)
    sh2[:,:, 1,3] = sh111 * sh102 + sh101 * sh112
    sh2[:,:, 1,4] = kSqrt01_04 * (
                (sh112 * sh102 - sh110 * sh100) + (sh102 * sh112 - sh100 * sh110))

    sh_rotated[:,:, 5] = dp(5, sh.clone(), sh2[:,:, 1].clone())

    sh2[:,:, 2,0] = kSqrt01_03 * (sh112 * sh110 + sh110 * sh112) - kSqrt01_12 * (
                (sh122 * sh120 + sh120 * sh122) + (sh102 * sh100 + sh100 * sh102))
    sh2[:,:, 2,1] = kSqrt04_03 * sh111 * sh110 - kSqrt01_03 * (sh121 * sh120 + sh101 * sh100)
    sh2[:,:, 2,2] = sh111 * sh111 - kSqrt01_04 * (sh121 * sh121 + sh101 * sh101)
    sh2[:,:, 2,3] = kSqrt04_03 * sh111 * sh112 - kSqrt01_03 * (sh121 * sh122 + sh101 * sh102)
    sh2[:,:, 2,4] = kSqrt01_03 * (sh112 * sh112 - sh110 * sh110) - kSqrt01_12 * (
                (sh122 * sh122 - sh120 * sh120) + (sh102 * sh102 - sh100 * sh100))

    sh_rotated[:,:, 6] = dp(5, sh.clone(), sh2[:,:, 2].clone())

    sh2[:,:, 3,0] = kSqrt01_04 * (
                (sh112 * sh120 + sh110 * sh122) + (sh122 * sh110 + sh120 * sh112))
    sh2[:,:, 3,1] = sh111 * sh120 + sh121 * sh110
    sh2[:,:, 3,2] = kSqrt03_04 * (sh111 * sh121 + sh121 * sh111)
    sh2[:,:, 3,3] = sh111 * sh122 + sh121 * sh112
    sh2[:,:, 3,4] = kSqrt01_04 * (
                (sh112 * sh122 - sh110 * sh120) + (sh122 * sh112 - sh120 * sh110))

    sh_rotated[:,:, 7] = dp(5, sh.clone(), sh2[:,:, 3].clone())

    sh2[:,:, 4,0] = kSqrt01_04 * (
                (sh122 * sh120 + sh120 * sh122) - (sh102 * sh100 + sh100 * sh102))
    sh2[:,:, 4,1] = (sh121 * sh120 - sh101 * sh100)
    sh2[:,:, 4,2] = kSqrt03_04 * (sh121 * sh121 - sh101 * sh101)
    sh2[:,:, 4,3] = (sh121 * sh122 - sh101 * sh102)
    sh2[:,:, 4,4] = kSqrt01_04 * (
                (sh122 * sh122 - sh120 * sh120) - (sh102 * sh102 - sh100 * sh100))

    sh_rotated[:,:, 8] = dp(5, sh.clone(), sh2[:,:, 4].clone())

    if (n < 3):
        return sh_rotated

    sh3 = torch.zeros((sh.shape[0],sh.shape[1], 7, 7)).cuda()
    
    sh200 = sh2[:,:, 0,0]
    sh204 = sh2[:,:, 0,4]
    sh240 = sh2[:,:, 4,0]
    sh201 = sh2[:,:, 0,1]
    sh241 = sh2[:,:, 4,1]
    sh244 = sh2[:,:, 4,4]
    sh202 = sh2[:,:, 0,2]
    sh242 = sh2[:,:, 4,2]
    sh203 = sh2[:,:, 0,3]
    sh243 = sh2[:,:, 4,3]
    sh210 = sh2[:,:, 1,0]
    sh214 = sh2[:,:, 1,4]
    sh230 = sh2[:,:, 3,0]
    sh234 = sh2[:,:, 3,4]
    sh211 = sh2[:,:, 1,1]
    sh231 = sh2[:,:, 3,1]
    sh212 = sh2[:,:, 1,2]
    sh232 = sh2[:,:, 3,2]
    sh213 = sh2[:,:, 1,3]
    sh233 = sh2[:,:, 3,3]
    sh220 = sh2[:,:, 2,0]
    sh221 = sh2[:,:, 2,1]
    sh222 = sh2[:,:, 2,2]
    sh223 = sh2[:,:, 2,3]
    sh224 = sh2[:,:, 2,4]

    sh3[:,:, 0, 0] = kSqrt01_04 * (
                (sh122 * sh200 + sh120 * sh204) + (sh102 * sh240 + sh100 * sh244))
    sh3[:,:, 0, 1] = kSqrt03_02 * (sh121 * sh200 + sh101 * sh240)
    sh3[:,:, 0, 2] = kSqrt15_16 * (sh121 * sh201 + sh101 * sh241)
    sh3[:,:, 0, 3] = kSqrt05_06 * (sh121 * sh202 + sh101 * sh242)
    sh3[:,:, 0, 4] = kSqrt15_16 * (sh121 * sh203 + sh101 * sh243)
    sh3[:,:, 0, 5] = kSqrt03_02 * (sh121 * sh204 + sh101 * sh244)
    sh3[:,:, 0, 6] = kSqrt01_04 * (
                (sh122 * sh204 - sh120 * sh200) + (sh102 * sh244 - sh100 * sh240))

    sh_rotated[:,:, 9] = dp(7, sh.clone(), sh3[:,:, 0].clone())

    sh3[:,:, 1, 0] = kSqrt01_06 * (sh112 * sh200 + sh110 * sh204) + kSqrt01_06 * (
                (sh122 * sh210 + sh120 * sh214) + (sh102 * sh230 + sh100 * sh234))
    sh3[:,:, 1, 1] = sh111 * sh200 + (sh121 * sh210 + sh101 * sh230)
    sh3[:,:, 1, 2] = kSqrt05_08 * sh111 * sh201 + kSqrt05_08 * (sh121 * sh211 + sh101 * sh231)
    sh3[:,:, 1, 3] = kSqrt05_09 * sh111 * sh202 + kSqrt05_09 * (sh121 * sh212 + sh101 * sh232)
    sh3[:,:, 1, 4] = kSqrt05_08 * sh111 * sh203 + kSqrt05_08 * (sh121 * sh213 + sh101 * sh233)
    sh3[:,:, 1, 5] = sh111 * sh204 + (sh121 * sh214 + sh101 * sh234)
    sh3[:,:, 1, 6] = kSqrt01_06 * (sh112 * sh204 - sh110 * sh200) + kSqrt01_06 * (
                (sh122 * sh214 - sh120 * sh210) + (sh102 * sh234 - sh100 * sh230))

    sh_rotated[:,:, 10] = dp(7, sh.clone(), sh3[:,:, 1].clone())

    time_7js_s = time.time()
    sh3[:,:, 2, 0] = kSqrt04_15 * (sh112 * sh210 + sh110 * sh214) + kSqrt01_05 * (
                sh102 * sh220 + sh100 * sh224) - kSqrt01_60 * (
                            (sh122 * sh200 + sh120 * sh204) - (
                                sh102 * sh240 + sh100 * sh244))
    sh3[:,:, 2, 1] = kSqrt08_05 * sh111 * sh210 + kSqrt06_05 * sh101 * sh220 - kSqrt01_10 * (
                sh121 * sh200 - sh101 * sh240)
    sh3[:,:, 2, 2] = sh111 * sh211 + kSqrt03_04 * sh101 * sh221 - kSqrt01_16 * (
                sh121 * sh201 - sh101 * sh241)
    sh3[:,:, 2, 3] = kSqrt08_09 * sh111 * sh212 + kSqrt02_03 * sh101 * sh222 - kSqrt01_18 * (
                sh121 * sh202 - sh101 * sh242)
    sh3[:,:, 2, 4] = sh111 * sh213 + kSqrt03_04 * sh101 * sh223 - kSqrt01_16 * (
                sh121 * sh203 - sh101 * sh243)
    sh3[:,:, 2, 5] = kSqrt08_05 * sh111 * sh214 + kSqrt06_05 * sh101 * sh224 - kSqrt01_10 * (
                sh121 * sh204 - sh101 * sh244)
    sh3[:,:, 2, 6] = kSqrt04_15 * (sh112 * sh214 - sh110 * sh210) + kSqrt01_05 * (
                sh102 * sh224 - sh100 * sh220) - kSqrt01_60 * (
                            (sh122 * sh204 - sh120 * sh200) - (
                                sh102 * sh244 - sh100 * sh240))

    sh_rotated[:,:, 11] = dp(7, sh.clone(), sh3[:,:, 2].clone())

    sh3[:,:, 3, 0] = kSqrt03_10 * (sh112 * sh220 + sh110 * sh224) - kSqrt01_10 * (
                (sh122 * sh230 + sh120 * sh234) + (sh102 * sh210 + sh100 * sh214))
    sh3[:,:, 3, 1] = kSqrt09_05 * sh111 * sh220 - kSqrt03_05 * (sh121 * sh230 + sh101 * sh210)
    sh3[:,:, 3, 2] = kSqrt09_08 * sh111 * sh221 - kSqrt03_08 * (sh121 * sh231 + sh101 * sh211)
    sh3[:,:, 3, 3] = sh111 * sh222 - kSqrt01_03 * (sh121 * sh232 + sh101 * sh212)
    sh3[:,:, 3, 4] = kSqrt09_08 * sh111 * sh223 - kSqrt03_08 * (sh121 * sh233 + sh101 * sh213)
    sh3[:,:, 3, 5] = kSqrt09_05 * sh111 * sh224 - kSqrt03_05 * (sh121 * sh234 + sh101 * sh214)
    sh3[:,:, 3, 6] = kSqrt03_10 * (sh112 * sh224 - sh110 * sh220) - kSqrt01_10 * (
                (sh122 * sh234 - sh120 * sh230) + (sh102 * sh214 - sh100 * sh210))

    sh_rotated[:,:, 12] = dp(7, sh.clone(), sh3[:,:, 3].clone())

    sh3[:,:, 4, 0] = kSqrt04_15 * (sh112 * sh230 + sh110 * sh234) + kSqrt01_05 * (
                sh122 * sh220 + sh120 * sh224) - kSqrt01_60 * (
                            (sh122 * sh240 + sh120 * sh244) + (
                                sh102 * sh200 + sh100 * sh204))
    sh3[:,:, 4, 1] = kSqrt08_05 * sh111 * sh230 + kSqrt06_05 * sh121 * sh220 - kSqrt01_10 * (
                sh121 * sh240 + sh101 * sh200)
    sh3[:,:, 4, 2] = sh111 * sh231 + kSqrt03_04 * sh121 * sh221 - kSqrt01_16 * (
                sh121 * sh241 + sh101 * sh201)
    sh3[:,:, 4, 3] = kSqrt08_09 * sh111 * sh232 + kSqrt02_03 * sh121 * sh222 - kSqrt01_18 * (
                sh121 * sh242 + sh101 * sh202)
    sh3[:,:, 4, 4] = sh111 * sh233 + kSqrt03_04 * sh121 * sh223 - kSqrt01_16 * (
                sh121 * sh243 + sh101 * sh203)
    sh3[:,:, 4, 5] = kSqrt08_05 * sh111 * sh234 + kSqrt06_05 * sh121 * sh224 - kSqrt01_10 * (
                sh121 * sh244 + sh101 * sh204)
    sh3[:,:, 4, 6] = kSqrt04_15 * (sh112 * sh234 - sh110 * sh230) + kSqrt01_05 * (
                sh122 * sh224 - sh120 * sh220) - kSqrt01_60 * (
                            (sh122 * sh244 - sh120 * sh240) + (
                                sh102 * sh204 - sh100 * sh200))

    sh_rotated[:,:, 13] = dp(7, sh.clone(), sh3[:,:, 4].clone())

    sh3[:,:, 5, 0] = kSqrt01_06 * (sh112 * sh240 + sh110 * sh244) + kSqrt01_06 * (
                (sh122 * sh230 + sh120 * sh234) - (sh102 * sh210 + sh100 * sh214))
    sh3[:,:, 5, 1] = sh111 * sh240 + (sh121 * sh230 - sh101 * sh210)
    sh3[:,:, 5, 2] = kSqrt05_08 * sh111 * sh241 + kSqrt05_08 * (sh121 * sh231 - sh101 * sh211)
    sh3[:,:, 5, 3] = kSqrt05_09 * sh111 * sh242 + kSqrt05_09 * (sh121 * sh232 - sh101 * sh212)
    sh3[:,:, 5, 4] = kSqrt05_08 * sh111 * sh243 + kSqrt05_08 * (sh121 * sh233 - sh101 * sh213)
    sh3[:,:, 5, 5] = sh111 * sh244 + (sh121 * sh234 - sh101 * sh214)
    sh3[:,:, 5, 6] = kSqrt01_06 * (sh112 * sh244 - sh110 * sh240) + kSqrt01_06 * (
                (sh122 * sh234 - sh120 * sh230) - (sh102 * sh214 - sh100 * sh210))

    sh_rotated[:,:, 14] = dp(7, sh.clone(), sh3[:,:, 5].clone())

    sh3[:,:, 6, 0] = kSqrt01_04 * (
                (sh122 * sh240 + sh120 * sh244) - (sh102 * sh200 + sh100 * sh204))
    sh3[:,:, 6, 1] = kSqrt03_02 * (sh121 * sh240 - sh101 * sh200)
    sh3[:,:, 6, 2] = kSqrt15_16 * (sh121 * sh241 - sh101 * sh201)
    sh3[:,:, 6, 3] = kSqrt05_06 * (sh121 * sh242 - sh101 * sh202)
    sh3[:,:, 6, 4] = kSqrt15_16 * (sh121 * sh243 - sh101 * sh203)
    sh3[:,:, 6, 5] = kSqrt03_02 * (sh121 * sh244 - sh101 * sh204)
    sh3[:,:, 6, 6] = kSqrt01_04 * (
                (sh122 * sh244 - sh120 * sh240) - (sh102 * sh204 - sh100 * sh200))

    sh_rotated[:,:, 15] = dp(7, sh.clone(), sh3[:,:, 6].clone())

    if (n < 4):
        return sh_rotated

    return

def eval_sh(deg, sh, dirs, R=None, trans=False):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff


    if trans:
        sh = RotateSH(R, deg, sh)


    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])

    return result

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5